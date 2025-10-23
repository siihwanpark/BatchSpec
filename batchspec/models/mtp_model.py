"""Multi-Token Prediction (MTP) transformer implementation."""

from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor

from .configs import ModelArgs, LoRAConfig
from .modules import RoPEMixin, MTPAttention, StandardKVCache, SamplerHead
from .base_model import BaseTransformer, GatedLoRATransformerBlock


class MTPTransformer(BaseTransformer, RoPEMixin):
    """Multi-Token Prediction transformer with gated LoRA and sampling.
    This model supports gated LoRA and includes a sampler module for multi-token prediction.
    
    Args:
        config: Model configuration
        lora_config: LoRA configuration for adaptation
    """
    
    def __init__(self, config: ModelArgs, lora_config: LoRAConfig):
        self.lora_config = lora_config
        super().__init__(config)
        
        # Sampler Head for token prediction
        self.sampler_head = SamplerHead(config)
    
    def _build_layers(self) -> nn.ModuleList:
        """Build transformer layers with LoRA support.
        
        Returns:
            ModuleList of LoRA-enabled transformer blocks
        """
        attention_class = self._get_attention_class()
        
        layers = []
        for _ in range(self.config.n_layer):
            attention = attention_class(self.config, self.lora_config)
            block = GatedLoRATransformerBlock(self.config, attention, self.lora_config)
            layers.append(block)
            
        return nn.ModuleList(layers)
    
    def _get_attention_class(self):
        """Return MTPAttention class."""
        return MTPAttention
    
    def setup_caches(self, num_pages: int, page_size: int, **kwargs):
        """Setup KV caches and attention kernels.
        
        Args:
            num_pages: Total number of pages to allocate
            page_size: Size of each page (tokens per page)
            **kwargs: Additional arguments (ignored)
        """
        # Setup RoPE kernels (uses position IDs)
        self._setup_rope_kernels(use_position_ids=True)
        
        # Determine dtype for cache
        dtype = (
            self.output.weight.dtype 
            if self.output.weight.dtype == torch.float16 
            else torch.bfloat16
        )
        
        # Setup attention kernels and KV caches for each layer
        for layer in self.layers:
            attn = layer.attention
            
            # Initialize KV cache
            attn.kv_cache = StandardKVCache(
                num_pages, page_size,
                self.config.n_local_heads,
                self.config.head_dim,
                dtype
            )
            
            # Register MTP-specific attention kernels
            attn.attn_prefill = torch.ops.mylib.attn_prefill
            attn.attn_draft = torch.ops.mylib.attn_draft
            attn.attn_draft_and_verify = torch.ops.mylib.attn_draft_and_verify
            
            # Register RoPE kernel
            attn.rope = torch.ops.mylib.rope
    
    def get_tok_embeddings(self) -> nn.Embedding:
        """Get token embedding layer.
        
        Returns:
            Token embedding module
        """
        return self.tok_embeddings
    
    def get_lm_head(self) -> nn.Linear:
        """Get language model head (output projection).
        
        Returns:
            Output linear layer
        """
        return self.output
    
    def _forward_lm_head(
        self, 
        x: Tensor,
        gate_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass through language model head with optional masking.
        
        Args:
            x: Hidden states of shape (batch_size, seq_len, dim)
            gate_mask: Optional mask to select non-masked tokens only
            
        Returns:
            Logits tensor
        """
        if gate_mask is None:
            # Standard forward pass
            return self._maybe_all_gather_logits(self.output(x))
        else:
            # Forward only non-masked tokens for efficiency
            bsz, _, dim = x.shape
            
            # Find non-masked token positions
            non_mask_indices = (gate_mask.view(-1) == 0).nonzero(as_tuple=True)[0]
            
            # Select and process only non-masked tokens
            x_selected = x.reshape(-1, dim).index_select(0, non_mask_indices)
            logits = self._maybe_all_gather_logits(self.output(x_selected))
            
            # Reshape back to include batch dimension
            return logits.reshape(bsz, non_mask_indices.numel() // bsz, -1)
    
    def forward(
        self,
        idx: Tensor,
        gate_mask: Optional[Tensor],
        position_ids: Tensor,
        kv_append_indptr: Tensor,
        kv_page_indices: Tensor,
        kv_page_indptr: Tensor,
        kv_page_lastlen: Tensor,
        attn_type: str = "prefill"
    ) -> tuple[Tensor, Tensor]:
        """Forward pass through the MTP transformer.
        
        Args:
            idx: Input token indices of shape (batch_size, seq_len)
            gate_mask: Optional mask for LoRA gating
            position_ids: Position IDs for RoPE
            kv_append_indptr: Indices for KV cache appending
            kv_page_indices: Page indices for KV cache
            kv_page_indptr: Page indirection pointers
            kv_page_lastlen: Last length of each page
            attn_type: Type of attention ("prefill", "draft", or "draft_and_verify")
            
        Returns:
            Tuple of (logits, hidden_states)
        """
        # Embed tokens
        x = self.tok_embeddings(idx)
        
        # Forward pass through transformer layers with gated LoRA
        for layer in self.layers:
            x = layer(
                x, gate_mask, position_ids,
                kv_append_indptr, kv_page_indices,
                kv_page_indptr, kv_page_lastlen,
                attn_type=attn_type
            )
        
        # Final normalization
        x = self.norm(x)
        
        # Generate logits
        logits = self._forward_lm_head(x, gate_mask)
        return logits, x
    
    def sampler_forward(self, idx: Tensor, hidden_states: Tensor) -> Tensor:
        """Forward pass through sampler head for multi-token prediction.
        
        Args:
            idx: Previous token indices
            hidden_states: Current hidden states
            
        Returns:
            Predicted next token indices
        """
        # Get embeddings of previous tokens
        prev_embeds = self.tok_embeddings(idx)
        
        # Concatenate with current hidden states
        sampler_inputs = torch.cat([prev_embeds, hidden_states], dim=-1)
        
        # Forward pass through sampler head
        sampler_hidden_states = self.sampler_head(sampler_inputs)
        
        # Get local logits
        sampler_logits_local = self.output(sampler_hidden_states)
        
        # Handle distributed training
        if self.process_group is None:
            # Single process: return argmax
            return sampler_logits_local.argmax(dim=-1)
        else:
            # Multi-process: all-gather across all ranks and select global maximum
            val_l, idx_l = sampler_logits_local.max(dim=-1, keepdim=True)
            
            # Adjust indices for vocabulary partitioning
            idx_l = idx_l + self.vocab_start
            
            # Gather values and indices from all ranks
            world = dist.get_world_size(self.process_group)
            vals_g = [torch.empty_like(val_l) for _ in range(world)]
            idxs_g = [torch.empty_like(idx_l) for _ in range(world)]
            
            dist.all_gather(vals_g, val_l, group=self.process_group)
            dist.all_gather(idxs_g, idx_l, group=self.process_group)
            
            # Concatenate gathered tensors
            vals_cat = torch.cat(vals_g, dim=-1)  # [B, 1, world]
            idxs_cat = torch.cat(idxs_g, dim=-1)  # [B, 1, world]
            
            # Select the index with maximum value across all ranks
            best = vals_cat.argmax(dim=-1, keepdim=True)  # [B, 1, 1]
            tok = torch.gather(idxs_cat, -1, best)  # [B, 1, 1]
            
            return tok[..., 0]  # [B, 1]

    @classmethod
    def from_name(cls, name: str, lora_config: LoRAConfig):
        """Create MTP transformer model from configuration name.
        
        Args:
            name: Model configuration name
            lora_config: LoRA configuration
        """
        from .configs import get_config
        config = get_config(name)
        return cls(config, lora_config)
