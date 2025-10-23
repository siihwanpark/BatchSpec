"""EAGLE transformer implementation with unified chain/standard modes."""

from typing import Optional, Union, List

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor

from .configs import ModelArgs
from .modules import RoPEMixin, StandardAttention, EAGLEAttention, StandardKVCache, RMSNorm, FeedForward
from .base_model import BaseTransformerBlock, BaseTransformer


class EAGLEModel(nn.Module):
    """EAGLE speculation module for draft token generation.
    
    Processes target model hidden states and input embeddings to
    generate draft tokens for speculative decoding.
    
    Args:
        config: Configuration for EAGLE module
        use_chain_mode: Whether to use chain mode (different attention patterns)
    """
    
    def __init__(self, config: ModelArgs, use_chain_mode: bool = False):
        super().__init__()
        
        self.config = config
        self.use_chain_mode = use_chain_mode
        
        # Projection layer for target hidden states
        # Projects concatenated hidden states from multiple layers
        if hasattr(config, "target_hidden_size"):
            self.fc = nn.Linear(
                config.target_hidden_size * 3, 
                config.dim, 
                bias=False
            )
        else:
            self.fc = nn.Linear(
                config.dim * 3, 
                config.dim, 
                bias=False
            )
        
        # Normalization for hidden states and input embeddings
        self.hidden_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.input_norm = RMSNorm(config.dim, eps=config.norm_eps)
        
        # EAGLE attention layer
        self.attn = EAGLEAttention(config, use_chain_mode=use_chain_mode)
        
        # Post-attention processing
        self.post_attn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.feed_forward = FeedForward(config)
        
        # Output projection
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.draft_vocab_size, bias=False)

        # Vocabulary mapping buffers for draft-to-target token conversion
        d2t = torch.zeros((config.draft_vocab_size), dtype=torch.long)
        t2d = torch.zeros((config.vocab_size), dtype=torch.long)
        self.register_buffer('draft_to_target', d2t)
        self.register_buffer('target_to_draft', t2d)
        
        # Distributed training support
        self.process_group = None
        self.world_size = None
        self.rank = None

    def forward(
        self,
        hidden_states: Tensor,
        input_embeds: Tensor,
        input_pos: Union[Tensor, tuple],
        kv_append_indptr: Tensor,
        kv_page_indices: Tensor,
        kv_page_indptr: Tensor,
        kv_page_lastlen: Tensor,
        attn_type: str = "prefill"
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        """Forward pass through EAGLE module.
        
        Args:
            hidden_states: Hidden states from target model (concatenated from multiple layers)
            input_embeds: Input token embeddings
            input_pos: Position information (varies by mode)
            kv_append_indptr: Indices for KV cache appending
            kv_page_indices: Page indices for KV cache
            kv_page_indptr: Page indirection pointers
            kv_page_lastlen: Last length of each page
            attn_type: Type of attention computation
            
        Returns:
            Logits tensor, or tuple of (logits, hidden_states) for chain mode
        """
        # Project hidden states if dimension mismatch
        if hidden_states.shape[-1] != input_embeds.shape[-1]:
            hidden_states = self.fc(hidden_states)
        
        # Self-Attention with residual
        residual = hidden_states
        hidden_states = self.hidden_norm(hidden_states)
        input_embeds = self.input_norm(input_embeds)
        
        # Concatenate for EAGLE attention (expects 2*dim input)
        attn_input = torch.cat([input_embeds, hidden_states], dim=-1)
        
        hidden_states = self.attn(
            attn_input, input_pos, kv_append_indptr,
            kv_page_indices, kv_page_indptr, kv_page_lastlen,
            attn_type
        )
        hidden_states = residual + hidden_states

        # Feed Forward with residual
        residual = hidden_states
        hidden_states = self.post_attn_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        # Generate logits
        logits = self.output(self.norm(hidden_states))
        
        # Gather logits for distributed training
        if self.process_group is not None:
            gathered_logits = [
                torch.empty_like(logits) 
                for _ in range(self.world_size)
            ]
            dist.all_gather(gathered_logits, logits, group=self.process_group)
            logits = torch.cat(gathered_logits, dim=-1)

        # Return both logits and hidden states for chain mode
        if self.use_chain_mode:
            return logits, hidden_states
        else:
            return logits


class EAGLETransformer(BaseTransformer, RoPEMixin):
    """EAGLE-augmented transformer for speculative decoding.
    
    Combines a target model with an EAGLE module for efficient
    multi-token speculation.
    
    Args:
        config: Target model configuration
        eagle_config: EAGLE module configuration
        use_chain_mode: Whether to use chain mode
    """
    
    def __init__(
        self,
        config: ModelArgs,
        eagle_config: ModelArgs,
        use_chain_mode: bool = False
    ):
        super().__init__(config)
        self.use_chain_mode = use_chain_mode
        
        # EAGLE speculation module
        self.eagle = EAGLEModel(eagle_config, use_chain_mode=use_chain_mode)
    
    def _get_attention_class(self):
        """Return StandardAttention class for target model."""
        return StandardAttention
    
    def _get_block_class(self):
        """Return BaseTransformerBlock class."""
        return BaseTransformerBlock
    
    def setup_caches(self, num_pages: int, page_size: int):
        """Setup KV caches and attention kernels.
        
        Args:
            num_pages: Total number of pages to allocate
            page_size: Size of each page (tokens per page)
        """
        # Setup RoPE kernels
        # Chain mode uses offsets, standard mode uses position IDs
        self._setup_rope_kernels(use_position_ids=(not self.use_chain_mode))
        
        # Determine dtype for cache
        dtype = (
            self.output.weight.dtype 
            if self.output.weight.dtype == torch.float16 
            else torch.bfloat16
        )
        
        # Setup target model attention
        for layer in self.layers:
            attn = layer.attention
            
            # Initialize KV cache
            attn.kv_cache = StandardKVCache(
                num_pages, page_size,
                self.config.n_local_heads,
                self.config.head_dim,
                dtype
            )
            
            # Register target-specific attention kernels
            attn.attn_prefill = torch.ops.mylib.target_prefill_attn
            attn.attn_verify = torch.ops.mylib.target_verify_attn
            attn.rope = torch.ops.mylib.rope
            
            # Store whether we use position IDs for this attention
            attn.use_position_ids = (not self.use_chain_mode)
        
        # Setup EAGLE module attention
        eagle_attn = self.eagle.attn
        
        eagle_attn.kv_cache = StandardKVCache(
            num_pages, page_size,
            self.eagle.config.n_local_heads,
            self.eagle.config.head_dim,
            dtype
        )
        
        eagle_attn.prefill_attn = torch.ops.mylib.eagle_prefill_attn
        eagle_attn.rope = torch.ops.mylib.rope
        eagle_attn.use_position_ids = (not self.use_chain_mode)
        
        # Set mode-specific kernels
        if self.use_chain_mode:
            eagle_attn.decode_1_attn = torch.ops.mylib.eagle_decode_1_attn
            eagle_attn.decode_2_attn = torch.ops.mylib.eagle_decode_2_attn
        else:
            eagle_attn.init_speculate_attn = torch.ops.mylib.eagle_init_speculate_attn
            eagle_attn.sub_speculate_attn = torch.ops.mylib.eagle_sub_speculate_attn
    
    def forward(
        self,
        idx: Tensor,
        input_pos: Tensor,
        kv_append_indptr: Tensor,
        kv_page_indices: Tensor,
        kv_page_indptr: Tensor,
        kv_page_lastlen: Tensor,
        return_hidden_states: bool = False,
        attn_type: str = "prefill"
    ) -> Union[Tensor, tuple[Tensor, List[Tensor]]]:
        """Forward pass through target model.
        
        Args:
            idx: Input token indices
            input_pos: Position information
            kv_append_indptr: Indices for KV cache appending
            kv_page_indices: Page indices for KV cache
            kv_page_indptr: Page indirection pointers
            kv_page_lastlen: Last length of each page
            return_hidden_states: Whether to return intermediate hidden states
            attn_type: Type of attention
            
        Returns:
            Logits, or tuple of (logits, hidden_states) if return_hidden_states=True
        """
        if return_hidden_states:
            hidden_states = []
        
        # Embed tokens
        x = self.tok_embeddings(idx)
        
        # Process through transformer layers
        for i, layer in enumerate(self.layers):
            x = layer(
                x, input_pos, kv_append_indptr,
                kv_page_indices, kv_page_indptr, kv_page_lastlen,
                attn_type=attn_type,
                use_position_ids=(not self.use_chain_mode)
            )
            
            # Collect hidden states from specific layers for EAGLE
            if return_hidden_states and i in [2, len(self.layers)//2, len(self.layers)-3]:
                hidden_states.append(x)
        
        # Final normalization and projection
        x = self.norm(x)
        logits = self.output(x)
        
        # Gather logits for distributed training
        logits = self._gather_logits(logits)
        
        if return_hidden_states:
            return logits, hidden_states
        
        return logits
    
    def eagle_forward(
        self,
        hidden_states: Tensor,
        idx: Tensor,
        input_pos: Tensor,
        kv_append_indptr: Tensor,
        kv_page_indices: Tensor,
        kv_page_indptr: Tensor,
        kv_page_lastlen: Tensor,
        attn_type: str = "prefill"
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        """Forward pass through EAGLE module.
        
        Args:
            hidden_states: Hidden states from target model
            idx: Input token indices
            input_pos: Position information
            kv_append_indptr: Indices for KV cache appending
            kv_page_indices: Page indices for KV cache
            kv_page_indptr: Page indirection pointers
            kv_page_lastlen: Last length of each page
            attn_type: Type of attention
            
        Returns:
            EAGLE output (logits, or logits+hidden_states for chain mode)
        """
        input_embeds = self.tok_embeddings(idx)
        return self.eagle(
            hidden_states, input_embeds, input_pos,
            kv_append_indptr, kv_page_indices, 
            kv_page_indptr, kv_page_lastlen,
            attn_type
        )
    
    @classmethod
    def from_name(
        cls,
        target_name: str,
        drafter_name: str,
        use_chain_mode: bool = False
    ):
        """Create EAGLE model from configuration names.
        
        Args:
            target_name: Target model configuration name
            drafter_name: EAGLE module configuration name
            use_chain_mode: Whether to use chain mode
            
        Returns:
            EAGLETransformer instance
        """
        from .configs import get_config
        
        target_config = get_config(target_name)
        eagle_config = get_config(drafter_name)
        
        return cls(target_config, eagle_config, use_chain_mode)
