"""EAGLE transformer implementation with unified chain/tree modes.

References:
- https://github.com/SafeAILab/EAGLE/blob/main/eagle/model/cnets.py
"""

from typing import Union, List, Any, Optional, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor

from .configs import ModelArgs
from .modules import StandardAttentionWithNonCausalSupport, EAGLEAttention, StandardKVCache, RMSNorm, FeedForward, setup_rope_function
from .base_model import BaseTransformerBlock, BaseTransformer

if TYPE_CHECKING:
    from batchspec.backends.base.page_table import PageTable


class EAGLEModel(nn.Module):
    """EAGLE speculation module for draft token generation.
    
    Processes target model hidden states and input embeddings to
    generate draft tokens for speculative decoding.
    
    Args:
        config: Configuration for EAGLE module
    """
    
    def __init__(self, config: ModelArgs):
        super().__init__()
        
        self.config = config
        
        # Projection layer for target hidden states
        # Projects concatenated hidden states from multiple layers
        if config.target_hidden_size is not None:
            self.fc = nn.Linear(config.target_hidden_size * 3, config.dim, bias=False)
        else:
            self.fc = nn.Linear(config.dim * 3, config.dim, bias=False)
        
        # Normalization for hidden states and input embeddings
        self.hidden_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.input_norm = RMSNorm(config.dim, eps=config.norm_eps)
        
        # EAGLE attention layer
        self.attn = EAGLEAttention(config)
        
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

    
    def setup_target_id_to_draft_id_mapping(self):
        """Setup the mapping from target token IDs to draft token IDs after loading the state dict.
            (Custom buffer; not included in the original EAGLE implementation.)
        """
        print("Setting up target_to_draft_id mapping...")
        device = self.target_to_draft.device
        t2d_id = torch.full((self.config.vocab_size,), -1, dtype=torch.long, device=device)
        t2d_id[self.target_to_draft] = torch.arange(self.config.draft_vocab_size, dtype=torch.long, device=device)
        self.target_to_draft_id = t2d_id
        

    def _maybe_all_gather_logits(self, logits: Tensor) -> Tensor:
        """All-gather logits across all ranks for tensor parallel.
        
        Args:
            logits: Local logits tensor
            
        Returns:
            All-gathered logits from all ranks
        """
        if self.process_group is None:
            return logits

        gathered_logits = [
            torch.empty_like(logits) 
            for _ in range(self.world_size)
        ]
        dist.all_gather(gathered_logits, logits, group=self.process_group)
        return torch.cat(gathered_logits, dim=-1)

    def forward(
        self,
        input_embeds: Tensor,
        hidden_states: Tensor,
        *args,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass through EAGLE module.
        
        Args:
            input_embeds: Input token embeddings
            hidden_states: Hidden states from target model (concatenated from multiple layers)
            *args, **kwargs: Additional arguments for attention
            
        Returns:
            tuple of (logits, hidden_states)
        """
        # Project hidden states if dimension mismatch
        if hidden_states.shape[-1] != input_embeds.shape[-1]:
            hidden_states = self.fc(hidden_states)
        
        # Self-Attention with residual
        residual = hidden_states
        hidden_states = self.hidden_norm(hidden_states)
        input_embeds = self.input_norm(input_embeds)
        
        # Concatenate for EAGLE attention (expects 2*dim input)
        hidden_states = torch.cat([input_embeds, hidden_states], dim=-1)
        
        hidden_states = self.attn(hidden_states, *args, **kwargs)
        hidden_states = residual + hidden_states

        # Feed Forward with residual
        residual = hidden_states
        hidden_states = self.post_attn_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        # Generate logits
        logits = self.output(self.norm(hidden_states))
        
        # Gather logits for distributed training
        logits = self._maybe_all_gather_logits(logits)

        # Return both logits and hidden states for chain mode
        return logits, hidden_states

    
    def convert_draft_to_target(self, draft_tokens: Tensor) -> Tensor:
        return draft_tokens + self.draft_to_target[draft_tokens]
    
    def convert_target_to_draft(self, target_tokens: Tensor) -> Tensor:
        if not hasattr(self, 'target_to_draft_id'):
            self.setup_target_id_to_draft_id_mapping()
        return self.target_to_draft_id[target_tokens]


class EAGLETransformer(BaseTransformer):
    """EAGLE-augmented transformer for speculative decoding.
    
    Combines a target model with an EAGLE module for efficient speculative decoding.
    
    Args:
        config: Target model configuration
        eagle_config: EAGLE module configuration
    """
    
    def __init__(
        self,
        config: ModelArgs,
        eagle_config: ModelArgs,
    ):
        super().__init__(config)
        
        # EAGLE speculation module
        self.eagle = EAGLEModel(eagle_config)
    
    def _get_attention_class(self):
        """Return StandardAttentionWithNonCausalSupport class for target model."""
        return StandardAttentionWithNonCausalSupport
    
    def _get_block_class(self):
        """Return BaseTransformerBlock class."""
        return BaseTransformerBlock
    
    def setup_caches(self,
        target_num_pages: int,
        eagle_num_pages: int,
        page_size: int,
        causal_attn_kernel: Any,
        eagle_attn_kernel: Any,
        non_causal_attn_kernel: Optional[Any] = None
    ):
        """Setup KV caches and attention kernels.
        
        Args:
            target_num_pages: Total number of pages to allocate for target model
            eagle_num_pages: Total number of pages to allocate for EAGLE module
            page_size: Size of each page (tokens per page)
            causal_attn_kernel: Causal attention kernel for target model
            eagle_attn_kernel: Attention kernel for EAGLE module
            non_causal_attn_kernel: Non-causal attention kernel for target model (Not used in chain mode)
        """
        # Setup RoPE functions
        target_rope_func = setup_rope_function(self.config, use_position_ids=True)
        eagle_rope_func = setup_rope_function(self.eagle.config, use_position_ids=True)
        
        # Determine dtype for cache
        dtype = (
            self.output.weight.dtype 
            if self.output.weight.dtype == torch.float16 
            else torch.bfloat16
        )
        
        # Setup target model attention
        for layer in self.layers:
            attn = layer.attention
            
            attn.rope = target_rope_func
            attn.causal_attn_kernel = causal_attn_kernel
            attn.non_causal_attn_kernel = non_causal_attn_kernel
            attn.kv_cache = StandardKVCache(
                max_num_pages=target_num_pages,
                page_size=page_size,
                n_heads=self.config.n_local_heads,
                head_dim=self.config.head_dim,
                dtype=dtype
            )
        
        # Setup EAGLE module attention
        eagle_attn = self.eagle.attn
        eagle_attn.kv_cache = StandardKVCache(
            max_num_pages=eagle_num_pages,
            page_size=page_size,
            n_heads=self.eagle.config.n_local_heads,
            head_dim=self.eagle.config.head_dim,
            dtype=dtype
        )
        
        eagle_attn.rope = eagle_rope_func
        eagle_attn.attn_kernel = eagle_attn_kernel
    
    def forward(
        self,
        input_ids: Tensor,
        qo_indptr: Tensor,
        position_ids: Tensor,
        kv_page_table: "PageTable",
        causal: bool = True,
    ) -> Union[Tensor, tuple[Tensor, List[Tensor]]]:
        """Forward pass through target model.
        
        Args:
            input_ids: Input token indices
            qo_indptr: Index pointer for Query/Output tokens
            position_ids: Position IDs for RoPE
            kv_page_table: Page table for KV cache
            causal: Whether to use causal attention
            
        Returns:
            Logits, or tuple of (logits, hidden_states) if return_hidden_states=True
        """
        # Embed tokens
        if input_ids.dim() != 1:
            # Input shape: (batch_size, seq_len)
            bsz, seqlen = input_ids.shape
            x = self.tok_embeddings(input_ids.view(bsz * seqlen))
        else:
            # Input shape: (nnz)
            x = self.tok_embeddings(input_ids)
        
        # Process through transformer layers
        hidden_states = []
        for i, layer in enumerate(self.layers):
            x = layer(x,
                qo_indptr=qo_indptr,
                position_ids=position_ids,
                kv_page_table=kv_page_table,
                causal=causal,
            )
            
            # Collect hidden states from specific layers for EAGLE
            if i in [2, len(self.layers)//2, len(self.layers)-3]:
                hidden_states.append(x)
        hidden_states = torch.cat(hidden_states, dim=-1)

        # Final normalization and projection
        x = self.norm(x)
        logits = self.output(x)

        if input_ids.dim() != 1:
            logits = logits.view(bsz, seqlen, -1)
            hidden_states = hidden_states.view(bsz, seqlen, -1)

        # Gather logits for distributed training
        logits = self._maybe_all_gather_logits(logits)
        return logits, hidden_states
    
    def eagle_forward(
        self,
        input_ids: Tensor,
        hidden_states: Tensor,
        qo_indptr: Tensor,
        position_ids: Tensor,
        kv_page_table: "PageTable",
    ) -> tuple[Tensor, Tensor]:
        """Forward pass through EAGLE module.
        
        Args:
            input_ids: Input token indices [bsz, seq_len]
            hidden_states: Hidden states from target model [bsz, seq_len, hidden_size * 3]
            qo_indptr: Index pointer for Query/Output tokens [bsz + 1]
            position_ids: Position IDs for RoPE [bsz * seq_len]
            kv_page_table: Page table for KV cache
            
        Returns:
            EAGLE output (logits, hidden_states)
        """
        input_embeds = self.tok_embeddings(input_ids)
        return self.eagle(
            input_embeds=input_embeds,
            hidden_states=hidden_states,
            qo_indptr=qo_indptr,
            position_ids=position_ids,
            kv_page_table=kv_page_table,
        )
    
    @classmethod
    def from_name(
        cls,
        target_name: str,
        eagle_name: str,
    ):
        """Create EAGLE model from configuration names.
        
        Args:
            target_name: Target model configuration name
            eagle_name: EAGLE module configuration name
            
        Returns:
            EAGLETransformer instance
        """
        from .configs import get_config
        
        target_config = get_config(target_name)
        eagle_config = get_config(eagle_name)
        
        return cls(target_config, eagle_config)
