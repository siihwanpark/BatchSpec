"""Standard transformer implementation."""

from typing import Optional

import torch
from torch import Tensor

from .configs import ModelArgs
from .modules import RoPEMixin, StandardAttention, StandardKVCache
from .base_model import BaseTransformerBlock, BaseTransformer


class StandardTransformer(BaseTransformer, RoPEMixin):
    """Standard transformer model for autoregressive generation.
    
    Supports prefill, verify, and decode attention patterns.
    
    Args:
        config: Model configuration
    """
    
    def __init__(self, config: ModelArgs):
        super().__init__(config)
    
    def _get_attention_class(self):
        """Return StandardAttention class."""
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
        # Setup RoPE kernels (uses offsets, not position_ids)
        self._setup_rope_kernels(use_position_ids=False)
        
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

            # Register RoPE kernel
            attn.rope = torch.ops.mylib.rope
            
            # Register attention kernels
            attn.attn_prefill = torch.ops.mylib.attn_prefill
            attn.attn_decode = torch.ops.mylib.attn_decode 
            
            
    def forward(
        self,
        idx: Tensor,
        input_pos: Tensor,
        kv_append_indptr: Tensor,
        kv_page_indices: Tensor,
        kv_page_indptr: Tensor,
        kv_page_lastlen: Tensor,
        attn_type: str = "decode"
    ) -> Tensor:
        """Forward pass through the transformer.
        
        Args:
            idx: Input token indices of shape (batch_size, seq_len)
            input_pos: Position offsets for RoPE
            kv_append_indptr: Indices for KV cache appending
            kv_page_indices: Page indices for KV cache
            kv_page_indptr: Page indirection pointers
            kv_page_lastlen: Last length of each page
            attn_type: Type of attention ("prefill", "verify", or "decode")
            
        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size)
        """
        # Embed tokens
        x = self.tok_embeddings(idx)
        
        # Forward pass through transformer layers
        for layer in self.layers:
            x = layer(
                x, input_pos, kv_append_indptr,
                kv_page_indices, kv_page_indptr, kv_page_lastlen,
                attn_type=attn_type
            )
        
        # Final normalization and projection
        x = self.norm(x)
        logits = self.output(x)
        
        # All-gather logits across all ranks for tensor parallel
        return self._maybe_all_gather_logits(logits)

    @classmethod
    def from_name(cls, name: str):
        """Create Standard transformer model from configuration name.
        
        Args:
            name: Model configuration name
            
        Returns:
            Standard transformer model instance
        """
        from .configs import get_config
        config = get_config(name)
        return cls(config)