"""Standard transformer implementation."""

from typing import Any, Dict, Optional

import torch
from torch import Tensor

from .configs import ModelArgs
from .modules import RoPEMixin, StandardAttention, StandardKVCache
from .base_model import BaseTransformerBlock, BaseTransformer
from batchspec.backends.base.page_table import PageTable


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
    
    def setup_caches(self, num_pages: int, page_size: int, attn_kernel: Any):
        """Setup KV caches and attention kernels.
        
        Args:
            num_pages: Total number of pages to allocate
            page_size: Size of each page (tokens per page)
            attn_kernel: Attention kernel
        """
        # Setup RoPE function (uses offsets, not position_ids)
        rope_func = self._setup_rope_kernels(use_position_ids=False)
        
        # Determine dtype for cache
        dtype = (
            self.output.weight.dtype 
            if self.output.weight.dtype == torch.float16 
            else torch.bfloat16
        )
        
        # Setup attention kernels and KV caches for each layer
        for layer in self.layers:
            attn = layer.attention
            
            # Assign RoPE function and attention kernels
            attn.rope = rope_func
            attn.attn_kernel = attn_kernel

            # Initialize KV cache
            attn.kv_cache = StandardKVCache(
                max_num_pages=num_pages,
                page_size=page_size,
                n_heads=self.config.n_local_heads,
                head_dim=self.config.head_dim,
                dtype=dtype,
            )


    def forward(
        self,
        input_ids: Tensor,
        position_offsets: Tensor,
        qo_indptr: Tensor,
        kv_page_table: PageTable,
    ) -> Tensor:
        """Forward pass through the transformer.
        
        Args:
            input_ids: Input token indices of shape (batch_size, seq_len)
            position_offsets: Position offsets (start position id)
            qo_indptr: Index pointer for Query/Output tokens
            kv_page_table: Page table for KV cache
            
        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size)
        """
        # Embed tokens
        x = self.tok_embeddings(input_ids)
        
        # Forward pass through transformer layers
        for layer in self.layers:
            x = layer(x, 
                position_offsets=position_offsets,
                qo_indptr=qo_indptr,
                kv_page_table=kv_page_table,
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