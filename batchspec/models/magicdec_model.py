"""MagicDec transformer implementation with StreamingLLM drafter."""

from typing import Any, TYPE_CHECKING

import torch
from torch import Tensor

from .configs import ModelArgs
from .modules import MagicDecAttention, StandardKVCache, StreamingKVCache, setup_rope_function
from .base_model import BaseTransformerBlock, BaseTransformer

if TYPE_CHECKING:
    from batchspec.backends.base.page_table import PageTable


class MagicDecTransformer(BaseTransformer):
    """Transformer model for speculative decoding via MagicDec with StreamingLLM drafter."""
    
    def __init__(self, config: ModelArgs):
        super().__init__(config)
    
    def _get_attention_class(self):
        """Return MagicDecAttention class."""
        return MagicDecAttention
    
    def _get_block_class(self):
        """Return BaseTransformerBlock class."""
        return BaseTransformerBlock
    
    def setup_caches(self,
        num_pages: int,
        page_size: int,
        attn_kernel: Any,
        batch_size: int,
        stream_budget: int,
    ):
        """Setup KV caches and attention kernels.
        
        Args:
            num_pages: Total number of pages to allocate
            page_size: Size of each page (tokens per page)
            attn_kernel: Attention kernel
        """
        # Setup RoPE function (uses offsets, not position_ids)
        rope_func = setup_rope_function(self.config, use_position_ids=False)
        
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

            max_num_draft_pages_per_request = (stream_budget + page_size - 1) // page_size
            attn.draft_kv_cache = StreamingKVCache(
                max_num_pages=batch_size * max_num_draft_pages_per_request,
                page_size=page_size,
                n_heads=self.config.n_local_heads,
                head_dim=self.config.head_dim,
                dtype=dtype,
            )

            # Register RoPE-related variables
            attn.register_rope_variables(batch_size, page_size, stream_budget)


    def forward(
        self,
        input_ids: Tensor,
        position_offsets: Tensor,
        qo_indptr: Tensor,
        kv_page_table: "PageTable",
        draft: bool = False,
    ) -> Tensor:
        """Forward pass through the transformer.
        
        Args:
            input_ids: Input token indices of shape (batch_size, seq_len) or (nnz)
            position_offsets: Position offsets (start position id) or (nnz)
            qo_indptr: Index pointer for Query/Output tokens
            kv_page_table: Page table for KV cache
            draft: Whether to use draft KV cache
        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size)
        """
        # Embed tokens
        if input_ids.dim() != 1:
            # Input shape: (batch_size, seq_len)
            bsz, seqlen = input_ids.shape
            x = self.tok_embeddings(input_ids.view(bsz * seqlen))
        else:
            # Input shape: (nnz)
            x = self.tok_embeddings(input_ids)
        
        # Forward pass through transformer layers
        for layer in self.layers:
            x = layer(x, 
                position_offsets=position_offsets,
                qo_indptr=qo_indptr,
                kv_page_table=kv_page_table,
                draft=draft,
            )
        
        # Final normalization and projection
        x = self.norm(x)
        logits = self.output(x)

        if input_ids.dim() != 1:
            logits = logits.view(bsz, seqlen, -1)
        
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