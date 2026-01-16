"""EAGLE model attention implementation."""

from typing import TYPE_CHECKING

import torch.nn as nn
from torch import Tensor

from .base import BaseAttention
from ...configs import ModelArgs
from batchspec.profiler import rope_compute_timer, attention_compute_timer

if TYPE_CHECKING:
    from batchspec.backends.base.page_table import PageTable


class EAGLEAttention(BaseAttention):
    """Attention implementation for EAGLE speculative decoding.
    
    EAGLE uses a modified attention mechanism that processes concatenated
    input embeddings and hidden states for speculation.
    
    Args:
        config: Model configuration
    """
    
    def __init__(self, config: ModelArgs):
        # First initialize base class (sets up standard wqkv)
        super().__init__(config)
        
        # Override wqkv for EAGLE's concatenated input (2*dim input)
        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        self.wqkv = nn.Linear(config.dim * 2, total_head_dim, bias=config.qkv_bias)
        self.attn_kernel = None
    
    def forward(
        self,
        x: Tensor,
        qo_indptr: Tensor,
        position_ids: Tensor,
        kv_page_table: "PageTable",
    ) -> Tensor:
        """Forward pass through EAGLE attention layer.
        
        Args:
            x: Concatenated input tensor of shape (batch_size, seq_len, 2*dim)
            qo_indptr: Index pointer for Query/Output tokens
            position_ids: Position IDs for RoPE
            kv_page_table: Page table for KV cache
            
        Returns:
            Output tensor of shape (batch_size, seq_len, dim)
        """
        bsz, seqlen, _ = x.shape
        
        # Split QKV from concatenated input
        q, k, v = self._split_qkv(self.wqkv(x))
        
        # Note: EAGLE doesn't use Q/K normalization
        # Reshape for attention computation
        q = q.view(bsz * seqlen, self.n_head, self.head_dim)
        k = k.view(bsz * seqlen, self.n_local_heads, self.head_dim)
        v = v.contiguous().view(bsz * seqlen, self.n_local_heads, self.head_dim)
        
        # Apply RoPE
        with rope_compute_timer():
            q, k = self._apply_rope(q, k, position_ids)
        
        # Update KV cache
        kv_cache = self.kv_cache.update(k, v, qo_indptr, kv_page_table)
        
        # Compute attention
        with attention_compute_timer():
            y = self.attn_kernel.run(q, kv_cache)
        
        # Reshape and project output
        y = y.contiguous().view(bsz, seqlen, self.dim)
        y = self.wo(y)
        return self._maybe_all_reduce_output(y)
