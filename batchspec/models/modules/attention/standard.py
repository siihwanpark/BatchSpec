"""Standard attention implementation."""

from typing import Optional, TYPE_CHECKING

import torch
import torch.distributed as dist
from torch import Tensor

from .base import BaseAttention
from ...configs import ModelArgs
from batchspec.profiler import attention_compute_timer, rope_compute_timer

if TYPE_CHECKING:
    from batchspec.backends.base.page_table import PageTable


class StandardAttention(BaseAttention):
    """Standard attention.
    
    Args:
        config: Model configuration
    """
    
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        
        # Attention kernel to be set during setup
        self.attn_kernel = None
    
    def forward(
        self,
        x: Tensor,
        position_offsets: Tensor,
        qo_indptr: Tensor,
        kv_page_table: "PageTable",
    ) -> Tensor:
        """Forward pass through attention layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            position_offsets: position offsets
            qo_indptr: Index pointer for Query/Output tokens
            kv_page_table: Page table for KV cache
            
        Returns:
            Output tensor of shape (batch_size, seq_len, dim)
        """
        bsz, seqlen, _ = x.shape
        
        # Split QKV
        q, k, v = self._split_qkv(self.wqkv(x))
        
        # Apply Q/K normalization
        q = self.q_norm(q.view(bsz, seqlen, self.n_head, self.head_dim))
        k = self.k_norm(k.view(bsz, seqlen, self.n_local_heads, self.head_dim))

        # Reshape for attention computation (flatten the heads)
        q = q.view(bsz * seqlen, self.n_head, self.head_dim)
        k = k.view(bsz * seqlen, self.n_local_heads, self.head_dim)
        v = v.contiguous().view(bsz * seqlen, self.n_local_heads, self.head_dim)
        
        # Apply RoPE
        with rope_compute_timer():
            q, k = self._apply_rope(q, k, qo_indptr, position_offsets)
        
        # Update KV cache
        kv_cache = self.kv_cache.update(k, v, qo_indptr, kv_page_table)
        
        # Compute attention
        with attention_compute_timer():
            y = self.attn_kernel.run(q, kv_cache)
        
        # Reshape and project output (unflatten the heads)
        y = y.contiguous().view(bsz, seqlen, self.dim)
        y = self.wo(y)
        return self._maybe_all_reduce_output(y)
