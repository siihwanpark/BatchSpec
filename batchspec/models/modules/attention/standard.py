"""Standard attention implementation."""

from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor

from .base import BaseAttention, AttentionMixin
from ...configs import ModelArgs
from batchspec.profiler import attention_compute_timer, rope_compute_timer


class StandardAttention(BaseAttention, AttentionMixin):
    """Standard attention.
    
    Args:
        config: Model configuration
    """
    
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        
        # Attention kernels to be set during setup
        self.attn_prefill = None
        self.attn_decode = None
    
    def forward(
        self,
        x: Tensor,
        input_pos: Tensor,  # Can be position_ids or offsets depending on model
        kv_append_indptr: Tensor,
        kv_page_indices: Tensor,
        kv_page_indptr: Tensor,
        kv_page_lastlen: Tensor,
        attn_type: str = "decode",
    ) -> Tensor:
        """Forward pass through attention layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            input_pos: Position IDs or offsets
            kv_append_indptr: Indices for KV cache appending
            kv_page_indices: Page indices for KV cache
            kv_page_indptr: Page indirection pointers
            kv_page_lastlen: Last length of each page
            attn_type: Attention type ("prefill", "verify", or "decode")
            
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
            q, k = self._apply_rope(q, k, kv_append_indptr, input_pos)
        
        # Update KV cache
        kv_cache = self.kv_cache.update(
            k, v, kv_append_indptr, 
            kv_page_indices, kv_page_indptr, kv_page_lastlen
        )
        
        # Compute attention based on type
        attn_kernels = {
            "prefill": self.attn_prefill,
            "decode": self.attn_decode,
        }
        
        with attention_compute_timer():
            y = self._compute_attention(q, kv_cache, attn_type, attn_kernels)
        
        # Reshape and project output (unflatten the heads)
        y = y.contiguous().view(bsz, seqlen, self.dim)
        y = self.wo(y)
        return self._maybe_all_reduce_output(y)
