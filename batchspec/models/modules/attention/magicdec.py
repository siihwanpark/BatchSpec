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


class MagicDecAttention(BaseAttention):
    """Attention for MagicDec with StreamingLLM drafter.
    
    Args:
        config: Model configuration
    """
    
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        
        # Attention kernel to be set during setup
        self.attn_kernel = None
    

    def register_rope_variables(self, bsz: int, page_size: int, stream_budget: int):
        """Register RoPE variables for Streaming KV Cache.
        
        Args:
            bsz: Batch size
            page_size: Page size
            stream_budget: Stream budget
        """
        assert hasattr(self, 'draft_kv_cache'), "Please call after initializing the draft KV cache."
        device = self.draft_kv_cache.kv_cache.device
        
        self.page_size = page_size
        self.rope_indptr = torch.arange(bsz+1, dtype=torch.int32, device=device) * stream_budget
        self.rope_offsets = torch.zeros(bsz, dtype=torch.int32, device=device)


    def forward(
        self,
        x: Tensor,
        position_offsets: Tensor,
        qo_indptr: Tensor,
        kv_page_table: "PageTable",
        draft: bool = False,
    ) -> Tensor:
        """Forward pass through attention layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            position_offsets: position offsets
            qo_indptr: Index pointer for Query/Output tokens
            kv_page_table: Page table for KV cache
            draft: Whether to use draft KV cache
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
        
        # Apply RoPE and update KV cache
        with rope_compute_timer():
            if draft:
                q = self._apply_rope(q, k, qo_indptr, position_offsets)[0]
                kv_cache = self._update_draft_kv_and_apply_rope(k, v, qo_indptr, kv_page_table)
            else:
                q, k = self._apply_rope(q, k, qo_indptr, position_offsets)
                kv_cache = self.kv_cache.update(k, v, qo_indptr, kv_page_table)
        
        # Compute attention
        with attention_compute_timer():
            y = self.attn_kernel.run(q, kv_cache)
        
        # Reshape and project output (unflatten the heads)
        y = y.contiguous().view(bsz, seqlen, self.dim)
        y = self.wo(y)
        return self._maybe_all_reduce_output(y)

    def _update_draft_kv_and_apply_rope(
        self,
        k: Tensor,
        v: Tensor,
        qo_indptr: Tensor,
        kv_page_table: "PageTable",
    ) -> Tensor:
        """Update draft KV cache and apply RoPE.
        
        Args:
            k: Flat Key tensor (No RoPE applied)
            v: Flat Value tensor (No RoPE applied)
            position_offsets: position offsets
            qo_indptr: Index pointer for Query/Output tokens
            kv_page_table: Page table for draft KV cache

        Returns:
            Draft KV cache tensor (RoPE applied)
        """
        # Update draft KV cache
        self.draft_kv_cache.update(k, v, qo_indptr, kv_page_table)

        # Rotate the keys
        key_states = self.draft_kv_cache.kv_cache[:, 0].reshape(-1, self.n_local_heads, self.head_dim)
        value_states = self.draft_kv_cache.kv_cache[:, 1]

        rotated_key_states = self._apply_rope(key_states, key_states, self.rope_indptr, self.rope_offsets)[1]
        rotated_key_states = rotated_key_states.reshape(-1, self.page_size, self.n_local_heads, self.head_dim)
        return torch.cat([rotated_key_states[:, None], value_states[:, None]], dim=1)
