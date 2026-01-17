"""Key-Value cache implementations for efficient attention computation."""

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor
from flashinfer import (
    get_batch_indices_positions, 
    get_seq_lens,
    append_paged_kv_cache,
)

if TYPE_CHECKING:
    from batchspec.backends.base.page_table import PageTable


class StandardKVCache(nn.Module):
    """Standard paged key-value cache for attention mechanisms.
    
    Implements a paged memory system for storing key-value pairs,
    enabling efficient attention computation with reduced memory overhead.
    
    Args:
        max_num_pages: Maximum number of pages to allocate
        page_size: Size of each page (number of tokens per page)
        n_heads: Number of attention heads
        head_dim: Dimension of each attention head
        dtype: Data type for cache tensors
    """
    
    def __init__(
        self, 
        max_num_pages: int,
        page_size: int,
        n_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        
        # Initialize cache tensor: [pages, 2 (k/v), page_size, heads, head_dim]
        cache_shape = (max_num_pages, 2, page_size, n_heads, head_dim)
        self.register_buffer('kv_cache', torch.empty(cache_shape, dtype=dtype))
        self.page_size = page_size
        
    def update(
        self, 
        k: Tensor,
        v: Tensor,
        kv_append_indptr: Tensor,
        kv_page_table: "PageTable",
    ) -> Tensor:
        """Update cache with new key-value pairs.
        
        Args:
            k: Key tensor
            v: Value tensor
            kv_page_table: Page table for KV cache
            
        Returns:
            Updated KV cache tensor
        """
        batch_indices, positions = get_batch_indices_positions(
            append_indptr=kv_append_indptr,
            seq_lens=get_seq_lens(
                kv_indptr=kv_page_table.paged_kv_indptr,
                kv_last_page_len=kv_page_table.paged_kv_last_page_len,
                page_size=self.page_size,
            ), 
            nnz=kv_append_indptr[-1].item()
        )
        append_paged_kv_cache(
            append_key=k,
            append_value=v,
            batch_indices=batch_indices,
            positions=positions,
            paged_kv_cache=self.kv_cache,
            kv_indices=kv_page_table.paged_kv_indices,
            kv_indptr=kv_page_table.paged_kv_indptr,
            kv_last_page_len=kv_page_table.paged_kv_last_page_len,
            kv_layout='NHD',
        )
        return self.kv_cache


class StreamingKVCache(StandardKVCache):
    """Streaming KV cache with sliding window and sink token support.
    
    Extends the standard KV cache with streaming capabilities, maintaining
    a fixed budget of tokens while preserving important sink tokens.
    """

    def evict_kv(
        self,
        num_evicts: Tensor,
        num_sink_tokens: int,
        cachelens: Tensor,
        n_local_heads: int,
        head_dim: int
    ) -> None:
        """Evict tokens from cache while preserving sink tokens.
        
        Args:
            num_evicts: Number of tokens to evict per batch of shape [bsz]
            num_sink_tokens: Number of sink tokens to preserve
            cachelens: Current cache lengths per batch of shape [bsz]
            n_local_heads: Number of local attention heads
            head_dim: Dimension of each head
        """
        bsz = num_evicts.shape[0]
        
        # (max_num_pages, 2, page_size, n_heads, head_dim)
        # Get current cache state
        key_states = self.kv_cache[:, 0].clone().reshape(bsz, -1, n_local_heads, head_dim)
        value_states = self.kv_cache[:, 1].clone().reshape(bsz, -1, n_local_heads, head_dim)

        target_cachelens = cachelens - num_evicts
        assert torch.all(target_cachelens >= num_sink_tokens), f"num_evicts: {num_evicts} is too large, the target cache length must be greater than or equal to the number of sink tokens, but got {target_cachelens}"
        
        # Define read and write masks
        positions = torch.arange(key_states.shape[1], device=key_states.device)[None, :].expand(bsz, -1)
        
        start_read = (num_sink_tokens + num_evicts)[:, None]
        end_read = cachelens[:, None]
        mask_read = (positions >= start_read) & (positions < end_read)

        start_write = num_sink_tokens
        end_write = target_cachelens[:, None]
        mask_write = (positions >= start_write) & (positions < end_write)

        # Shift tokens
        key_states[mask_write] = key_states[mask_read]
        value_states[mask_write] = value_states[mask_read]

        # Update KV cache
        self.kv_cache[:, 0].copy_(key_states.reshape(-1, self.page_size, n_local_heads, head_dim))
        self.kv_cache[:, 1].copy_(value_states.reshape(-1, self.page_size, n_local_heads, head_dim))
