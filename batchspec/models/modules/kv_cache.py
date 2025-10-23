"""Key-Value cache implementations for efficient attention computation."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from flashinfer import (
    get_batch_indices_positions, 
    get_seq_lens,
    append_paged_kv_cache,
)


# Define custom operator for KV cache updates (only if not already registered)
if not hasattr(torch.ops.mylib, 'update_kv'):
    torch.library.define(
        "mylib::update_kv",
        "(Tensor k, Tensor v, Tensor kv_append_indptr, Tensor(a!) kv_cache, "
        "Tensor kv_page_indices, Tensor kv_page_indptr, Tensor cachelen, int page_size) -> ()",
    )

    @torch.library.impl("mylib::update_kv", "cuda")
    def update_kv(k, v, kv_append_indptr, kv_cache, kv_page_indices, 
                    kv_page_indptr, kv_page_last_len, page_size=128):
        """Update KV cache with new key-value pairs."""
        nnz_kv = kv_append_indptr[-1].item()
        batch_indices, positions = get_batch_indices_positions(
            kv_append_indptr, 
            get_seq_lens(kv_page_indptr, kv_page_last_len, page_size), 
            nnz_kv
        )
        append_paged_kv_cache(
            k, v, batch_indices, positions, kv_cache, 
            kv_page_indices, kv_page_indptr, kv_page_last_len
        )

    @torch.library.register_fake("mylib::update_kv")
    def update_kv_abstract(k, v, kv_append_indptr, kv_cache, kv_page_indices, 
                            kv_page_indptr, kv_page_last_len, page_size=128):
        """Abstract implementation for torch.compile."""
        return None


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
        self.register_buffer('kv_cache', torch.zeros(cache_shape, dtype=dtype))
        self.page_size = page_size
        
    def update(
        self, 
        k: Tensor,
        v: Tensor,
        kv_append_indptr: Tensor,
        kv_page_indices: Tensor,
        kv_page_indptr: Tensor,
        kv_page_lastlen: Tensor
    ) -> Tensor:
        """Update cache with new key-value pairs.
        
        Args:
            k: Key tensor
            v: Value tensor
            kv_append_indptr: Indices for appending KV pairs
            kv_page_indices: Page indices for KV storage
            kv_page_indptr: Page indirection pointer
            kv_page_lastlen: Last length of each page
            
        Returns:
            Updated KV cache tensor
        """
        torch.ops.mylib.update_kv(
            k, v, kv_append_indptr, self.kv_cache, 
            kv_page_indices, kv_page_indptr, kv_page_lastlen, 
            self.page_size
        )
        return self.kv_cache


class StreamingKVCache(StandardKVCache):
    """Streaming KV cache with sliding window and sink token support.
    
    Extends the standard KV cache with streaming capabilities, maintaining
    a fixed budget of tokens while preserving important sink tokens.
    
    Args:
        max_num_pages: Maximum number of pages
        page_size: Size of each page
        n_heads: Number of attention heads
        head_dim: Dimension of each head
        streaming_budget: Maximum number of tokens to keep in cache
        num_sink_tokens: Number of initial tokens to always preserve
        dtype: Data type for cache tensors
    """
    
    def __init__(
        self,
        max_num_pages: int,
        page_size: int,
        n_heads: int,
        head_dim: int,
        streaming_budget: int,
        num_sink_tokens: int,
        dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__(max_num_pages, page_size, n_heads, head_dim, dtype)
        self.streaming_budget = streaming_budget
        self.num_sink_tokens = num_sink_tokens
    
    def update_for_prefill(
        self,
        k: Tensor,
        v: Tensor,
        query_lens: Tensor,
        kv_append_indptr: Tensor,
        kv_page_indices: Tensor,
        kv_page_indptr: Tensor,
        kv_page_lastlen: Tensor,
        bsz: int,
        context_len: int,
        seq_len: int,
        n_local_heads: int,
        head_dim: int,
        rope
    ) -> Tensor:
        """Update cache during prefill phase with streaming support.
        
        Implements sliding window attention with sink token preservation.
        When the cache exceeds the budget, older tokens (except sink tokens)
        are evicted to make room for new ones.
        """
        if context_len + seq_len > self.streaming_budget:
            # Reshape for batch processing
            k = k.reshape(bsz, -1, n_local_heads, head_dim)
            v = v.reshape(bsz, -1, n_local_heads, head_dim)
            
            # Get current cache state
            key_states = self.kv_cache[:, 0].clone().reshape(
                bsz, -1, n_local_heads, head_dim
            )
            value_states = self.kv_cache[:, 1].clone().reshape(
                bsz, -1, n_local_heads, head_dim
            )
            
            max_kv_len = key_states.shape[1]
            positions_cached_kv = torch.arange(
                max_kv_len, device=key_states.device
            ).unsqueeze(0)
            positions_new_kv = torch.arange(
                seq_len, device=k.device
            ).unsqueeze(0)
            
            query_lens_ext = query_lens[:, None]
            active_mask = (query_lens_ext > 0)

            # Handle requests that fit within budget
            plain_req_mask = active_mask & (
                context_len + query_lens_ext <= self.streaming_budget
            )
            plain_read = plain_req_mask & (positions_new_kv < query_lens_ext)
            plain_write = plain_req_mask & (
                (positions_cached_kv >= context_len) & 
                (positions_cached_kv < (context_len + query_lens_ext))
            )
            
            key_states[plain_write] = k[plain_read]
            value_states[plain_write] = v[plain_read]

            # Handle rolling/streaming updates
            rolling_req_mask = active_mask & (
                context_len + query_lens_ext > self.streaming_budget
            )
            
            if rolling_req_mask.any():
                upper_bound = min(context_len, self.streaming_budget)

                # Shift existing KV states (preserve sink tokens)
                shift_write_end = self.streaming_budget - query_lens_ext
                shift_write = rolling_req_mask & (
                    (positions_cached_kv >= self.num_sink_tokens) & 
                    (positions_cached_kv < shift_write_end)
                )

                shift_read_lens = (
                    self.streaming_budget - query_lens_ext - self.num_sink_tokens
                )
                shift_read_start = upper_bound - shift_read_lens
                shift_read = rolling_req_mask & (
                    (positions_cached_kv >= shift_read_start) & 
                    (positions_cached_kv < upper_bound)
                )
                
                key_states[shift_write] = key_states[shift_read]
                value_states[shift_write] = value_states[shift_read]
                
                # Append new KV states
                append_write_start = self.streaming_budget - query_lens_ext
                append_write = rolling_req_mask & (
                    (positions_cached_kv >= append_write_start) & 
                    (positions_cached_kv < self.streaming_budget)
                )
                append_read = rolling_req_mask & (positions_new_kv < query_lens_ext)
                
                key_states[append_write] = k[append_read]
                value_states[append_write] = v[append_read]

            # Update cache
            self.kv_cache[:, 0].copy_(
                key_states.reshape(-1, self.page_size, n_local_heads, head_dim)
            )
            self.kv_cache[:, 1].copy_(
                value_states.reshape(-1, self.page_size, n_local_heads, head_dim)
            )
        else:
            # Standard update when within budget
            self.update(
                k, v, kv_append_indptr, kv_page_indices, 
                kv_page_indptr, kv_page_lastlen
            )
            key_states = self.kv_cache[:, 0].clone().reshape(
                bsz, -1, n_local_heads, head_dim
            )
            value_states = self.kv_cache[:, 1].clone().reshape(
                bsz, -1, n_local_heads, head_dim
            )

        # Apply RoPE to cached keys
        keys_to_rotate = key_states[:, :self.streaming_budget].reshape(
            -1, n_local_heads, head_dim
        )
        key_states[:, :self.streaming_budget] = rope(
            q=keys_to_rotate,
            k=keys_to_rotate,
            indptr=kv_append_indptr // seq_len * self.streaming_budget,
            offsets=torch.zeros(bsz, dtype=torch.int32, device=kv_append_indptr.device),
        )[1].reshape(bsz, -1, n_local_heads, head_dim)
        
        key_states = key_states.reshape(-1, self.page_size, n_local_heads, head_dim)
        value_states = value_states.reshape(-1, self.page_size, n_local_heads, head_dim)
        
        return torch.stack((key_states, value_states), dim=1)
    
    def evict_kv(
        self,
        num_evicts: Tensor,
        cachelens: Tensor,
        n_local_heads: int,
        head_dim: int
    ) -> None:
        """Evict tokens from cache while preserving sink tokens.
        
        Args:
            num_evicts: Number of tokens to evict per batch
            cachelens: Current cache lengths per batch
            n_local_heads: Number of local attention heads
            head_dim: Dimension of each head
        """
        bsz = num_evicts.shape[0]
        
        # Get current cache state
        key_states = self.kv_cache[:, 0].clone().reshape(
            bsz, -1, n_local_heads, head_dim
        )
        value_states = self.kv_cache[:, 1].clone().reshape(
            bsz, -1, n_local_heads, head_dim
        )

        target_cachelens = cachelens - num_evicts
        positions = torch.arange(
            key_states.shape[1], device=key_states.device
        )[None, :].expand(bsz, -1)
        
        # Define read and write masks
        start_read = (self.num_sink_tokens + num_evicts)[:, None]
        end_read = cachelens[:, None]
        mask_read = (positions >= start_read) & (positions < end_read)

        start_write = self.num_sink_tokens
        end_write = target_cachelens[:, None]
        mask_write = (positions >= start_write) & (positions < end_write)

        # Shift tokens
        key_states[mask_write] = key_states[mask_read]
        value_states[mask_write] = value_states[mask_read]

        # Update cache
        self.kv_cache[:, 0].copy_(
            key_states.reshape(-1, self.page_size, n_local_heads, head_dim)
        )
        self.kv_cache[:, 1].copy_(
            value_states.reshape(-1, self.page_size, n_local_heads, head_dim)
        )
