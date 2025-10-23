"""KV cache management mixin for backends."""

import torch
from torch import Tensor

from ..utils import PageManager

class KVCacheMixin:
    """Mixin providing KV cache management functionality.
    
    This mixin provides common KV cache operations including:
    - Inserting new KV entries
    - Deleting KV entries
    - Clearing all KV caches
    - Converting length values to tensors
    
    Classes using this mixin must have:
    - self.device: Device to place tensors on
    - self.batch_size: Batch size
    - self.page_size: Page size for paged attention
    - self.model: Model with layers containing KV caches (for clear_kv)
    """
    
    def _init_kv_cache_state(
        self,
        max_batch_size: int,
        max_num_pages: int,
        max_num_pages_per_request: int,
        device: torch.device
    ):
        """Initialize KV cache state tensors.
        
        Args:
            max_batch_size: Maximum batch size
            max_num_pages: Total number of pages
            max_num_pages_per_request: Maximum pages per request
            device: Device to place tensors on
        """
        self.cachelens = torch.zeros(max_batch_size, dtype=torch.int32, device=device)
        self.num_pages_per_request = torch.zeros(max_batch_size, device=device, dtype=torch.int32)
        
        self.qo_indptr = torch.arange(max_batch_size + 1, dtype=torch.int32, device=device)
        self.paged_kv_indptr = torch.arange(max_batch_size + 1, dtype=torch.int32, device=device)
        self.paged_kv_indices = torch.empty(max_num_pages, dtype=torch.int32, device=device)
        self.paged_kv_last_page_len = torch.zeros((max_batch_size), dtype=torch.int32, device=device)
        
        self.page_manager = PageManager(max_batch_size, max_num_pages_per_request, device)
    
    def _as_len_tensor(self, lens) -> Tensor:
        """Convert length value(s) to tensor.
        
        Args:
            lens: Length value(s), can be int, list, or tensor
            
        Returns:
            Tensor of lengths with proper device and dtype
        """
        if isinstance(lens, torch.Tensor):
            t = lens.to(device=self.device, dtype=self.paged_kv_last_page_len.dtype)
        else:
            t = torch.tensor(lens, device=self.device, dtype=self.paged_kv_last_page_len.dtype)
        
        # Expand scalar to batch size
        if t.dim() == 0:
            t = t.expand_as(self.paged_kv_last_page_len)
        
        return t

    def insert_kv(self, dec_lens):
        """Insert KV entries for new tokens.
        
        Args:
            dec_lens: Number of tokens to insert per batch
        """
        dec = self._as_len_tensor(dec_lens)
        
        # Skip if no insertions needed
        if torch.all(dec <= 0):
            return

        # Calculate current state
        old_full = self.num_pages_per_request.clone() - 1
        old_tail = self.paged_kv_last_page_len.clone()
        ps = self.page_size

        # Calculate new state after insertion
        total_after = old_full * ps + old_tail + dec
        new_full = torch.where(
            total_after > 0,
            torch.div(total_after - 1, ps, rounding_mode='floor').to(old_full.dtype),
            torch.zeros_like(old_full),
        )
        new_tail = torch.where(
            total_after > 0,
            (((total_after - 1) % ps) + 1).to(old_tail.dtype),
            torch.zeros_like(old_tail),
        )

        # Allocate new pages if needed
        add_pages = (new_full - old_full).clamp_min(0).to(torch.int32)
        if add_pages.max().item() > 0:
            self.paged_kv_indptr, self.paged_kv_indices = self.page_manager.allocate_counts(
                add_pages, self.paged_kv_indices, self.paged_kv_indptr
            )
            self.num_pages_per_request += add_pages

        # Update state
        self.paged_kv_last_page_len = new_tail
        self.cachelens = (self.cachelens + dec).clamp_min(0)

    def delete_kv(self, del_lens):
        """Delete KV entries.
        
        Args:
            del_lens: Number of tokens to delete per batch
        """
        dec = self._as_len_tensor(del_lens)
        
        # Skip if no deletions needed
        if torch.all(dec <= 0):
            return

        # Calculate current state
        old_full = self.num_pages_per_request.clone() - 1
        old_tail = self.paged_kv_last_page_len.clone()
        ps = self.page_size

        # Calculate new state after deletion
        total_before = old_full * ps + old_tail
        total_after = (total_before - dec).clamp_min(0)

        new_full = torch.where(
            total_after > 0,
            torch.div(total_after - 1, ps, rounding_mode='floor').to(old_full.dtype),
            torch.zeros_like(old_full),
        )
        new_tail = torch.where(
            total_after > 0,
            (((total_after - 1) % ps) + 1).to(old_tail.dtype),
            torch.zeros_like(old_tail),
        )

        # Free pages if needed
        free_pages = (old_full - new_full).clamp_min(0).to(torch.int32)
        if free_pages.max().item() > 0:
            self.paged_kv_indptr, self.paged_kv_indices = self.page_manager.free_counts(
                free_pages, self.paged_kv_indices, self.paged_kv_indptr
            )
            self.num_pages_per_request -= free_pages

        # Update state
        self.paged_kv_last_page_len = new_tail
        self.cachelens = (self.cachelens - dec).clamp_min(0)

    def clear_kv(self):
        """Clear all KV caches and reset state."""
        # Zero out KV cache in model
        for layer in self.model.layers:
            layer.attention.kv_cache.kv_cache.zero_()
        
        # Reset state
        self.cachelens = torch.zeros(self.batch_size, dtype=torch.int32, device=self.device)
        self.qo_indptr = torch.arange(self.batch_size + 1, dtype=torch.int32, device=self.device)
        
        # Reset page manager and allocate initial pages
        self.page_manager.reset()
        self.num_pages_per_request = torch.ones((self.batch_size), device=self.device, dtype=torch.int32)
        self.paged_kv_indptr = torch.arange(self.batch_size + 1, dtype=torch.int32, device=self.device)
        self.paged_kv_indices = self.page_manager.allocate(
            torch.arange(self.batch_size, dtype=torch.int32, device=self.device)
        )
        self.paged_kv_last_page_len = torch.zeros((self.batch_size), dtype=torch.int32, device=self.device)
