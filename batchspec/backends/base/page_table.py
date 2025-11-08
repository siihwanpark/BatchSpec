from typing import Optional

import torch
from torch import Tensor

from ..utils.paging import PageManager

class PageTable:
    """
    Page table for the KV cache.
    Includes page table index pointers, page table indices, and last page length.
    Also includes a page manager for allocating and freeing pages.
    """
    def __init__(
        self,
        page_size: int,
        max_batch_size: int,
        max_num_pages_per_request: int,
        device: torch.device
    ):
        self.page_size = page_size
        self.max_batch_size = max_batch_size
        self.max_num_pages_per_request = max_num_pages_per_request
        self.device = device
        
        self.page_manager = PageManager(max_batch_size, max_num_pages_per_request, device)
        self.cachelens = torch.zeros(self.max_batch_size, dtype=torch.int32, device=self.device)
        
        self.page_manager.reset()
        self.paged_kv_indptr = torch.arange(self.max_batch_size + 1, dtype=torch.int32, device=self.device)
        self.paged_kv_indices = torch.empty(max_batch_size * max_num_pages_per_request, dtype=torch.int32, device=self.device)
        self.paged_kv_last_page_len = torch.empty(self.max_batch_size, dtype=torch.int32, device=self.device)


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
        old_full = self.page_manager.num_allocated_pages.clone() - 1
        old_tail = self.paged_kv_last_page_len.clone()
        page_size = self.page_size

        # Calculate new state after insertion
        total_after = old_full * page_size + old_tail + dec
        new_full = torch.where(
            total_after > 0,
            torch.div(total_after - 1, page_size, rounding_mode='floor').to(old_full.dtype),
            torch.zeros_like(old_full),
        )
        new_tail = torch.where(
            total_after > 0,
            (((total_after - 1) % page_size) + 1).to(old_tail.dtype),
            torch.zeros_like(old_tail),
        )

        # Allocate new pages if needed
        add_pages = (new_full - old_full).clamp_min(0).to(torch.int32)
        if add_pages.max().item() > 0:
            self.paged_kv_indptr, self.paged_kv_indices = self.page_manager.allocate(
                add_pages, self.paged_kv_indices, self.paged_kv_indptr
            )

        # Update state
        self.paged_kv_last_page_len = new_tail
        self.cachelens = (self.cachelens + dec).clamp_min(0)

    def delete_kv(self, del_lens):
        """Delete KV entries.
        
        Args:
            del_lens: Number of tokens to delete per batch
        """
        del_lens = self._as_len_tensor(del_lens)
        
        # Skip if no deletions needed
        if torch.all(del_lens <= 0):
            return

        # Calculate current state
        old_full = self.page_manager.num_allocated_pages.clone() - 1
        old_tail = self.paged_kv_last_page_len.clone()
        page_size = self.page_size

        # Calculate new state after deletion
        total_before = old_full * page_size + old_tail
        total_after = (total_before - del_lens).clamp_min(0)

        new_full = torch.where(
            total_after > 0,
            torch.div(total_after - 1, page_size, rounding_mode='floor').to(old_full.dtype),
            torch.zeros_like(old_full),
        )
        new_tail = torch.where(
            total_after > 0,
            (((total_after - 1) % page_size) + 1).to(old_tail.dtype),
            torch.zeros_like(old_tail),
        )

        # Free pages if needed
        free_pages = (old_full - new_full).clamp_min(0).to(torch.int32)
        if free_pages.max().item() > 0:
            self.paged_kv_indptr, self.paged_kv_indices = self.page_manager.free(
                free_pages, self.paged_kv_indices, self.paged_kv_indptr
            )

        # Update state
        self.paged_kv_last_page_len = new_tail
        self.cachelens = (self.cachelens - del_lens).clamp_min(0)

    def clear_kv(self, model: Optional[torch.nn.Module] = None):
        """Clear all KV caches and reset state.
        
        Args:
            model: Model to zero out KV cache from
        """
        # Zero out KV cache in model
        if model is not None:
            for layer in model.layers:
                layer.attention.kv_cache.kv_cache.zero_()
        
        self.cachelens.zero_()
        self.page_manager.reset()
        self.paged_kv_indptr = torch.arange(self.max_batch_size + 1, dtype=torch.int32, device=self.device)
        self.paged_kv_indices = self.page_manager.allocate(torch.ones(self.max_batch_size, dtype=torch.int32, device=self.device))
        self.paged_kv_last_page_len.zero_()
        