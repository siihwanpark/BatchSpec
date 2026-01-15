"""Paged memory management for KV caches.

"""

from typing import Optional, Tuple

import torch


class PageManager:
    """Deterministic, vectorized page allocator for paged KV caches.

    Overview
    --------
    This class manages page allocation/deallocation **per request** (batch element)
    for a paged KV cache. Pages for request `r` are assigned **deterministic IDs**

        page_id = r * max_num_pages_per_request + k,   k = 0,1,2,...

    and are always maintained in **append-only order** (allocate appends at tail,
    free removes from tail). This yields stable, reproducible page tables and
    enables O(N) vectorized updates.

    Data Model (CSR-style Page Table)
    ---------------------------------
    - `kv_page_indices` : 1-D int32 tensor of length `sum_r count_r`.
                          Concatenation of page IDs for each request, in order.
    - `kv_page_indptr`  : 1-D int32 tensor of shape [B+1], prefix sums of counts.
                          For request r, its pages live in
                              kv_page_indices[ indptr[r] : indptr[r+1] )
    - `num_allocated_pages` : 1-D int32 tensor of per-request page counts,
                              kept in sync with the table.

    Contract with the Cache
    -----------------------
    Many paged-KV designs keep **at least one page per request at all times**,
    even when the total token length is zero. With that convention:
        total_tokens[r] = (pages[r] - 1) * page_size + last_page_len[r]
    and for the empty state `total_tokens=0` we have `pages=1` and `last_page_len=0`.
    This manager supports that contract by **forbidding frees that would drop
    below 1 page** (see `free_counts`).

    API
    ---
    *Vectorized one-shot ops* (preferred):
      - `allocate_counts(add_counts, kv_page_indices=None, kv_page_indptr=None)`
          Add `add_counts[r] >= 0` pages to each request r in a single pass.
          If `kv_page_indices/kv_page_indptr` are **None**, returns a 1-D tensor
          containing the **newly created page IDs** (useful for cold init).
          Otherwise, returns the **updated** `(kv_page_indptr, kv_page_indices)`.

      - `free_counts(remove_counts, kv_page_indices, kv_page_indptr)`
          Remove `remove_counts[r] >= 0` pages from the **tail** of each request r.
          Enforces the "keep ≥1 page" rule: raises if any removal would drop a
          request below 1 page. Returns updated `(kv_page_indptr, kv_page_indices)`.

    *Backward-compatible wrappers*:
      - `allocate(requested_indices, ...)`
          Accepts a 1-D tensor of request indices; duplicates mean "allocate
          multiple pages" for that request. Internally reduces to `allocate_counts`
          via `bincount`.
      - `free(requested_indices, ...)`
          Symmetric to `allocate`, reducing to `free_counts`.

    Determinism & Invariants
    ------------------------
    - For each request r, after any sequence of ops, the current pages are
          [r*max + 0, r*max + 1, ..., r*max + (count_r-1)]
      i.e., **contiguous IDs starting from 0** within the request's ID domain.
    - Existing pages preserve order; appends come after old tail; frees pop from tail.
    - `kv_page_indptr` is monotonically non-decreasing, `len = B+1`.
    - `num_allocated_pages == (kv_page_indptr[1:] - kv_page_indptr[:-1])`.

    Complexity
    ----------
    - `allocate_counts` / `free_counts` run in **O(total_old_pages + total_delta_pages)**.
      They rebuild the CSR once per call (vectorized), avoiding per-page Python loops.
    - Memory moves are linear and coalesced; suitable for large batch updates on GPU.

    Errors & Edge Cases
    -------------------
    - `allocate_counts`: raises if any request would exceed `max_num_pages_per_request`.
    - `free_counts`    : raises if any request would drop below **1** page.
    - Zero deltas are allowed and are no-ops.
    - All inputs are expected to be `int32` on the same device as the manager state.

    Minimal Usage
    -------------
    >>> B = 2
    >>> pm = PageManager(bsz=B, max_num_pages_per_request=64, device=torch.device("cpu"))

    # Cold init: give each request exactly 1 page (IDs only; no table yet)
    >>> new_ids = pm.allocate_counts(torch.tensor([1, 1], dtype=torch.int32))
    >>> new_ids
    tensor([ 0, 64], dtype=torch.int32)
    >>> pm.num_allocated_pages
    tensor([1, 1], dtype=torch.int32)

    # Build a table representing that state
    >>> kv_page_indptr = torch.tensor([0, 1, 2], dtype=torch.int32)
    >>> kv_page_indices = torch.tensor([0, 64], dtype=torch.int32)

    # Allocate one more page to request 0 (append at tail)
    >>> indptr, indices = pm.allocate_counts(torch.tensor([1, 0], dtype=torch.int32),
    ...                                      kv_page_indices, kv_page_indptr)
    >>> indptr
    tensor([0, 2, 3], dtype=torch.int32)
    >>> indices
    tensor([ 0,  1, 64], dtype=torch.int32)
    >>> pm.num_allocated_pages
    tensor([2, 1], dtype=torch.int32)

    # Free one page from request 0 (pop from tail)
    >>> indptr, indices = pm.free_counts(torch.tensor([1, 0], dtype=torch.int32),
    ...                                  indices, indptr)
    >>> indptr
    tensor([0, 1, 2], dtype=torch.int32)
    >>> indices
    tensor([ 0, 64], dtype=torch.int32)
    >>> pm.num_allocated_pages
    tensor([1, 1], dtype=torch.int32)

    Wrapper Example
    ---------------
    >>> # Same as allocating counts=[2,1]:
    >>> indptr, indices = pm.allocate(torch.tensor([0, 0, 1], dtype=torch.int32),
    ...                               kv_page_indices=indices, kv_page_indptr=indptr)
    >>> pm.num_allocated_pages
    tensor([3, 2], dtype=torch.int32)

    Notes
    -----
    - Thread-safety is NOT provided; callers must serialize updates.
    - The manager is agnostic to `page_size`; that belongs to the higher-level cache.
    - The "keep ≥1 page" rule matches common paged-KV designs where even an empty
      sequence holds a tail page (with `last_page_len=0`), simplifying index math.
    """

    def __init__(self, bsz: int, max_num_pages_per_request: int, device: torch.device):
        """Initialize page manager.
        
        Args:
            bsz: Batch size (number of requests)
            max_num_pages_per_request: Maximum pages each request can allocate
            device: Device to place tensors on
        """
        self.bsz = int(bsz)
        self.max_num_pages_per_request = int(max_num_pages_per_request)
        self.num_allocated_pages = torch.zeros(self.bsz, dtype=torch.int32, device=device)

    def reset(self):
        """Reset all allocations to zero."""
        self.num_allocated_pages.fill_(0)

    def allocate(
        self,
        add_counts: torch.Tensor,
        kv_page_indices: Optional[torch.Tensor] = None,
        kv_page_indptr: Optional[torch.Tensor] = None,
    ):
        """Add pages to requests.
        
        Args:
            add_counts: Number of pages to add per request [B], int32, >=0
            kv_page_indices: Current page indices [sum(old_counts)] or None
            kv_page_indptr: Current page indptr [B+1] or None
            
        Returns:
            If indices/indptr are None: newly created page IDs [sum(add_counts)]
            Otherwise: (updated_indptr, updated_indices)
        """
        device = self.num_allocated_pages.device
        add_counts = add_counts.to(device=device, dtype=torch.int32).clamp_min(0)

        old_counts = self.num_allocated_pages
        new_counts = old_counts + add_counts
        if (new_counts > self.max_num_pages_per_request).any():
            raise RuntimeError("allocate_counts would exceed max_num_pages_per_request.")

        # If caller only needs the IDs of newly created pages (no table update)
        if kv_page_indices is None or kv_page_indptr is None:
            reqs_add = torch.nonzero(add_counts > 0, as_tuple=False).squeeze(-1)
            if reqs_add.numel() == 0:
                self.num_allocated_pages = new_counts
                return torch.empty(0, dtype=torch.int32, device=device)
            counts = add_counts[reqs_add].to(torch.int64)
            total_new = int(counts.sum().item())

            # Repeat req ids per their counts
            reqs_rep = torch.repeat_interleave(reqs_add, counts)
            
            # 0..(k_i-1) per request
            within = torch.arange(total_new, device=device) - torch.repeat_interleave(
                torch.cumsum(counts, 0) - counts, counts
            )
            base = (reqs_rep * self.max_num_pages_per_request) + old_counts[reqs_rep].to(torch.int64)
            new_ids = (base + within).to(torch.int32)
            self.num_allocated_pages = new_counts
            return new_ids

        # Build updated indptr = prefix-sum of new_counts
        updated_kv_page_indptr = torch.empty(self.bsz + 1, dtype=torch.int32, device=device)
        updated_kv_page_indptr[0] = 0
        torch.cumsum(new_counts, dim=0, out=updated_kv_page_indptr[1:])

        # Allocate output indices buffer
        total_old = int(kv_page_indices.numel())
        total_new = int(updated_kv_page_indptr[-1].item())
        updated_kv_page_indices = torch.empty(total_new, dtype=kv_page_indices.dtype, device=device)

        # Place OLD pages at the beginning of each request slice
        global_old = torch.arange(total_old, device=device)
        req_of_old = torch.searchsorted(kv_page_indptr, global_old, right=True) - 1
        intra_old = global_old - kv_page_indptr[req_of_old]
        write_pos_old = updated_kv_page_indptr[req_of_old] + intra_old
        updated_kv_page_indices[write_pos_old] = kv_page_indices

        # Append NEW pages directly after the old tail
        reqs_add = torch.nonzero(add_counts > 0, as_tuple=False).squeeze(-1)
        if reqs_add.numel() > 0:
            counts = add_counts[reqs_add].to(torch.int64)
            total_app = int(counts.sum().item())
            reqs_rep = torch.repeat_interleave(reqs_add, counts)
            within = torch.arange(total_app, device=device) - torch.repeat_interleave(
                torch.cumsum(counts, 0) - counts, counts
            )

            # start write positions for appends = start_of_req + old_count(req)
            starts = (updated_kv_page_indptr[reqs_add] + old_counts[reqs_add]).to(torch.int64)
            write_pos_new = torch.repeat_interleave(starts, counts) + within

            base = (reqs_rep * self.max_num_pages_per_request) + old_counts[reqs_rep].to(torch.int64)
            new_ids = (base + within).to(updated_kv_page_indices.dtype)
            updated_kv_page_indices[write_pos_new] = new_ids

        self.num_allocated_pages = new_counts
        return updated_kv_page_indptr, updated_kv_page_indices

    def free(
        self,
        remove_counts: torch.Tensor,
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Remove pages from requests (vectorized one-shot operation).
        
        Args:
            remove_counts: Number of pages to remove per request [B], int32, >=0
            kv_page_indices: Current page indices
            kv_page_indptr: Current page indptr
            
        Returns:
            Tuple of (updated_indptr, updated_indices)
        """
        device = self.num_allocated_pages.device
        remove_counts = remove_counts.to(device=device, dtype=torch.int32).clamp_min(0)

        old_counts = self.num_allocated_pages
        if (remove_counts > old_counts).any():
            raise RuntimeError("free_counts would free more pages than allocated.")
        new_counts = old_counts - remove_counts

        # New indptr
        updated_kv_page_indptr = torch.empty(self.bsz + 1, dtype=torch.int32, device=device)
        updated_kv_page_indptr[0] = 0
        torch.cumsum(new_counts, dim=0, out=updated_kv_page_indptr[1:])

        # Keep first new_counts per request
        total_old = int(kv_page_indices.numel())
        global_old = torch.arange(total_old, device=device)
        req_of_old = torch.searchsorted(kv_page_indptr, global_old, right=True) - 1
        intra_old = global_old - kv_page_indptr[req_of_old]
        keep = intra_old < new_counts[req_of_old]

        updated_kv_page_indices = kv_page_indices[keep]
        self.num_allocated_pages = new_counts
        return updated_kv_page_indptr, updated_kv_page_indices

