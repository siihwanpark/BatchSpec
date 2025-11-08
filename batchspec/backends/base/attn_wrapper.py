"""Attention wrapper management mixin for backends."""

from typing import Dict, Any, Optional

import torch
import flashinfer
from torch import Tensor


class AttentionWrapperMixin:
    """Mixin for managing FlashInfer attention wrappers.
    
    Provides utilities for creating and registering attention wrappers
    with different configurations (prefill, decode, verify, etc.).
    
    Classes using this mixin must have:
    - self.device: Device to place tensors on
    - self.qo_indptr, self.paged_kv_indptr, self.paged_kv_indices, 
      self.paged_kv_last_page_len: KV cache state tensors
    """
    
    def _create_attention_buffer(self, size_mb: int = 128) -> torch.Tensor:
        """Create attention buffer for FlashInfer.
        
        Args:
            size_mb: Size in megabytes
            
        Returns:
            Empty buffer tensor
        """
        return torch.empty(size_mb * 1024 * 1024, dtype=torch.uint8, device=self.device)
    
    def _create_attention_wrapper(
        self,
        buffer: Tensor,
        qo_indptr: Tensor,
        use_custom_mask: bool = False,
        custom_mask_buf: Optional[Tensor] = None,
        mask_indptr_buf: Optional[Tensor] = None
    ) -> Any:
        """Create a FlashInfer attention wrapper.
        
        Args:
            buffer: Pre-allocated buffer for attention computation
            qo_indptr: Index pointer for Query/Output tokens
            use_custom_mask: Whether to use custom attention mask
            custom_mask_buf: Custom mask buffer (if use_custom_mask=True)
            mask_indptr_buf: Mask indirection pointer buffer (if use_custom_mask=True)
            
        Returns:
            BatchPrefillWithPagedKVCacheWrapper instance
        """
        wrapper_kwargs = dict(
            float_workspace_buffer=buffer,
            kv_layout="NHD",
            use_cuda_graph=True,
            qo_indptr_buf=qo_indptr,
            paged_kv_indptr_buf=self.kv_page_table.paged_kv_indptr,
            paged_kv_indices_buf=self.kv_page_table.paged_kv_indices,
            paged_kv_last_page_len_buf=self.kv_page_table.paged_kv_last_page_len
        )
        
        # Add custom mask buffers if needed
        if use_custom_mask:
            if custom_mask_buf is None or mask_indptr_buf is None:
                raise ValueError("custom_mask_buf and mask_indptr_buf must be provided when use_custom_mask=True")
            wrapper_kwargs.update(
                custom_mask_buf=custom_mask_buf,
                mask_indptr_buf=mask_indptr_buf
            )
        
        return flashinfer.BatchPrefillWithPagedKVCacheWrapper(**wrapper_kwargs)
