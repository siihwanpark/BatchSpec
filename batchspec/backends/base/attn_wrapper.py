"""Attention wrapper management mixin for backends."""

from typing import Dict, Any, Optional

import torch
import flashinfer


# Global registry for attention wrappers
# This allows us to update wrappers without re-registering operators
_ATTENTION_WRAPPER_REGISTRY: Dict[str, Any] = {}


def _register_custom_attn_op(name: str, wrapper, device: str = "cuda"):
    """Register a custom attention operator with FlashInfer wrapper.
    
    This creates a custom PyTorch operator that wraps a FlashInfer
    attention wrapper, enabling its use in compiled models.
    
    The wrapper is stored in a global registry, so subsequent calls
    will update the wrapper without re-registering the operator.
    
    Args:
        name: Name of the custom operator (e.g., "mylib::attn_prefill")
        wrapper: FlashInfer attention wrapper instance
        device: Device type for the implementation
    """
    # Store wrapper in global registry (always update)
    _ATTENTION_WRAPPER_REGISTRY[name] = wrapper
    
    # Check if already registered to avoid duplicate registration
    namespace, op_name = name.split("::")
    if hasattr(getattr(torch.ops, namespace, None), op_name):
        return  # Already registered, just updated the wrapper in registry
    
    # Define the operator schema (only once)
    torch.library.define(
        name,
        "(Tensor q, Tensor kv_cache) -> Tensor",
    )

    # Register CUDA implementation (fetches from registry)
    @torch.library.impl(name, device)
    def _impl(q, kv_cache):
        current_wrapper = _ATTENTION_WRAPPER_REGISTRY[name]
        return current_wrapper.run(q, kv_cache)

    # Register fake/abstract implementation for torch.compile
    @torch.library.register_fake(name)
    def _abstract(q, kv_cache):
        return torch.empty_like(q)


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
        buffer: torch.Tensor,
        qo_length: int,
        use_custom_mask: bool = False,
        custom_mask_buf: Optional[torch.Tensor] = None,
        mask_indptr_buf: Optional[torch.Tensor] = None
    ) -> Any:
        """Create a FlashInfer attention wrapper.
        
        Args:
            buffer: Pre-allocated buffer for attention computation
            qo_length: Query length multiplier for qo_indptr
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
            qo_indptr_buf=self.qo_indptr * qo_length,
            paged_kv_indptr_buf=self.paged_kv_indptr,
            paged_kv_indices_buf=self.paged_kv_indices,
            paged_kv_last_page_len_buf=self.paged_kv_last_page_len
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
    
    def _register_attention_wrappers(self, wrappers: Dict[str, Any], prefix: str = "mylib"):
        """Register attention wrappers as custom ops.
        
        Args:
            wrappers: Dictionary mapping wrapper names to wrapper objects
            prefix: Prefix for custom op names (e.g., "mylib")
        """
        for name, wrapper in wrappers.items():
            op_name = f"{prefix}::{name}"
            _register_custom_attn_op(op_name, wrapper)
