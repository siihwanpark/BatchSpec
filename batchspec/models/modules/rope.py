"""Rotary Position Embedding (RoPE) utilities."""

import torch
import flashinfer


class RoPEMixin:
    """Mixin for setting up RoPE kernels in transformers.
    
    This mixin provides a common implementation for registering
    RoPE kernels that can be shared across different model types.
    """
    
    def _setup_rope_kernels(self, use_position_ids: bool = False):
        """Setup RoPE kernels for attention computation.
        
        Args:
            use_position_ids: Whether to use position IDs directly (True)
                            or compute from indptr/offsets (False)
        """
        config = self.config
        
        # Define schema based on whether position IDs are used
        if use_position_ids:
            schema = "(Tensor q, Tensor k, Tensor position_ids) -> (Tensor ropeq, Tensor ropek)"
        else:
            schema = "(Tensor q, Tensor k, Tensor indptr, Tensor offsets) -> (Tensor ropeq, Tensor ropek)"
        
        # RoPE configuration
        rope_kwargs = dict(
            interleave=True,
            rope_scale=config.scaling_factor,
            rope_theta=config.rope_base
        )

        # Handle Llama 3.1 style RoPE with frequency factors
        has_freq_factors = (
            config.high_freq_factor is not None and 
            config.low_freq_factor is not None
        )
        
        if has_freq_factors:
            rope_kwargs.update(
                low_freq_factor=config.low_freq_factor,
                high_freq_factor=config.high_freq_factor,
                old_context_len=config.original_max_position_embeddings
            )
            
            if use_position_ids:
                backend = lambda q, k, position_ids: flashinfer.rope.apply_llama31_rope_pos_ids(
                    q, k, position_ids, **rope_kwargs
                )
            else:
                backend = lambda q, k, indptr, offsets: flashinfer.rope.apply_llama31_rope(
                    q, k, indptr, offsets, **rope_kwargs
                )
        else:
            if use_position_ids:
                backend = lambda q, k, position_ids: flashinfer.rope.apply_rope_pos_ids(
                    q, k, position_ids, **rope_kwargs
                )
            else:
                backend = lambda q, k, indptr, offsets: flashinfer.rope.apply_rope(
                    q, k, indptr, offsets, **rope_kwargs
                )
        
        # Register custom operator (only if not already registered)
        if not hasattr(torch.ops.mylib, 'rope'):
            torch.library.define("mylib::rope", schema)
            
            @torch.library.impl("mylib::rope", "cuda")
            def rope_impl(*args):
                return backend(*args)
            
            @torch.library.register_fake("mylib::rope")
            def rope_fake(*args):
                q, k = args[0], args[1]
                return torch.empty_like(q), torch.empty_like(k)
