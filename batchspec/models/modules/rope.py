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
                            
        Returns:
            RoPE function that can be directly called
        """
        config = self.config
        
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
                rope_func = lambda q, k, position_ids: flashinfer.rope.apply_llama31_rope_pos_ids(
                    q, k, position_ids, **rope_kwargs
                )
            else:
                rope_func = lambda q, k, indptr, offsets: flashinfer.rope.apply_llama31_rope(
                    q, k, indptr, offsets, **rope_kwargs
                )
        else:
            if use_position_ids:
                rope_func = lambda q, k, position_ids: flashinfer.rope.apply_rope_pos_ids(
                    q, k, position_ids, **rope_kwargs
                )
            else:
                rope_func = lambda q, k, indptr, offsets: flashinfer.rope.apply_rope(
                    q, k, indptr, offsets, **rope_kwargs
                )
        
        return rope_func
