"""Rotary Position Embedding (RoPE) utilities."""

import flashinfer

from torch import Tensor
from typing import Callable
from ..configs import ModelArgs


def setup_rope_function(
    config: ModelArgs,
    use_position_ids: bool = False
) -> Callable[...,tuple[Tensor, Tensor]]:
    """Setup RoPE function for attention computation.
    
    Args:
        config: Model configuration
        use_position_ids: Whether to use position IDs directly (True) or compute from indptr/offsets (False)
                        
    Returns:
        RoPE function that can be directly called
    """
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
