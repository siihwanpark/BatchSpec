"""Model configuration registry."""

from typing import Dict, Any, Optional
from .model_config import ModelArgs


MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # Llama 2 series
    "llama-2-7b": dict(
        block_size=4096, n_layer=32, n_head=32, dim=4096
    ),
    "llama-2-7b-32k": dict(
        block_size=32768, n_layer=32, dim=4096, vocab_size=32000, scaling_factor=8
    ),
    "llama-2-13b": dict(
        block_size=4096, n_layer=40, n_head=40, dim=5120
    ),
    "llama-2-70b": dict(
        block_size=4096, n_layer=80, n_head=64, dim=8192, 
        n_local_heads=8, intermediate_size=28672
    ),
    
    # Llama 3 series
    "llama-3-8b": dict(
        block_size=8192, n_layer=32, n_head=32, n_local_heads=8, 
        dim=4096, intermediate_size=14336, vocab_size=128256, rope_base=500000
    ),
    "llama-3-70b": dict(
        block_size=8192, n_layer=80, n_head=64, n_local_heads=8, 
        dim=8192, intermediate_size=28672, vocab_size=128256, rope_base=500000
    ),
    
    # Llama 3.1 series
    "llama-3.1-8b": dict(
        block_size=131072, n_layer=32, n_head=32, n_local_heads=8, 
        dim=4096, intermediate_size=14336, vocab_size=128256, 
        rope_base=500000.0, scaling_factor=8, high_freq_factor=4, 
        low_freq_factor=1, original_max_position_embeddings=8192
    ),
    "llama-3.1-70b": dict(
        block_size=131072, n_layer=80, n_head=64, n_local_heads=8, 
        dim=8192, intermediate_size=28672, vocab_size=128256, 
        rope_base=500000.0, scaling_factor=8, high_freq_factor=4, 
        low_freq_factor=1, original_max_position_embeddings=8192
    ),
    "llama-3.2-1b": dict(
        block_size=131072, n_layer=16, n_head=32, n_local_heads=8, 
        dim=2048, intermediate_size=8192, vocab_size=128256, 
        rope_base=500000.0, scaling_factor=32, high_freq_factor=4, 
        low_freq_factor=1, original_max_position_embeddings=8192
    ),
    
    # Qwen series
    "Qwen2.5-7b": dict(
        block_size=131072, n_layer=28, n_head=28, n_local_heads=4, 
        dim=3584, intermediate_size=18944, vocab_size=152064, 
        rope_base=1000000.0, qkv_bias=True, norm_eps=1e-6
    ),
    "Qwen2.5-14b": dict(
        block_size=131072, n_layer=48, n_head=40, n_local_heads=8, 
        dim=5120, intermediate_size=13824, vocab_size=152064, 
        rope_base=1000000.0, qkv_bias=True, norm_eps=1e-6
    ),
    "Qwen2.5-32b": dict(
        block_size=131072, n_layer=64, n_head=40, n_local_heads=8, 
        dim=5120, intermediate_size=27648, vocab_size=152064, 
        rope_base=1000000.0, qkv_bias=True, norm_eps=1e-6
    ),
    "Qwen3-0.6B": dict(
        block_size=40960, n_layer=28, n_head=16, n_local_heads=8, 
        head_dim=128, dim=1024, intermediate_size=3072, vocab_size=151936, 
        rope_base=1000000.0, norm_eps=1e-6, qk_norm=True
    ),
    "Qwen3-1.7B": dict(
        block_size=40960, n_layer=28, n_head=16, n_local_heads=8, 
        dim=2048, intermediate_size=6144, vocab_size=151936, 
        rope_base=1000000.0, norm_eps=1e-6, qk_norm=True
    ),
    "Qwen3-8B": dict(
        block_size=40960, n_layer=36, n_head=32, n_local_heads=8, 
        dim=4096, intermediate_size=12288, vocab_size=151936, 
        rope_base=1000000.0, norm_eps=1e-6, qk_norm=True
    ),
    "Qwen3-14B": dict(
        block_size=40960, n_layer=40, n_head=40, n_local_heads=8, 
        dim=5120, intermediate_size=17408, vocab_size=151936, 
        rope_base=1000000.0, norm_eps=1e-6, qk_norm=True
    ),
    "Qwen3-32B": dict(
        block_size=40960, n_layer=64, n_head=64, n_local_heads=8, 
        head_dim=128, dim=5120, intermediate_size=25600, vocab_size=151936, 
        rope_base=1000000.0, norm_eps=1e-6, qk_norm=True
    ),
    
    # DeepSeek series
    "DeepSeek-R1-Distill-Qwen-1.5B": dict(
        block_size=131072, n_layer=28, n_head=12, n_local_heads=2, 
        dim=1536, intermediate_size=8960, vocab_size=151936, 
        rope_base=10000.0, qkv_bias=True, norm_eps=1e-6
    ),
    "DeepSeek-R1-Distill-Qwen-7B": dict(
        block_size=131072, n_layer=28, n_head=28, n_local_heads=4, 
        dim=3584, intermediate_size=18944, vocab_size=152064, 
        rope_base=10000.0, qkv_bias=True, norm_eps=1e-6
    ),
    "DeepSeek-R1-Distill-Llama-8B": dict(
        block_size=131072, n_layer=32, n_head=32, n_local_heads=8, 
        dim=4096, intermediate_size=14336, vocab_size=128256, 
        rope_base=500000.0, scaling_factor=8, high_freq_factor=4, 
        low_freq_factor=1, original_max_position_embeddings=8192, norm_eps=1e-5
    ),
    
    # Extended context models
    "Llama-3-8B-Instruct-262k": dict(
        block_size=262144, n_layer=32, n_head=32, n_local_heads=8, 
        dim=4096, intermediate_size=14336, vocab_size=128256, 
        rope_base=283461213.0, norm_eps=1e-5
    ),
    "longspec-Llama-3-8B-Instruct-262k": dict(
        block_size=262144, n_layer=32, n_head=32, n_local_heads=8, 
        dim=4096, intermediate_size=14336, vocab_size=128256, 
        rope_base=283461213.0, norm_eps=1e-5
    ),
    
    # Small/Test models
    "68m": dict(
        block_size=2048, n_layer=2, n_head=12, n_local_heads=12, 
        dim=768, intermediate_size=3072, vocab_size=32000
    ),
    "tinyllama": dict(
        block_size=2048, n_layer=22, n_head=32, n_local_heads=4, 
        dim=2048, intermediate_size=5632, vocab_size=32000
    ),
    
    # Drafter models
    "Qwen3-8B_eagle3": dict(
        block_size=40960, n_layer=1, n_head=32, n_local_heads=8, 
        dim=4096, intermediate_size=12288, vocab_size=151936, 
        draft_vocab_size=32000, rope_base=1000000.0, norm_eps=1e-6,
    ),
    "Qwen3-14B_eagle3": dict(
        block_size=40960, n_layer=1, n_head=40, n_local_heads=8, 
        dim=5120, intermediate_size=17408, vocab_size=151936, 
        draft_vocab_size=32000, rope_base=1000000.0, norm_eps=1e-6,
    ),
    "Qwen3-32B_eagle3": dict(
        block_size=40960, n_layer=1, n_head=64, n_local_heads=8, 
        dim=5120, intermediate_size=25600, vocab_size=151936, 
        draft_vocab_size=32000, rope_base=1000000.0, norm_eps=1e-6,
    ),
    "EAGLE3-DeepSeek-R1-Distill-LLaMA-8B": dict(
        block_size=2048, n_layer=1, n_head=32, n_local_heads=8, 
        dim=4096, intermediate_size=14336, vocab_size=128256, 
        draft_vocab_size=32000, norm_eps=1e-5
    ),
}


def register_config(name: str, config: Dict[str, Any]) -> None:
    """Register a new model configuration.
    
    Args:
        name: Configuration name identifier
        config: Dictionary of configuration parameters
    """
    MODEL_CONFIGS[name] = config


def get_config(name: str) -> ModelArgs:
    """Get model configuration by name.
    
    Args:
        name: Configuration name identifier
        
    Returns:
        ModelArgs instance with the requested configuration
        
    Raises:
        ValueError: If configuration name not found
    """
    if name in MODEL_CONFIGS:
        return ModelArgs.from_dict(MODEL_CONFIGS[name])
    
    # Fuzzy search for partial matches
    matches = [
        config_name for config_name in MODEL_CONFIGS 
        if config_name.lower() in name.lower()
    ]
    
    if not matches:
        raise ValueError(f"No configuration found for '{name}'")
    
    # If multiple matches, prefer the longest match (more specific)
    if len(matches) > 1:
        matches.sort(key=len, reverse=True)
        # Ensure there's only one "best" match
        if len(matches[0]) == len(matches[1]):
            raise ValueError(
                f"Ambiguous configuration name '{name}'. "
                f"Matches: {', '.join(matches[:2])}"
            )
    
    print(f"Using configuration: {matches[0]}")
    return ModelArgs.from_dict(MODEL_CONFIGS[matches[0]])
