"""BatchSpec Models - Refactored speculative decoding models.

This package provides a clean, modular implementation of various
speculative decoding models with minimal code duplication and
good extensibility.

Directory Structure:
    - configs/: Model and LoRA configuration management
    - modules/: Shared neural network components (normalization, feedforward, etc.)
    - modules/attention/: Concrete attention mechanism implementations
    - models/: Complete transformer model implementations

Example Usage:
    >>> from batchspec.models import get_model
    >>> model = get_model("llama-3-8b", "standard")
    >>> model.setup_caches(num_pages=1000, page_size=128)
    >>> logits = model(tokens, positions, ...)

Model Types:
    - StandardTransformer: Standard autoregressive generation
    - EAGLETransformer: EAGLE speculative decoding (chain & standard modes)
    - MTPTransformer: Multi-token prediction with LoRA

For detailed documentation, see individual module docstrings.
"""

__version__ = "1.0.0"

# Configuration management
from .configs import (
    ModelArgs,
    LoRAConfig,
    MODEL_CONFIGS,
    register_config,
    get_config,
)

# Shared modules
from .modules import (
    BaseAttention,
    GatedLoRAAttention,
    AttentionMixin,
    RoPEMixin,
    RMSNorm,
    FeedForward,
    GatedLoRAFeedForward,
    StandardKVCache,
    StreamingKVCache,
    GatedLoRALinear,
    SamplerHead,
    SamplerHeadBlock,
)

# Complete transformer models
from .base_model import BaseTransformer, BaseTransformerBlock, GatedLoRATransformerBlock
from .standard_model import StandardTransformer
from .eagle_model import EAGLETransformer, EAGLEModel
from .mtp_model import MTPTransformer

__all__ = [
    # Version
    "__version__",
    
    # Configurations
    "ModelArgs",
    "LoRAConfig",
    "MODEL_CONFIGS",
    "register_config",
    "get_config",
    
    # Modules
    "RMSNorm",
    "FeedForward",
    "GatedLoRAFeedForward",
    "StandardKVCache",
    "StreamingKVCache",
    "GatedLoRALinear",
    "SamplerHead",
    "SamplerHeadBlock",
    
    # Base classes
    "BaseAttention",
    "AttentionMixin",
    "BaseTransformer",
    "BaseTransformerBlock",
    "GatedLoRATransformerBlock",
    "RoPEMixin",
    
    # Attention
    "StandardAttention",
    "EAGLEAttention",
    "MTPAttention",
    
    # Transformers
    "StandardTransformer",
    "EAGLETransformer",
    "EAGLEModel",
    "MTPTransformer",
]

from typing import Optional

def get_model(
    model_name: str, 
    model_type: str,
    drafter_name: Optional[str] = None,
    use_chain_mode: bool = False,
    lora_config: Optional[LoRAConfig] = None
):
    """Factory function to create models by type name.
    
    Args:
        model_name: Name of the model configuration
        model_type: Type of model ("standard", "eagle", "mtp")
        drafter_name: Name of the drafter configuration (for EAGLE model only)
        use_chain_mode: Whether to use chain mode (for EAGLE model only)
        lora_config: LoRA configuration (for MTP model only)
    Returns:
        Model instance
        
    Raises:
        ValueError: If model_type is not recognized
        
    Example:
        >>> model = get_model("llama-3-8b", "standard")
        >>> model = get_model("llama-3-8b", "eagle", drafter_name="eagle-3-8b", use_chain_mode=True)
        >>> model = get_model("llama-3-8b", "mtp", lora_config=LoRAConfig(rank=16, alpha=32))
    """    
    model_type = model_type.lower()
    if model_type == "standard":
        return StandardTransformer.from_name(model_name)
    
    elif model_type == "eagle":
        if drafter_name is None:
            raise ValueError("Drafter configuration is required for EAGLE model.")
        return EAGLETransformer.from_name(model_name, drafter_name, use_chain_mode)

    elif model_type == "mtp":
        if lora_config is None:
            raise ValueError("LoRA configuration is required for MTP model.")
        return MTPTransformer.from_name(model_name, lora_config)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}.")


__all__.append("get_model")
