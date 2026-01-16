"""Model configuration classes."""

import math
from dataclasses import dataclass
from typing import Optional, Dict, Any
from types import SimpleNamespace


@dataclass
class ModelArgs:
    """Configuration for transformer models."""
    
    # Core architecture
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: Optional[int] = None
    n_local_heads: int = -1
    head_dim: int = -1
    
    # RoPE configuration
    rope_base: float = 10000.0
    scaling_factor: float = 1.0
    low_freq_factor: Optional[int] = None
    high_freq_factor: Optional[int] = None
    original_max_position_embeddings: Optional[int] = None
    
    # Model-specific options
    norm_eps: float = 1e-5
    qkv_bias: bool = False
    qk_norm: bool = False
    
    # For EAGLE models
    draft_vocab_size: int = 32000
    target_hidden_size: Optional[int] = None

    def __post_init__(self):
        """Initialize derived values after dataclass initialization."""
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            # Round to nearest multiple of 256 for better performance
            self.intermediate_size = (
                n_hidden if n_hidden % 256 == 0 
                else n_hidden + 256 - (n_hidden % 256)
            )
        
        if self.head_dim == -1:
            self.head_dim = self.dim // self.n_head

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelArgs":
        """Create ModelArgs from a dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert ModelArgs to a dictionary."""
        return {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith('_')
        }


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation) layers."""
    
    rank: int = 16
    alpha: float = 32.0
    lora_bias: bool = False
    use_rslora: bool = False
    lora_scaling: Optional[float] = None
    
    def __post_init__(self):
        """Calculate LoRA scaling factor based on configuration."""
        if self.lora_scaling is None:
            if self.use_rslora:
                self.lora_scaling = self.alpha / math.sqrt(self.rank)
            else:
                self.lora_scaling = self.alpha / self.rank

    @classmethod
    def from_args(cls, args: SimpleNamespace, prefix: str = "lora_") -> "LoRAConfig":
        """Create LoRAConfig from args with automatic prefix stripping.
        
        Args:
            args: Parsed arguments namespace
            prefix: Prefix to strip from arg names (default: "lora_")
        """
        # Get all fields from dataclass
        from dataclasses import fields as dataclass_fields
        
        kwargs = {}
        for field in dataclass_fields(cls):
            # Skip computed fields
            if field.name == "lora_scaling":
                continue
            
            # Try with prefix first
            prefixed_name = f"{prefix}{field.name}"
            if hasattr(args, prefixed_name):
                kwargs[field.name] = getattr(args, prefixed_name)
            # Try without prefix
            elif hasattr(args, field.name):
                kwargs[field.name] = getattr(args, field.name)
        
        return cls(**kwargs)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LoRAConfig":
        """Create LoRAConfig from a dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert LoRAConfig to a dictionary."""
        return {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith('_')
        }
