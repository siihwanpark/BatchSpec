"""Base attention classes and mixins."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor

from ...configs import ModelArgs, LoRAConfig
from ..normalization import RMSNorm
from ..lora import GatedLoRALinear


class BaseAttention(nn.Module, ABC):
    """Base class for all attention implementations.
    
    Provides common functionality for attention mechanisms including
    QKV projections, normalization, and distributed inference support.
    
    Args:
        config: Model configuration
    """
    
    def __init__(self, config: ModelArgs):
        super().__init__()
        
        assert config.dim % config.n_head == 0
        
        # Calculate total dimensions for QKV
        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        
        # Combined QKV projection for efficiency
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=config.qkv_bias)
        
        # Output projection
        # Note: head_dim * n_head may differ from dim in some models (e.g., Qwen3-0.6B)
        self.wo = nn.Linear(config.head_dim * config.n_head, config.dim, bias=False)
        
        # Optional Q/K normalization
        if config.qk_norm:
            self.q_norm = RMSNorm(config.head_dim, config.norm_eps)
            self.k_norm = RMSNorm(config.head_dim, config.norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
        
        # Store dimensions
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = self.n_head * self.head_dim
        
        # Suppose to be set during setup
        self.kv_cache = None
        self.rope = None
        self.process_group = None
        
        # Register load hook for compatibility with different checkpoint formats
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        """Hook to handle loading from checkpoints with separate Q/K/V matrices."""
        # Handle separate wq, wk, wv weights
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])
            
        # Handle separate wq, wk, wv biases
        if prefix + "wq.bias" in state_dict:
            bq = state_dict.pop(prefix + "wq.bias")
            bk = state_dict.pop(prefix + "bk.bias")
            bv = state_dict.pop(prefix + "bv.bias")
            state_dict[prefix + "wqkv.bias"] = torch.cat([bq, bk, bv])
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError("Each attention subclass must implement forward()")

    def _split_qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Split combined QKV tensor into separate Q, K, V tensors.
        
        Args:
            x: Combined QKV tensor from wqkv projection
            
        Returns:
            Tuple of (queries, keys, values) tensors
        """
        kv_size = self.n_local_heads * self.head_dim
        q, k, v = x.split([self.dim, kv_size, kv_size], dim=-1)
        return q, k, v
    
    def _apply_rope(self, q: Tensor, k: Tensor, *rope_args) -> tuple[Tensor, Tensor]:
        """Apply rotary position embeddings.
        
        Args:
            q: Query tensor
            k: Key tensor
            *rope_args: Additional arguments for RoPE (position_ids or indptr/offsets)
            
        Returns:
            Tuple of (q, k) with RoPE applied
        """
        if self.rope is not None:
            return self.rope(q, k, *rope_args)
        return q, k
    
    def _maybe_all_reduce_output(self, y: Tensor) -> Tensor:
        """Apply distributed all-reduce if configured.
        
        Args:
            y: Output tensor
            
        Returns:
            All-reduced tensor if using distributed inference, otherwise unchanged
        """
        if self.process_group is not None:
            dist.all_reduce(y, group=self.process_group)
        return y


class GatedLoRAAttention(BaseAttention):
    """Attention layer with gated LoRA.
    
    Args:
        config: Model configuration
        lora_config: LoRA configuration
    """
    
    def __init__(self, config: ModelArgs, lora_config: LoRAConfig):
        # Initialize base attention first
        super().__init__(config)
        
        # Replace linear layers with gated LoRA-enabled versions
        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        
        # Create gated LoRA config for QKV (3x parameters)
        wqkv_lora_config = LoRAConfig(
            rank=3 * lora_config.rank,
            alpha=3 * lora_config.alpha,
            lora_bias=lora_config.lora_bias,
            use_rslora=lora_config.use_rslora
        )
        
        # Replace wqkv and wo with gated LoRA versions
        self.wqkv = GatedLoRALinear(config.dim, total_head_dim, bias=config.qkv_bias, lora_config=wqkv_lora_config)
        self.wo = GatedLoRALinear(config.head_dim * config.n_head, config.dim, bias=False, lora_config=lora_config)


class AttentionMixin:
    """Mixin providing common attention computation patterns.
    
    This mixin can be used by concrete attention implementations to
    share common patterns like computing attention with different kernels.
    """
    
    def _compute_attention(
        self,
        q: Tensor,
        kv_cache: Any,
        attn_type: str,
        attn_kernels: Dict[str, Any]
    ) -> Tensor:
        """Compute attention using specified kernel.
        
        Args:
            q: Query tensor
            kv_cache: KV cache object
            attn_type: Type of attention computation
            attn_kernels: Dictionary mapping attention types to kernels

        Returns:
            Attention output
            
        Raises:
            ValueError: If attn_type is not supported
        """
        if attn_type not in attn_kernels:
            raise ValueError(
                f"Unsupported attention type: {attn_type}. "
                f"Supported types: {list(attn_kernels.keys())}"
            )
        
        kernel = attn_kernels[attn_type]
        if kernel is None:
            raise RuntimeError(
                f"Attention kernel for '{attn_type}' is not initialized. "
                f"Please ensure setup_caches() has been called."
            )
        
        return kernel(q, kv_cache)
