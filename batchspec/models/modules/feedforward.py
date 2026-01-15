"""Feed-forward network modules."""

from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from flashinfer.activation import silu_and_mul

from ..configs import ModelArgs, LoRAConfig
from .lora import GatedLoRALinear


class FeedForward(nn.Module):
    """Standard feed-forward network uses the SwiGLU activation function.
    
    Args:
        config: Model configuration containing dimensions
    """
    
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        # Fused w1 and w3 for efficiency (gate and up projections)
        self.w13 = nn.Linear(config.dim, 2 * config.intermediate_size, bias=False)
        # Down projection
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)
        self.process_group = None
        
        # Register hook for loading legacy checkpoints
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        """Hook to handle loading from checkpoints with separate w1/w3."""
        if prefix + "w1.weight" in state_dict:
            w1 = state_dict.pop(prefix + "w1.weight")
            w3 = state_dict.pop(prefix + "w3.weight")
            state_dict[prefix + "w13.weight"] = torch.cat([w1, w3])
    
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
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, dim)
        """
        y = self.w2(silu_and_mul(self.w13(x)))
        return self._maybe_all_reduce_output(y)


class GatedLoRAFeedForward(FeedForward):
    """Feed-forward network with gated LoRA support.
    
    Args:
        config: Model configuration
        lora_config: gated LoRA configuration for adaptation layers
    """
    
    def __init__(self, config: ModelArgs, lora_config: LoRAConfig):
        super().__init__(config)
        
        # Create gated LoRA config for w13 (needs 2x rank for fused gate+up)
        w13_lora_config = LoRAConfig(
            rank=2 * lora_config.rank,
            alpha=2 * lora_config.alpha,
            lora_bias=lora_config.lora_bias,
            use_rslora=lora_config.use_rslora
        )
        
        # gated LoRA-enabled linear layers
        self.w13 = GatedLoRALinear(config.dim, 2 * config.intermediate_size, bias=False, lora_config=w13_lora_config)
        self.w2 = GatedLoRALinear(config.intermediate_size, config.dim, bias=False, lora_config=lora_config)
    
    def forward(self, x: Tensor, gate_mask: Tensor) -> Tensor:
        """Forward pass with gated LoRA.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            gate_mask: Mask for gated LoRA
            
        Returns:
            Output tensor of shape (batch_size, seq_len, dim)
        """
        y = self.w2(silu_and_mul(self.w13(x, gate_mask)), gate_mask)
        return self._maybe_all_reduce_output(y)