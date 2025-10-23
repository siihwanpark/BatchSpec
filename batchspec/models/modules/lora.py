"""LoRA implementation with gating support."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configs import LoRAConfig
from batchspec.profiler import cuda_bucket_timer


class GatedLoRALinear(nn.Module):
    """Linear layer with gated LoRA adaptation.
    
    Implements Linear layer with Gated LoRA for selective application of LoRA updates.
    This enables parameter-efficient fine-tuning while maintaining the base model's capabilities.
    
    Args:
        in_features: Size of input features
        out_features: Size of output features
        lora_config: Configuration for LoRA adaptation
        bias: Whether to include bias in base layer
        device: Device to place tensors on
        dtype: Data type for tensors
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_config: LoRAConfig,
        bias: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        assert not lora_config.lora_bias, "lora_bias is not supported for GatedLoRALinear"

        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_config.rank
        self.lora_scaling = lora_config.lora_scaling

        # Base linear layer
        self.base_layer = nn.Linear(in_features, out_features, bias=bias)
        
        # LoRA adaptation layers
        self.lora_A = nn.Linear(in_features, self.lora_rank, bias=False, device=device, dtype=dtype)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))

        # Hold lora.B.T() for faster forward operation
        self.register_buffer(
            "lora_BT", 
            torch.zeros(self.lora_rank, out_features, device=device, dtype=dtype),
            persistent=True
        )
        
        # Register hook for loading state dicts
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args, **kwargs):
        """Hook to handle loading from different checkpoint formats."""
        # Handle loading from non-LoRA checkpoints
        if prefix + "weight" in state_dict:
            W = state_dict.pop(prefix + "weight")
            state_dict[prefix + "base_layer.weight"] = W
        if prefix + "bias" in state_dict:
            b = state_dict.pop(prefix + "bias")
            state_dict[prefix + "base_layer.bias"] = b
            
        # Handle loading B matrix (transpose if needed)
        if prefix + "lora_B.weight" in state_dict:
            W = state_dict.pop(prefix + "lora_B.weight")
            state_dict[prefix + "lora_BT"] = W.t().contiguous()
    
    def forward(self, x: torch.Tensor, gate_mask: Optional[torch.Tensor] = None):
        """Forward pass with optional gated LoRA.
        
        Args:
            x: Input tensor of shape (..., in_features)
            gate_mask: Optional mask for gated LoRA updates.
                      Shape should be broadcastable with x.
                      
        Returns:
            Output tensor of shape (..., out_features)
        """
        # Get profiling buckets if configured
        base_bucket = getattr(self, "_prof_base_bucket", None)
        lora_bucket = getattr(self, "_prof_lora_bucket", None)

        # Base layer forward pass
        with cuda_bucket_timer(base_bucket):
            y = self.base_layer(x)
            
        # If no gate mask, return base output only
        if gate_mask is None:
            return y

        # Apply LoRA with gating
        with cuda_bucket_timer(lora_bucket):
            # Project to low-rank space
            z = self.lora_A(x)
            # Apply gating in the subspace
            z.mul_(gate_mask)

            # Efficient matrix multiplication with transposed B
            out_local = y.size(-1)
            y2d = y.reshape(-1, out_local)
            z2d = z.reshape(-1, self.lora_rank)
            
            # Add scaled LoRA update to base output
            y2d.addmm_(z2d, self.lora_BT, alpha=self.lora_scaling)
            y = y2d.view_as(y)
            
        return y
