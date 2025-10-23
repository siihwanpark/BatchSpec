"""Sampler Head for multi-token prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configs import ModelArgs
from .normalization import RMSNorm


class SamplerHeadBlock(nn.Module):
    """Basic building block for sampler head.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        eps: Epsilon for layer normalization
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.norm = RMSNorm(out_features, eps=eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through sampler head block.
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized activated output
        """
        return self.norm(F.silu(self.linear(x)))


class SamplerHead(nn.Module):
    """Sampler head for multi-token prediction.
    
    Args:
        config: Model configuration
        use_residual: Whether to use residual connections between blocks
    """
    
    def __init__(
        self, 
        config: ModelArgs,
        use_residual: bool = True,
    ):
        super().__init__()
        
        norm_eps = config.norm_eps
        hidden_size = config.dim
        self.use_residual = use_residual

        self.layers = nn.ModuleList([
            SamplerHeadBlock(2 * hidden_size, hidden_size, eps=norm_eps),
            SamplerHeadBlock(hidden_size, hidden_size, eps=norm_eps),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through sampler head.
        
        Args:
            x: Input tensor containing concatenated embeddings and hidden states
               Shape: (batch_size, seq_len, 2 * hidden_size)
               
        Returns:
            Processed features for multi-token prediction
            Shape: (batch_size, seq_len, hidden_size)
        """
        x = self.layers[0](x)
        y = self.layers[1](x)
        
        if self.use_residual:
            return x + y
        else:
            return y
