"""Normalization layers."""

import torch
import torch.nn as nn
from torch import Tensor


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    This normalization technique normalizes the inputs by the root mean square,
    providing stability during training while being computationally efficient.
    
    Args:
        dim: The dimension of the input features
        eps: Small epsilon value to prevent division by zero
    """
    
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: Tensor) -> Tensor:
        """Apply RMS normalization."""
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of RMS normalization.
        
        Args:
            x: Input tensor of shape (..., dim)
            
        Returns:
            Normalized tensor with same shape as input
        """
        # Cast to float32 for numerical stability, then back to original dtype
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
