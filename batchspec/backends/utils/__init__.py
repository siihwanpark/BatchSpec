import random
import numpy as np
import torch

from .paging import PageManager
from .sampling import get_sampling_probs, sample, fix_invalid_draft_tokens
from .tensor_parallel import init_dist, apply_tp, apply_tp_eagle

__all__ = [
    "PageManager",
    "get_sampling_probs",
    "sample",
    "fix_invalid_draft_tokens",
    "init_dist",
    "apply_tp",
    "apply_tp_eagle",
]

def setup_seed(seed: int):
    """Setup random seeds for reproducibility.
    
    Sets seeds for:
    - PyTorch (CPU and CUDA)
    - NumPy
    - Python random
    - CuDNN deterministic mode
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

__all__.append("setup_seed")