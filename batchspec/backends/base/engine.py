"""Base backend class for language model inference."""

from abc import abstractmethod
from typing import List

import torch
from transformers import PreTrainedTokenizer

from .kv_cache import KVCacheMixin
from .attn_wrapper import AttentionWrapperMixin


class BaseEngine(KVCacheMixin, AttentionWrapperMixin):
    """Base class for language model engines.
    
    Provides common functionality for all engines including:
    - Model loading
    - Cache setup
    - KV cache management (via KVCacheMixin)
    - Attention wrapper management (via AttentionWrapperMixin)
    - Sampling parameter setup
    - Compilation support
    
    Args:
        dtype: Data type for computations
        device: Device to run computations on
        tokenizer: HuggingFace tokenizer
    """
    
    def __init__(self,
        tokenizer: PreTrainedTokenizer,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda:0",
    ):
        """Initialize the engine.
        
        Args:
            tokenizer: HuggingFace tokenizer
            dtype: Data type for computations
            device: Device to run computations on
        """
        self.dtype = dtype
        self.device = device
        self.tokenizer = tokenizer
        
        self.model = None
        self.batch_size = None
        self.page_size = None
        self.prefill_chunk_size = None
        
        # Sampling parameters
        self.temperature = 0.0
        self.top_k = -1
        self.top_p = 1.0
        self.greedy = True
        self.force_budget = False
    
    @abstractmethod
    def load_model(self, *args, **kwargs):
        """Load model from checkpoint.
        
        Must be implemented by subclasses.
        """
        raise NotImplementedError()

    def setup_caches(
        self,
        max_batch_size: int = 1,
        max_cache_length: int = 2048,
        page_size: int = 16,
        prefill_chunk_size: int = 128,
        **kwargs
    ):
        """Setup KV caches and page management.
        
        Args:
            max_batch_size: Maximum batch size
            max_cache_length: Maximum cache length
            page_size: Size of each page
            prefill_chunk_size: Chunk size for prefill
            **kwargs: Additional backend-specific arguments
        """
        self.batch_size = max_batch_size
        self.page_size = page_size
        self.prefill_chunk_size = prefill_chunk_size

        # Calculate page requirements
        self.max_num_pages_per_request = (max_cache_length + page_size - 1) // page_size
        self.max_num_pages = max_batch_size * self.max_num_pages_per_request
        
        # Initialize KV cache state (from KVCacheMixin)
        self._init_kv_cache_state(
            max_batch_size,
            self.max_num_pages,
            self.max_num_pages_per_request,
            self.device
        )
    
    def setup_sampling_params(
        self,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 0.95,
        force_budget: bool = False,
    ):
        """Setup sampling parameters.
        
        Args:
            temperature: Sampling temperature (0.0 for greedy)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            force_budget: Force the generation until the budget is exhausted
        """
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.greedy = (temperature == 0.0)
        self.force_budget = force_budget

    def setup_special_tokens(
        self,
        suppress_token: str = '</think>',
        replace_tokens: List[str] = ['Wait', 'Alternatively']
    ):
        """Setup suppress and replace tokens.
        
        Args:
            suppress_token: Token to suppress
            replace_tokens: Tokens to replace
        """
        self.eos_token_id = self.tokenizer.eos_token_id
        self.suppress_token_id = self.tokenizer.encode(suppress_token, add_special_tokens=False)[0]
        self.replace_token_ids = torch.tensor(
            [self.tokenizer.encode(token, add_special_tokens=False)[0] for token in replace_tokens],
            dtype=torch.long,
            device=self.device
        )

    def compile(self):
        """Enable torch.compile for model inference.
        
        Configures torch compilation settings for optimal performance.
        """
        import torch._dynamo.config
        import torch._inductor.config
        
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.fx_graph_cache = True
        torch._functorch.config.enable_autograd_cache = True
