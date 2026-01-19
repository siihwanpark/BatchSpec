"""Base backend class for language model inference."""

from abc import abstractmethod
from typing import List, Optional

import torch
from torch import Tensor
from transformers import PreTrainedTokenizer
from flashinfer.sampling import chain_speculative_sampling

from .attn_wrapper import AttentionWrapperMixin
from .page_table import PageTable
from ..utils import get_sampling_probs, fix_invalid_draft_tokens


class BaseEngine(AttentionWrapperMixin):
    """Base class for language model engines.
    
    Provides common functionality for all engines including:
    - Model loading
    - Cache setup
    - KV page table management (via PageTable)
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
        batch_size: int = 1,
        max_cache_length: int = 2048,
        page_size: int = 16,
        prefill_chunk_size: int = 128,
    ):
        """Setup KV page table.
        
        Args:
            batch_size: Batch size
            max_cache_length: Maximum cache length
            page_size: Size of each page
            prefill_chunk_size: Chunk size for prefill
        """
        self.batch_size = batch_size
        self.prefill_chunk_size = prefill_chunk_size
        
        # Initialize KV page table
        self.kv_page_table = PageTable(
            page_size=page_size,
            batch_size=batch_size,
            max_num_pages_per_request=(max_cache_length + page_size - 1) // page_size,
            device=self.device
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
            suppress_token: Token to suppress (by default, ['</think>'])
            replace_tokens: Tokens to replace (by default, ['Wait', 'Alternatively'])
        """
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.suppress_token_ids = torch.tensor(
            [self.tokenizer.encode(suppress_token, add_special_tokens=False)[0], self.eos_token_id],
            dtype=torch.long, device=self.device)
        self.replace_token_ids = torch.tensor(
            [self.tokenizer.encode(token, add_special_tokens=False)[0] for token in replace_tokens],
            dtype=torch.long,device=self.device)

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

    # =============== Common Helper functions for Speculative Decoding =================
    def evaluate_posterior(
        self,
        draft_tokens: Tensor,
        target_preds_or_logits: Tensor,
        draft_logits: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Evaluate the posterior for the draft tokens.

        Args:
            draft_tokens (torch.Tensor): The tokens that are generated by the sampler. Shape: [bsz, draft_length]
            target_preds_or_logits (torch.Tensor):
                - (Greedy) The predicted tokens by the target model. Shape: [bsz, draft_length + 1]
                - (Sampling) The predicted logits by the target model. Shape: [bsz, draft_length + 1, target_vocab_size]
            draft_logits (torch.Tensor): The predicted logits by the draft model. Shape: [bsz, draft_length, target_vocab_size]

        Returns:
            bonus_tokens (torch.Tensor): The bonus tokens. Shape: [bsz, 1]
            accept_nums (torch.Tensor): The number of accepted tokens. Shape: [bsz, 1]
            eos_accepted (torch.Tensor): The flag of whether the EOS tokens are accepted. Shape: [bsz, 1]
        """
        bsz, draft_length = draft_tokens.shape

        if self.greedy:
            assert target_preds_or_logits.dim() == 2, "Target predicted tokens must be a 2D tensor (bsz, seq_len)"

            eos_condition = (draft_tokens == self.eos_token_id) # [bsz, draft_length]
            accept_flags_matrix = target_preds_or_logits[:, :draft_length] == draft_tokens # [bsz, draft_length]
            accept_flags_matrix = accept_flags_matrix.int().cumprod(dim=1).bool()
            accept_nums = accept_flags_matrix.sum(dim=1, keepdim=True) + 1  # [bsz, 1]
            
            bonus_tokens = target_preds_or_logits.gather(1, accept_nums - 1) # [bsz, 1]
            eos_accepted = (eos_condition & accept_flags_matrix).any(dim=1, keepdim=True) # [bsz, 1]
            
            return bonus_tokens, accept_nums[:, 0], eos_accepted

        else:
            """ Note that all the tokens here are in the draft vocabulary, not the target vocabulary. """
            assert target_preds_or_logits.dim() == 3, "Target predicted logits must be a 3D tensor (bsz, seq_len, vocab_size)"

            target_probs = get_sampling_probs(target_preds_or_logits, top_p=self.top_p, top_k=self.top_k, temperature=self.temperature)
            if draft_logits is not None:
                draft_probs = get_sampling_probs(draft_logits, top_p=self.top_p, top_k=self.top_k, temperature=self.temperature)
                draft_tokens = fix_invalid_draft_tokens(draft_tokens, draft_probs)
            else:
                # Construct the draft probability distribution (point mass distribution)
                # Used for N-gram drafter and MTP drafter
                vocab_size = target_preds_or_logits.shape[-1]
                draft_probs = torch.zeros((bsz, draft_length, vocab_size), dtype=target_preds_or_logits.dtype, device=target_preds_or_logits.device)
                draft_probs.scatter_(-1, draft_tokens[..., None], 1.0)
            
            output_tokens, _, emitted_nums = chain_speculative_sampling(draft_probs, draft_tokens, target_probs)
            accept_nums = emitted_nums + 1
            
            last_valid_idx = ((output_tokens != -1).to(torch.long) * torch.arange(output_tokens.size(1), device=output_tokens.device)).argmax(dim=1, keepdim=True)
            bonus_tokens = output_tokens.gather(1, last_valid_idx)
            if hasattr(self, 'draft_eos_token_id'):
                # Only for EAGLE
                eos_accepted = (self.draft_eos_token_id == output_tokens).any(dim=1, keepdim=True)
            else:
                eos_accepted = (self.eos_token_id == output_tokens).any(dim=1, keepdim=True)

            return bonus_tokens, accept_nums, eos_accepted
    
    
    def budget_forcing(
        self,
        draft_tokens: Tensor,
        bonus_tokens: Tensor,
        accept_nums: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Limits the number of accepted draft tokens when certain tokens appear.

        The accepted draft tokens are truncated at the first occurrence of a suppressed token
        in the draft. The bonus tokens are also forced to avoid suppressed tokens by
        replacing them with a safe alternative when necessary.

        Args:
            draft_tokens: [bsz, k] draft token ids
            bonus_tokens: [bsz, 1] bonus token ids
            accept_nums: [bsz] accept numbers

        Returns:
            updated_bonus_tokens: [bsz, 1] updated bonus token ids
            updated_accept_nums: [bsz] updated accept numbers
        """
        suppressed_accept_nums = accept_nums.clone()
        suppress_mask = (draft_tokens[..., None] == self.suppress_token_ids).any(dim=-1) # [bsz, k]

        suppress_indices = torch.argmax(suppress_mask.int(), dim=-1) # [bsz]
        suppress_indices[~suppress_mask.any(dim=-1)] = -1
        rows_to_update = suppress_indices != -1
        if rows_to_update.any():
            suppressed_accept_nums[rows_to_update] = suppress_indices[rows_to_update].to(suppressed_accept_nums.dtype)

        bonus_is_suppress = (bonus_tokens == self.suppress_token_ids).any(dim=-1) # [bsz]
        bonus_update_mask = rows_to_update | bonus_is_suppress
        if bonus_update_mask.any():
            num_to_replace = bonus_update_mask.sum()
            random_indices = torch.randint(0, self.replace_token_ids.shape[0], (num_to_replace,), device=bonus_tokens.device)
            bonus_tokens[bonus_update_mask, 0] = self.replace_token_ids[random_indices].to(bonus_tokens.dtype)

        return bonus_tokens, suppressed_accept_nums   
