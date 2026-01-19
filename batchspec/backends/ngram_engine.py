"""Backend engine for speculative decoding with N-gram draft (Prompt Lookup Decoding)."""

from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from transformers import PreTrainedTokenizer
from flashinfer.sampling import chain_speculative_sampling

from .base import BaseEngine, PageTable
from .utils import sample, apply_tp, get_sampling_probs
from batchspec.profiler import get_active_profiler, cpu_bucket_timer
from batchspec.models import get_model

class NGramDraftEngine(BaseEngine):
    """Backend engine for Speculative Decoding with N-gram draft.
    
    References:
        - https://github.com/apoorvumang/prompt-lookup-decoding/
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda:0",
        draft_length: int = 4,
        max_ngram_size: int = 3,
    ):
        super().__init__(tokenizer, dtype, device)
        self.draft_length = draft_length
        self.max_ngram_size = max_ngram_size
    
    def load_model(
        self,
        model_name: str,
        checkpoint_path: Path,
        use_tp: bool = False,
        rank_group = None,
        group = None
    ):
        """Load model from checkpoint.
        
        Args:
            model_name: Name of model configuration
            checkpoint_path: Path to target model checkpoint
            use_tp: Whether to use tensor parallelism
            rank_group: Rank group for TP
            group: Process group for distributed training
        """
        # Load model on meta device first
        with torch.device('meta'):
            model = get_model(model_name, "standard")
        
        # Load checkpoint
        checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
        model.load_state_dict(checkpoint, assign=True, strict=True)

        # Apply tensor parallelism if requested
        if use_tp:
            print("Applying tensor parallel to target model...")
            apply_tp(model, rank_group, group=group)
        
        # Move to device and set to eval mode
        model = model.to(device=self.device, dtype=self.dtype)
        self.model = model.eval()
    
    def setup_caches(
        self,
        batch_size: int = 1,
        max_seq_length: int = 2048,
        page_size: int = 16,
        prefill_chunk_size: int = 128,
        attn_buffer_size_mb: int = 384,
    ):
        """Setup KV caches and attention wrappers.
        
        Args:
            batch_size: Batch size
            max_seq_length: Maximum sequence length
            page_size: Size of each page
            prefill_chunk_size: Chunk size for prefill
            attn_buffer_size_mb: Size of attention buffer in MB
        """
        # Setup base caches (add 1 for potential decode token)
        self.max_seq_len = max_seq_length
        self.page_size = page_size

        max_cachelen = max_seq_length + self.draft_length + 1
        super().setup_caches(batch_size, max_cachelen, page_size, prefill_chunk_size)

        # Create target attention wrapper
        self.attn_buffer = self._create_attention_buffer(attn_buffer_size_mb)
        self.attn_wrapper = self._create_attention_wrapper(batch_size, self.attn_buffer)

        # Setup model caches
        max_num_pages = self.kv_page_table.max_num_pages_per_request * batch_size
        with torch.device(self.device):
            self.model.setup_caches(
                num_pages=max_num_pages,
                page_size=page_size,
                attn_kernel=self.attn_wrapper,
            )


    def init_cache(self):
        """Initialize the KV cache for the target model."""
        self.kv_page_table.clear_kv(self.model)
        

    def compile(self):
        """Enable torch.compile for forward functions."""
        super().compile()
        self.forward = torch.compile(self.forward)


    def forward(self, input_ids: Tensor) -> Tensor:
        """Single step forward through target model.
        
        Args:
            input_ids: Input token IDs [bsz, seq_len]
        Returns:
            Logits of shape (bsz, seq_len, vocab_size)
        """
        bsz, seq_len = input_ids.shape
        qo_indptr = torch.arange(bsz + 1, device=self.device, dtype=torch.int32) * seq_len
        past_cachelens = self.kv_page_table.cachelens.clone()
        
        self.pre_forward(qo_indptr)
        with torch.inference_mode():
            logits = self.model(
                input_ids=input_ids,
                position_offsets=past_cachelens,
                qo_indptr=qo_indptr,
                kv_page_table=self.kv_page_table,
            ) # [bsz, seq_len, vocab_size]
        
        return logits


    def pre_forward(self, qo_indptr: Tensor):
        """Prepare for target forward step."""
        self.kv_page_table.insert_kv(qo_indptr[1:] - qo_indptr[:-1]) # insert KV cache for the new tokens
        self.attn_wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=self.kv_page_table.paged_kv_indptr,
            paged_kv_indices=self.kv_page_table.paged_kv_indices,
            paged_kv_last_page_len=self.kv_page_table.paged_kv_last_page_len,
            num_qo_heads=self.model.config.n_head,
            num_kv_heads=self.model.config.n_local_heads,
            head_dim_qk=self.model.config.head_dim,
            page_size=self.page_size,
            q_data_type=self.dtype,
            causal=True,
        )


    def prefill(self, input_ids: Tensor) -> Tensor:
        """Execute the chunked prefill with the pre-defined prefill chunk size and return the last token predictions.
        
        Args:
            input_ids: Input token IDs [bsz, seq_len]
            
        Returns:
            Next token predictions [bsz, 1]
        """
        _, seq_len = input_ids.shape
        assert seq_len % self.prefill_chunk_size == 0, f"The sequence length must be divisible by the prefill chunk size, but got seq_len={seq_len} and prefill_chunk_size={self.prefill_chunk_size}"
        
        chunk_size = self.prefill_chunk_size
        num_chunks = seq_len // chunk_size
        for i in range(num_chunks):
            logits = self.forward(input_ids[:, i*chunk_size:(i+1)*chunk_size])
        return sample(logits[:, -1], top_p=self.top_p, top_k=self.top_k, temperature=self.temperature)


    def draft(self, input_ids: Tensor, max_ngram_size: int, lengths: Tensor) -> Tensor:
        """Find N-gram continuation candidates from padded batched input_ids.

        Changes vs. original PLD (for higher acceptance):
        1) Pick the MOST RECENT match (last_idx) instead of the earliest match.
        2) Relax overlap rule: only exclude the self-match window at the end,
            i.e., require idx <= (L - n - 1). (Keep continuation-in-bounds constraint.)

        Args:
            input_ids: [bsz, seqlen] right-padded
            max_ngram_size: maximum n-gram size to try
            lengths: [bsz] valid lengths (excluding padding)
        Returns:
            draft_tokens: [bsz, draft_length] (pad-filled if not found)
        """
        assert input_ids.dim() == 2, "input_ids must be [bsz, seqlen]"
        bsz, seqlen = input_ids.shape
        n_cont = self.draft_length
        device = input_ids.device
        dtype = input_ids.dtype

        draft_tokens = torch.full((bsz, n_cont), self.pad_token_id, dtype=dtype, device=device)
        found = torch.zeros((bsz,), dtype=torch.bool, device=device)

        lengths = lengths.to(device=device, dtype=torch.long)

        # Valid token mask: pos < lengths (right-padding)
        pos = torch.arange(seqlen, device=device)
        valid_tok = pos[None, :] < lengths[:, None]

        for n in range(min(max_ngram_size, seqlen), 0, -1):
            need = ~found
            if not need.any():
                break

            # Must be able to take a suffix of length n and a continuation of length n_cont
            active = need & (lengths >= n) & (lengths >= n + n_cont)
            if not active.any():
                continue

            num_windows = seqlen - n + 1
            if num_windows <= 0:
                continue

            # windows: [bsz, num_windows, n]
            windows = input_ids.unfold(dimension=1, size=n, step=1)

            # window validity: exclude any window touching padding
            win_valid = valid_tok.unfold(dimension=1, size=n, step=1).all(dim=2)

            # suffix: last n tokens per sample (excluding padding)
            base = torch.clamp(lengths - n, min=0)
            offs = torch.arange(n, device=device)[None, :]
            suffix_pos = base[:, None] + offs
            suffix = input_ids.gather(1, suffix_pos)

            matches = (windows == suffix[:, None, :]).all(dim=2)
            matches = matches & win_valid & active[:, None]

            idxs = torch.arange(num_windows, device=device)

            # Continuation must fit in valid length:
            # start = idx + n, end = start + n_cont <= L  => idx <= L - n - n_cont
            max_idx_for_cont = lengths - n - n_cont

            # Relaxed overlap rule: exclude only self-match (the last window at idx = L - n)
            # So require idx <= L - n - 1
            max_idx_exclude_self = lengths - n - 1
            max_valid_idx = torch.minimum(max_idx_for_cont, max_idx_exclude_self)

            valid_pos = idxs[None, :] <= max_valid_idx[:, None]
            valid_matches = matches & valid_pos

            # Pick MOST RECENT match (largest idx). If none, get -1.
            neg1 = torch.tensor(-1, device=device, dtype=torch.long)
            idx_matrix = idxs[None, :].expand(bsz, -1)
            masked_idx = torch.where(valid_matches, idx_matrix, neg1)
            last_idx = masked_idx.max(dim=1).values

            update = active & (last_idx >= 0)
            if not update.any():
                continue

            start = last_idx[update] + n
            cont_pos = start[:, None] + torch.arange(n_cont, device=device)[None, :]
            draft_tokens[update] = input_ids[update].gather(1, cont_pos)

            found[update] = True

        return draft_tokens


    def generate_batch(self, input_ids, full_input_ids, max_gen_len: int, prefix_len: int):
        """
        Generate a batch of tokens using the standalone speculative decoding.
        
        If the force_budget is not set, the generation will terminate whenever at least one EOS token is generated.
        Otherwise, the generation will proceed until the budget is exhausted by replacing </think> or EOS tokens to 'Wait' or 'Alternatively'.
        
        Args:
            input_ids: [bsz, seq_len]
            full_input_ids: [bsz, prefix_len]
            max_gen_len: Maximum generation length
            prefix_len: Length of the prefix

        Returns:
            output: [bsz, max_gen_len * (draft_length+1)] # upper bound of the output length (when all the drafts are accepted)
            num_generated_tokens: [bsz]
            model_steps: int
        """
        # Pre-link local variables to reduce indexing time
        bsz = input_ids.shape[0]
        k = self.draft_length
        max_ngram_size = self.max_ngram_size
        device = self.device

        force_budget = self.force_budget
        eos_token_id = self.eos_token_id
        greedy = self.greedy

        # Define local variables
        model_steps = 0
        num_generated_tokens = torch.zeros(bsz, device=device, dtype=torch.int32)
        batch_indices = torch.arange(bsz, device=device)
        batch_indices_2d = torch.arange(bsz, device=device)[:, None]
        
        output = torch.zeros(bsz, max_gen_len * (k+1), device=device, dtype=torch.long)
        tokens_buffer = torch.zeros(bsz, 1+k, device=device, dtype=torch.long) # one is the prev step's next token, k for draft tokens
        
        # Prefill
        next_tokens = self.prefill(input_ids)
        output[:, :1] = next_tokens
        num_generated_tokens += 1
        model_steps += 1

        tokens_buffer[:, :1] = next_tokens

        # SANITY CHECK: The KV cache length must be equal to the prefix length
        assert torch.all(self.kv_page_table.cachelens == prefix_len), "The target model's KV cache length must be equal to the prefix length"

        # Initialize the profiler (NullProfiler when profiling=False)
        profiler = get_active_profiler()
        profiler.begin_run(bsz=bsz, label="decode", prefix_len=prefix_len)

        terminal = False
        while num_generated_tokens.mean(dtype=torch.float32) < max_gen_len and not terminal:
            with profiler.step_timing_ctx():
                # N-gram Draft
                full_sequence = torch.cat([full_input_ids, output[:, :num_generated_tokens.max()]], dim=1)
                tokens_buffer[:, 1:] = self.draft(full_sequence, max_ngram_size, prefix_len + num_generated_tokens)

                # Target verification
                target_logits = self.forward(tokens_buffer)

                # Evaluate the posterior
                if greedy:
                    bonus_tokens, accept_nums, eos_accepted = self.evaluate_posterior(tokens_buffer[:, 1:], target_logits.argmax(dim=-1))
                else:
                    bonus_tokens, accept_nums, eos_accepted = self.evaluate_posterior(tokens_buffer[:, 1:], target_logits)
                
                # Force budget
                if force_budget:
                    bonus_tokens, accept_nums = self.budget_forcing(tokens_buffer[:, 1:], bonus_tokens, accept_nums)

                # Register accept_nums to Profiler
                profiler.set_step_tokens(int(accept_nums.sum().item()))

                # Delete the KV cache for the rejected tokens
                self.kv_page_table.delete_kv((k+1) - accept_nums)
                
                # Update output
                write_indices = num_generated_tokens[:, None] + torch.arange(k, device=device)[None, :] # [B, k]
                output[batch_indices_2d, write_indices] = tokens_buffer[:, 1:]
                num_generated_tokens += accept_nums - 1
                model_steps += 1

                # Prepare inputs for the next iteration
                tokens_buffer[:, :1] = bonus_tokens
                output[batch_indices, num_generated_tokens] = bonus_tokens[:, 0]
                num_generated_tokens += 1
                
                # Check the terminal condition
                eos_accepted_or_generated = eos_accepted | (tokens_buffer[:, 0] == eos_token_id)
                if (not force_budget) and eos_accepted_or_generated.any():
                    # On the terminal step, we need to write the bonus tokens to the output
                    terminal = True
                    output[batch_indices, num_generated_tokens] = bonus_tokens[:, 0]
                    num_generated_tokens += 1
        
        profiler.end_run()
        self.kv_page_table.delete_kv(self.kv_page_table.cachelens - prefix_len) # revert the KV cache to proceed next run with longer prefix
        
        # SANITY CHECK: The KV cache length must be equal to the prefix length
        assert torch.all(self.kv_page_table.cachelens == prefix_len), "The target model's KV cache length must be equal to the prefix length"

        return output, num_generated_tokens, model_steps

