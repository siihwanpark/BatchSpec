"""Standalone backend engine for speculative decoding with standalone drafter."""

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

class StandaloneEngine(BaseEngine):
    """Backend engine for speculative decoding with standalone drafter."""
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda:0",
        draft_length: int = 4,
    ):
        super().__init__(tokenizer, dtype, device)
        self.draft_length = draft_length
    
    def load_model(
        self,
        model_name: str,
        drafter_name: str,
        checkpoint_path: Path,
        drafter_checkpoint_path: Path,
        use_tp: bool = False,
        use_drafter_tp: bool = False,
        rank_group = None,
        group = None
    ):
        """Load model from checkpoint.
        
        Args:
            model_name: Name of model configuration
            drafter_name: Name of drafter model configuration
            checkpoint_path: Path to target model checkpoint
            drafter_checkpoint_path: Path to drafter module checkpoint
            use_tp: Whether to use tensor parallelism
            use_drafter_tp: Whether to use tensor parallelism for drafter module (If set to True, drafter shares the same rank group with target model)
            rank_group: Rank group for TP
            group: Process group for distributed training
        """
        # Load model on meta device first
        with torch.device('meta'):
            model = get_model(model_name, "standard")
            drafter = get_model(drafter_name, "standard")
        
        # Load checkpoint
        checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
        model.load_state_dict(checkpoint, assign=True, strict=True)

        drafter_checkpoint = torch.load(str(drafter_checkpoint_path), mmap=True, weights_only=True)
        drafter.load_state_dict(drafter_checkpoint, assign=True, strict=True)

        # Apply tensor parallelism if requested
        if use_tp:
            print("Applying tensor parallel to target model...")
            apply_tp(model, rank_group, group=group)
        
        if use_tp and use_drafter_tp:
            print("Applying tensor parallel to drafter model...")
            apply_tp(drafter, rank_group, group=group)
        
        # Move to device and set to eval mode
        model = model.to(device=self.device, dtype=self.dtype)
        drafter = drafter.to(device=self.device, dtype=self.dtype)
        
        self.model = model.eval()
        self.drafter = drafter.eval()
    
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
        self.target_attn_buffer = self._create_attention_buffer(attn_buffer_size_mb)
        self.target_attn_wrapper = self._create_attention_wrapper(batch_size, self.target_attn_buffer)
        
        # Create drafter attention wrapper
        self.drafter_attn_buffer = self._create_attention_buffer(attn_buffer_size_mb)
        self.drafter_attn_wrapper = self._create_attention_wrapper(batch_size, self.drafter_attn_buffer)
        
        # Create drafter KV Page Table
        self.drafter_kv_page_table = PageTable(
            page_size=page_size,
            batch_size=batch_size,
            max_num_pages_per_request=(max_cachelen + page_size - 1) // page_size,
            device=self.device
        )

        # Setup model caches
        target_max_num_pages = self.kv_page_table.max_num_pages_per_request * batch_size
        drafter_max_num_pages = self.drafter_kv_page_table.max_num_pages_per_request * batch_size
        with torch.device(self.device):
            self.model.setup_caches(
                num_pages=target_max_num_pages,
                page_size=page_size,
                attn_kernel=self.target_attn_wrapper,
            )

            self.drafter.setup_caches(
                num_pages=drafter_max_num_pages,
                page_size=page_size,
                attn_kernel=self.drafter_attn_wrapper,
            )


    def init_cache(self):
        """Initialize the KV cache for the target and drafter models."""
        self.kv_page_table.clear_kv(self.model)
        self.drafter_kv_page_table.clear_kv(self.drafter)
        

    def compile(self):
        """Enable torch.compile for forward functions."""
        super().compile()
        self.target_forward = torch.compile(self.target_forward)
        self.drafter_forward = torch.compile(self.drafter_forward)


    def target_forward(self, input_ids: Tensor) -> Tensor:
        """Single step forward through target model.
        
        Args:
            input_ids: Input token IDs [bsz, seq_len]
        Returns:
            Logits of shape (bsz, seq_len, vocab_size)
        """
        bsz, seq_len = input_ids.shape
        qo_indptr = torch.arange(bsz + 1, device=self.device, dtype=torch.int32) * seq_len
        past_cachelens = self.kv_page_table.cachelens.clone()
        
        self.pre_target_forward(qo_indptr)
        with torch.inference_mode():
            logits = self.model(
                input_ids=input_ids,
                position_offsets=past_cachelens,
                qo_indptr=qo_indptr,
                kv_page_table=self.kv_page_table,
            ) # [bsz, seq_len, vocab_size]
        
        return logits


    def pre_target_forward(self, qo_indptr: Tensor):
        """Prepare for target forward step."""
        self.kv_page_table.insert_kv(qo_indptr[1:] - qo_indptr[:-1]) # insert KV cache for the new tokens
        self.target_attn_wrapper.plan(
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


    def drafter_forward(self, input_ids: Tensor) -> Tensor:
        """Single step forward through drafter model.
        
        Args:
            input_ids: Input token IDs [bsz, seq_len]
        Returns:
            Logits of shape (bsz, seq_len, vocab_size)
        """
        bsz, seq_len = input_ids.shape
        qo_indptr = torch.arange(bsz + 1, device=self.device, dtype=torch.int32) * seq_len
        past_cachelens = self.drafter_kv_page_table.cachelens.clone()
        
        self.pre_drafter_forward(qo_indptr)
        with torch.inference_mode():
            logits = self.drafter(
                input_ids=input_ids,
                position_offsets=past_cachelens,
                qo_indptr=qo_indptr,
                kv_page_table=self.drafter_kv_page_table,
            ) # [bsz, seq_len, vocab_size]
        
        return logits


    def pre_drafter_forward(self, qo_indptr: Tensor):
        """Prepare for drafter forward step."""
        self.drafter_kv_page_table.insert_kv(qo_indptr[1:] - qo_indptr[:-1]) # insert KV cache for the new tokens
        self.drafter_attn_wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=self.drafter_kv_page_table.paged_kv_indptr,
            paged_kv_indices=self.drafter_kv_page_table.paged_kv_indices,
            paged_kv_last_page_len=self.drafter_kv_page_table.paged_kv_last_page_len,
            num_qo_heads=self.drafter.config.n_head,
            num_kv_heads=self.drafter.config.n_local_heads,
            head_dim_qk=self.drafter.config.head_dim,
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
            start_idx = i*chunk_size
            end_idx = (i+1)*chunk_size
            chunk_input_ids = input_ids[:, start_idx:end_idx]
            
            # Target prefill
            logits = self.target_forward(chunk_input_ids)
            
            # Drafter prefill
            self.drafter_forward(chunk_input_ids)
            
        return sample(logits[:, -1], top_p=self.top_p, top_k=self.top_k, temperature=self.temperature)


    def draft(self, input_ids: Tensor) -> Tensor:
        """Execute the draft step and return the next token predictions.
        
        Args:
            input_ids: Input token IDs [bsz, seq_len]
        Returns:
            Logits of shape (bsz, seq_len, vocab_size)
        """
        _, seq_len = input_ids.shape
        assert seq_len == 1 or seq_len == 2, f"The input length must be 1 or 2 for draft, but got {seq_len}"

        logits = self.drafter_forward(input_ids)
        draft_tokens = sample(logits, top_p=self.top_p, top_k=self.top_k, temperature=self.temperature)

        return draft_tokens, logits


    def generate_batch(self, input_ids, max_gen_len: int, prefix_len: int):
        """
        Generate a batch of tokens using the standalone speculative decoding.
        
        If the force_budget is not set, the generation will terminate whenever at least one EOS token is generated.
        Otherwise, the generation will proceed until the budget is exhausted by replacing </think> or EOS tokens to 'Wait' or 'Alternatively'.
        
        Args:
            input_ids: [bsz, seq_len]
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
        device = self.device
        dtype = self.dtype

        force_budget = self.force_budget
        eos_token_id = self.eos_token_id
        greedy = self.greedy
        vocab_size = self.model.config.vocab_size

        # Define local variables
        model_steps = 0
        num_generated_tokens = torch.zeros(bsz, device=device, dtype=torch.int32)
        batch_indices = torch.arange(bsz, device=device)
        batch_indices_2d = torch.arange(bsz, device=device)[:, None]
        output = torch.zeros(bsz, max_gen_len * (k+1), device=device, dtype=torch.long)
        
        tokens_buffer = torch.zeros(bsz, 1+k, device=device, dtype=torch.long) # one is the prev step's next token, k for draft tokens
        logits_buffer = torch.zeros(bsz, k, vocab_size, device=device, dtype=dtype) if not greedy else None
        
        # Prefill
        next_tokens = self.prefill(input_ids)
        tokens_buffer[:, :1] = next_tokens

        # SANITY CHECK: The KV cache length must be equal to the prefix length
        assert torch.all(self.kv_page_table.cachelens == prefix_len), "The target model's KV cache length must be equal to the prefix length"
        assert torch.all(self.drafter_kv_page_table.cachelens == prefix_len), "The drafter's KV cache length must be equal to the prefix length"

        # Initialize the profiler (NullProfiler when profiling=False)
        profiler = get_active_profiler()
        profiler.begin_run(bsz=bsz, label="decode", prefix_len=prefix_len)

        terminal = False
        use_double_buffer = False
        while num_generated_tokens.mean(dtype=torch.float32) < max_gen_len and not terminal:
            with profiler.step_timing_ctx():
                # Standalone Draft
                # First draft, we need to handle the double buffer case
                if use_double_buffer:
                    pad_len = (~full_accept_mask).int() # indicate the padding length for the double buffer (0 for full accept sequences, 1 for others)
                    double_draft_tokens, double_draft_logits = self.draft(double_buffer)
                    self.drafter_kv_page_table.delete_kv(pad_len) # delete the KV cache for the padded tokens

                    draft_tokens = double_draft_tokens[batch_indices, pad_len-1][:, None]
                    tokens_buffer[:, 1:2] = draft_tokens
                    if not greedy: logits_buffer[:, 0] = double_draft_logits[batch_indices, pad_len-1]
                    use_double_buffer = False
                else:
                    draft_tokens, draft_logits = self.draft(tokens_buffer[:, :1])
                    tokens_buffer[:, 1:2] = draft_tokens
                    if not greedy: logits_buffer[:, :1] = draft_logits

                # Subsequent drafts
                for i in range(1, k):
                    draft_tokens, draft_logits = self.draft(tokens_buffer[:, i:i+1])
                    tokens_buffer[:, i+1:i+2] = draft_tokens
                    if not greedy: logits_buffer[:, i:i+1] = draft_logits

                # Target verification
                target_logits = self.target_forward(tokens_buffer)

                # Evaluate the posterior
                if greedy:
                    bonus_tokens, accept_nums, eos_accepted = self.evaluate_posterior(tokens_buffer[:, 1:], target_logits.argmax(dim=-1))
                else:
                    bonus_tokens, accept_nums, eos_accepted = self.evaluate_posterior(tokens_buffer[:, 1:], target_logits, logits_buffer)
                    bonus_tokens, accept_nums = self.sampling_fault_handler(tokens_buffer, bonus_tokens, accept_nums)
                
                # Force budget
                if force_budget:
                    bonus_tokens, accept_nums = self.budget_forcing(tokens_buffer[:, 1:], bonus_tokens, accept_nums)

                # Register accept_nums to Profiler
                profiler.set_step_tokens(int(accept_nums.sum().item()))

                # Delete the KV cache for the rejected tokens
                # Note that the drafter's KV cache follows the target model's KV cache with a delay of 1 token.
                self.kv_page_table.delete_kv((k+1) - accept_nums)
                self.drafter_kv_page_table.delete_kv(torch.clamp(k - accept_nums, min=0))
                
                # SANITY CHECK: The KV cache delay must be 1 for the partially accepted sequences, and 2 for the fully accepted sequences.
                kv_cache_delay = self.kv_page_table.cachelens - self.drafter_kv_page_table.cachelens
                full_accept_mask = (accept_nums == k+1)
                assert torch.all(kv_cache_delay[~full_accept_mask] == 0), f"The KV cache delay must be 0 for the partially accepted sequences, but got {kv_cache_delay[~full_accept_mask]}"
                assert torch.all(kv_cache_delay[full_accept_mask] == 1), f"The KV cache delay must be 1 for the fully accepted sequences, but got {kv_cache_delay[full_accept_mask]}"

                # Update output
                write_indices = num_generated_tokens[:, None] + torch.arange(k + 1, device=device)[None, :] # [B, k+1]
                output[batch_indices_2d, write_indices] = tokens_buffer
                num_generated_tokens += accept_nums
                model_steps += 1

                # Prepare inputs for the next iteration
                tokens_buffer[:, :1] = bonus_tokens
                if accept_nums.max() == k+1:
                    """ When there exists a fully accepted sequence, we need to use the double token buffer to handle the delayed forward of the drafter.
                        For instance, suppose that t0 is the bonus token from the previous iteration, then in this iteration, the draft would be:
                        t0 -> Drafter -> t1 -> Drafter -> t2 -> Drafter -> t3. At this point, the Drafter's KV cache has KV entries for [t0, t1, t2].
                        Next, the target model would verify [t0, t1, t2, t3] and suppose that all tokens are accepted with a bonus token t4.
                        Then, in the next iteration, the drafter should start from generating KV entries for [t3, t4] instead of [t4].
                        Thus, we need to use the double token buffer to handle this case."""
                    
                    use_double_buffer = True
                    full_accept_mask = (accept_nums == k+1)
                    double_buffer = torch.stack([
                        torch.where(full_accept_mask, tokens_buffer[:, -1], bonus_tokens[:, 0]),
                        torch.where(full_accept_mask, bonus_tokens[:, 0], 0),
                    ], dim=1)
                
                # Check the terminal condition
                eos_accepted_or_generated = eos_accepted | (tokens_buffer[:, 0] == eos_token_id)
                if (not force_budget) and eos_accepted_or_generated.any():
                    # On the terminal step, we need to write the bonus tokens to the output
                    terminal = True
                    output[batch_indices, num_generated_tokens] = bonus_tokens[:, 0].long()
                    num_generated_tokens += 1
        
        profiler.end_run()
        self.kv_page_table.delete_kv(self.kv_page_table.cachelens - prefix_len) # revert the KV cache to proceed next run with longer prefix
        self.drafter_kv_page_table.delete_kv(self.drafter_kv_page_table.cachelens - prefix_len) # revert the drafter's KV cache to proceed next run with longer prefix
        
        # SANITY CHECK: The KV cache length must be equal to the prefix length
        assert torch.all(self.kv_page_table.cachelens == prefix_len), "The target model's KV cache length must be equal to the prefix length"
        assert torch.all(self.drafter_kv_page_table.cachelens == prefix_len), "The drafter's KV cache length must be equal to the prefix length"

        return output, num_generated_tokens, model_steps

