"""Standard backend engine for autoregressive generation."""

from pathlib import Path
from typing import Optional

import torch
import flashinfer
from torch import Tensor
from transformers import PreTrainedTokenizer
from flashinfer.sampling import chain_speculative_sampling

from .base import BaseEngine, PageTable
from .utils import sample, apply_tp, get_sampling_probs
from batchspec.profiler import get_active_profiler, cpu_bucket_timer
from batchspec.models import get_model

class EAGLEChainEngine(BaseEngine):
    """Chain speculative decoding mode EAGLE backend engine for autoregressive generation.
    
    Supports prefill and decode phases with FlashInfer attention in chain mode.
    """
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
        eagle_name: str,
        checkpoint_path: Path,
        eagle_checkpoint_path: Path,
        use_tp: bool = False,
        use_eagle_tp: bool = False,
        rank_group = None,
        group = None
    ):
        """Load model from checkpoint.
        
        Args:
            model_name: Name of model configuration
            eagle_name: Name of EAGLE module configuration
            checkpoint_path: Path to target model checkpoint
            eagle_checkpoint_path: Path to EAGLE module checkpoint
            use_tp: Whether to use tensor parallelism
            use_eagle_tp: Whether to use tensor parallelism for EAGLE module (If set to True, EAGLE shares the same rank group with target model)
            rank_group: Rank group for TP
            group: Process group for distributed training
        """
        # Load model on meta device first
        with torch.device('meta'):
            model = get_model(model_name, "eagle", drafter_name=eagle_name)
        
        # Load checkpoint
        checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
        model.load_state_dict(checkpoint, assign=True, strict=False)

        eagle_checkpoint = torch.load(str(eagle_checkpoint_path), mmap=True, weights_only=True)
        model.eagle.load_state_dict(eagle_checkpoint, assign=True, strict=True)

        # Apply tensor parallelism if requested
        if use_tp:
            print("Applying tensor parallel to model...")
            apply_tp(model, rank_group, group=group)
        
        if use_eagle_tp:
            raise NotImplementedError("Tensor parallelism for EAGLE module is not implemented yet")
            # print("Applying tensor parallel to EAGLE module...")
            # apply_tp_eagle(model.eagle, rank_group, group=group)
        
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

        # Create target attention wrapper (causal attention kernel)
        self.target_attn_buffer = self._create_attention_buffer(attn_buffer_size_mb)
        self.target_attn_wrapper = self._create_attention_wrapper(batch_size, self.target_attn_buffer)
        
        # Create EAGLE attention wrapper (causal attention kernel)
        self.eagle_attn_buffer = self._create_attention_buffer(attn_buffer_size_mb)
        self.eagle_attn_wrapper = self._create_attention_wrapper(batch_size, self.eagle_attn_buffer)
        
        # Create EAGLE KV Page Table
        self.eagle_kv_page_table = PageTable(
            page_size=page_size,
            max_batch_size=batch_size,
            max_num_pages_per_request=(max_cachelen + page_size - 1) // page_size,
            device=self.device
        )

        # Setup model caches
        target_max_num_pages = self.kv_page_table.max_num_pages_per_request * batch_size
        eagle_max_num_pages = self.eagle_kv_page_table.max_num_pages_per_request * batch_size
        with torch.device(self.device):
            self.model.setup_caches(
                target_num_pages=target_max_num_pages,
                eagle_num_pages=eagle_max_num_pages,
                page_size=page_size,
                causal_attn_kernel=self.target_attn_wrapper,
                eagle_attn_kernel=self.eagle_attn_wrapper,
            )


    def setup_special_tokens(self):
        """Setup special tokens."""
        super().setup_special_tokens()
        self.draft_eos_token_id = self.model.eagle.convert_target_to_draft(self.eos_token_id)
        

    def compile(self):
        """Enable torch.compile for forward functions."""
        super().compile()
        self.forward = torch.compile(self.forward)
        self.eagle_forward = torch.compile(self.eagle_forward)


    def target_forward(self, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Single step forward through target model.
        
        Args:
            input_ids: Input token IDs [bsz, seq_len]
        Returns:
            Logits of shape (bsz, seq_len, target_vocab_size), Hidden states of shape (bsz, seq_len, hidden_size)
        """
        bsz, seq_len = input_ids.shape
        qo_indptr = torch.arange(bsz + 1, device=self.device, dtype=torch.int32) * seq_len
        position_ids = (self.kv_page_table.cachelens[:, None] + torch.arange(seq_len, device=self.device)[None, :]).flatten()

        self.pre_target_forward(qo_indptr)
        with torch.inference_mode():
            output = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                qo_indptr=qo_indptr,
                kv_page_table=self.kv_page_table,
            ) # [bsz, seq_len, target_vocab_size], [bsz, seq_len, hidden_size]
        
        return output


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


    def eagle_forward(self, input_ids: Tensor, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        """Single step forward through EAGLE module.
        
        Args:
            input_ids: Input token IDs [bsz, seq_len]
            hidden_states: Hidden states from target model [bsz, seq_len, hidden_size]
        Returns:
            Logits of shape (bsz, seq_len, draft_vocab_size), Hidden states of shape (bsz, seq_len, hidden_size)
        """
        bsz, seq_len = input_ids.shape
        qo_indptr = torch.arange(bsz + 1, device=self.device, dtype=torch.int32) * seq_len
        position_ids = (self.eagle_kv_page_table.cachelens[:, None] + torch.arange(seq_len, device=self.device)[None, :]).flatten()

        self.pre_eagle_forward(qo_indptr)
        with torch.inference_mode():
            output = self.model.eagle_forward(
                input_ids=input_ids,
                hidden_states=hidden_states,
                qo_indptr=qo_indptr,
                position_ids=position_ids,
                kv_page_table=self.eagle_kv_page_table,
            ) # [bsz, seq_len, draft_vocab_size], [bsz, seq_len, hidden_size]
        
        return output


    def pre_eagle_forward(self, qo_indptr: Tensor):
        """Prepare for EAGLE forward step."""
        self.eagle_kv_page_table.insert_kv(qo_indptr[1:] - qo_indptr[:-1]) # insert KV cache for the new tokens
        self.eagle_attn_wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=self.eagle_kv_page_table.paged_kv_indptr,
            paged_kv_indices=self.eagle_kv_page_table.paged_kv_indices,
            paged_kv_last_page_len=self.eagle_kv_page_table.paged_kv_last_page_len,
            num_qo_heads=self.model.eagle.config.n_head,
            num_kv_heads=self.model.eagle.config.n_local_heads,
            head_dim_qk=self.model.eagle.config.head_dim,
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
            logits, hidden_states = self.target_forward(chunk_input_ids)
            
            # EAGLE prefill
            # Shift input_ids to left by 1 token for EAGLE prefill
            if end_idx < seq_len:
                eagle_input_ids = input_ids[:, start_idx+1:end_idx+1]
                self.eagle_forward(eagle_input_ids, hidden_states)
            else:
                # For the last chunk, we drop the last hidden_states to match the input_ids length
                eagle_input_ids = input_ids[:, start_idx+1:seq_len]
                self.eagle_forward(eagle_input_ids, hidden_states[:, :-1])
            
        return (
            sample(logits[:, -1], top_p=self.top_p, top_k=self.top_k, temperature=self.temperature),
            hidden_states[:, -1:]
        )


    def draft(self, input_ids: Tensor, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Execute the draft step and return the next token predictions.
        
        Args:
            input_ids: Input token IDs [bsz, seq_len]
            hidden_states: Hidden states from target model [bsz, seq_len, hidden_size]
        Returns:
            Next token predictions [bsz, 1]
        """
        _, seq_len = input_ids.shape
        assert seq_len == 1 or seq_len == 2, f"The input length must be 1 or 2 for draft, but got {seq_len}"

        logits, hidden_states = self.eagle_forward(input_ids, hidden_states)
        draft_tokens = sample(logits[:, 0], top_p=self.top_p, top_k=self.top_k, temperature=self.temperature)
        draft_tokens = self.model.eagle.convert_draft_to_target(draft_tokens)

        return draft_tokens, logits, hidden_states


    def generate_batch(self, input_ids, max_gen_len: int, prefix_len: int):
        """
        Generate a batch of tokens using the EAGLE chain speculative decoding.
        
        If the force_budget is not set, the generation will terminate whenever at least one EOS token is generated.
        Otherwise, the generation will proceed until the budget is exhausted by replacing </think> tokens to 'Wait' or 'Alternatively'.
        
        Args:
            input_ids: [bsz, seq_len]
            max_gen_len: Maximum generation length
            prefix_len: Length of the prefix

        Returns:
            output: [bsz, max_gen_len * 3]
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
        draft_vocab_size = self.model.eagle.config.draft_vocab_size
        target_vocab_size = self.model.config.vocab_size

        # Transformations between target and draft vocabularies
        target_to_draft = self.model.eagle.target_to_draft
        
        # Define local variables
        model_steps = 0
        num_generated_tokens = torch.zeros(bsz, device=device, dtype=torch.int32)
        batch_indices = torch.arange(bsz, device=device)
        batch_indices_2d = torch.arange(bsz, device=device)[:, None]
        output = torch.zeros(bsz, max_gen_len * 3, device=device, dtype=torch.long)
        
        tokens_buffer = torch.zeros(bsz, 1+k, device=device, dtype=torch.long) # one is the prev step's next token, k for draft tokens
        logits_buffer = torch.zeros(bsz, k, draft_vocab_size, device=device, dtype=dtype) if not greedy else None
        
        # Prefill
        next_tokens, hidden_states = self.prefill(input_ids)
        
        # SANITY CHECK: The KV cache length must be equal to the prefix length
        assert torch.all(self.kv_page_table.cachelens == prefix_len), "The target model's KV cache length must be equal to the prefix length"
        assert torch.all(self.eagle_kv_page_table.cachelens == prefix_len-1), "The drafter's KV cache length must be equal to the prefix length minus 1"

        tokens_buffer[:, :1] = next_tokens
        
        # Initialize the profiler (NullProfiler when profiling=False)
        profiler = get_active_profiler()
        profiler.begin_run(bsz=bsz, label="decode", prefix_len=prefix_len)

        terminal = False
        use_double_buffer = False
        while num_generated_tokens.mean(dtype=torch.float32) < max_gen_len and not terminal:
            with profiler.step_timing_ctx():
                profiler.set_step_seq_len(self.kv_page_table.cachelens)
        
                # EAGLE Chain Draft
                # First draft, we need to handle the double buffer case
                if use_double_buffer:
                    pad_len = (~full_accept_mask).int() # indicate the padding length for the double buffer (0 for full accept sequences, 1 for others)
                    double_draft_tokens, double_draft_logits, double_hidden_states = self.draft(double_buffer, hidden_states)
                    self.eagle_kv_page_table.delete_kv(pad_len) # delete the KV cache for the padded tokens

                    draft_tokens = double_draft_tokens[batch_indices, pad_len-1][:, None]
                    hidden_states = double_hidden_states[batch_indices, pad_len-1][:, None]
                    
                    tokens_buffer[:, 1:2] = draft_tokens
                    if not greedy: logits_buffer[:, 0] = double_draft_logits[batch_indices, pad_len-1]
                    use_double_buffer = False
                else:
                    draft_tokens, draft_logits, hidden_states = self.draft(tokens_buffer[:, :1], hidden_states)
                    tokens_buffer[:, 1:2] = draft_tokens
                    if not greedy: logits_buffer[:, :1] = draft_logits

                # Subsequent drafts
                for i in range(1, k):
                    draft_tokens, draft_logits, hidden_states = self.draft(tokens_buffer[:, i:i+1], hidden_states)
                    tokens_buffer[:, i+1:i+2] = draft_tokens
                    if not greedy: logits_buffer[:, i:i+1] = draft_logits

                # Target verification
                target_logits, hidden_states = self.target_forward(tokens_buffer)

                # Evaluate the posterior
                if greedy:
                    bonus_tokens, accept_nums, eos_accepted = self.evaluate_posterior(tokens_buffer[:, 1:], target_logits.argmax(dim=-1))
                else:
                    # ========== Use Draft Vocabulary ===========
                    # Project the target logits to the draft vocabulary
                    target_logits = target_logits[..., target_to_draft] # [bsz, k+1, draft_vocab_size]
                    draft_tokens = self.model.eagle.convert_target_to_draft(tokens_buffer[:, 1:]) # [bsz, k]

                    bonus_tokens, accept_nums, eos_accepted = self.evaluate_posterior(draft_tokens, target_logits, logits_buffer)
                    # ==== Convert back to Target Vocabulary ====
                    bonus_tokens = self.model.eagle.convert_draft_to_target(bonus_tokens) # [bsz, 1]
                
                # Force budget
                if force_budget:
                    bonus_tokens, accept_nums = self.budget_forcing(tokens_buffer[:, 1:], bonus_tokens, accept_nums)
                
                # Register accept_nums to Profiler
                profiler.set_step_tokens(int(accept_nums.sum().item()))

                # Delete the KV cache for the rejected tokens
                # Note that the drafter's KV cache follows the target model's KV cache with a delay of 1 token.
                self.kv_page_table.delete_kv((k+1) - accept_nums)
                self.eagle_kv_page_table.delete_kv(torch.clamp(k - accept_nums, min=0))
                
                # SANITY CHECK: The KV cache delay must be 1 for the partially accepted sequences, and 2 for the fully accepted sequences.
                kv_cache_delay = self.kv_page_table.cachelens - self.eagle_kv_page_table.cachelens
                full_accept_mask = (accept_nums == k+1)
                assert torch.all(kv_cache_delay[~full_accept_mask] == 1), f"The KV cache delay must be 1 for the partially accepted sequences, but got {kv_cache_delay[~full_accept_mask]}"
                assert torch.all(kv_cache_delay[full_accept_mask] == 2), f"The KV cache delay must be 2 for the fully accepted sequences, but got {kv_cache_delay[full_accept_mask]}"

                # Update output
                write_indices = num_generated_tokens[:, None] + torch.arange(k + 1, device=device)[None, :] # [B, k+1]
                output[batch_indices_2d, write_indices] = tokens_buffer
                num_generated_tokens += accept_nums
                model_steps += 1

                # print(f"[Decode] tokens_buffer: {tokens_buffer}")
                # print(f"[Decode] bonus_tokens: {bonus_tokens}")

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
                    double_buffer = torch.zeros((bsz, 2), device=device, dtype=torch.long)
                    
                    full_accept_mask = (accept_nums == k+1)
                    double_buffer[:, 0] = torch.where(full_accept_mask, tokens_buffer[:, -1], bonus_tokens[:, 0])
                    double_buffer[:, 1] = torch.where(full_accept_mask, bonus_tokens[:, 0], 0)

                    hidden_states = torch.stack([
                        hidden_states[batch_indices, accept_nums-1-full_accept_mask.int()],
                        hidden_states[batch_indices, accept_nums-1],
                    ], dim=1)
                else:
                    hidden_states = hidden_states[batch_indices, accept_nums-1][:, None]
                
                # Check the terminal condition
                eos_accepted_or_generated = eos_accepted | (tokens_buffer[:, 0] == eos_token_id)
                if (not force_budget) and eos_accepted_or_generated.any():
                    # On the terminal step, we need to write the bonus tokens to the output
                    terminal = True
                    output[batch_indices, num_generated_tokens] = bonus_tokens[:, 0]
                    num_generated_tokens += 1
        
        profiler.end_run()
        self.kv_page_table.delete_kv(num_generated_tokens) # revert the KV cache to proceed next run with longer prefix
        self.eagle_kv_page_table.delete_kv(self.eagle_kv_page_table.cachelens - prefix_len) # revert the drafter's KV cache to proceed next run with longer prefix
        
        # SANITY CHECK: The KV cache length must be equal to the prefix length
        assert torch.all(self.kv_page_table.cachelens == prefix_len), "The target model's KV cache length must be equal to the prefix length"
        assert torch.all(self.eagle_kv_page_table.cachelens == prefix_len), "The drafter's KV cache length must be equal to the prefix length"

        return output, num_generated_tokens, model_steps


    # =============================== Helper functions ===============================
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
        _, draft_length = draft_tokens.shape
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
            assert draft_logits is not None, "Draft logits must be provided for non-greedy mode"
            assert target_preds_or_logits.dim() == 3, "Target predicted logits must be a 3D tensor (bsz, seq_len, vocab_size)"

            target_probs = get_sampling_probs(target_preds_or_logits, top_p=self.top_p, top_k=self.top_k, temperature=self.temperature)
            draft_probs = get_sampling_probs(draft_logits, top_p=self.top_p, top_k=self.top_k, temperature=self.temperature)
            output_tokens, _, emitted_nums = chain_speculative_sampling(draft_probs, draft_tokens, target_probs)
            accept_nums = emitted_nums + 1
            
            last_valid_idx = ((output_tokens != -1).to(torch.long) * torch.arange(output_tokens.size(1), device=output_tokens.device)).argmax(dim=1, keepdim=True)
            bonus_tokens = output_tokens.gather(1, last_valid_idx)
            eos_accepted = (self.draft_eos_token_id == output_tokens).any(dim=1, keepdim=True)
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