"""MagicDec backend engine for speculative decoding with StreamingLLM drafter."""

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

class MagicDecEngine(BaseEngine):
    """
    Backend engine for speculative decoding using a MagicDec-style drafter
    integrated with StreamingLLM for long-context generation.

    Reference:
        MagicDec: https://arxiv.org/abs/2408.11049

    This engine adapts MagicDec to long-generation regimes by redesigning
    the drafter to operate under a constrained KV-cache budget, following
    the StreamingLLM paradigm.

    Design overview:
        - The drafter is instantiated as the target model with a compressed
        KV cache maintained under a fixed budget (`kv_budget`).

        - Speculative decoding is *not* applied until the target model's
        KV cache size reaches twice the `kv_budget`.
        * Rationale: starting speculation immediately at `kv_budget`
            yields little benefit, as the drafter and target have comparable
            latency in this regime.

        - Once the target KV cache exceeds `2 × kv_budget`:
            - A compressed KV cache for the drafter is constructed following
            StreamingLLM (i.e., a small set of sink tokens and the most
            recent tokens).
            - Additional slack is reserved in the drafter KV cache to
            accommodate future speculative decoding steps.
            - Speculative decoding is performed using the drafter model.

        - If the drafter's KV cache reaches `kv_budget` during decoding,
        a portion of past tokens is evicted to free space for subsequent
        speculative steps.

        - This process is repeated until generation terminates.
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
            model = get_model(model_name, "magicdec")
        
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
        num_sink_tokens: int = 16,
        stream_budget: int = 1024,
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

        # Create attention wrapper
        self.attn_buffer = self._create_attention_buffer(attn_buffer_size_mb)
        self.attn_wrapper = self._create_attention_wrapper(batch_size, self.attn_buffer)
        
        # Create draft KV Page Table
        assert num_sink_tokens < stream_budget, f"The number of sink tokens must be less than the stream budget, but got num_sink_tokens={num_sink_tokens} and stream_budget={stream_budget}"
        self.num_sink_tokens = num_sink_tokens
        self.stream_budget = stream_budget
        self.draft_kv_page_table = PageTable(
            page_size=page_size,
            batch_size=batch_size,
            max_num_pages_per_request=(stream_budget + page_size - 1) // page_size,
            device=self.device
        )

        # Setup model caches
        max_num_pages = self.kv_page_table.max_num_pages_per_request * batch_size
        with torch.device(self.device):
            self.model.setup_caches(
                num_pages=max_num_pages,
                page_size=page_size,
                attn_kernel=self.attn_wrapper,
                batch_size=batch_size,
                stream_budget=stream_budget,
            )


    def init_cache(self):
        """Initialize the KV cache for the target and buffer KV caches."""
        self.kv_page_table.clear_kv(self.model)
        self.draft_kv_page_table.clear_kv()
        

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
            Logits of shape (bsz, seq_len, target_vocab_size)
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


    def drafter_forward(self, input_ids: Tensor) -> Tensor:
        """Single step forward through StreamingLLM drafter model.
        
        Args:
            input_ids: Input token IDs [bsz, seq_len]
        Returns:
            Logits of shape (bsz, seq_len, vocab_size)
        """
        bsz, seq_len = input_ids.shape
        qo_indptr = torch.arange(bsz + 1, device=self.device, dtype=torch.int32) * seq_len
        past_cachelens = self.draft_kv_page_table.cachelens.clone()

        self.pre_drafter_forward(qo_indptr)
        with torch.inference_mode():
            logits = self.model(
                input_ids=input_ids,
                position_offsets=past_cachelens,
                qo_indptr=qo_indptr,
                kv_page_table=self.draft_kv_page_table,
                draft=True,
            ) # [bsz, seq_len, vocab_size]
        
        return logits


    def pre_drafter_forward(self, qo_indptr: Tensor):
        """Prepare for StreamingLLM drafter forward step."""
        self.draft_kv_page_table.insert_kv(qo_indptr[1:] - qo_indptr[:-1]) # insert KV cache for the new tokens
        self.attn_wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=self.draft_kv_page_table.paged_kv_indptr,
            paged_kv_indices=self.draft_kv_page_table.paged_kv_indices,
            paged_kv_last_page_len=self.draft_kv_page_table.paged_kv_last_page_len,
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
            start_idx = i*chunk_size
            end_idx = (i+1)*chunk_size
            chunk_input_ids = input_ids[:, start_idx:end_idx]
            
            # Target prefill
            logits = self.target_forward(chunk_input_ids)

            # Drafter prefill
            self.evict_draft_kv(required_length=chunk_size)
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

        # Counter for the number of KV cache appends for the drafter
        # Since the StreamingKVCache makes the KV cache length dynamic,
        # we need to keep track of the number of KV cache appends to revert the KV cache correctly
        draft_kv_append_cnt = torch.zeros(bsz, device=device, dtype=torch.int32) 
        
        # Prefill
        next_tokens = self.prefill(input_ids)
        tokens_buffer[:, :1] = next_tokens

        # SANITY CHECK: The KV cache length must be equal to the prefix length
        assert torch.all(self.kv_page_table.cachelens == prefix_len), "The target model's KV cache length must be equal to the prefix length"
        assert torch.all(self.draft_kv_page_table.cachelens == prefix_len) \
                or torch.all(self.draft_kv_page_table.cachelens == self.stream_budget), \
                f"The drafter's KV cache length must be equal to the prefix length or the stream budget, but got {self.draft_kv_page_table.cachelens}"

        # Initialize the profiler (NullProfiler when profiling=False)
        profiler = get_active_profiler()
        profiler.begin_run(bsz=bsz, label="decode", prefix_len=prefix_len)

        terminal = False
        use_double_buffer = False
        while num_generated_tokens.mean(dtype=torch.float32) < max_gen_len and not terminal:
            with profiler.step_timing_ctx():
                # MagicDec Draft
                # Evict the draft KV cache at the start of the draft generation
                self.evict_draft_kv(required_length=k+1 if use_double_buffer else k)
                assert torch.all(self.stream_budget - self.draft_kv_page_table.cachelens >= (k+1 if use_double_buffer else k)), \
                        f"The spare KV cache must be enough for the draft generation, but got {self.stream_budget - self.draft_kv_page_table.cachelens}"

                # First draft, we need to handle the double buffer case
                if use_double_buffer:
                    pad_len = (~full_accept_mask).int() # indicate the padding length for the double buffer (0 for full accept sequences, 1 for others)
                    double_draft_tokens, double_draft_logits = self.draft(double_buffer)
                    self.draft_kv_page_table.delete_kv(pad_len) # delete the KV cache for the padded tokens

                    draft_tokens = double_draft_tokens[batch_indices, pad_len-1][:, None]
                    tokens_buffer[:, 1:2] = draft_tokens
                    if not greedy: logits_buffer[:, 0] = double_draft_logits[batch_indices, pad_len-1]
                    use_double_buffer = False
                    draft_kv_append_cnt += 2 - pad_len
                else:
                    draft_tokens, draft_logits = self.draft(tokens_buffer[:, :1])
                    tokens_buffer[:, 1:2] = draft_tokens
                    if not greedy: logits_buffer[:, :1] = draft_logits
                    draft_kv_append_cnt += 1

                # Subsequent drafts
                for i in range(1, k):
                    draft_tokens, draft_logits = self.draft(tokens_buffer[:, i:i+1])
                    tokens_buffer[:, i+1:i+2] = draft_tokens
                    if not greedy: logits_buffer[:, i:i+1] = draft_logits
                    draft_kv_append_cnt += 1

                # Target verification
                target_logits = self.target_forward(tokens_buffer)

                # Evaluate the posterior
                if greedy:
                    bonus_tokens, accept_nums, eos_accepted = self.evaluate_posterior(tokens_buffer[:, 1:], target_logits.argmax(dim=-1))
                else:
                    bonus_tokens, accept_nums, eos_accepted = self.evaluate_posterior(tokens_buffer[:, 1:], target_logits, logits_buffer)
                
                # Force budget
                if force_budget:
                    bonus_tokens, accept_nums = self.budget_forcing(tokens_buffer[:, 1:], bonus_tokens, accept_nums)

                # Register accept_nums to Profiler
                profiler.set_step_tokens(int(accept_nums.sum().item()))

                # Delete the KV cache for the rejected tokens
                # Note that the drafter's KV cache follows the target model's KV cache with a delay of 1 token.
                self.kv_page_table.delete_kv((k+1) - accept_nums)
                self.draft_kv_page_table.delete_kv(torch.clamp(k - accept_nums, min=0))
                draft_kv_append_cnt -= torch.clamp(k - accept_nums, min=0)
                
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
                    output[batch_indices, num_generated_tokens] = bonus_tokens[:, 0]
                    num_generated_tokens += 1
        
        profiler.end_run()
        self.kv_page_table.delete_kv(self.kv_page_table.cachelens - prefix_len) # revert the target model's KV cache to proceed next run with longer prefix
        self.draft_kv_page_table.delete_kv(draft_kv_append_cnt) # revert the drafter's KV cache to proceed next run with longer prefix
        
        # SANITY CHECK: The KV cache length must be equal to the prefix length
        assert torch.all(self.kv_page_table.cachelens == prefix_len), "The target model's KV cache length must be equal to the prefix length"

        return output, num_generated_tokens, model_steps


    # =============================== Helper functions ===============================
    def evict_draft_kv(self, required_length: int):
        """
        Evict the KV cache for the StreamingLLM drafter to secure enough margin for the required length.

        Args:
            required_length: The required length of the KV cache
        """
        draft_cachelens = self.draft_kv_page_table.cachelens
        num_evicts = torch.clamp(draft_cachelens - (self.stream_budget - required_length), min=0)
        if torch.all(num_evicts == 0):
            return # no need to evict the KV cache

        for layer in self.model.layers:
            layer.attention.draft_kv_cache.evict_kv(
                num_evicts=num_evicts,
                num_sink_tokens=self.num_sink_tokens,
                cachelens=draft_cachelens,
                n_local_heads=self.model.config.n_local_heads,
                head_dim=self.model.config.head_dim,
            )
        
        self.draft_kv_page_table.delete_kv(num_evicts)
    
