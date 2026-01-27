"""MTP (Multi-Token Prediction) engine for self-speculative decoding (Continuous version).

This engine implements self-speculative decoding with LoRA and gated sampling.
"""

from typing import Tuple, List, Optional
from pathlib import Path

import torch
from torch import Tensor
from transformers import PreTrainedTokenizer
from flashinfer.sampling import chain_speculative_sampling

from batchspec.backends import MTPEngine
from batchspec.backends.utils import get_sampling_probs, sample, apply_tp
from batchspec.models import get_model, LoRAConfig
from batchspec.profiler import get_active_profiler, cpu_bucket_timer
from .base import (
    Sequence, Status, BatchPack, MTPBatchBuilderMixin, MTPScheduler, SchedulerTracer, PrefillScheduler,
)


class MTPContinuousEngine(MTPEngine, MTPBatchBuilderMixin):
    """MTP engine for self-speculative decoding with multi-token prediction (Continuous version)."""
    
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
        # Setup base caches
        self.max_seq_len = max_seq_length
        self.draft_and_verify_len = (self.draft_length + 1) ** 2
        self.max_cache_len = max_seq_length + self.draft_and_verify_len + 1
        self.page_size = page_size

        super().setup_caches(batch_size, self.max_cache_len, page_size, prefill_chunk_size)
        
        # Set common attention masks and position ids
        self.setup_common_attn_mask_and_position_ids()

        # Create non-causal attention wrapper
        self.non_causal_attn_buffer = self._create_attention_buffer(attn_buffer_size_mb)
        max_bytes_for_attn_masks = (batch_size * (self.draft_and_verify_len * self.max_cache_len)) // 8 + 1
        self.custom_mask_buf = torch.empty(max_bytes_for_attn_masks, dtype=torch.uint8, device=self.device)
        self.non_causal_attn_wrapper = self._create_attention_wrapper(
            batch_size, self.non_causal_attn_buffer,
            use_custom_mask=True,
            custom_mask_buf=self.custom_mask_buf,
        )

        # Setup model caches
        max_num_pages = self.kv_page_table.max_num_pages_per_request * batch_size
        with torch.device(self.device):
            self.model.setup_caches(
                num_pages=max_num_pages,
                page_size=page_size,
                causal_attn_kernel=None,
                non_causal_attn_kernel=self.non_causal_attn_wrapper,
            )
        
        # Setup scheduler for continuous batching
        self.scheduler = MTPScheduler(
            max_concurrency=batch_size,
            prefill_chunk_len=prefill_chunk_size,
            draft_length=self.draft_length,
        )

        # Setup tracer
        self.tracer = SchedulerTracer(self.scheduler, color=True, show_queues=True)


    def set_stop_on_tail(self, stop_on_tail: bool):
        """Set whether to stop the generation immediately upon reaching the tail generation."""
        self.stop_on_tail = stop_on_tail


    def forward(self, batch: BatchPack) -> Tensor:
        """Single step MTP forward of continuous batching.
        
        Args:
            batch: BatchPack of input data
            
        Returns:
            logits: [nnz, vocab_size]
            hidden_states: [nnz, hidden_size]
        """
        self.pre_forward(qo_indptr=batch.qo_indptr, attn_mask=batch.attn_mask)
        with torch.inference_mode():
            logits, hidden_states = self.model(
                input_ids=batch.input_ids,
                gate_mask=batch.gate_mask,
                qo_indptr=batch.qo_indptr,
                position_ids=batch.position_ids_or_offsets,
                kv_page_table=self.kv_page_table,
                causal=False,
            ) # [nnz, vocab_size], [nnz, hidden_size]
        
        return logits, hidden_states
    

    def generate(self, sequences: List[Sequence], clear_kv: bool = True):
        """
        Generate tokens from prompts.

        Args:
            sequences: List[Sequence], the sequences to generate from.

        Returns:
            List[Sequence], the completed sequences.
        """
        # Clear KV cache and reset steps counter
        steps = 0
        draft_len = self.draft_length
        if clear_kv:
            self.kv_page_table.clear_kv(self.model)
        
        # Add sequences to the scheduler
        self.scheduler.add_sequences(sequences)
        
        # Setup buffers (next_tokens for PREFILL, other buffers for FIRST_DRAFT and DRAFT_VERIFY)
        next_tokens = torch.full((self.batch_size,), -1, device=self.device, dtype=torch.long)
        draft_buffer = torch.empty((self.batch_size, draft_len+1), device=self.device, dtype=torch.long)
        verify_buffer = torch.empty((self.batch_size, draft_len+1), device=self.device, dtype=torch.long)
        accept_nums_buffer = torch.zeros((self.batch_size,), device=self.device, dtype=torch.int32)

        # Initialize the profiler (NullProfiler when profiling=False)
        profiler = get_active_profiler()
        profiler.begin_run(bsz=self.batch_size, label="generate", prefix_len=0)

        # Main loop
        while not self.scheduler.is_done():
            with profiler.step_timing_ctx():
                # Flush the accept numbers from the previous step
                accept_nums_buffer.fill_(0)

                # Plan the workloads (only when realloc is required)
                if self.scheduler.realloc:
                    with cpu_bucket_timer("engine.plan"):
                        workloads, tail = self.scheduler.plan()
                    if tail and self.stop_on_tail:
                        print("Operation has reached the tail generation. Stopping generation immediately.")
                        break
                
                # Trace the plan
                self.tracer.on_plan(steps, workloads, self.kv_page_table)

                # Record the mean sequence length in this step
                profiler.set_step_mean_seqlen(self.kv_page_table.cachelens.float().mean().item())
                profiler.set_step_seqlen_std(self.kv_page_table.cachelens.float().std().item())

                # Build the batch and forward the model
                prev_cachelens = self.kv_page_table.cachelens.clone()
                with cpu_bucket_timer("engine.build_batch"):
                    batch = self.build_batch(workloads, self.kv_page_table, draft_buffer)
                with cpu_bucket_timer("engine.forward"):
                    logits, hidden_states = self.forward(batch) # [nnz, vocab_size], [nnz, hidden_size]

                # Get the qo_indptr for the logits
                logits_qo_indptr = self.get_logits_qo_indptr(batch.qo_indptr, batch.gate_mask)

                # PREFILL: Generate the next tokens
                if (prefill_indices := batch.status_map["PREFILL"]) is not None:
                    last_token_indices = logits_qo_indptr[1:][prefill_indices] - 1 # [n_pre]
                    last_logits = logits[last_token_indices, :] # [n_pre, vocab_size]
                    next_tokens[prefill_indices] = sample(last_logits, top_p=self.top_p, top_k=self.top_k, temperature=self.temperature)[:, 0] # [n_pre]
                
                # FIRST_DRAFT: Generate the draft tokens
                if (first_draft_indices := batch.status_map["FIRST_DRAFT"]) is not None:
                    # Sample the next token from the first logits
                    first_logits = logits[logits_qo_indptr[first_draft_indices], :] # [_, vocab_size]
                    draft_buffer[first_draft_indices, :1] = sample(first_logits, top_p=self.top_p, top_k=self.top_k, temperature=self.temperature) # [_, 1]

                    # Extract the hidden states for the draft tokens
                    starts = batch.qo_indptr[first_draft_indices] + 1
                    offsets = torch.arange(draft_len, device=self.device)
                    flat_indices = (starts[:, None] + offsets[None, :]).reshape(-1)
                    selected_hidden_states = hidden_states.index_select(0, flat_indices).reshape(-1, draft_len, hidden_states.shape[-1]) # [_, k, hidden_size]

                    # Generate the draft tokens from the mask tokens (i.e., `draft_len`-token following the x0)
                    draft_buffer[first_draft_indices, 1:] = self.sampler_draft(draft_buffer[first_draft_indices, :1], selected_hidden_states) # [_, k]

                    # Delete the KV cache entries for the draft tokens
                    delete_lens = torch.zeros(self.batch_size, device=self.device, dtype=torch.int32)
                    delete_lens.scatter_(0, first_draft_indices, draft_len)
                    self.kv_page_table.delete_kv(delete_lens)

                # DRAFT_VERIFY: Verify the draft tokens (from the previous step) and generate the draft tokens (for the next step)
                if (draft_verify_indices := batch.status_map["DRAFT_VERIFY"]) is not None:
                    # ============= VERIFY =============
                    # Extract target logits for the verification
                    starts = logits_qo_indptr[draft_verify_indices]
                    offsets = torch.arange(1+draft_len, device=self.device)
                    logits_flat_indices = (starts[:, None] + offsets[None, :]).reshape(-1)
                    target_logits = logits.index_select(0, logits_flat_indices).reshape(-1, 1+draft_len, logits.shape[-1]) # [_, 1+draft_len, vocab_size]
                    
                    # Evaluate the posterior for the draft tokens
                    if self.greedy:
                        bonus_tokens, accept_nums, _ = self.evaluate_posterior(draft_buffer[draft_verify_indices, 1:], target_logits.argmax(dim=-1))
                    else:
                        bonus_tokens, accept_nums, _ = self.evaluate_posterior(draft_buffer[draft_verify_indices, 1:], target_logits)

                    # Force budget
                    if self.force_budget:
                        bonus_tokens, accept_nums = self.budget_forcing(draft_buffer[draft_verify_indices, 1:], bonus_tokens, accept_nums)

                    # Collate the accepted KV cache entries
                    with cpu_bucket_timer("engine.collate_kv"):
                        self.collate_accepted_kv_cache(draft_verify_indices, accept_nums, prev_cachelens[draft_verify_indices])

                    # Update the verify buffer with the draft tokens and accept numbers
                    verify_buffer[draft_verify_indices] = draft_buffer[draft_verify_indices]
                    accept_nums_buffer.scatter_(0, draft_verify_indices, accept_nums)

                    # ============= DRAFT =============
                    # Update the draft buffer with the bonus tokens
                    draft_buffer[draft_verify_indices, :1] = bonus_tokens.long()

                    # Extract the hidden states for the valid draft tokens (i.e., the draft tokens following the accepted token)
                    starts = batch.qo_indptr[draft_verify_indices]
                    offsets = torch.arange(self.draft_and_verify_len, device=self.device)
                    hidden_flat_indices = (starts[:, None] + offsets[None, :]).reshape(-1)
                    dv_hidden_states = hidden_states.index_select(0, hidden_flat_indices).reshape(-1, draft_len+1, draft_len+1, hidden_states.shape[-1])
                    
                    batch_indices = torch.arange(draft_verify_indices.shape[0], device=self.device)
                    selected_hidden_states = dv_hidden_states[batch_indices, accept_nums-1, 1:, :] # [_, k, hidden_size]
                    
                    # Generate the draft tokens from the mask tokens
                    draft_buffer[draft_verify_indices, 1:] = self.sampler_draft(draft_buffer[draft_verify_indices, :1], selected_hidden_states) # [_, k], [_, k, hidden_size]

                # Record the number of tokens generated in this step
                profiler.set_step_tokens(int(accept_nums_buffer.sum().item()))

                # Remove the KV cache entries for the NULL sequences
                if batch.status_map["NULL"] is not None:
                    delete_lens = torch.zeros(self.batch_size, device=self.device, dtype=torch.int32)
                    delete_lens.scatter_(0, batch.status_map["NULL"], 1)
                    self.kv_page_table.delete_kv(delete_lens)

                # Process the results
                with cpu_bucket_timer("engine.process_results"):
                    self.scheduler.process_results(workloads, next_tokens, verify_buffer, accept_nums_buffer, self.eos_token_id, self.kv_page_table)

            self.tracer.on_update(steps, workloads, self.kv_page_table, accept_nums_tensor=accept_nums_buffer)
            steps += 1
        
        profiler.end_run()
        return steps


    # ========================= For benchmarking =========================
    def prefill(self, sequences: List[Sequence]):
        """
        Prefill tokens from prompts.

        Args:
            sequences: List[Sequence], the sequences to prefill.

        Returns:
            None
        """
        # Clear the KV cache
        self.kv_page_table.clear_kv(self.model)

        # Initialize the prefill scheduler
        prefill_scheduler = PrefillScheduler(max_concurrency=self.batch_size, prefill_chunk_len=self.prefill_chunk_size)
        prefill_scheduler.add_sequences(sequences)

        # Initialize the next tokens buffer
        next_tokens = torch.full((self.batch_size,), -1, device=self.device, dtype=torch.long)

        # Main loop
        while not prefill_scheduler.is_done():
            # Plan the workloads (only when realloc is required)
            if prefill_scheduler.realloc:
                workloads = prefill_scheduler.plan()

            # Build the batch and forward the model
            batch = self.build_batch(workloads, self.kv_page_table, None)
            logits, _ = self.forward(batch) # [nnz, vocab_size], [nnz, hidden_size]

            # Get the qo_indptr for the logits
            logits_qo_indptr = self.get_logits_qo_indptr(batch.qo_indptr, batch.gate_mask)
            if (prefill_indices := batch.status_map["PREFILL"]) is not None:
                last_token_indices = logits_qo_indptr[1:][prefill_indices] - 1 # [n_pre]
                last_logits = logits[last_token_indices, :] # [n_pre, vocab_size]
                next_tokens[prefill_indices] = sample(last_logits, top_p=self.top_p, top_k=self.top_k, temperature=self.temperature)[:, 0] # [n_pre]
            
            # Remove the KV cache entries for the NULL sequences
            if batch.status_map["NULL"] is not None:
                delete_lens = torch.zeros(self.batch_size, device=self.device, dtype=torch.int32)
                delete_lens.scatter_(0, batch.status_map["NULL"], 1)
                self.kv_page_table.delete_kv(delete_lens)

            # Process the results
            prefill_scheduler.process_results(workloads, next_tokens, move_to=Status.FIRST_DRAFT)
        
        return None


    # =============================== Helper Functions ===============================
    @staticmethod
    def get_logits_qo_indptr(qo_indptr: Tensor, gate_mask: Tensor) -> Tensor:
        """
        Compute the qo_indptr for the logits based on the qo_indptr and gate_mask.
        Essential because the logits are computed only for the non-masked tokens, so we need to adjust the qo_indptr accordingly.
        
        Args:
            qo_indptr (torch.Tensor): [n_seqs+1] shape tensor indicating sequence boundaries.
            gate_mask (torch.Tensor): [N] shape 1D tensor of 0/1 values.
                                    1 means the token should be removed.
        Returns:
            torch.Tensor: new qo_indptr tensor of shape [n_seqs+1].
        """
        # Compute per-token sequence id: seq_id[i] = s if qo_indptr[s] <= i < qo_indptr[s+1]
        pos = torch.arange(gate_mask.numel(), device=gate_mask.device)
        seq_id = torch.searchsorted(qo_indptr, pos, right=True) - 1

        # Count remaining tokens (where gate_mask == 0) for each sequence
        kept_counts = torch.bincount(seq_id[gate_mask == 0], minlength=qo_indptr.numel() - 1)

        # Build new indptr
        new_qo_indptr = torch.zeros_like(qo_indptr)
        new_qo_indptr[1:] = kept_counts.cumsum(0).to(qo_indptr.dtype)

        return new_qo_indptr
    

    def collate_accepted_kv_cache(
        self,
        seq_indices: Tensor,
        accept_nums: Tensor,
        prev_cachelens: Tensor
    ):
        """
        Collate the accepted KV cache entries based on the accepted numbers.
        Assume that k(k+1) KV entries were inserted to the KV cache from the previous step (draft_and_verify) where k is the draft length.
        Here, we need to 
            1. collate the accepted KV cache entries based on the accepted numbers.
            2. delete the KV cache entries that are (1) not accepted, (2) from the mask tokens.
        
        For example, suppose that the k=3, accepted_nums=[2, 3], prev_cachelens=[10, 15].
        Then, the number of newly inserted KV entries from the previous step is 12.
        In this case, we need to compute the following indices:
            1. save indices : [[10, 14], [15, 19, 23]]
            2. delete indices : [[11, 12, 13, 15, 16, 17, 18, 19, 20, 21], [16, 17, 18, 20, 21, 22, 24, 25, 26]]
        The accepted KV cache entries from save indices will be appended to the back-front of the KV cache.

        Args:
            seq_indices: The indices of the sequences. Shape: [n_seqs] where n_seqs <= bsz
            accept_nums: The number of accepted tokens. Shape: [n_seqs]
            prev_cachelens: The number of KV cache entries from the previous step. Shape: [n_seqs]

        Returns:
            None
        """ 
        assert accept_nums.dim() == 1 and prev_cachelens.dim() == 1, f"The accept_nums and prev_cachelens are expected to be a 1D tensor but got {accept_nums.dim()}D and {prev_cachelens.dim()}D."

        n_seqs = seq_indices.shape[0]
        n_local_heads, head_dim = self.model.config.n_local_heads, self.model.config.head_dim
        draft_len = self.draft_length
        stride = draft_len + 1
        max_accept_len = int(accept_nums.max().item())

        base = torch.arange(max_accept_len, device=self.device, dtype=torch.long) * stride # [max_accept_len]
        src = (prev_cachelens[:, None] + base[None, :]).reshape(-1) # [n_seqs * max_accept_len]
        dst = (prev_cachelens[:, None] + torch.arange(max_accept_len, device=self.device, dtype=torch.long)[None, :]).reshape(-1) # [n_seqs * max_accept_len]

        cols = torch.arange(max_accept_len, device=self.device, dtype=torch.long).expand(n_seqs, -1) # [n_seqs, max_accept_len]
        valid = (cols < accept_nums[:, None]).reshape(-1) # [n_seqs * max_accept_len]

        sidx = seq_indices.repeat_interleave(max_accept_len) # [n_seqs * max_accept_len]
        sidx, src, dst = sidx[valid], src[valid], dst[valid]

        for layer in self.model.layers:
            kv = layer.attention.kv_cache.kv_cache
            kv = kv.permute(0, 2, 1, 3, 4) # [num_pages, page_size, 2, n_local_heads, head_dim]
            orig = kv.shape
            kv = kv.reshape(self.batch_size, -1, 2, n_local_heads, head_dim) # [bsz, num_pages * page_size, 2, n_local_heads, head_dim]
            kv[sidx, dst] = kv[sidx, src] # RHS copy → LHS write (overlap-safe) 
            kv = kv.reshape(orig).permute(0, 2, 1, 3, 4) # [num_pages, 2, page_size, n_local_heads, head_dim]
            layer.attention.kv_cache.kv_cache = kv

        inserted_len = (draft_len + 1) ** 2
        delete_lens = torch.zeros(self.batch_size, device=self.device, dtype=torch.int32)
        delete_lens.scatter_(0, seq_indices, inserted_len - accept_nums) # [bsz]
        self.kv_page_table.delete_kv(delete_lens)
