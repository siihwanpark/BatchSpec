"""Standard backend engine for autoregressive generation with continuous batching."""

from pathlib import Path
from typing import List

import torch
from torch import Tensor

from batchspec.backends import StandardEngine
from batchspec.backends.utils import sample, apply_tp
from batchspec.profiler import get_active_profiler, cpu_bucket_timer
from batchspec.models import get_model
from .base import (
    Sequence, Status, Scheduler, BatchPack, BatchBuilderMixin, SchedulerTracer, PrefillScheduler
)

class StandardContinuousEngine(StandardEngine, BatchBuilderMixin):
    """Standard backend engine for autoregressive generation with continuous batching."""

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
        super().setup_caches(batch_size, max_seq_length, page_size, prefill_chunk_size, attn_buffer_size_mb)

        # Setup scheduler
        self.scheduler = Scheduler(
            max_concurrency=batch_size,
            prefill_chunk_len=prefill_chunk_size,
        )

        # Setup tracer
        self.tracer = SchedulerTracer(self.scheduler, color=True, show_queues=True)

    
    def set_stop_on_tail(self, stop_on_tail: bool):
        """Set whether to stop the generation immediately upon reaching the tail generation."""
        self.stop_on_tail = stop_on_tail


    def forward(self, batch: BatchPack) -> Tensor:
        """Single step forward of continuous batching.
        
        Args:
            batch: BatchPack of input data
            
        Returns:
            Logits of shape (nnz, vocab_size)
        """
        self.pre_forward(batch.qo_indptr)
        with torch.inference_mode():
            logits = self.model(
                input_ids=batch.input_ids,
                position_offsets=batch.position_ids_or_offsets,
                qo_indptr=batch.qo_indptr,
                kv_page_table=self.kv_page_table,
            ) # [nnz, vocab_size]

        return logits
    

    def generate(self, sequences: List[Sequence], clear_kv: bool = True):
        """
        Generate tokens from prompts.

        Args:
            sequences: List[Sequence], the sequences to generate from.
            clear_kv: Whether to clear the KV cache before generating. (Only for benchmarking, default to True)

        Returns:
            List[Sequence], the completed sequences.
        """
        steps = 0
        if clear_kv:
            # Clear KV cache and reset steps counter
            self.kv_page_table.clear_kv(self.model)
        
        # Add sequences to the scheduler
        self.scheduler.add_sequences(sequences)

        # Initialize the profiler (NullProfiler when profiling=False)
        profiler = get_active_profiler()
        profiler.begin_run(bsz=self.batch_size, label="generate", prefix_len=0)

        # Main loop
        while not self.scheduler.is_done():
            with profiler.step_timing_ctx():
                # Plan the workloads (only when realloc is required)
                if self.scheduler.realloc:
                    with cpu_bucket_timer("engine.plan"):
                        workloads, tail = self.scheduler.plan()
                    if tail and self.stop_on_tail:
                        print("Operation has reached the tail generation. Stopping generation immediately.")
                        break
                
                import pdb; pdb.set_trace()
                
                # Trace the plan
                self.tracer.on_plan(steps, workloads, self.kv_page_table)

                # Record the mean sequence length in this step
                cachelens = self.kv_page_table.cachelens
                mean_seqlen = cachelens[cachelens > 16].float().mean().item() if (cachelens > 16).any() else 0.0
                profiler.set_step_mean_seqlen(mean_seqlen)
            
                # Build the batch and forward the model
                with cpu_bucket_timer("engine.build_batch"):
                    batch = self.build_batch(workloads, self.kv_page_table)
                with cpu_bucket_timer("engine.forward"):
                    logits = self.forward(batch) # [nnz, vocab_size]
                with cpu_bucket_timer("engine.sample"):
                    next_tokens = sample(logits[batch.qo_indptr[1:]-1, :], top_p=self.top_p, top_k=self.top_k, temperature=self.temperature) # [n_seqs, 1]
                
                if self.force_budget:
                    next_tokens = self.budget_forcing(next_tokens)

                # Record the number of tokens generated in this step
                step_tokens = int(batch.status_map["DECODE"].shape[0]) if batch.status_map["DECODE"] is not None else 0
                profiler.set_step_tokens(step_tokens)

                # Remove the KV cache entries for the NULL sequences
                if batch.status_map["NULL"] is not None:
                    delete_lens = torch.zeros(self.batch_size, device=self.device, dtype=torch.int32)
                    delete_lens.scatter_(0, batch.status_map["NULL"], 1)
                    self.kv_page_table.delete_kv(delete_lens)

                # Process the results
                with cpu_bucket_timer("engine.process_results"):
                    self.scheduler.process_results(workloads, next_tokens, self.eos_token_id, self.kv_page_table)
                if steps % 100 == 0:
                    self.tracer.on_update(steps, workloads, self.kv_page_table, next_tokens) # update the tracer
                steps += 1
        
        profiler.end_run()
        return steps


    # ========================= For benchmarking =========================
    def prefill(self, sequences: List[Sequence]):
        """
        Prefill tokens from prompts.

        Args:
            sequences: List[Sequence], the sequences to prefill from.

        Returns:
            PrefillScheduler, the prefill scheduler.
        """
        # Clear KV cache and reset steps counter
        self.kv_page_table.clear_kv(self.model)
        steps = 0

        # Initialize the prefill scheduler
        prefill_scheduler = PrefillScheduler(max_concurrency=self.batch_size, prefill_chunk_len=self.prefill_chunk_size)
        prefill_scheduler.add_sequences(sequences)

        # Main loop
        while not prefill_scheduler.is_done():
            # Plan the workloads (only when realloc is required)
            if prefill_scheduler.realloc:
                workloads = prefill_scheduler.plan()
            
            # Trace the plan
            self.tracer.on_plan(steps, workloads, self.kv_page_table)
        
            # Build the batch and forward the model
            batch = self.build_batch(workloads, self.kv_page_table)
            logits = self.forward(batch) # [nnz, vocab_size]
            next_tokens = sample(logits[batch.qo_indptr[1:]-1, :], top_p=self.top_p, top_k=self.top_k, temperature=self.temperature) # [n_seqs, 1]
            
            # Remove the KV cache entries for the NULL sequences
            if batch.status_map["NULL"] is not None:
                delete_lens = torch.zeros(self.batch_size, device=self.device, dtype=torch.int32)
                delete_lens.scatter_(0, batch.status_map["NULL"], 1)
                self.kv_page_table.delete_kv(delete_lens)

            # Process the results
            prefill_scheduler.process_results(workloads, next_tokens)
            steps += 1

        return prefill_scheduler


    def generate_with_repeated_prefill(self, sequences: List[Sequence]):
        """
        Generate tokens from prompts.

        Args:
            sequences: List[Sequence], the sequences to generate from.
            clear_kv: Whether to clear the KV cache before generating. (Only for benchmarking, default to True)

        Returns:
            List[Sequence], the completed sequences.
        """
        steps = 0
        self.scheduler.add_sequences(sequences)

        # Initialize the profiler (NullProfiler when profiling=False)
        profiler = get_active_profiler()
        profiler.begin_run(bsz=self.batch_size, label="generate", prefix_len=0)

        # Main loop
        while not self.scheduler.is_done():
            with profiler.step_timing_ctx():
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
                cachelens = self.kv_page_table.cachelens
                mean_seqlen = cachelens[cachelens > 16].float().mean().item() if (cachelens > 16).any() else 0.0
                profiler.set_step_mean_seqlen(mean_seqlen)
            
                # Build the batch and forward the model
                with cpu_bucket_timer("engine.build_batch"):
                    batch = self.build_batch(workloads, self.kv_page_table)
                with cpu_bucket_timer("engine.forward"):
                    logits = self.forward(batch) # [nnz, vocab_size]
                with cpu_bucket_timer("engine.sample"):
                    next_tokens = sample(logits[batch.qo_indptr[1:]-1, :], top_p=self.top_p, top_k=self.top_k, temperature=self.temperature) # [n_seqs, 1]
                
                if self.force_budget:
                    next_tokens = self.budget_forcing(next_tokens)

                # Record the number of tokens generated in this step
                step_tokens = int(batch.status_map["DECODE"].shape[0]) if batch.status_map["DECODE"] is not None else 0
                profiler.set_step_tokens(step_tokens)

                # Process the results
                with cpu_bucket_timer("engine.process_results"):
                    self.scheduler.process_results(workloads, next_tokens, self.eos_token_id, self.kv_page_table)

                # Continue the prefill with fixed sequence length.
                if batch.status_map["PREFILL"] is not None:
                    # Delete the KV cache entries for the PREFILL sequences.
                    processed_lens = torch.tensor([workload.n_tokens for workload in workloads], device=self.device, dtype=torch.int32,)
                    delete_lens = torch.zeros(self.batch_size, device=self.device, dtype=torch.int32)
                    delete_lens[batch.status_map["PREFILL"]] = processed_lens[batch.status_map["PREFILL"]]
                    self.kv_page_table.delete_kv(delete_lens)
                    
                    # Revert the progress of the PREFILL sequences.
                    for workload in workloads:
                        seq = workload.seq
                        n_tokens = workload.n_tokens
                        if seq.status == Status.PREFILL:
                            seq.num_prefilled_tokens -= n_tokens
                            seq.cur_pos -= n_tokens

                self.tracer.on_update(steps, workloads, self.kv_page_table, next_tokens) # update the tracer
                steps += 1
        
        profiler.end_run()
        return steps


    # =============================== Helper Functions ===============================
    def budget_forcing(self, next_tokens: Tensor) -> Tensor:
        """Limits the number of accepted draft tokens when certain tokens appear.

        The accepted draft tokens are truncated at the first occurrence of a suppressed token
        in the draft. The bonus tokens are also forced to avoid suppressed tokens by
        replacing them with a safe alternative when necessary.

        Args:
            next_tokens: [bsz, 1] next token ids

        Returns:
            updated_next_tokens: [bsz, 1] updated next token ids
        """
        suppress_mask = (next_tokens[..., None] == self.suppress_token_ids).any(dim=-1)[..., 0] # [bsz]
        if suppress_mask.any():
            num_to_replace = suppress_mask.sum()
            random_indices = torch.randint(0, self.replace_token_ids.shape[0], (num_to_replace,), device=next_tokens.device)
            next_tokens[suppress_mask, 0] = self.replace_token_ids[random_indices].to(next_tokens.dtype)
        
        return next_tokens