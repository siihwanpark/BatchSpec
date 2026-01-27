"""Standard backend engine for autoregressive generation with continuous batching."""

from pathlib import Path
from typing import List

import torch
from torch import Tensor

from batchspec.backends import StandardEngine
from batchspec.backends.utils import sample, apply_tp
from batchspec.profiler import get_active_profiler, cpu_bucket_timer
from batchspec.models import get_model
from .base import Sequence, Scheduler, BatchPack, BatchBuilderMixin, SchedulerTracer

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
    

    def generate(self, sequences: List[Sequence]):
        """
        Generate tokens from prompts.

        Args:
            sequences: List[Sequence], the sequences to generate from.

        Returns:
            List[Sequence], the completed sequences.
        """
        # Clear KV cache and reset steps counter
        self.kv_page_table.clear_kv(self.model)
        steps = 0

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
                
                # Trace the plan
                self.tracer.on_plan(steps, workloads, self.kv_page_table)
            
                # Build the batch and forward the model
                with cpu_bucket_timer("engine.build_batch"):
                    batch = self.build_batch(workloads, self.kv_page_table)
                with cpu_bucket_timer("engine.forward"):
                    logits = self.forward(batch) # [nnz, vocab_size]
                with cpu_bucket_timer("engine.sample"):
                    next_tokens = sample(logits[batch.qo_indptr[1:]-1, :], top_p=self.top_p, top_k=self.top_k, temperature=self.temperature) # [n_seqs, 1]

                # Process the results
                with cpu_bucket_timer("engine.process_results"):
                    self.scheduler.process_results(workloads, next_tokens, self.eos_token_id, self.kv_page_table)
                if steps % 100 == 0:
                    self.tracer.on_update(steps, workloads, self.kv_page_table, next_tokens) # update the tracer
                steps += 1
                profiler.set_step_tokens(int(next_tokens.shape[0]))
        
        profiler.end_run()
        return steps