"""Standard backend engine for autoregressive generation."""

from pathlib import Path
from typing import List

import torch
import flashinfer
from torch import Tensor
from transformers import PreTrainedTokenizer

from .base import BaseEngine
from .utils import sample, apply_tp
from .continuous import Sequence, Scheduler, BatchPack, BatchBuilderMixin, SchedulerTracer
from batchspec.profiler import get_active_profiler, cpu_bucket_timer
from batchspec.models import get_model

class StandardEngine(BaseEngine, BatchBuilderMixin):
    """Standard backend engine for autoregressive generation.
    
    Supports prefill and decode phases with FlashInfer attention.
    """

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
            target_checkpoint: Path to checkpoint
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
            print("Applying tensor parallel to model...")
            apply_tp(model, rank_group, group=group)
        
        # Move to device and set to eval mode
        model = model.to(device=self.device, dtype=self.dtype)
        self.model = model.eval()
    
    def setup_caches(
        self,
        max_batch_size: int = 1,
        max_seq_length: int = 2048,
        page_size: int = 16,
        prefill_chunk_size: int = 128,
        decode_length: int = 1
    ):
        """Setup KV caches and attention wrappers.
        
        Args:
            max_batch_size: Maximum batch size
            max_seq_length: Maximum sequence length
            page_size: Size of each page
            prefill_chunk_size: Chunk size for prefill
            decode_length: number of tokens to decode
        """
        # Setup base caches (add 1 for potential decode token)
        self.max_seq_len = max_seq_length
        self.page_size = page_size
        super().setup_caches(max_batch_size, max_seq_length + 1, page_size, prefill_chunk_size)

        # Create attention wrappers
        self.attn_buffer = self._create_attention_buffer(384)
        self.attn_wrapper = self._create_attention_wrapper(self.attn_buffer, qo_indptr=self.qo_indptr)
        
        # Setup model caches
        max_num_pages = self.kv_page_table.max_num_pages_per_request * max_batch_size
        with torch.device(self.device):
            self.model.setup_caches(
                num_pages=max_num_pages,
                page_size=page_size,
                attn_kernel=self.attn_wrapper
            )

        # Setup scheduler
        self.scheduler = Scheduler(
            max_concurrency=max_batch_size,
            prefill_chunk_len=prefill_chunk_size,
            decode_len=decode_length,
        )

        # Setup tracer
        self.tracer = SchedulerTracer(self.scheduler, color=True, show_queues=True)


    def compile(self):
        """Enable torch.compile for forward functions."""
        super().compile()
        self.forward = torch.compile(self.forward)


    def forward(self, batch: BatchPack) -> Tensor:
        """Single step forward of continuous batching.
        
        Args:
            batch: BatchPack of input data
            
        Returns:
            Next token predictions of shape (nnz)
        """
        self.pre_forward(batch)
        with torch.inference_mode():
            logits = self.model(
                input_ids=batch.input_ids,
                position_offsets=batch.position_ids_or_offsets,
                qo_indptr=batch.qo_indptr,
                kv_page_table=self.kv_page_table,
            ) # [nnz, vocab_size]
        
        last_logits = logits[batch.qo_indptr[1:]-1, :] # [n_seqs, vocab_size]
        return sample(last_logits, top_p=self.top_p, top_k=self.top_k, temperature=self.temperature)

    def pre_forward(self, batch: BatchPack):
        """Prepare for forward step."""
        self.attn_wrapper.plan(
            qo_indptr=batch.qo_indptr,
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

    def generate(self, sequences: List[Sequence]):
        """
        Generate tokens from prompts.

        Args:
            sequences: List[Sequence], the sequences to generate from.

        Returns:
            List[Sequence], the completed sequences.
        """
        # clear KV cache
        self.kv_page_table.clear_kv(self.model)
        
        step = 0
        self.scheduler.add_sequences(sequences)
        while not self.scheduler.is_done():
            if self.scheduler.realloc:
                workloads = self.scheduler.plan()
            self.tracer.on_plan(step, workloads, self.kv_page_table)
        
            batch = self.build_batch(workloads, self.kv_page_table)
            next_tokens = self.forward(batch)

            self.scheduler.process_results(workloads, next_tokens, self.eos_token_id, self.kv_page_table)
            self.tracer.on_update(step, workloads, self.kv_page_table, next_tokens)
            step += 1
        
        return self.scheduler.completed

    def generate_batch(self, input_ids, query_lens):
        """
        Generate a batch of tokens using the standard autoregressive decoding.
        
        If the force_budget is not set, the generation will terminate whenever at least one EOS token is generated.
        Otherwise, the generation will proceed until the budget is exhausted by replacing </think> tokens to 'Wait' or 'Alternatively'.
        
        Args:
            input_ids: [bsz, max_query_len]
            query_lens: [bsz]

        Returns:
            output: [bsz, max_len+1]
            num_generated_tokens: [bsz]
            num_total_tokens: [bsz]
            model_steps: int
        """

        # Pre-link local variables to reduce indexing time
        bsz = input_ids.shape[0]
        max_len = self.max_seq_len
        device = self.device
        force_budget = self.force_budget
        eos_token_id = self.eos_token_id
        suppress_token_id = self.suppress_token_id
        replace_token_ids = self.replace_token_ids
        
        # Define local variables
        model_steps = 0
        max_query_len = query_lens.max()
        num_total_tokens = query_lens.clone()
        batch_indices = torch.arange(bsz, device=device)

        output = torch.zeros(bsz, max_len+1, device=device, dtype=torch.long)
        output[:, :max_query_len] = input_ids[:, :max_query_len]
        
        # Prefill
        next_tokens = self.prefill(input_ids=input_ids, query_lens=query_lens)
        output[batch_indices, num_total_tokens] = next_tokens[:, 0]
        num_total_tokens += 1
        model_steps += 1
        
        # Initialize the profiler (NullProfiler when profiling=False)
        profiler = get_active_profiler()
        profiler.begin_run(bsz=bsz, label="generate_batch")

        terminal = False
        while num_total_tokens.max() < max_len and not terminal:
            with profiler.step_timing_ctx():
                profiler.set_step_seq_len(num_total_tokens)

                # Decode
                next_tokens = self.decode(input_ids=next_tokens)
                profiler.set_step_tokens(bsz)
                
                # Force budget
                if force_budget:
                    replace_mask = (next_tokens[:, 0] == suppress_token_id)
                    next_tokens[replace_mask, 0] = replace_token_ids[torch.randint(0, replace_token_ids.shape[0], (replace_mask.sum(),), device=self.device)]

                # Update output
                output[batch_indices, num_total_tokens] = next_tokens[:, 0]
                num_total_tokens += 1
                model_steps += 1
            
            if (not force_budget) and (next_tokens[:, 0] == eos_token_id).any(): terminal = True
        
        profiler.end_run()
        num_generated_tokens = num_total_tokens - query_lens
        return output, num_generated_tokens, num_total_tokens, model_steps