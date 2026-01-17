"""Standard backend engine for autoregressive generation."""

from pathlib import Path

import torch
from torch import Tensor

from .base import BaseEngine
from .utils import sample, apply_tp
from batchspec.profiler import get_active_profiler, cpu_bucket_timer
from batchspec.models import get_model

class StandardEngine(BaseEngine):
    """Standard backend engine for autoregressive generation."""

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
        super().setup_caches(batch_size, max_seq_length + 1, page_size, prefill_chunk_size)

        # Create attention wrappers
        self.attn_buffer = self._create_attention_buffer(attn_buffer_size_mb)
        self.attn_wrapper = self._create_attention_wrapper(batch_size, self.attn_buffer)
        
        # Setup model caches
        max_num_pages = self.kv_page_table.max_num_pages_per_request * batch_size
        with torch.device(self.device):
            self.model.setup_caches(
                num_pages=max_num_pages,
                page_size=page_size,
                attn_kernel=self.attn_wrapper
            )
    

    def init_cache(self):
        """Initialize the KV cache for the model."""
        self.kv_page_table.clear_kv(self.model)
    

    def compile(self):
        """Enable torch.compile for forward functions."""
        super().compile()
        self.forward = torch.compile(self.forward)


    def forward(self, input_ids: Tensor) -> Tensor:
        """Single step forward.
        
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
        """Prepare for forward step."""
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
            chunk_input_ids = input_ids[:, i*chunk_size:(i+1)*chunk_size]
            logits = self.forward(chunk_input_ids) # [bsz, chunk_seq_len, vocab_size]
        return sample(logits[:, -1, :], top_p=self.top_p, top_k=self.top_k, temperature=self.temperature)

    
    def decode(self, input_ids: Tensor) -> Tensor:
        """Execute the decode step and return the next token predictions.
        
        Args:
            input_ids: Input token IDs [bsz, seq_len]
            
        Returns:
            Next token predictions [bsz, 1]
        """
        _, seq_len = input_ids.shape
        assert seq_len == 1, f"The input length must be 1 for decode, but got {seq_len}"

        logits = self.forward(input_ids) # [bsz, 1, vocab_size]
        return sample(logits[:, 0, :], top_p=self.top_p, top_k=self.top_k, temperature=self.temperature)


    def generate_batch(self, input_ids, max_gen_len: int, prefix_len: int):
        """
        Generate a batch of tokens using the standard autoregressive decoding.
        
        If the force_budget is not set, the generation will terminate whenever at least one EOS token is generated.
        Otherwise, the generation will proceed until the budget is exhausted by replacing </think> tokens to 'Wait' or 'Alternatively'.
        
        Args:
            input_ids: [bsz, seq_len]
            max_gen_len: Maximum generation length

        Returns:
            output: [bsz, max_gen_len]
            num_generated_tokens: int
            model_steps: int
        """

        # Pre-link local variables to reduce indexing time
        bsz = input_ids.shape[0]
        device = self.device

        force_budget = self.force_budget
        eos_token_id = self.eos_token_id
        suppress_token_ids = self.suppress_token_ids
        replace_token_ids = self.replace_token_ids
        
        # Define local variables
        model_steps = 0
        num_generated_tokens = 0
        output = torch.zeros(bsz, max_gen_len+1, device=device, dtype=torch.long)

        # Prefill
        next_tokens = self.prefill(input_ids=input_ids)
        output[:, 0] = next_tokens[:, 0]
        
        # Initialize the profiler (NullProfiler when profiling=False)
        profiler = get_active_profiler()
        profiler.begin_run(bsz=bsz, label="decode", prefix_len=prefix_len)

        terminal = False
        while num_generated_tokens < max_gen_len and not terminal:
            with profiler.step_timing_ctx():
                profiler.set_step_seq_len(self.kv_page_table.cachelens)

                # Decode
                next_tokens = self.decode(input_ids=next_tokens)
                profiler.set_step_tokens(bsz)
                
            # Force budget
            if force_budget:
                replace_mask = torch.isin(next_tokens[:, 0], suppress_token_ids)
                num_replace = replace_mask.sum().item()
                if num_replace > 0:
                    rand_idx = torch.randint(0, replace_token_ids.shape[0], (num_replace,), device=self.device)
                    next_tokens[replace_mask, 0] = replace_token_ids[rand_idx]

            # Update output
            output[:, num_generated_tokens+1] = next_tokens[:, 0]
            num_generated_tokens += 1
            model_steps += 1
            
            if (not force_budget) and (next_tokens[:, 0] == eos_token_id).any(): terminal = True
        
        profiler.end_run()
        self.kv_page_table.delete_kv(self.kv_page_table.cachelens - prefix_len) # revert the KV cache to proceed next run with longer prefix
        assert torch.all(self.kv_page_table.cachelens == prefix_len), "The KV cache length must be equal to the prefix length"

        num_generated_tokens = torch.full((bsz,), num_generated_tokens, device=device, dtype=torch.int32)

        return output, num_generated_tokens, model_steps