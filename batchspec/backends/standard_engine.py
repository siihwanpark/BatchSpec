"""Standard backend engine for autoregressive generation."""

from pathlib import Path
from typing import Optional

import torch
import flashinfer
from torch import Tensor
from transformers import PreTrainedTokenizer

from .base import BaseEngine
from .utils import sample, apply_tp
from batchspec.profiler import get_active_profiler, cpu_bucket_timer
from batchspec.models import get_model

class StandardEngine(BaseEngine):
    """Standard backend engine for autoregressive generation.
    
    Supports prefill and decode phases with FlashInfer attention.
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizer, dtype: torch.dtype = torch.bfloat16, device: str = "cuda:0"):
        """Initialize the engine.
        
        Args:
            tokenizer: HuggingFace tokenizer
            dtype: Data type for computations
            device: Device to run computations on
        """
        super().__init__(tokenizer, dtype, device)

        self.prefill_forward = lambda model, x, input_pos, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen: \
            model(x, input_pos, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen, attn_type='prefill')
        self.decode_forward = lambda model, x, input_pos, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen: \
            model(x, input_pos, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen, attn_type='decode')
    
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
        prefill_chunk_size: int = 128
    ):
        """Setup KV caches and attention wrappers.
        
        Args:
            max_batch_size: Maximum batch size
            max_seq_length: Maximum sequence length
            page_size: Size of each page
            prefill_chunk_size: Chunk size for prefill
        """
        # Setup base caches (add 1 for potential decode token)
        self.max_seq_len = max_seq_length
        super().setup_caches(max_batch_size, max_seq_length + 1, page_size, prefill_chunk_size)
        
        # Create attention buffers
        self.prefill_buffer = self._create_attention_buffer(384)  # 3 * 128MB
        self.decode_buffer = self._create_attention_buffer(384)
        
        # Create attention wrappers
        self.attn_wrappers = {
            "prefill": self._create_attention_wrapper(
                self.prefill_buffer,
                qo_length=prefill_chunk_size
            ),
            "decode": self._create_attention_wrapper(
                self.decode_buffer,
                qo_length=1
            ),
        }
        
        # Register with custom ops
        self._register_attention_wrappers({
            "attn_prefill": self.attn_wrappers["prefill"],
            "attn_decode": self.attn_wrappers["decode"],
        })
        
        # Setup model caches
        with torch.device(self.device):
            self.model.setup_caches(num_pages=self.max_num_pages, page_size=self.page_size)
    
    def compile(self):
        """Enable torch.compile for forward functions."""
        super().compile()
        self.prefill_forward = torch.compile(self.prefill_forward)
        self.decode_forward = torch.compile(self.decode_forward)
    
    def prefill(self, input_ids: Tensor, query_lens: Tensor) -> Tensor:
        """Prefill input sequence and return first predicted token.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            query_lens: Actual length of each sequence
            
        Returns:
            Next token predictions of shape (batch_size, 1)
        """
        self.clear_kv()
        bsz, seq_len = input_ids.shape
        
        # Validate sequence length
        assert seq_len % self.prefill_chunk_size == 0, \
            f"Sequence length ({seq_len}) must be divisible by prefill_chunk_size ({self.prefill_chunk_size})"
        
        # Initialize state
        last_logits = None  # Lazy init for tensor parallel
        logit_recorded = torch.zeros(bsz, dtype=torch.bool, device=self.device)
        
        chunk_size = self.prefill_chunk_size
        num_chunks = seq_len // chunk_size
        
        # Process in chunks
        for i in range(num_chunks):
            chunk_input_ids = input_ids[:, i*chunk_size:(i+1)*chunk_size]
            chunk_query_lens = torch.clamp(query_lens - (i * chunk_size), min=0, max=chunk_size)
            
            # Skip if all padding
            if torch.all(chunk_query_lens == 0):
                continue
            
            # Prepare and run forward
            self.pre_prefill()
            with torch.inference_mode():
                logits = self.prefill_forward(
                    model=self.model,
                    x=chunk_input_ids,
                    input_pos=torch.full((bsz,), i*chunk_size, dtype=torch.int32, device=self.device),
                    kv_append_indptr=self.qo_indptr * chunk_size,
                    kv_page_indices=self.paged_kv_indices,
                    kv_page_indptr=self.paged_kv_indptr,
                    kv_page_lastlen=self.paged_kv_last_page_len,
                )
            
            # Lazy init last_logits
            if last_logits is None:
                last_logits = torch.full(
                    (bsz, logits.shape[-1]), float('nan'),
                    device=self.device, dtype=self.dtype
                )
            
            # Extract logits for sequences ending in this chunk
            target_indices_in_chunk = chunk_query_lens - 1
            finishes_in_this_chunk = (query_lens > i*chunk_size) & (query_lens <= (i+1)*chunk_size)
            target_sequences_mask = finishes_in_this_chunk & (~logit_recorded)
            
            target_batch_indices = torch.where(target_sequences_mask)[0]
            if target_batch_indices.numel() > 0:
                indices_in_chunk_to_grab = target_indices_in_chunk[target_batch_indices]
                last_logits[target_batch_indices] = logits[target_batch_indices, indices_in_chunk_to_grab, :]
                logit_recorded[target_batch_indices] = True
            
            # Handle padding
            exists_padding = (chunk_query_lens < chunk_size)
            if exists_padding.any():
                self.delete_kv(chunk_size - chunk_query_lens)
        
        # Handle any remaining NaN (e.g., sequences with query_len=0)
        if torch.isnan(last_logits).any():
            print("Warning: Found NaN in last_logits. Replacing with zeros.")
            last_logits = torch.nan_to_num(last_logits, nan=0.0)
        
        # Sample next tokens
        return sample(last_logits, top_p=self.top_p, top_k=self.top_k, temperature=self.temperature)
    
    def pre_prefill(self):
        """Prepare for prefill step."""
        self.insert_kv(self.prefill_chunk_size)
        self.attn_wrappers["prefill"].plan(
            qo_indptr=self.qo_indptr * self.prefill_chunk_size,
            paged_kv_indptr=self.paged_kv_indptr,
            paged_kv_indices=self.paged_kv_indices,
            paged_kv_last_page_len=self.paged_kv_last_page_len,
            num_qo_heads=self.model.config.n_head,
            num_kv_heads=self.model.config.n_local_heads,
            head_dim_qk=self.model.config.head_dim,
            page_size=self.page_size,
            q_data_type=self.dtype,
            causal=True,
        )
    
    def decode(self, input_ids: Tensor) -> Tensor:
        """Decode one step.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, 1)
            
        Returns:
            Next token predictions of shape (batch_size, 1)
        """
        dec_len = input_ids.shape[1]
        assert dec_len == 1, f"Decode expects seq_len=1, got {dec_len}"
        
        # Prepare and run forward
        self.pre_decode()
        with torch.inference_mode():
            logits = self.decode_forward(
                model=self.model,
                x=input_ids,
                input_pos=self.cachelens - 1,
                kv_append_indptr=self.qo_indptr,
                kv_page_indices=self.paged_kv_indices,
                kv_page_indptr=self.paged_kv_indptr,
                kv_page_lastlen=self.paged_kv_last_page_len,
            )
        
        # Sample next tokens
        return sample(logits, top_p=self.top_p, top_k=self.top_k, temperature=self.temperature)
    
    def pre_decode(self):
        """Prepare for decode step."""
        self.insert_kv(1)
        self.attn_wrappers["decode"].plan(
            qo_indptr=self.qo_indptr,
            paged_kv_indptr=self.paged_kv_indptr,
            paged_kv_indices=self.paged_kv_indices,
            paged_kv_last_page_len=self.paged_kv_last_page_len,
            num_qo_heads=self.model.config.n_head,
            num_kv_heads=self.model.config.n_local_heads,
            head_dim_qk=self.model.config.head_dim,
            page_size=self.page_size,
            q_data_type=self.dtype,
            causal=True,
        )

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