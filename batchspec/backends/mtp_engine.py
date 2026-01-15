"""MTP (Multi-Token Prediction) engine for self-speculative decoding.

This engine implements self-speculative decoding with LoRA and gated sampling.
"""

from typing import Tuple, List, Optional
from pathlib import Path

import torch
from torch import Tensor
from transformers import PreTrainedTokenizer
from flashinfer.sampling import chain_speculative_sampling

from .base import BaseEngine
from .utils import get_sampling_probs, sample, apply_tp
from batchspec.models import get_model, LoRAConfig
from batchspec.profiler import get_active_profiler, cpu_bucket_timer


class MTPEngine(BaseEngine):
    """MTP engine for self-speculative decoding with multi-token prediction.
    
    Key features:
    - LoRA supported
    - Gated sampling with mask tokens
    - Draft and verify in single forward pass
    - Supports single or dual draft lengths
    
    Args:
        dtype: Data type for computations
        device: Device to run on
        draft_length: List of draft lengths (1 or 2 elements)
        tokenizer: Tokenizer for handling mask tokens
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
        self.mask_token_id = None
        
    def load_model(
        self,
        model_name: str,
        checkpoint_path: Path,
        lora_checkpoint_path: Path,
        lora_config: LoRAConfig,
        use_tp: bool = False,
        rank_group = None,
        group = None,
    ):
        """Load MTP model with LoRA adapters.
        
        Args:
            model_name: Name of model configuration
            target_checkpoint: Path to base model checkpoint
            lora_checkpoint: Path to LoRA checkpoint
            lora_config: LoRA configuration
            use_tp: Whether to use tensor parallelism
            rank_group: Rank group for TP
            group: Process group
        """
        with torch.device('meta'):
            model = get_model(model_name, "mtp", lora_config=lora_config)

        # Load base weights
        checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
        load_result = model.load_state_dict(checkpoint, assign=True, strict=False)
        if load_result.unexpected_keys:
            raise ValueError(
                f"Base checkpoint contains unexpected keys not found in model:\n"
                f"{load_result.unexpected_keys}"
            )

        # Load LoRA adapter and sampler head weights
        lora_checkpoint = torch.load(str(lora_checkpoint_path), mmap=True, weights_only=True)
        load_result = model.load_state_dict(lora_checkpoint, assign=True, strict=False)
        if load_result.unexpected_keys:
            raise ValueError(
                f"LoRA checkpoint contains unexpected keys not found in model:\n"
                f"{load_result.unexpected_keys}"
            )

        # Append <mask> token and resize token embeddings
        if '<mask>' not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({'additional_special_tokens': ['<mask>']})
            print(f"Added <mask> token with ID: {self.tokenizer.convert_tokens_to_ids('<mask>')}")
        
        self.mask_token_id = self.tokenizer.convert_tokens_to_ids('<mask>')
        model.resize_token_embeddings(len(self.tokenizer))

        if use_tp:
            print("Applying tensor parallel to model ...")
            apply_tp(model, rank_group, group=group)

        model = model.to(device=self.device, dtype=self.dtype)
        self.model = model.eval()
    
    def setup_caches(
        self,
        batch_size: int = 1,
        max_seq_length: int = 2048,
        page_size: int = 16,
        prefill_chunk_size: int = 128
    ):
        """Setup KV caches and attention wrappers.
        
        Args:
            batch_size: Batch size
            max_seq_length: Maximum sequence length
            page_size: Size of each page
            prefill_chunk_size: Chunk size for prefill
        """
        # Setup base caches
        self.max_seq_len = max_seq_length
        self.draft_and_verify_len = (self.draft_length + 1) ** 2
        self.max_cache_len = max_seq_length + self.draft_and_verify_len + 1
        self.page_size = page_size

        super().setup_caches(batch_size, self.max_cache_len, page_size, prefill_chunk_size)
        
        # Set common attention masks and position ids
        self.setup_common_attn_mask_and_position_ids()

        # Create causal attention wrapper
        self.causal_attn_buffer = self._create_attention_buffer(384)
        self.causal_attn_wrapper = self._create_attention_wrapper(
            self.causal_attn_buffer,
            qo_indptr=self.qo_indptr,
        )

        # Create non-causal attention wrapper
        self.non_causal_qo_indptr = torch.arange(self.batch_size + 1, dtype=torch.int32, device=self.device)
        self.non_causal_attn_buffer = self._create_attention_buffer(384)
        
        max_bytes_for_attn_masks = (self.batch_size * (self.draft_and_verify_len * self.max_cache_len)) // 8 + 1
        self.custom_mask_buf = torch.empty(max_bytes_for_attn_masks, dtype=torch.uint8, device=self.device)
        self.mask_indptr_buf = torch.arange(self.batch_size + 1, dtype=torch.int32, device=self.device)
        
        self.non_causal_attn_wrapper = self._create_attention_wrapper(
            self.non_causal_attn_buffer,
            qo_indptr=self.non_causal_qo_indptr,
            use_custom_mask=True,
            custom_mask_buf=self.custom_mask_buf,
            mask_indptr_buf=self.mask_indptr_buf,
        )

        # Setup model caches
        max_num_pages = self.kv_page_table.max_num_pages_per_request * batch_size
        with torch.device(self.device):
            self.model.setup_caches(
                num_pages=max_num_pages,
                page_size=page_size,
                causal_attn_kernel=self.causal_attn_wrapper,
                non_causal_attn_kernel=self.non_causal_attn_wrapper,
            )


    def compile(self):
        """Enable torch.compile for forward functions.
        """
        super().compile()
        self.forward = torch.compile(self.forward)


    def pre_forward(self, qo_indptr: Tensor, attn_mask: Tensor):
        """Prepare for forward step."""
        self.kv_page_table.insert_kv(qo_indptr[1:] - qo_indptr[:-1]) # insert KV cache for the new tokens
        wrapper_plan_kwargs = {
            "qo_indptr": qo_indptr,
            "paged_kv_indptr": self.kv_page_table.paged_kv_indptr,
            "paged_kv_indices": self.kv_page_table.paged_kv_indices,
            "paged_kv_last_page_len": self.kv_page_table.paged_kv_last_page_len,
            "num_qo_heads": self.model.config.n_head,
            "num_kv_heads": self.model.config.n_local_heads,
            "head_dim_qk": self.model.config.head_dim,
            "page_size": self.page_size,
            "q_data_type": self.dtype,
        }

        if attn_mask is not None:
            wrapper_plan_kwargs["custom_mask"] = attn_mask
            wrapper_plan_kwargs["causal"] = False
            self.non_causal_attn_wrapper.plan(**wrapper_plan_kwargs)
        else:
            wrapper_plan_kwargs["causal"] = True
            self.causal_attn_wrapper.plan(**wrapper_plan_kwargs)


    def forward(self,
        input_ids: Tensor,
        qo_indptr: Tensor,
        position_ids: Tensor,
        gate_mask: Optional[Tensor]=None,
        attn_mask: Optional[Tensor]=None,
    ) -> Tuple[Tensor, Tensor]:
        """Single step forward.
        
        Args:
            input_ids: Input token indices of shape [bsz, seq_len]
            qo_indptr: Index pointer for Query/Output tokens of shape [bsz + 1]
            position_ids: Position IDs for RoPE of shape [bsz * seq_len]
            gate_mask: Optional mask for LoRA gating
            attn_mask: Optional attention mask

        Returns:
            Tuple of (logits, hidden_states)
            - logits: [bsz, seq_len, vocab_size]
            - hidden_states: [bsz, seq_len, hidden_size]
        """
        self.pre_forward(qo_indptr=qo_indptr, attn_mask=attn_mask)
        with torch.inference_mode():
            logits, hidden_states = self.model(
                input_ids=input_ids,
                gate_mask=gate_mask,
                position_ids=position_ids,
                qo_indptr=qo_indptr,
                kv_page_table=self.kv_page_table,
                causal=attn_mask is None,
            ) # [bsz, seq_len, vocab_size], [bsz, seq_len, hidden_size]
        
        return logits, hidden_states
    

    def prefill(self, input_ids: Tensor) -> Tensor:
        """Execute the chunked prefill with the pre-defined prefill chunk size and return the last token predictions.
        
        Args:
            input_ids: Input token IDs [bsz, seq_len]
            
        Returns:
            Next token predictions [bsz, 1]
        """        
        bsz, seq_len = input_ids.shape
        assert seq_len % self.prefill_chunk_size == 0, f"The sequence length must be divisible by the prefill chunk size, but got seq_len={seq_len} and prefill_chunk_size={self.prefill_chunk_size}"
        
        chunk_size = self.prefill_chunk_size
        num_chunks = seq_len // chunk_size
        for i in range(num_chunks):
            chunk_input_ids = input_ids[:, i*chunk_size:(i+1)*chunk_size]
            chunk_seq_len = chunk_input_ids.shape[1]
            
            # Target prefill
            qo_indptr = torch.arange(bsz + 1, device=self.device, dtype=torch.int32) * chunk_seq_len
            position_ids = (self.kv_page_table.cachelens[:, None] + torch.arange(chunk_seq_len, device=self.device)[None, :]).flatten()
            logits, _ = self.forward(
                input_ids=chunk_input_ids,
                qo_indptr=qo_indptr,
                position_ids=position_ids,
            ) # [bsz, chunk_seq_len, vocab_size]

        return sample(logits[:, -1, :], top_p=self.top_p, top_k=self.top_k, temperature=self.temperature)


    def draft(self, input_ids: Tensor, gate_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """First draft after prefill / Fallback draft
        
        Assume that a single token is generated in the previous step (x_0).
        Then the input sequence is [x_0, m_1, ..., m_k] where m_i is the <mask> token, and k is the draft length.
        And the corresponding gate_mask is [0, 1, ..., 1] where the first 0 is for x_0.
        
        Note that in this case, we use causal attention mask and normal position_ids.
        
        Args:
            input_ids: [x_0, m_1, ..., m_k]
            gate_mask: Gate mask for LoRA
            
        Returns:
            Tuple of (logits, hidden_states)
        """
        bsz, dec_len = input_ids.shape
        assert dec_len == self.draft_length + 1, f"The input sequence length must be equal to the draft length + 1, but got dec_len={dec_len} and draft_length={self.draft_length}"

        # prepare position_ids
        qo_indptr = torch.arange(bsz + 1, device=self.device, dtype=torch.int32) * dec_len
        position_ids = (self.kv_page_table.cachelens[:, None] + torch.arange(dec_len, device=self.device)[None, :]).flatten()

        # model forward for draft
        logits, hidden_states = self.forward(
            input_ids=input_ids,
            gate_mask=gate_mask,
            qo_indptr=qo_indptr,
            position_ids=position_ids,
        ) # [bsz, dec_len, vocab_size], [bsz, dec_len, hidden_size]
        
        # delete the KV cache entries for the draft(mask) tokens
        self.kv_page_table.delete_kv(self.draft_length)

        return logits, hidden_states
    
    
    def draft_and_verify(self, input_ids: Tensor, gate_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Regular routine for self-speculative decoding with MTP.

        Assume that k draft tokens are generated in the previous step (x_0, ..., x_k) where k is the draft length.
        Then the input sequence is [x_0, m_1, ..., m_k, x_1, m_1, ..., m_k, ..., x_k, m_1, ..., m_k] where m_i is the <mask> token.
        And the corresponding gate_mask is [0, 1, ..., 1, 0, 1, ..., 1, ..., 0, 1, ..., 1] where the 0's for x_0, x_1, ..., x_k and 1's for m_1, ..., m_k.
        
        Note that in this case, we use non-causal attention mask and custom position_ids.
        The attention mask follows the rules:
            1. Every regular token (x_i) only attends to the previous regular tokens,
            2. Every mask token (m_i) only attends to the previous regular tokens and previous mask tokens within the same block of mask tokens (m_1, ..., m_k).
        
        For example, when k = 2, the input sequence would be [x_0, m_1, m_2, x_1, m_1, m_2] and the gate_mask would be [0, 1, 1, 0, 1, 1].
        Then, the corresponding attention mask would be:
            [[1,0,0,0,0,0],
             [1,1,0,0,0,0],
             [1,1,1,0,0,0],
             [1,0,0,1,0,0],
             [1,0,0,1,1,0],
             [1,0,0,1,1,1]]
        , and the corresponding position_ids would be [0, 1, 2, 1, 2, 3] which can be derived from the attention mask.
        
        Args:
            input_ids: Interleaved tokens and masks
            gate_mask: Gate mask for LoRA
            
        Returns:
            Tuple of (logits, hidden_states)
        """
        bsz, dec_len = input_ids.shape
        assert dec_len == self.draft_and_verify_len, f"The input sequence length must be equal to the draft_and_verify length, but got dec_len={dec_len} and draft_and_verify_len={self.draft_and_verify_len}"
        
        # prepare qo_indptr, position_ids, and attn_mask
        qo_indptr = torch.arange(bsz + 1, device=self.device, dtype=torch.int32) * dec_len
        position_ids = (self.kv_page_table.cachelens[:, None] + self.common_position_ids[None, :]).flatten()
        
        mask_arr = []
        for i in range(bsz):
            ones_mask = torch.ones((dec_len, self.kv_page_table.cachelens[i]), device=self.device)
            mask_i = torch.cat((ones_mask, self.common_attn_mask), dim=-1)
            mask_arr.append(mask_i.flatten())
        attn_mask = torch.cat(mask_arr, dim=0)
        attn_mask = attn_mask.contiguous().to(device=self.device, dtype=torch.bool)

        # model forward for draft_and_verify
        logits, hidden_states = self.forward(
            input_ids=input_ids,
            gate_mask=gate_mask,
            qo_indptr=qo_indptr,
            position_ids=position_ids,
            attn_mask=attn_mask,
        ) # [bsz, dec_len, vocab_size], [bsz, dec_len, hidden_size]

        return logits, hidden_states
    
    
    def sampler_draft(self, next_tokens: Tensor, draft_hidden_states: Tensor) -> Tensor:
        """
        Draft with sampler.
        Assume that the next_tokens is one generated from the prefill or last accepted token from the evaluate posterior.
        And the draft_hidden_states is obtained by the model forward for draft(mask) tokens proceeding the one that gives the next_tokens.
        
        In other words, in the previous step, the input sequence was either [x_0, m_1, ..., m_k] (prefill) or [x_0, m_1, ..., m_k, ..., x_k, m_1, ..., m_k] (draft_and_verify).
        For the former, the next_tokens is x_1 and the draft_hidden_states is model(m_1, ..., m_k).
        For the latter, for example, when [x_0, x_1, x_2] is accepted, the next_tokens is x_3 and the draft_hidden_states is model(x_2, m_1, ..., m_k)[1:] (drop the first one).

        This function generates the actual draft tokens by sampling from the sampler.

        Args:
            next_tokens (torch.Tensor): The next tokens to be generated. Shape: [bsz, 1]
            draft_hidden_states (torch.Tensor): The hidden states of the draft tokens proceeding the one that gives the next_tokens. Shape: [bsz, draft_length, hidden_size]

        Returns:
            draft_tokens (torch.Tensor): The actual draft tokens. Shape: [bsz, draft_length]
        """
        # placeholder for draft tokens (actually, 1 regular token + draft_length draft tokens)
        tokens_buffer = torch.zeros((self.batch_size, 1 + self.draft_length), dtype=torch.long, device=self.device) # [bsz, 1 + draft_length]
        tokens_buffer[:, :1] = next_tokens # [bsz, 1]
        
        # model forward for sampler
        with torch.inference_mode():
            for j in range(self.draft_length):
                tokens_buffer[:, j+1:j+2] = self.model.sampler_forward(tokens_buffer[:, j:j+1], draft_hidden_states[:, j:j+1, :]) # [bsz, 1, vocab_size]

        return tokens_buffer[:, 1:]
    

    def generate_batch(self, input_ids: Tensor, max_gen_len: int, prefix_len: int):
        """
        Generate a batch of tokens using the self-speculative decoding with MTP.

        If the force_budget is not set, the generation will terminate whenever at least one EOS token is accepted.
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

        force_budget = self.force_budget
        eos_token_id = self.eos_token_id
        greedy = self.greedy

        # Define local variables
        model_steps = 0
        num_generated_tokens = torch.zeros(bsz, device=device, dtype=torch.int32)
        batch_indices = torch.arange(bsz, device=device)
        batch_indices_2d = torch.arange(bsz, device=device)[:, None]
        output = torch.zeros(bsz, max_gen_len * 3, device=device, dtype=torch.long)
        
        # Prefill
        tokens_buffer = torch.zeros(bsz, 1+k, device=device, dtype=torch.long) # one is the prev step's next token, k for draft tokens
        next_tokens = self.prefill(input_ids=input_ids)
        output[:, 0] = next_tokens[:, 0]

        # Initialize the profiler (NullProfiler when profiling=False)
        profiler = get_active_profiler()
        profiler.begin_run(bsz=bsz, label="decode", prefix_len=prefix_len)

        terminal = False
        first_draft = True
        while num_generated_tokens.sum() < bsz * max_gen_len and not terminal:
            with profiler.step_timing_ctx():
                profiler.set_step_seq_len(self.kv_page_table.cachelens)

                if first_draft:
                    # First draft (no verification) -> Effectively no token is generated
                    input_ids, gate_mask = self.interleave_mask_tokens(input_ids=next_tokens) # [bsz, 1+k], [bsz, 1+k]
                    logits, hidden_states = self.draft(input_ids=input_ids, gate_mask=gate_mask) # [bsz, 1+k, vocab_size], [bsz, 1+k, hidden_size]
                    
                    # tokens_buffer[:, 0] : next_tokens / tokens_buffer[:, 1:] : draft_tokens
                    tokens_buffer[:, :1] = sample(logits[:, 0], top_p=self.top_p, top_k=self.top_k, temperature=self.temperature)
                    tokens_buffer[:, 1:] = self.sampler_draft(tokens_buffer[:, :1], hidden_states[:, 1:]) # [bsz, k], [bsz, k, vocab_size]
                    first_draft = False
                else:
                    # Proceeding drafts (with verification)
                    # Prepare inputs
                    input_ids, gate_mask = self.interleave_mask_tokens(input_ids=tokens_buffer) # [bsz, (k+1)^2], [bsz, (k+1)^2]
                    past_cachelens = self.kv_page_table.cachelens.clone()

                    # Model forward for draft and verify
                    target_logits, hidden_states = self.draft_and_verify(input_ids=input_ids, gate_mask=gate_mask) # [bsz, (k+1)^2, vocab_size], [bsz, (k+1)^2, hidden_size]
                    
                    # Evaluate the posterior
                    if greedy:
                        bonus_tokens, accept_nums, eos_accepted = self.evaluate_posterior(tokens_buffer[:, 1:], target_logits.argmax(dim=-1))
                    else:
                        bonus_tokens, accept_nums, eos_accepted = self.evaluate_posterior(tokens_buffer[:, 1:], target_logits)
                    
                    # Force budget
                    if force_budget:
                        bonus_tokens, accept_nums = self.budget_forcing(tokens_buffer[:, 1:], bonus_tokens, accept_nums)

                    # Collate the accepted KV cache entries
                    self.collate_accepted_kv_cache(accept_nums, past_cachelens)

                    # Register accept_nums to Profiler
                    profiler.set_step_tokens(int(accept_nums.sum().item()))
                    
                    # Write the accepted tokens to the output
                    write_indices = (num_generated_tokens + 1)[:, None] + torch.arange(k + 1, device=device)[None, :] # [B, k+1]
                    output[batch_indices_2d, write_indices] = tokens_buffer
                    num_generated_tokens += accept_nums
                    
                    # Prepare for next iteration
                    tokens_buffer[:, :1] = bonus_tokens
                    hidden_states = hidden_states.reshape(bsz, k+1, k+1, -1) # [bsz, k+1, k+1, hidden_size]
                    selected_hidden_states = hidden_states[batch_indices, accept_nums-1, 1:, :] # [bsz, k, hidden_size]
                    tokens_buffer[:, 1:] = self.sampler_draft(tokens_buffer[:, :1], selected_hidden_states) # [bsz, k], [bsz, k, vocab_size]
                    if (not force_budget) and (eos_accepted).any(): terminal = True
                    model_steps += 1
                    
            if (not force_budget) and (tokens_buffer[:, 0] == eos_token_id).any(): terminal = True
        
        profiler.end_run()
        self.kv_page_table.delete_kv(num_generated_tokens) # revert the KV cache to proceed next run with longer prefix

        return output, num_generated_tokens, model_steps

    # =============================== Helper functions ===============================
    def get_causal_mask(self, bsz: int, cachelens: int, declen: int) -> Tensor:
        """
        Get the causal mask for the attention.
        Args:
            bsz: The batch size.
            cachelens: The number of cached tokens.
            declen: The length of the decoded tokens.
        Returns:
            The causal mask for the attention. Shape: [declen, cachelens+declen]
        """
        return torch.tril(torch.ones(bsz, declen, cachelens+declen, device=self.device, dtype=torch.bool))
    
    def setup_common_attn_mask_and_position_ids(self):
        """
        Build non-causal MTP attention mask (bool) and position_ids.

        Sequence (length S = k*(k+1)) consists of k groups:
        group g in [0..k-1]: [x_g, m1, m2, ..., mk]
        index i:
            g(i) = i // (k+1)      # group id
            r(i) = i %  (k+1)      # 0 -> regular, 1..k -> mask id

        Mask rule:
        - row r(i)=0 (regular): allow columns with r(j)=0 and j <= i
        - row r(i)>0 (mask):    allow columns with r(j)=0 and j <= i  (all prev regulars)
                                + same-group masks with 1 <= r(j) <= r(i)
        position_ids:
        pos[i] = g(i) + r(i)
        """

        S = (self.draft_length + 1) ** 2
        idx = torch.arange(S, device=self.device)

        # Closed-form position ids
        g = torch.div(idx, (self.draft_length + 1), rounding_mode='floor')
        r = idx % (self.draft_length + 1)
        position_ids = (g + r).to(torch.long)

        # Vectorized boolean mask
        I, J = idx[:, None], idx[None, :]
        g_i, g_j = g[:, None], g[None, :]
        r_i, r_j = r[:, None], r[None, :]

        allowed_regulars = (r_j == 0) & (J <= I)
        same_group_masks = (g_i == g_j) & (r_j > 0) & (r_j <= r_i)
        attn_mask = allowed_regulars | same_group_masks

        self.common_attn_mask = attn_mask
        self.common_position_ids = position_ids
    

    @torch.no_grad()
    def interleave_mask_tokens(
        self,
        input_ids: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Interleave mask tokens with input tokens.
        
        Transforms [x_0, x_1, ...] into [x_0, m_1, ..., m_k, x_1, m_1, ..., m_k, ...]
        where m_i are mask tokens and k is the draft length.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            
        Returns:
            Tuple of (interleaved_ids, gate_mask)
            - interleaved_ids: Shape (batch_size, seq_len * (draft_length + 1))
            - gate_mask: 0 for tokens, 1 for masks
        """
        B, L = input_ids.shape
        D = self.draft_length

        # Output length = L * (D + 1)
        out_len = L * (D + 1)

        # 1) Allocate and fill with mask_token_id
        out_ids = torch.empty((B, out_len), dtype=input_ids.dtype, device=self.device)
        out_ids.fill_(self.mask_token_id)

        # View as [B, L, D+1] and write tokens at slot 0 of each block
        view_ids = out_ids.view(B, L, D + 1)
        view_ids[:, :, 0] = input_ids  # tokens at the first position of each (D+1)-block

        # 2) gate_mask: 1 for masks, 0 for tokens
        gate_mask = torch.ones((B, out_len), dtype=self.dtype, device=self.device)
        view_gate = gate_mask.view(B, L, D + 1)
        view_gate[:, :, 0] = 0  # token slots

        return out_ids, gate_mask[..., None]


    def evaluate_posterior(
        self,
        draft_tokens: Tensor,
        target_preds: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Evaluate the posterior for the draft tokens.

        Args:
            draft_tokens (torch.Tensor): The tokens that are generated by the sampler. Shape: [bsz, draft_length]
            target_preds (torch.Tensor)
                - (Greedy) The predicted tokens by the target model. Shape: [bsz, draft_length + 1]
                - (Sampling) The predicted logits by the target model. Shape: [bsz, draft_length + 1]

        Returns:
            bonus_tokens (torch.Tensor): The bonus tokens. Shape: [bsz, 1]
            accept_nums (torch.Tensor): The number of accepted tokens. Shape: [bsz, 1]
            eos_accepted (torch.Tensor): The flag of whether the EOS tokens are accepted. Shape: [bsz, 1]
        """
        bsz, draft_length = draft_tokens.shape
        if self.greedy:
            eos_condition = (draft_tokens == self.eos_token_id) # [bsz, draft_length]
            accept_flags_matrix = target_preds[:, :draft_length] == draft_tokens # [bsz, draft_length]
            accept_nums = accept_flags_matrix.sum(dim=1, keepdim=True) + 1  # [bsz, 1]
            
            bonus_tokens = target_preds.gather(1, accept_nums - 1) # [bsz, 1]
            eos_accepted = (eos_condition & accept_flags_matrix).any(dim=1, keepdim=True) # [bsz, 1]
            return bonus_tokens, accept_nums[:, 0], eos_accepted
        else:
            vocab_size = target_preds.shape[-1]

            target_probs = get_sampling_probs(target_preds, top_p=self.top_p, top_k=self.top_k, temperature=self.temperature) # [bsz, draft_length+1, V]
            draft_probs = torch.zeros((bsz * draft_length, vocab_size), dtype=target_preds.dtype, device=target_preds.device)
            draft_probs.scatter_(1, draft_tokens.reshape(-1, 1), 1.0)
            draft_probs = draft_probs.reshape(bsz, draft_length, vocab_size)

            output_tokens, _, emitted_nums = chain_speculative_sampling(draft_probs, draft_tokens, target_probs)
            accept_nums = emitted_nums + 1 # [bsz, 1]
            
            last_valid_idx = ((output_tokens != -1).to(torch.long) * torch.arange(output_tokens.size(1), device=output_tokens.device)).argmax(dim=1, keepdim=True)
            bonus_tokens = output_tokens.gather(1, last_valid_idx) # [bsz, 1]
            eos_accepted = (self.eos_token_id == output_tokens).any(dim=1, keepdim=True) # [bsz, 1]
            return bonus_tokens, accept_nums, eos_accepted
    

    def collate_accepted_kv_cache(
        self,
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
            accept_nums (torch.Tensor): The number of accepted tokens. Shape: [bsz]
            prev_cachelens (torch.Tensor): The number of KV cache entries from the previous step. Shape: [bsz]

        Returns:
            None
        """ 
        assert accept_nums.dim() == 1 and prev_cachelens.dim() == 1, f"The accept_nums and prev_cachelens are expected to be a 1D tensor but got {accept_nums.dim()}D and {prev_cachelens.dim()}D."

        bsz = accept_nums.shape[0]
        n_local_heads, head_dim = self.model.config.n_local_heads, self.model.config.head_dim
        draft_length = self.draft_length
        stride = draft_length + 1
        max_accept_len = int(accept_nums.max().item())

        base = torch.arange(max_accept_len, device=self.device, dtype=torch.long) * stride # [max_accept_len]
        src = (prev_cachelens[:, None] + base[None, :]).reshape(-1) # [bsz * max_accept_len]
        dst = (prev_cachelens[:, None] + torch.arange(max_accept_len, device=self.device, dtype=torch.long)[None, :]).reshape(-1) # [bsz * max_accept_len]

        cols = torch.arange(max_accept_len, device=self.device, dtype=torch.long).expand(bsz, -1) # [bsz, max_accept_len]
        valid = (cols < accept_nums[:, None]).reshape(-1) # [bsz * max_accept_len]

        bidx = torch.arange(bsz, device=self.device, dtype=torch.long).repeat_interleave(max_accept_len) # [bsz * max_accept_len]
        bidx, src, dst = bidx[valid], src[valid], dst[valid] # [nnz]

        for layer in self.model.layers:
            kv = layer.attention.kv_cache.kv_cache
            kv = kv.permute(0, 2, 1, 3, 4) # [num_pages, page_size, 2, n_local_heads, head_dim]
            orig = kv.shape
            kv = kv.reshape(bsz, -1, 2, n_local_heads, head_dim) # [bsz, num_pages * page_size, 2, n_local_heads, head_dim]
            kv[bidx, dst] = kv[bidx, src] # RHS copy → LHS write (overlap-safe)
            kv = kv.reshape(orig).permute(0, 2, 1, 3, 4) # [num_pages, 2, page_size, n_local_heads, head_dim]
            layer.attention.kv_cache.kv_cache = kv

        inserted_len = (self.draft_length + 1) ** 2
        self.kv_page_table.delete_kv(inserted_len - accept_nums)
    

    def budget_forcing(
        self,
        draft_tokens: Tensor,
        bonus_tokens: Tensor,
        accept_nums: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Force budget by suppressing specific tokens.
        
        Args:
            draft_tokens: Draft tokens
            bonus_tokens: Bonus tokens
            accept_nums: Accept counts
            
        Returns:
            Tuple of (updated_bonus_tokens, updated_accept_nums)
        """
        suppressed_accept_nums = accept_nums.clone()
        suppress_mask_in_accepted = (draft_tokens == self.suppress_token_id)
        suppress_indices = torch.argmax(suppress_mask_in_accepted.int(), dim=-1)
        suppress_indices[~suppress_mask_in_accepted.any(dim=-1)] = -1
        
        rows_to_update = suppress_indices != -1
        if rows_to_update.any():
            suppressed_accept_nums[rows_to_update] = suppress_indices[rows_to_update].to(suppressed_accept_nums.dtype)
        
        bonus_update_mask = rows_to_update | (bonus_tokens[:, 0] == self.suppress_token_id)
        if bonus_update_mask.any():
            num_to_replace = bonus_update_mask.sum()
            random_indices = torch.randint(0, self.replace_token_ids.shape[0], (num_to_replace,), device=bonus_tokens.device)
            bonus_tokens[bonus_update_mask, 0] = self.replace_token_ids[random_indices].to(bonus_tokens.dtype)
        
        return bonus_tokens, suppressed_accept_nums