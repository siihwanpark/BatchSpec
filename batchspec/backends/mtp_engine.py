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
        
        # Forward functions
        self.prefill_forward = lambda model, x, position_ids, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen: \
            model(x, None, position_ids, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen, attn_type="prefill")
        self.draft_forward = lambda model, x, gate_mask, position_ids, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen: \
            model(x, gate_mask, position_ids, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen, attn_type="draft")
        self.draft_and_verify_forward = lambda model, x, gate_mask, position_ids, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen: \
            model(x, gate_mask, position_ids, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen, attn_type="draft_and_verify")
    
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
        # Setup base caches
        self.max_seq_len = max_seq_length
        self.draft_and_verify_len = (self.draft_length + 1) ** 2
        self.max_cache_len = max_seq_length + self.draft_and_verify_len + 1
        super().setup_caches(max_batch_size, self.max_cache_len, page_size, prefill_chunk_size)
        
        # Set common attention masks and position ids
        self._setup_common_attn_mask_and_position_ids()

        # Create attention buffers
        self.prefill_buffer = self._create_attention_buffer(384)  # 3 * 128MB
        self.draft_buffer = self._create_attention_buffer(384)
        self.draft_and_verify_buffer = self._create_attention_buffer(384)
        self.custom_mask_buf = torch.empty(max_batch_size * self.draft_and_verify_len * self.max_cache_len // 8 + 1, dtype=torch.uint8, device=self.device)
        self.mask_indptr_buf = torch.arange(max_batch_size + 1, dtype=torch.int32, device=self.device)
        
        # Create attention wrappers
        self.attn_wrappers = {
            "prefill": self._create_attention_wrapper(
                self.prefill_buffer,
                qo_length=prefill_chunk_size
            ),
            "draft": self._create_attention_wrapper(
                self.draft_buffer,
                qo_length=self.draft_length
            ),
            "draft_and_verify": self._create_attention_wrapper(
                self.draft_and_verify_buffer,
                qo_length=self.draft_and_verify_len,
                use_custom_mask=True,
                custom_mask_buf=self.custom_mask_buf,
                mask_indptr_buf=self.mask_indptr_buf,
            ),
        }
        
        # Register with custom ops
        self._register_attention_wrappers({
            "attn_prefill": self.attn_wrappers["prefill"],
            "attn_draft": self.attn_wrappers["draft"],
            "attn_draft_and_verify": self.attn_wrappers["draft_and_verify"],
        })
        
        # Setup model caches
        with torch.device(self.device):
            self.model.setup_caches(num_pages=self.max_num_pages, page_size=self.page_size)

    def compile(self):
        """Enable torch.compile for forward functions.
        """
        super().compile()
        self.prefill_forward = torch.compile(self.prefill_forward)
        self.draft_forward = torch.compile(self.draft_forward)
        self.draft_and_verify_forward = torch.compile(self.draft_and_verify_forward)
    
    def _setup_common_attn_mask_and_position_ids(self):
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
    
    def prefill(
        self,
        input_ids: Tensor,
        query_lens: Tensor
    ) -> Tensor:
        """Prefill input sequence and return first predicted token.
        
        Similar to standard prefill but uses position_ids and handles padding differently.
        
        Args:
            input_ids: Input token IDs [bsz, seq_len]
            query_lens: Actual length per sequence in each batch
            
        Returns:
            Next token predictions [bsz, 1]
        """
        # Clear KV cache
        self.clear_kv()
        
        # Initialize logits and last_recorded
        logits = None
        bsz, seq_len = input_ids.shape
        assert seq_len % self.prefill_chunk_size == 0, f"The sequence length must be divisible by the prefill chunk size, but got seq_len={seq_len} and prefill_chunk_size={self.prefill_chunk_size}"

        last_logits = None # For lazy initialization
        last_recorded = torch.zeros(bsz, dtype=torch.bool, device=self.device)

        chunk_size = self.prefill_chunk_size
        num_chunks = seq_len // chunk_size

        for i in range(num_chunks):
            chunk_input_ids = input_ids[:, i*chunk_size:(i+1)*chunk_size]
            chunk_query_lens = query_lens - (i * chunk_size)
            chunk_query_lens = torch.clamp(chunk_query_lens, min=0, max=chunk_size)
            
            # if every query in chunk only has pad tokens, skip the chunk
            if torch.all(chunk_query_lens == 0): continue
            
            # Target prefill
            position_ids = (self.cachelens[:, None] + torch.arange(chunk_size, device=self.cachelens.device)[None, :]).flatten()
            self.pre_prefill(dec_len=chunk_size)
            with torch.inference_mode():
                logits, _ = self.prefill_forward(
                    model=self.model,
                    x=chunk_input_ids,
                    position_ids=position_ids,
                    kv_append_indptr=self.qo_indptr*chunk_size,
                    kv_page_indices=self.paged_kv_indices,
                    kv_page_indptr=self.paged_kv_indptr,
                    kv_page_lastlen=self.paged_kv_last_page_len,
                )

            # Lazy initialization since the hidden_dim is not known until the first chunk is processed (for TP)
            if last_logits is None:
                last_logits = torch.full((bsz, logits.shape[-1]), float('nan'), device=self.device, dtype=self.dtype)

            # Grab the last token's logits and hidden states for each sequence in the chunk
            target_indices_in_chunk = chunk_query_lens - 1
            finishes_in_this_chunk = (query_lens > i*chunk_size) & (query_lens <= (i+1)*chunk_size)
            target_sequences_mask = finishes_in_this_chunk & (~last_recorded)
            
            target_batch_indices = torch.where(target_sequences_mask)[0]
            if target_batch_indices.numel() > 0:
                indices_in_chunk_to_grab = target_indices_in_chunk[target_batch_indices]
                last_logits[target_batch_indices] = logits[target_batch_indices, indices_in_chunk_to_grab, :]
                last_recorded[target_batch_indices] = True
            
            exists_padding = (chunk_query_lens < chunk_size)
            if exists_padding.any():
                self.delete_kv(chunk_size - chunk_query_lens)
        
        assert not torch.isnan(last_logits).any(), "Found NaN in last_logits."
        return sample(last_logits, top_p=self.top_p, top_k=self.top_k, temperature=self.temperature) # [bsz, 1]
    

    def pre_prefill(self, dec_len: int):
        """Prepare for prefill step.
        
        Args:
            dec_len: Decode length
        """
        self.insert_kv(dec_len)
        self.attn_wrappers["prefill"].plan(
            qo_indptr=self.qo_indptr * dec_len,
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
    
    def draft_and_verify(
        self,
        input_ids: Tensor,
        gate_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
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
        
        # prepare position_ids
        position_ids = (self.cachelens[:, None] + self.common_position_ids[None, :]).flatten()

        # model forward for draft_and_verify
        self.pre_draft_and_verify(bsz, dec_len)
        with torch.inference_mode():
            logits, hidden_states = self.draft_and_verify_forward(
                model=self.model, 
                x=input_ids,
                gate_mask=gate_mask,
                position_ids=position_ids,
                kv_append_indptr=self.qo_indptr*dec_len,
                kv_page_indices=self.paged_kv_indices,
                kv_page_indptr=self.paged_kv_indptr,
                kv_page_lastlen=self.paged_kv_last_page_len,
            )

        return logits, hidden_states
    
    def pre_draft_and_verify(self, bsz: int, dec_len: int):
        """Prepare for draft_and_verify step.
        
        Builds custom attention mask by concatenating:
        - ones_mask for attending to cached KV
        - common_attn_mask for MTP attention pattern
        
        Args:
            bsz: Batch size
            dec_len: Decode length
        """
        mask_arr = []
        for i in range(bsz):
            ones_mask = torch.ones((dec_len, self.cachelens[i]), device=self.device)
            mask_i = torch.cat((ones_mask, self.common_attn_mask), dim=-1)
            mask_arr.append(mask_i.flatten())

        attn_mask = torch.cat(mask_arr, dim=0)
        attn_mask = attn_mask.contiguous().to(device=self.device, dtype=torch.bool)

        self.insert_kv(dec_len)
        self.attn_wrappers["draft_and_verify"].plan(
            qo_indptr=self.qo_indptr*dec_len,
            paged_kv_indptr=self.paged_kv_indptr,
            paged_kv_indices=self.paged_kv_indices,
            paged_kv_last_page_len=self.paged_kv_last_page_len,
            num_qo_heads=self.model.config.n_head, 
            num_kv_heads=self.model.config.n_local_heads, 
            head_dim_qk=self.model.config.head_dim, 
            page_size=self.page_size, 
            q_data_type=self.dtype, 
            causal=False,
            custom_mask=attn_mask
        )
    
    def draft(
        self,
        input_ids: Tensor,
        gate_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
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
        dec_len = input_ids.shape[1]
        assert dec_len == self.draft_length + 1, f"The input sequence length must be equal to the draft length + 1, but got dec_len={dec_len} and draft_length={self.draft_length}"

        # prepare position_ids
        position_ids = (self.cachelens[:, None] + torch.arange(dec_len, device=self.cachelens.device)[None, :]).flatten()

        # model forward for draft
        self.pre_draft(dec_len)
        with torch.inference_mode():
            logits, hidden_states = self.draft_forward(
                model=self.model, 
                x=input_ids,
                gate_mask=gate_mask,
                position_ids=position_ids,
                kv_append_indptr=self.qo_indptr * dec_len,
                kv_page_indices=self.paged_kv_indices,
                kv_page_indptr=self.paged_kv_indptr,
                kv_page_lastlen=self.paged_kv_last_page_len,
            ) # [bsz, dec_len, vocab_size], [bsz, dec_len, hidden_size]
        
        # delete the KV cache entries for the draft(mask) tokens
        self.delete_kv(self.draft_length)

        return logits, hidden_states

    def pre_draft(self, dec_len: int):
        """Prepare for draft step.
        
        Args:
            dec_len: draft length + 1
        """
        self.insert_kv(dec_len)
        self.attn_wrappers["draft"].plan(
            qo_indptr=self.qo_indptr*dec_len,
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
    
    def sampler_draft(
        self,
        next_tokens: Tensor,
        draft_hidden_states: Tensor
    ) -> Tensor:
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
        self.delete_kv(inserted_len - accept_nums)
    
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

    def generate_batch(self, input_ids: Tensor, query_lens: Tensor):
        """
        Generate a batch of tokens using the self-speculative decoding with MTP.

        If the force_budget is not set, the generation will terminate whenever at least one EOS token is accepted.
        Otherwise, the generation will proceed until the budget is exhausted by replacing </think> tokens to 'Wait' or 'Alternatively'.
        
        Args:
            input_ids: [bsz, max_query_len]
            query_lens: [bsz]

        Returns:
            output: [bsz, max_len+draft_length+1]
            num_generated_tokens: [bsz]
            num_total_tokens: [bsz]
            model_steps: int
        """

        # Pre-link local variables to reduce indexing time
        bsz = input_ids.shape[0]
        k = self.draft_length
        device = self.device
        force_budget = self.force_budget
        eos_token_id = self.eos_token_id
        greedy = self.greedy
        max_len = self.max_seq_len

        # Define local variables
        model_steps = 0
        max_query_len = query_lens.max()
        num_total_tokens = query_lens.clone()
        batch_indices = torch.arange(bsz, device=device)
        batch_indices_2d = torch.arange(bsz, device=device)[:, None]

        output = torch.zeros(bsz, max_len+k+1, device=device, dtype=torch.long)
        output[:, :max_query_len] = input_ids[:, :max_query_len]
        
        # Prefill
        tokens_buffer = torch.zeros(bsz, 1+k, device=device, dtype=torch.long) # one is the prev step's next token, k for draft tokens
        next_tokens = self.prefill(input_ids=input_ids, query_lens=query_lens)
        output[batch_indices, num_total_tokens] = next_tokens[:, 0]
        num_total_tokens += 1
        model_steps += 1

        # Initialize the profiler (NullProfiler when profiling=False)
        profiler = get_active_profiler()
        profiler.begin_run(bsz=bsz, label="generate_batch")

        terminal = False
        first_draft = True
        while num_total_tokens.max() < max_len and not terminal:
            with profiler.step_timing_ctx():
                profiler.set_step_seq_len(num_total_tokens)

                if first_draft:
                    # First draft (no verification)
                    input_ids, gate_mask = self.interleave_mask_tokens(input_ids=next_tokens) # [bsz, 1+k], [bsz, 1+k]
                    logits, hidden_states = self.draft(input_ids=input_ids, gate_mask=gate_mask) # [bsz, 1+k, vocab_size], [bsz, 1+k, hidden_size]
                    
                    # tokens_buffer[:, 0] : next_tokens / tokens_buffer[:, 1:] : draft_tokens
                    tokens_buffer[:, :1] = sample(logits[:, 0], top_p=self.top_p, top_k=self.top_k, temperature=self.temperature)
                    tokens_buffer[:, 1:] = self.sampler_draft(tokens_buffer[:, :1], hidden_states[:, 1:]) # [bsz, k], [bsz, k, vocab_size]

                    # register accept_nums to Profiler
                    profiler.set_step_tokens(bsz)
                    
                    output[batch_indices, num_total_tokens] = tokens_buffer[:, 0]
                    num_total_tokens += 1
                    first_draft = False
                else:
                    # Proceeding drafts (with verification)
                    assert torch.all(num_total_tokens-1 == self.cachelens), "The number of total tokens must be equal to the cachelens+1."
                    
                    # Prepare inputs
                    input_ids, gate_mask = self.interleave_mask_tokens(input_ids=tokens_buffer) # [bsz, (k+1)^2], [bsz, (k+1)^2]

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
                    self.collate_accepted_kv_cache(accept_nums, num_total_tokens-1)

                    # Register accept_nums to Profiler
                    profiler.set_step_tokens(int(accept_nums.sum().item()))
                    
                    # Write the accepted tokens to the output
                    write_indices = num_total_tokens[:, None] + torch.arange(k + 1, device=device)[None, :] # [B, k+1]
                    output[batch_indices_2d, write_indices] = tokens_buffer
                    num_total_tokens += accept_nums
                    
                    # Prepare for next iteration
                    tokens_buffer[:, :1] = bonus_tokens
                    hidden_states = hidden_states.reshape(bsz, k+1, k+1, -1) # [bsz, k+1, k+1, hidden_size]
                    selected_hidden_states = hidden_states[batch_indices, accept_nums-1, 1:, :] # [bsz, k, hidden_size]
                    tokens_buffer[:, 1:] = self.sampler_draft(tokens_buffer[:, :1], selected_hidden_states) # [bsz, k], [bsz, k, vocab_size]
                    if (not force_budget) and (eos_accepted).any(): terminal = True

            model_steps += 1
            if (not force_budget) and (tokens_buffer[:, 0] == eos_token_id).any(): terminal = True
        
        profiler.end_run()
        
        num_generated_tokens = num_total_tokens - query_lens
        return output, num_generated_tokens, num_total_tokens, model_steps