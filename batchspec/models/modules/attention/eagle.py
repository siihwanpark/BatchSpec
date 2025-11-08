"""EAGLE model attention implementation."""

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor

from .base import BaseAttention
from ...configs import ModelArgs
from batchspec.profiler import attention_compute_timer, rope_compute_timer


class EAGLEAttention(BaseAttention):
    """Attention implementation for EAGLE speculative decoding.
    
    EAGLE uses a modified attention mechanism that processes concatenated
    input embeddings and hidden states for efficient speculation.
    
    Args:
        config: Model configuration
        use_chain_mode: Whether to use chain mode (different attention patterns)
    """
    
    def __init__(self, config: ModelArgs, use_chain_mode: bool = False):
        # First initialize base class (sets up standard wqkv)
        super().__init__(config)
        
        # Override wqkv for EAGLE's concatenated input (2*dim input)
        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        self.wqkv = nn.Linear(config.dim * 2, total_head_dim, bias=config.qkv_bias)
        
        self.use_chain_mode = use_chain_mode
        
        # EAGLE-specific attention kernels
        self.prefill_attn = None
        
        if use_chain_mode:
            # Chain mode uses different kernels
            self.decode_1_attn = None
            self.decode_2_attn = None
        else:
            # Standard EAGLE mode
            self.init_speculate_attn = None
            self.sub_speculate_attn = None
    
    def forward(
        self,
        x: Tensor,
        input_pos: Union[Tensor, tuple],  # Can be position_ids or (indptr, offsets)
        kv_append_indptr: Tensor,
        kv_page_indices: Tensor,
        kv_page_indptr: Tensor,
        kv_page_lastlen: Tensor,
        attn_type: str = "prefill"
    ) -> Tensor:
        """Forward pass through EAGLE attention layer.
        
        Args:
            x: Concatenated input tensor of shape (batch_size, seq_len, 2*dim)
               Contains [input_embeds, hidden_states] concatenated
            input_pos: Position information (varies by configuration)
            kv_append_indptr: Indices for KV cache appending
            kv_page_indices: Page indices for KV cache
            kv_page_indptr: Page indirection pointers
            kv_page_lastlen: Last length of each page
            attn_type: Type of attention computation
            
        Returns:
            Output tensor of shape (batch_size, seq_len, dim)
        """
        bsz, seqlen, _ = x.shape
        
        # Split QKV from concatenated input
        q, k, v = self._split_qkv(self.wqkv(x))
        
        # Note: EAGLE doesn't use Q/K normalization
        # Reshape for attention computation
        q = q.view(bsz * seqlen, self.n_head, self.head_dim)
        k = k.view(bsz * seqlen, self.n_local_heads, self.head_dim)
        v = v.contiguous().view(bsz * seqlen, self.n_local_heads, self.head_dim)
        
        # Apply RoPE
        if isinstance(input_pos, tuple):
            # Unpack indptr and offsets
            indptr, offsets = input_pos
            q, k = self._apply_rope(q, k, indptr, offsets)
        else:
            # Single tensor (position_ids or offsets)
            # Check if we need indptr based on model configuration
            if hasattr(self, 'use_position_ids') and self.use_position_ids:
                q, k = self._apply_rope(q, k, input_pos)
            else:
                q, k = self._apply_rope(q, k, kv_append_indptr, input_pos)
        
        # Update KV cache
        kv_cache = self.kv_cache.update(
            k, v, kv_append_indptr,
            kv_page_indices, kv_page_indptr, kv_page_lastlen
        )
        
        # Select attention kernels based on mode
        if self.use_chain_mode:
            attn_kernels = {
                "prefill": self.prefill_attn,
                "decode_1": self.decode_1_attn,
                "decode_2": self.decode_2_attn,
            }
        else:
            attn_kernels = {
                "prefill": self.prefill_attn,
                "initial_speculate": self.init_speculate_attn,
                "subsequent_speculate": self.sub_speculate_attn,
            }
        
        with attention_compute_timer():
            y = self._compute_attention(q, kv_cache, attn_type, attn_kernels)
        
        # Reshape and project output
        y = y.contiguous().view(bsz, seqlen, self.dim)
        y = self.wo(y)
        
        return self._maybe_all_reduce_output(y)
