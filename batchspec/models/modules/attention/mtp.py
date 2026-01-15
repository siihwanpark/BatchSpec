"""MTP (Multi-Token Prediction) attention implementation."""

from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor

from .base import GatedLoRAAttention
from ...configs import ModelArgs, LoRAConfig
from batchspec.profiler import rope_compute_timer, attention_compute_timer


class MTPAttention(GatedLoRAAttention):
    """Attention implementation for Multi-Token Prediction models.
    
    Extends GatedLoRAAttention to support multi-token prediction.
    
    Args:
        config: Model configuration
        lora_config: LoRA configuration for adaptation
    """
    
    def __init__(self, config: ModelArgs, lora_config: LoRAConfig):
        super().__init__(config, lora_config)
        
        # Attention kernel to be set during setup
        self.attn_kernel = None
    
    def forward(
        self,
        x: Tensor,
        gate_mask: Tensor,
        position_ids: Tensor,
        qo_indptr: Tensor,
        kv_page_table: "PageTable",
    ) -> Tensor:
        """Forward pass through MTP attention layer with LoRA gating.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            gate_mask: Optional mask for LoRA gating
            position_ids: Position IDs for RoPE
            qo_indptr: Index pointer for Query/Output tokens
            kv_page_table: Page table for KV cache
            
        Returns:
            Output tensor of shape (batch_size, seq_len, dim)
        """
        bsz, seqlen, _ = x.shape
        
        # Split QKV
        q, k, v = self._split_qkv(self.wqkv(x, gate_mask))
        
        # Apply normalization
        q = self.q_norm(q.view(bsz, seqlen, self.n_head, self.head_dim))
        k = self.k_norm(k.view(bsz, seqlen, self.n_local_heads, self.head_dim))
        
        # Reshape for attention computation
        q = q.view(bsz * seqlen, self.n_head, self.head_dim)
        k = k.view(bsz * seqlen, self.n_local_heads, self.head_dim)
        v = v.contiguous().view(bsz * seqlen, self.n_local_heads, self.head_dim)
        
        # Apply RoPE with position IDs
        with rope_compute_timer():
            q, k = self._apply_rope(q, k, position_ids)
        
        # Update KV cache
        kv_cache = self.kv_cache.update(k, v, qo_indptr, kv_page_table)
        
        # Compute attention
        with attention_compute_timer():
            y = self.attn_kernel.run(q, kv_cache)
        
        # Reshape and project output with gated LoRA
        y = y.contiguous().view(bsz, seqlen, self.dim)
        y = self.wo(y, gate_mask)
        return self._maybe_all_reduce_output(y)
