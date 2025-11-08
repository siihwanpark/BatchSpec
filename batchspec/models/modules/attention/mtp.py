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
        
        # MTP-specific attention kernels
        self.attn_prefill = None
        self.attn_draft = None
        self.attn_draft_and_verify = None
    
    def forward(
        self,
        x: Tensor,
        gate_mask: Optional[Tensor],
        position_ids: Tensor,
        kv_append_indptr: Tensor,
        kv_page_indices: Tensor,
        kv_page_indptr: Tensor,
        kv_page_lastlen: Tensor,
        attn_type: str = "prefill"
    ) -> Tensor:
        """Forward pass through MTP attention layer with LoRA gating.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            gate_mask: Optional mask for LoRA gating
            position_ids: Position IDs for RoPE
            kv_append_indptr: Indices for KV cache appending
            kv_page_indices: Page indices for KV cache
            kv_page_indptr: Page indirection pointers
            kv_page_lastlen: Last length of each page
            attn_type: Type of attention ("prefill", "draft", or "draft_and_verify")
            
        Returns:
            Output tensor of shape (batch_size, seq_len, dim)
        """
        bsz, seqlen, _ = x.shape
        
        # Apply QKV projection with LoRA gating
        qkv = self.wqkv(x, gate_mask)
        
        # Split QKV
        q, k, v = self._split_qkv(qkv)
        
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
        kv_cache = self.kv_cache.update(
            k, v, kv_append_indptr,
            kv_page_indices, kv_page_indptr, kv_page_lastlen
        )
        
        # Compute attention based on type
        attn_kernels = {
            "prefill": self.attn_prefill,
            "draft": self.attn_draft,
            "draft_and_verify": self.attn_draft_and_verify,
        }

        with attention_compute_timer():
            y = self._compute_attention(q, kv_cache, attn_type, attn_kernels)
        
        # Reshape and project output with gated LoRA
        y = y.contiguous().view(bsz, seqlen, self.dim)
        y = self.wo(y, gate_mask)
        return self._maybe_all_reduce_output(y)
