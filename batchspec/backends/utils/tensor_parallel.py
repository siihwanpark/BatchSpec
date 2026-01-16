"""Tensor parallelism utilities with automatic LoRA detection.

This module provides clean TP utilities that automatically detect and handle
LoRA layers (GatedLoRALinear) without breaking regular Linear layers.
"""

from __future__ import annotations

import os
from itertools import accumulate
from typing import List, Optional, Sequence

import torch
import torch.distributed as dist
from torch import nn


# =============================================================================
# Rank / Environment Utilities
# =============================================================================

def _get_global_rank() -> int:
    """Get global rank from environment."""
    return int(os.environ.get("LOCAL_RANK", "0"))


def _get_world_size() -> int:
    """Get world size from environment."""
    return int(os.environ.get("LOCAL_WORLD_SIZE", "1"))


def _ensure_cuda_device_set() -> None:
    """Set CUDA device to match global rank."""
    torch.cuda.set_device(_get_global_rank())


# =============================================================================
# Generic TP Helpers
# =============================================================================

def _validate_tp_size(num_kv_heads: int, num_heads: int) -> bool:
    """Validate that heads are divisible by world size.
    
    Args:
        num_kv_heads: Number of KV heads
        num_heads: Number of query heads
        
    Returns:
        True if valid for TP
    """
    world_size = _get_world_size()
    return num_kv_heads % world_size == 0 and num_heads % world_size == 0


def _rank_index_in_group(rank_group: Sequence[int]) -> tuple[int, int, int]:
    """Get rank information within a group.
    
    Args:
        rank_group: List of ranks in the group
        
    Returns:
        tuple of (global_rank, index_in_group, group_world_size)
    """
    gr = _get_global_rank()
    assert rank_group and gr in rank_group, "rank_group must include current rank"
    return gr, rank_group.index(gr), len(rank_group)


def _even_splits(total: int, parts: int) -> List[int]:
    """Split total into roughly even parts.
    
    Remainder goes to lower-indexed ranks.
    
    Args:
        total: Total value to split
        parts: Number of parts
        
    Returns:
        List of sizes for each part
    """
    base, rem = divmod(total, parts)
    out = [base] * parts
    for i in range(rem):
        out[i] += 1
    return out


def _kv_slice_for_rank(num_kv_heads: int, rank_group: Sequence[int]) -> tuple[int, int]:
    """Calculate KV head slice for current rank.
    
    Args:
        num_kv_heads: Total number of KV heads
        rank_group: Rank group
        
    Returns:
        tuple of (start_idx, end_idx) for this rank's KV heads
    """
    _, idx, world = _rank_index_in_group(rank_group)
    sizes = _even_splits(num_kv_heads, world)
    cum = list(accumulate(sizes))
    start = 0 if idx == 0 else cum[idx - 1]
    end = cum[idx]
    return start, end


def _view_slice(x: torch.Tensor, dim: int, start: int, end: int) -> torch.Tensor:
    """Slice tensor along dimension.
    
    Args:
        x: Tensor to slice
        dim: Dimension to slice (0 or 1)
        start: Start index
        end: End index
        
    Returns:
        Sliced tensor view
    """
    if dim == 0:
        return x[start:end]
    if dim == 1:
        return x[:, start:end]
    raise ValueError(f"Unsupported shard dim={dim}")


def _qkv_split_slice_cat(
    t: torch.Tensor,
    dim: int,
    splits: Sequence[int],
    q_slice: tuple[int, int],
    kv_slice: tuple[int, int],
) -> torch.Tensor:
    """Split [Q|K|V] tensor, slice blocks, and concatenate back.
    
    Args:
        t: QKV tensor
        dim: Dimension to split along
        splits: Sizes for Q, K, V splits
        q_slice: (start, end) for Q
        kv_slice: (start, end) for K and V
        
    Returns:
        Sliced and concatenated tensor
    """
    assert len(splits) == 3
    q, k, v = t.split(splits, dim=dim)
    q = _view_slice(q, dim, q_slice[0], q_slice[1])
    k = _view_slice(k, dim, kv_slice[0], kv_slice[1])
    v = _view_slice(v, dim, kv_slice[0], kv_slice[1])
    return torch.cat((q, k, v), dim=dim)


# =============================================================================
# LoRA Detection and Helpers
# =============================================================================

def _is_lora_linear(mod: nn.Module) -> bool:
    """Detect if module is a LoRA linear layer.
    
    Duck-typing approach: checks for base_layer, lora_A, and lora_B/lora_BT.
    
    Args:
        mod: Module to check
        
    Returns:
        True if module appears to be LoRA linear
    """
    return (
        hasattr(mod, "base_layer") and
        hasattr(mod, "lora_A") and
        (hasattr(mod, "lora_B") or hasattr(mod, "lora_BT"))
    )


def _has_lora_BT(mod: nn.Module) -> bool:
    """Check if module has lora_BT (transposed B matrix).
    
    Args:
        mod: Module to check
        
    Returns:
        True if has lora_BT
    """
    return hasattr(mod, "lora_BT")


def _assign_buffer(mod: nn.Module, name: str, tensor: torch.Tensor) -> None:
    """Assign buffer without breaking _buffers dict.
    
    Args:
        mod: Module
        name: Buffer name
        tensor: Tensor to assign
    """
    if getattr(mod, "_buffers", None) is not None and name in mod._buffers:
        mod._buffers[name] = tensor
    else:
        setattr(mod, name, tensor)


def _sync_lora_shapes(ll: nn.Module) -> None:
    """Synchronize in/out/rank dimensions of LoRA module.
    
    Args:
        ll: LoRA linear module
    """
    if hasattr(ll, "base_layer"):
        ll.in_features = ll.base_layer.in_features
        ll.out_features = ll.base_layer.out_features
    if hasattr(ll, "lora_A"):
        ll.lora_rank = ll.lora_A.out_features


def _split2_and_chunk(
    t: torch.Tensor,
    dim: int,
    world: int,
    idx: int
) -> torch.Tensor:
    """Split w13 tensor [Gate|Up] and chunk by rank.
    
    For feed-forward layers with fused gate and up projections.
    
    Args:
        t: Tensor to split
        dim: Dimension to split along
        world: World size
        idx: Rank index
        
    Returns:
        Chunked tensor for this rank
    """
    assert t.shape[dim] % 2 == 0, "w13 split assumes even split"
    half = t.shape[dim] // 2
    a, b = torch.split(t, [half, half], dim=dim)
    a = torch.chunk(a, world, dim=dim)[idx]
    b = torch.chunk(b, world, dim=dim)[idx]
    return torch.cat((a, b), dim=dim)


def _set_param(t: Optional[torch.Tensor]) -> Optional[nn.Parameter]:
    """Convert tensor to parameter.
    
    Args:
        t: Tensor or None
        
    Returns:
        Parameter with requires_grad=False, or None
    """
    return None if t is None else nn.Parameter(t, requires_grad=False)


# =============================================================================
# Linear Layer Sharding (Plain and LoRA)
# =============================================================================

def _linear_colwise(linear: nn.Linear, start: int, end: int) -> None:
    """Apply column-parallel sharding to linear layer.
    
    Args:
        linear: Linear layer
        start: Start index for output dimension
        end: End index for output dimension
    """
    linear.weight = _set_param(_view_slice(linear.weight, 0, start, end))
    if hasattr(linear, "scales"):  # For quantized models
        linear.scales = _view_slice(linear.scales, 0, start, end)
    if linear.bias is not None:
        linear.bias = _set_param(_view_slice(linear.bias, 0, start, end))
    linear.out_features = linear.weight.shape[0]


def _linear_rowwise(linear: nn.Linear, start: int, end: int) -> None:
    """Apply row-parallel sharding to linear layer.
    
    Args:
        linear: Linear layer
        start: Start index for input dimension
        end: End index for input dimension
    """
    linear.weight = _set_param(_view_slice(linear.weight, 1, start, end))
    linear.in_features = linear.weight.shape[1]


def _lora_colwise(ll: nn.Module, start: int, end: int) -> None:
    """Apply column-parallel sharding to LoRA linear layer.
    
    Column-parallel: base & LoRA-B (out-dim) split, LoRA-A replicate.
    
    Args:
        ll: LoRA linear module
        start: Start index for output dimension
        end: End index for output dimension
    """
    assert _is_lora_linear(ll)
    base = ll.base_layer
    _linear_colwise(base, start, end)

    if _has_lora_BT(ll):
        # lora_BT: [rank, out] → out-dim at dim=1
        sliced = _view_slice(ll.lora_BT, 1, start, end).contiguous()
        _assign_buffer(ll, "lora_BT", sliced)
    else:
        # lora_B: weight [out, rank] → out-dim at dim=0
        B = ll.lora_B
        B.weight = _set_param(_view_slice(B.weight, 0, start, end))
        if B.bias is not None:
            B.bias = _set_param(_view_slice(B.bias, 0, start, end))
        B.out_features = B.weight.shape[0]

    _sync_lora_shapes(ll)


def _lora_rowwise(ll: nn.Module, start: int, end: int) -> None:
    """Apply row-parallel sharding to LoRA linear layer.
    
    Row-parallel: base & LoRA-A (in-dim) split, LoRA-B/BT replicate.
    
    Args:
        ll: LoRA linear module
        start: Start index for input dimension
        end: End index for input dimension
    """
    assert _is_lora_linear(ll)
    base, A = ll.base_layer, ll.lora_A
    _linear_rowwise(base, start, end)
    # A: weight [rank, in] so in-dim at dim=1
    A.weight = _set_param(_view_slice(A.weight, 1, start, end))
    A.in_features = A.weight.shape[1]
    # B or BT are replicated (no change)
    _sync_lora_shapes(ll)


def _lora_qkv_colwise(
    ll: nn.Module,
    qkv_splits: Sequence[int],
    q_slice: tuple[int, int],
    kv_slice: tuple[int, int],
) -> None:
    """Apply column-parallel sharding to fused QKV LoRA layer.
    
    Fused QKV: base & LoRA-B/BT have same [Q|K|V] out-dim slice; LoRA-A replicate.
    
    Args:
        ll: LoRA linear module
        qkv_splits: Sizes for Q, K, V
        q_slice: (start, end) for Q dimension
        kv_slice: (start, end) for K and V dimensions
    """
    assert _is_lora_linear(ll)
    base = ll.base_layer
    
    # Base layer: out-dim at dim=0
    base.weight = _set_param(_qkv_split_slice_cat(base.weight, 0, qkv_splits, q_slice, kv_slice))
    if hasattr(base, "scales"):
        base.scales = _qkv_split_slice_cat(base.scales, 0, qkv_splits, q_slice, kv_slice)
    if base.bias is not None:
        base.bias = _set_param(_qkv_split_slice_cat(base.bias, 0, qkv_splits, q_slice, kv_slice))
    base.out_features = base.weight.shape[0]

    if _has_lora_BT(ll):
        # lora_BT: [rank, out] → out-dim at dim=1
        new_BT = _qkv_split_slice_cat(ll.lora_BT, 1, qkv_splits, q_slice, kv_slice).contiguous()
        _assign_buffer(ll, "lora_BT", new_BT)
    else:
        # lora_B: weight [out, rank] → out-dim at dim=0
        B = ll.lora_B
        B.weight = _set_param(_qkv_split_slice_cat(B.weight, 0, qkv_splits, q_slice, kv_slice))
        if B.bias is not None:
            B.bias = _set_param(_qkv_split_slice_cat(B.bias, 0, qkv_splits, q_slice, kv_slice))
        B.out_features = B.weight.shape[0]

    _sync_lora_shapes(ll)


# =============================================================================
# Slice Calculators
# =============================================================================

def _row_slice_inputs(linear: nn.Linear, rank_group: Sequence[int]) -> tuple[int, int]:
    """Calculate row-parallel slice on input dimension.
    
    Args:
        linear: Linear layer
        rank_group: Rank group
        
    Returns:
        tuple of (start, end) for input features
    """
    _, idx, world = _rank_index_in_group(rank_group)
    sizes = _even_splits(linear.in_features, world)
    cum = list(accumulate(sizes))
    start = 0 if idx == 0 else cum[idx - 1]
    end = cum[idx]
    return start, end


def _col_slice_outputs(linear: nn.Linear, rank_group: Sequence[int]) -> tuple[int, int]:
    """Calculate column-parallel slice on output dimension.
    
    Args:
        linear: Linear layer
        rank_group: Rank group
        
    Returns:
        tuple of (start, end) for output features
    """
    _, idx, world = _rank_index_in_group(rank_group)
    sizes = _even_splits(linear.out_features, world)
    cum = list(accumulate(sizes))
    start = 0 if idx == 0 else cum[idx - 1]
    end = cum[idx]
    return start, end


# =============================================================================
# Feed-Forward Network (w13 + w2)
# =============================================================================

def _apply_tp_ffn(mlp, rank_group: Sequence[int], process_group) -> None:
    """Apply tensor parallelism to feed-forward network.
    
    Handles both plain Linear and LoRA layers automatically.
    
    Args:
        mlp: Feed-forward module with w13 and w2
        rank_group: Rank group
        process_group: Process group for all-reduce
    """
    w13 = mlp.w13
    _, idx, world = _rank_index_in_group(rank_group)

    # w13: column-parallel (split fused [Gate|Up] projection)
    if _is_lora_linear(w13):
        base = w13.base_layer
        # Base: out-dim (dim=0) split into [Gate|Up] half, chunk by rank
        base.weight = _set_param(_split2_and_chunk(base.weight, dim=0, world=world, idx=idx))
        if hasattr(base, "scales"):
            base.scales = _split2_and_chunk(base.scales, dim=0, world=world, idx=idx)
        if base.bias is not None:
            base.bias = _set_param(_split2_and_chunk(base.bias, dim=0, world=world, idx=idx))
        base.out_features = base.weight.shape[0]

        if _has_lora_BT(w13):
            # BT: [rank, out] → out-dim at dim=1
            new_BT = _split2_and_chunk(w13.lora_BT, dim=1, world=world, idx=idx).contiguous()
            _assign_buffer(w13, "lora_BT", new_BT)
        else:
            # B: [out, rank] → out-dim at dim=0
            B = w13.lora_B
            B.weight = _set_param(_split2_and_chunk(B.weight, dim=0, world=world, idx=idx))
            if B.bias is not None:
                B.bias = _set_param(_split2_and_chunk(B.bias, dim=0, world=world, idx=idx))
            B.out_features = B.weight.shape[0]
    else:
        # Plain nn.Linear
        w = w13.weight
        a, b = torch.split(w, [w.shape[0] // 2, w.shape[0] // 2], dim=0)
        a = torch.chunk(a, world, dim=0)[idx]
        b = torch.chunk(b, world, dim=0)[idx]
        w13.weight = _set_param(torch.cat((a, b), dim=0))
        if hasattr(w13, "scales"):
            s1, s3 = torch.split(w13.scales, [w13.scales.shape[0] // 2, w13.scales.shape[0] // 2], dim=0)
            w13.scales = torch.cat((torch.chunk(s1, world, dim=0)[idx],
                                    torch.chunk(s3, world, dim=0)[idx]), dim=0)
        if w13.bias is not None:
            b1, b3 = torch.split(w13.bias, [w13.bias.shape[0] // 2, w13.bias.shape[0] // 2], dim=0)
            w13.bias = _set_param(torch.cat((torch.chunk(b1, world, dim=0)[idx],
                                             torch.chunk(b3, world, dim=0)[idx]), dim=0))
        w13.out_features = w13.weight.shape[0]

    # w2: row-parallel (LoRA-A split / LoRA-B/BT replicate)
    w2 = mlp.w2
    if _is_lora_linear(w2):
        start, end = _row_slice_inputs(w2.base_layer, rank_group)
        _lora_rowwise(w2, start, end)
    else:
        start, end = _row_slice_inputs(w2, rank_group)
        _linear_rowwise(w2, start, end)

    mlp.process_group = process_group


# =============================================================================
# Attention (wqkv + wo)
# =============================================================================

def _apply_tp_attn(attn, rank_group: Sequence[int], config, process_group) -> None:
    """Apply tensor parallelism to attention module.
    
    Handles both plain Linear and LoRA layers automatically.
    
    Args:
        attn: Attention module
        rank_group: Rank group
        config: Model configuration
        process_group: Process group for all-reduce
    """
    kv_h_start, kv_h_end = _kv_slice_for_rank(attn.n_local_heads, rank_group)
    num_group = attn.n_head // attn.n_local_heads

    q_start = kv_h_start * num_group * attn.head_dim
    q_end = kv_h_end * num_group * attn.head_dim
    kv_start_elem = kv_h_start * attn.head_dim
    kv_end_elem = kv_h_end * attn.head_dim

    kv_size = attn.n_local_heads * attn.head_dim
    qkv_splits = [attn.dim, kv_size, kv_size]

    # wqkv → column-parallel Q/K/V slicing (LoRA-aware)
    wqkv = attn.wqkv
    if _is_lora_linear(wqkv):
        _lora_qkv_colwise(wqkv, qkv_splits, (q_start, q_end), (kv_start_elem, kv_end_elem))
    else:
        w = wqkv
        w.weight = _set_param(_qkv_split_slice_cat(w.weight, 0, qkv_splits, (q_start, q_end), (kv_start_elem, kv_end_elem)))
        if hasattr(w, "scales"):
            w.scales = _qkv_split_slice_cat(w.scales, 0, qkv_splits, (q_start, q_end), (kv_start_elem, kv_end_elem))
        if w.bias is not None:
            w.bias = _set_param(_qkv_split_slice_cat(w.bias, 0, qkv_splits, (q_start, q_end), (kv_start_elem, kv_end_elem)))
        w.out_features = w.weight.shape[0]

    # wo → row-parallel (LoRA-A split / LoRA-B replicate)
    wo = attn.wo
    if _is_lora_linear(wo):
        start, end = _row_slice_inputs(wo.base_layer, rank_group)
        _lora_rowwise(wo, start, end)
    else:
        start, end = _row_slice_inputs(wo, rank_group)
        _linear_rowwise(wo, start, end)

    # Update attention config
    attn.n_head = config.n_head
    attn.dim = config.dim
    attn.head_dim = attn.dim // attn.n_head
    attn.n_local_heads = config.n_local_heads
    attn.process_group = process_group


# =============================================================================
# Transformer-Level Configuration
# =============================================================================

def _apply_tp_transformer_config(model, rank_group: Sequence[int], process_group) -> None:
    """Apply TP configuration to transformer model.
    
    Updates config and shards output projection.
    
    Args:
        model: Transformer model
        rank_group: Rank group
        process_group: Process group
    """
    num_heads = model.config.n_head
    num_kv_heads = model.config.n_local_heads
    num_group = num_heads // num_kv_heads

    kv_s, kv_e = _kv_slice_for_rank(num_kv_heads, rank_group)
    local_kv = kv_e - kv_s
    local_num_heads = local_kv * num_group
    local_dim = model.config.head_dim * local_num_heads

    # Update config for this rank
    model.config.n_head = local_num_heads
    model.config.dim = local_dim
    model.config.n_local_heads = local_kv

    # Output projection → column-parallel (LoRA-aware)
    out = model.output
    if _is_lora_linear(out):
        start, end = _col_slice_outputs(out.base_layer, rank_group)
        _lora_colwise(out, start, end)
    else:
        start, end = _col_slice_outputs(out, rank_group)
        _linear_colwise(out, start, end)
    
    # Track vocabulary range for this rank
    model.vocab_start = int(start)
    model.vocab_end = int(end)

    # Set distributed training attributes
    model.process_group = process_group
    model.world_size = dist.get_world_size(process_group)
    model.rank = dist.get_rank(process_group)


# =============================================================================
# Public API
# =============================================================================

def init_dist(draft_ranks: Optional[Sequence[int]] = None):
    """Initialize NCCL process groups for distributed training.
    
    Args:
        draft_ranks: Optional list of ranks for draft model group
        
    Returns:
        If draft_ranks is None: (global_rank, global_group)
        Otherwise: (global_rank, global_group, draft_group)
    """
    _ensure_cuda_device_set()
    dist.init_process_group(
        backend="nccl",
        rank=_get_global_rank(),
        world_size=_get_world_size(),
        device_id=torch.device(f"cuda:{_get_global_rank()}"),
    )
    global_group = dist.group.WORLD
    
    if draft_ranks is not None:
        draft_group = dist.new_group(draft_ranks)
        return _get_global_rank(), global_group, draft_group
    
    return _get_global_rank(), global_group


def apply_tp(model, rank_group: Sequence[int], group) -> None:
    """Apply tensor parallelism to transformer model.
    
    Automatically detects LoRA layers and applies appropriate sharding rules:
    - Regular Linear: Standard TP sharding
    - LoRA Linear: LoRA-aware sharding (base + adapter)
    
    Args:
        model: Transformer model
        rank_group: Sequence of ranks in this TP group
        group: Process group for all-reduce
        
    Raises:
        ValueError: If heads not divisible by world size
    """
    if not _validate_tp_size(model.config.n_local_heads, model.config.n_head):
        raise ValueError(
            f"num_kv_heads {model.config.n_local_heads} and num_heads {model.config.n_head} "
            f"must be divisible by world_size {_get_world_size()}"
        )
    
    # Apply TP to transformer config and output head
    _apply_tp_transformer_config(model, rank_group, group)
    
    # Apply TP to each layer
    for block in model.layers:
        _apply_tp_ffn(block.feed_forward, rank_group, group)
        _apply_tp_attn(block.attention, rank_group, model.config, group)


def apply_tp_eagle(model, rank_group: Sequence[int], process_group) -> None:
    """Apply tensor parallelism to EAGLE model.
    
    Similar to apply_tp but for EAGLE module structure.
    
    Args:
        model: EAGLE model
        rank_group: Sequence of ranks in this TP group
        process_group: Process group for all-reduce
        
    Raises:
        ValueError: If heads not divisible by world size
    """
    if not _validate_tp_size(model.config.n_local_heads, model.config.n_head):
        raise ValueError(
            f"num_kv_heads {model.config.n_local_heads} and num_heads {model.config.n_head} "
            f"must be divisible by world_size {_get_world_size()}"
        )
    
    num_heads = model.config.n_head
    num_kv_heads = model.config.n_local_heads
    num_group = num_heads // num_kv_heads

    kv_s, kv_e = _kv_slice_for_rank(num_kv_heads, rank_group)
    local_kv = kv_e - kv_s
    local_num_heads = local_kv * num_group
    local_dim = model.config.dim * local_kv // num_kv_heads

    # Update config for this rank
    model.config.n_head = local_num_heads
    model.config.dim = local_dim
    model.config.n_local_heads = local_kv

    # Output colwise (LoRA-aware)
    out = model.output
    if _is_lora_linear(out):
        start, end = _col_slice_outputs(out.base_layer, rank_group)
        _lora_colwise(out, start, end)
    else:
        start, end = _col_slice_outputs(out, rank_group)
        _linear_colwise(out, start, end)

    # Apply TP to EAGLE's FFN and attention
    _apply_tp_ffn(model.feed_forward, rank_group, process_group)
    _apply_tp_attn(model.attn, rank_group, model.config, process_group)

    # Track vocabulary range
    model.vocab_start = int(start)
    model.vocab_end = int(end)

    # Set distributed training attributes
    model.process_group = process_group
    model.world_size = dist.get_world_size(process_group)
    model.rank = dist.get_rank(process_group)
