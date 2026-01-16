"""Utility functions for profiler - statistics, formatting, etc."""

import math
import os
from typing import Any, Dict, List

import torch.distributed as dist


# ============================================================================
# Statistics Helpers
# ============================================================================

def mean_std(vals: List[float]) -> tuple[float, float, int]:
    """Calculate mean and standard deviation."""
    vals = [float(v) for v in vals if v is not None]
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), 0
    if n == 1:
        return vals[0], 0.0, 1
    m = sum(vals) / n
    var = sum((x - m) * (x - m) for x in vals) / (n - 1)
    return m, math.sqrt(var), n


def percentile(sorted_vals: List[float], q: float) -> float:
    """Calculate percentile from sorted values."""
    if not sorted_vals:
        return float("nan")
    n = len(sorted_vals)
    k = max(0, min(n - 1, math.ceil(q / 100.0 * n) - 1))
    return sorted_vals[k] if n > 1 else sorted_vals[0]


# ============================================================================
# Distributed Helpers
# ============================================================================

def dist_ready() -> bool:
    """Check if distributed training is initialized."""
    return dist.is_available() and dist.is_initialized()


def rank_world() -> tuple[int, int]:
    """Get current rank and world size."""
    if dist_ready():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


# ============================================================================
# Bucket Utilities
# ============================================================================

def canon_bucket(bucket: str) -> tuple[str, str]:
    """
    Return (canonical_name, domain) where domain in {"model","engine"}.
    - Lowercases
    - engine.* -> strip "engine." prefix and mark domain="engine"
    - Normalize aliases (out_proj->o_proj, attn.qknorm->attn.qk_norm)
    """
    s = bucket.lower()
    if s.startswith("engine."):
        return s.split("engine.", 1)[1], "engine"
    # model domain normalization
    s = s.replace("attn.out_proj", "attn.o_proj")
    s = s.replace("attn.qknorm", "attn.qk_norm")
    return s, "model"


def order_and_compact(avg_dict: Dict[str, float], order: List[str]) -> Dict[str, float]:
    """
    Return ordered dict (insertion-ordered) with only keys in 'order' first.
    Other keys are kept in their original order.
    """
    ordered: Dict[str, float] = {}
    
    # First, add keys in specified order
    for k in order:
        if k in avg_dict:
            ordered[k] = avg_dict[k]
    
    # Then add remaining keys
    for k, v in avg_dict.items():
        if k not in ordered:
            ordered[k] = float(v)
    
    return ordered


# ============================================================================
# Formatting Helpers
# ============================================================================

def fmt(x: Any) -> Any:
    """Format number to 3 decimal places."""
    try:
        return f"{float(x):.3f}"
    except Exception:
        return x if x is not None else ""


def now_s() -> str:
    """Get current timestamp as ISO string."""
    from datetime import datetime
    return datetime.now().isoformat(timespec="seconds")


# ============================================================================
# Run Name Generation
# ============================================================================

def generate_run_name(args: Any) -> str:
    """Generate a descriptive run name from arguments."""
    model_name = args.model_name.split("/")[-1]
    run_name = f"{model_name}/{args.dataset}/{args.backend}/"
    run_name += f"bsz{args.batch_size}-gen{args.max_gen_len}-tp{len(args.rank_group)}"
    if args.backend == "mtp":
        run_name += f"-r{args.lora_rank}-k{args.draft_length}"
    
    if args.temperature > 0:
        run_name += f"-sampling"
    else:
        run_name += f"-greedy"
    
    return run_name

