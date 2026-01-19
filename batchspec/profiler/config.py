"""Profiler configuration and constants."""

from dataclasses import dataclass
from typing import Optional
from types import SimpleNamespace


# ============================================================================
# Bucket Ordering
# ============================================================================

MODEL_BUCKET_ORDER = [
    "embed",
    "attn.qkv_proj",
    "attn.qkv_proj.lora",
    "attn.qk_norm",
    "attn.rope",
    "attn.compute",
    "attn.o_proj",
    "attn.o_proj.lora",
    "mlp.gate_up_proj",
    "mlp.gate_up_proj.lora",
    "mlp.silu_and_mul",
    "mlp.down_proj",
    "mlp.down_proj.lora",
    "norm",
    "lm_head",
    "sampler.linear.0",
    "sampler.linear.1",
    "sampler.norm",
    "communication",
]

ENGINE_BUCKET_ORDER = [
    "pre_decode",
    "decode",
    "pre_draft",
    "draft",
    "pre_draft_and_verify",
    "draft_and_verify",
    "sampler_draft",
    "interleave_mask_tokens",
    "evaluate_posterior",
    "collate_kv",
    "insert_kv",
    "delete_kv",
    "update_output",
]


# ============================================================================
# Profiler Configuration
# ============================================================================

@dataclass
class ProfilerConfig:
    """Configuration for the profiler."""
    
    output_dir: str = "profiler_out"
    collect_on_rank0_only: bool = True   # collect only on rank0
    strict_sync: bool = True             # cuda sync at step boundaries (accuracy↑)
    dist_barrier: bool = False           # barrier at step boundaries (outside timing)
    model_profiling: bool = False        # model/module breakdown
    engine_profiling: bool = False       # engine call breakdown
    num_total_runs: int = 10             # total number of runs (only for reporting)
    print_per_run: bool = True           # print run summary
    run_name: Optional[str] = None       # folder name override

    @classmethod
    def from_args(cls, args: SimpleNamespace, prefix: str = "prof_") -> "ProfilerConfig":
        """Create ProfilerConfig from args with automatic prefix stripping.
        
        Args:
            args: Parsed arguments namespace
            prefix: Prefix to strip from arg names (default: "prof_")
        """
        # Get all fields from dataclass
        from dataclasses import fields as dataclass_fields
        
        kwargs = {}
        for field in dataclass_fields(cls):
            # Try with prefix first
            prefixed_name = f"{prefix}{field.name}"
            if hasattr(args, prefixed_name):
                kwargs[field.name] = getattr(args, prefixed_name)
            # Special mappings
            elif field.name == "engine_profiling" and hasattr(args, "backend_profiling"):
                kwargs[field.name] = args.backend_profiling
            # Try without prefix
            elif hasattr(args, field.name):
                kwargs[field.name] = getattr(args, field.name)
        
        return cls(**kwargs)