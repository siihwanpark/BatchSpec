"""Model and engine instrumentation hooks."""

import time
from typing import Any
import torch
import torch.nn as nn
import torch.distributed as dist

from .timers import get_active_profiler, NullProfiler
from .utils import dist_ready

# ============================================================================
# Utils
# ============================================================================

def is_profiler_active(profiler: "Profiler") -> bool:
    """Check if the profiler is active."""
    if isinstance(profiler, NullProfiler): return False
    if profiler.disabled: return False
    if not profiler._is_measuring: return False
    return True


# ============================================================================
# Model Instrumentation
# ============================================================================

def attach_model_hooks(profiler: "Profiler", model: nn.Module, use_gated_lora: bool = False) -> None:
    """
    Attach profiling hooks to model modules.
    
    Args:
        profiler: Profiler instance
        model: Model to instrument
        use_gated_lora: Whether to split base/LoRA timing for gated LoRA modules
    """
    if profiler.disabled or not profiler.cfg.model_profiling:
        if profiler.rank == 0 and not profiler.disabled:
            print("[Profiler] model attach skipped (model_profiling=False).")
        return

    for name, m in model.named_modules():
        lname = name.lower()
        cls = m.__class__.__name__.lower()

        # Embedding / LM head
        if lname.endswith("tok_embeddings") or "embedding" in cls:
            _wrap_module_forward(m, "embed")
            continue
        if lname.endswith("output") or "output" in lname:
            _wrap_module_forward(m, "lm_head")
            continue

        # Attention projections (base vs LoRA split)
        if "wqkv" in lname and lname.endswith("wqkv"):
            if use_gated_lora:
                _set_linear_buckets(m, "attn.qkv_proj", "attn.qkv_proj.lora")
            else:
                _wrap_module_forward(m, "attn.qkv_proj")
            continue
        if "wo" in lname and lname.endswith("wo"):
            if use_gated_lora:
                _set_linear_buckets(m, "attn.out_proj", "attn.out_proj.lora")
            else:
                _wrap_module_forward(m, "attn.out_proj")
            continue

        # MLP projections (base vs LoRA)
        if "w13" in lname and lname.endswith("w13"):
            if use_gated_lora:
                _set_linear_buckets(m, "mlp.gate_up_proj", "mlp.gate_up_proj.lora")
            else:
                _wrap_module_forward(m, "mlp.gate_up_proj")
            continue
        if "w2" in lname and lname.endswith("w2"):
            if use_gated_lora:
                _set_linear_buckets(m, "mlp.down_proj", "mlp.down_proj.lora")
            else:
                _wrap_module_forward(m, "mlp.down_proj")
            continue

        # Norms
        if lname.endswith("q_norm") or lname.endswith("k_norm"):
            _wrap_module_forward(m, "attn.qk_norm")
            continue
        elif "sampler_head" not in lname and "rmsnorm" in cls:
            _wrap_module_forward(m, "norm")
            continue

        # Sampler
        if "sampler_head." in lname:
            if ".layers." in lname and lname.endswith(".linear"):
                try:
                    idx = int(lname.split(".layers.")[1].split(".")[0])
                    _wrap_module_forward(m, f"sampler_head.linear.{idx}")
                    continue
                except Exception:
                    pass
            if lname.endswith(".norm"):
                _wrap_module_forward(m, "sampler_head.norm")
                continue

    _patch_communication_ops()
    if profiler.rank == 0:
        print("[Profiler] model attached.")


def _set_linear_buckets(mod: nn.Module, base_bucket: str, lora_bucket: str) -> None:
    """Set bucket names for base and LoRA parts of a module."""
    try:
        setattr(mod, "_prof_base_bucket", base_bucket)
        setattr(mod, "_prof_lora_bucket", lora_bucket)
    except Exception:
        pass


def _wrap_module_forward(module: nn.Module, bucket: str) -> None:
    """Wrap module's forward method with timing."""
    # Guard duplicate wraps
    if getattr(module, "_prof_wrapped", False):
        return
    setattr(module, "_prof_wrapped", True)
    
    orig_fwd = module.forward
    
    def wrapped(*args, **kwargs):
        prof = get_active_profiler()
        if not is_profiler_active(prof) or not prof.cfg.model_profiling:
            return orig_fwd(*args, **kwargs)

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = orig_fwd(*args, **kwargs)
        e.record()
        prof._step_events.append(("cuda", s, e, bucket))
        return out
    
    module.forward = wrapped


def _patch_communication_ops() -> None:
    """Patch distributed communication ops with timing."""
    if not dist_ready():
        return
    
    def _wrap(name: str):
        if not hasattr(dist, name):
            return
        orig = getattr(dist, name)
        
        def wrapped(*args, **kwargs):
            prof = get_active_profiler()
            if not is_profiler_active(prof) or not prof.cfg.model_profiling:
                return orig(*args, **kwargs)
            
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            out = orig(*args, **kwargs)
            e.record()
            prof._step_events.append(("cuda", s, e, "communication"))
            return out
        
        setattr(dist, name, wrapped)
    
    for n in ("all_reduce", "reduce_scatter_tensor", "all_gather", "broadcast"):
        _wrap(n)


# ============================================================================
# Engine Instrumentation
# ============================================================================

def attach_engine_hooks(profiler: "Profiler", engine_obj: Any) -> None:
    """
    Attach profiling hooks to engine methods.
    
    Args:
        profiler: Profiler instance
        engine_obj: Engine object to instrument
    """
    if profiler.disabled or not profiler.cfg.engine_profiling:
        if profiler.rank == 0 and not profiler.disabled:
            print("[Profiler] engine attach skipped (engine_profiling=False).")
        return
    
    name_map = {
        # Standard
        "prefill": "prefill",
        "decode": "decode",
        "budget_forcing": "budget_forcing",

        # Standalone, EAGLE, MagicDec, MTP
        "draft": "draft",
        "evaluate_posterior": "evaluate_posterior",

        # MTP
        "draft_and_verify": "draft_and_verify",
        "sampler_draft": "sampler_draft",

        "interleave_mask_tokens": "interleave_mask_tokens",
        
        "collate_kv": "collate_accepted_kv_cache",
        "update_output": "update_output",
        
    }
    
    for method_name, short in name_map.items():
        if hasattr(engine_obj, method_name):
            _wrap_engine_method(engine_obj, method_name, f"engine.{short}")
    
    if profiler.rank == 0:
        print("[Profiler] engine attached.")


def _wrap_engine_method(obj: Any, method_name: str, bucket: str) -> None:
    """Wrap a engine method with CPU timing."""
    orig = getattr(obj, method_name)
    
    def wrapped(*args, **kwargs):
        prof = get_active_profiler()
        if not is_profiler_active(prof) or not prof.cfg.engine_profiling:
            return orig(*args, **kwargs)
        
        t0 = time.perf_counter()
        try:
            return orig(*args, **kwargs)
        finally:
            dt_ms = (time.perf_counter() - t0) * 1e3
            prof._step_events.append(("cpu", dt_ms, None, bucket))
    
    setattr(obj, method_name, wrapped)

