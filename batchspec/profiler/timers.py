"""Timer contexts and global profiler access."""

import time
import torch


# ============================================================================
# No-op Context Manager
# ============================================================================

class _NoopCtx:
    """No-op context manager for disabled profiling."""
    __slots__ = ()
    
    def __enter__(self):
        return None
    
    def __exit__(self, exc_type, exc, tb):
        return False


_NOOP = _NoopCtx()


# ============================================================================
# Null Object Pattern for Profiler
# ============================================================================

class NullProfiler:
    """Null object profiler that does nothing.
    
    All methods are no-ops so calling code doesn't need to check for None.
    """
    __slots__ = ()
    
    disabled = True
    rank = 0
    
    class _NullConfig:
        model_profiling = False
        engine_profiling = False
    
    cfg = _NullConfig()
    
    def begin_run(self, **kwargs) -> None:
        """No-op."""
        pass
    
    def end_run(self) -> None:
        """No-op."""
        pass
    
    def step_timing_ctx(self):
        """Return no-op context manager."""
        return _NOOP
    
    def set_step_seq_len(self, *args, **kwargs) -> None:
        """No-op."""
        pass
    
    def set_step_tokens(self, *args, **kwargs) -> None:
        """No-op."""
        pass
    
    def __bool__(self) -> bool:
        """Always False so 'if profiler' checks work."""
        return False


_NULL_PROFILER = NullProfiler()


# ============================================================================
# Global Active Profiler
# ============================================================================

_ACTIVE_PROFILER: "Profiler" = _NULL_PROFILER


def register_active_profiler(prof: "Profiler") -> None:
    """Register an active profiler so helper timers can find it without plumbing."""
    global _ACTIVE_PROFILER
    _ACTIVE_PROFILER = prof


def release_active_profiler() -> None:
    """Release the active profiler and set it to NullProfiler."""
    global _ACTIVE_PROFILER
    _ACTIVE_PROFILER = _NULL_PROFILER


def get_active_profiler() -> "Profiler":
    """Get the currently active profiler."""
    return _ACTIVE_PROFILER

# ============================================================================
# Timer Contexts
# ============================================================================

class _CudaTimerCtx:
    """CUDA event-based timer context."""
    __slots__ = ("prof", "bucket", "s", "e")
    
    def __init__(self, prof: "Profiler", bucket: str):
        self.prof = prof
        self.bucket = bucket
        self.s = torch.cuda.Event(enable_timing=True)
        self.e = torch.cuda.Event(enable_timing=True)
    
    def __enter__(self):
        self.s.record()
        return None
    
    def __exit__(self, exc_type, exc, tb):
        self.e.record()
        self.prof._iter_events.append(("cuda", self.s, self.e, self.bucket))
        return False


class _CpuTimerCtx:
    """CPU wall-clock timer context."""
    __slots__ = ("prof", "bucket", "t0")
    
    def __init__(self, prof: "Profiler", bucket: str):
        self.prof = prof
        self.bucket = bucket
        self.t0 = 0.0
    
    def __enter__(self):
        if self.prof.cfg.strict_sync:
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return None
    
    def __exit__(self, exc_type, exc, tb):
        if self.prof.cfg.strict_sync:
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - self.t0) * 1e3
        self.prof._iter_events.append(("cpu", dt_ms, None, self.bucket))
        return False


# ============================================================================
# Timer Factory Functions
# ============================================================================

def _maybe_cuda_timer(bucket: str, need_model: bool = True):
    """
    Return an active CUDA timer context for the bucket, or a no-op.
    
    Args:
        bucket: Bucket name
        need_model: True -> gated by cfg.model_profiling
                   False -> gated by cfg.engine_profiling
                   None -> ungated (rare)
    """
    prof = get_active_profiler()
    if (prof is None or prof.disabled or 
        not prof._active_measure):
        return _NOOP
    
    if need_model is True and not prof.cfg.model_profiling:
        return _NOOP
    elif need_model is False and not prof.cfg.engine_profiling:
        return _NOOP
    
    if bucket is None:
        return _NOOP
    
    return _CudaTimerCtx(prof, bucket)


def _maybe_cpu_timer(bucket: str, need_engine: bool = True):
    """Return a CPU timer context or no-op."""
    prof = get_active_profiler()
    if (prof is None or prof.disabled or 
        not prof._active_measure):
        return _NOOP
    
    if need_engine and not prof.cfg.engine_profiling:
        return _NOOP
    
    if bucket is None:
        return _NOOP
    
    return _CpuTimerCtx(prof, bucket)


# ============================================================================
# Public Timer Functions (for model/engine code)
# ============================================================================

def attention_compute_timer():
    """Timer for attention computation."""
    return _maybe_cuda_timer("attn.compute", need_model=True)


def rope_compute_timer():
    """Timer for RoPE computation."""
    return _maybe_cuda_timer("attn.rope", need_model=True)


def cuda_bucket_timer(bucket: str):
    """Generic CUDA event bucket timer."""
    return _maybe_cuda_timer(bucket, need_model=True)


def cpu_bucket_timer(bucket: str):
    """Generic CPU wall-clock timer; gated by cfg.engine_profiling."""
    return _maybe_cpu_timer(bucket, need_engine=True)

