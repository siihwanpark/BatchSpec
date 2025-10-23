"""Profiler for speculative decoding - clean modular implementation."""

from .config import ProfilerConfig, MODEL_BUCKET_ORDER, ENGINE_BUCKET_ORDER
from .core import Profiler
from .utils import generate_run_name
from .timers import (
    get_active_profiler,
    register_active_profiler,
    release_active_profiler,
    attention_compute_timer,
    rope_compute_timer,
    cuda_bucket_timer,
    cpu_bucket_timer,
)


__all__ = [
    "ProfilerConfig",
    "MODEL_BUCKET_ORDER",
    "ENGINE_BUCKET_ORDER",
    "Profiler",
    "generate_run_name",
    "get_active_profiler",
    "register_active_profiler",
    "release_active_profiler",
    "attention_compute_timer",
    "rope_compute_timer",
    "cuda_bucket_timer",
    "cpu_bucket_timer",
]

