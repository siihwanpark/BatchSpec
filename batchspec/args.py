"""Command-line arguments for BatchSpec runners.

This module provides dataclass-based argument parsing using HfArgumentParser.
Common arguments are shared across run_benchmark.py and run_e2e.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Literal
from types import SimpleNamespace

import torch
from transformers import HfArgumentParser


@dataclass
class CommonArguments:
    """Common arguments shared across all runners."""
    model_name: str = field(
        metadata={"help": "Model name (e.g., meta-llama/Meta-Llama-3.1-8B, Qwen3-8B)."}
    )
    checkpoint_path: str = field(
        metadata={"help": "Path to the model checkpoint."}
    )
    tokenizer_path: str = field(
        metadata={"help": "Path to the tokenizer."}
    )
    backend: Literal["standard", "eagle", "mtp"] = field(
        default="standard",
        metadata={"help": "Backend name (standard, eagle, mtp)."}
    )
    dataset: str = field(
        default="AIME2025",
        metadata={"help": "Dataset name (AIME2025, GSM8K, MATH-500, LiveMathBench, GPQA-Diamond, LiveCodeBench-lite, CodeForces)."}
    )
    
    batch_size: int = field(
        default=16,
        metadata={"help": "Batch size for inference."}
    )
    max_gen_len: int = field(
        default=128,
        metadata={"help": "Maximum number of tokens to generate."}
    )
    
    dtype: Literal["bfloat16", "float16", "float32"] = field(
        default="bfloat16",
        metadata={"help": "Data type for model execution."}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducibility."}
    )
    compile: bool = field(
        default=False,
        metadata={"help": "Enable torch.compile() for the model."}
    )
    
    printoutput: bool = field(
        default=False,
        metadata={"help": "Print the generated output of each sequence."}
    )
    


@dataclass
class BenchmarkArguments:
    """Arguments specific to benchmark runner."""
    
    prefix_len_list: List[int] = field(
        default_factory=lambda: [1024, 2048, 4096, 8192, 12288, 16384, 20480, 24576, 28672],
        metadata={"help": "List of input prompt sequence lengths."}
    )
    num_total_runs: int = field(
        default=11,
        metadata={"help": "Total number of runs for the benchmark."}
    )


@dataclass
class SamplingArguments:
    """Sampling configuration for generation."""
    
    temperature: float = field(
        default=0.6,
        metadata={"help": "Temperature for sampling. 0 means greedy decoding."}
    )
    top_p: float = field(
        default=0.95,
        metadata={"help": "Top-p (nucleus) sampling."}
    )
    top_k: int = field(
        default=20,
        metadata={"help": "Top-k sampling."}
    )
    force_budget: bool = field(
        default=False,
        metadata={"help": "Force the generation until the budget (max_gen_len) is exhausted."}
    )


@dataclass
class SpecDecArguments:
    """Speculative decoding arguments."""
    
    draft_length: int = field(
        default=4,
        metadata={"help": "Number of draft tokens to generate in one step."}
    )

@dataclass
class EAGLEArguments:
    """EAGLE-specific arguments."""
    eagle_name: str = field(
        metadata={"help": "Name of the EAGLE drafter model."}
    )
    eagle_checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the EAGLE drafter checkpoint weights (for EAGLE backend)."}
    )
    use_eagle_tp: bool = field(
        default=False,
        metadata={"help": "Use tensor parallelism for EAGLE drafter module."}
    )


@dataclass
class MTPArguments:
    """MTP-specific arguments."""
    
    lora_checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the LoRA checkpoint weights (for MTP backend)."}
    )


@dataclass
class LoRAArguments:
    """LoRA configuration arguments.
    
    These map directly to LoRAConfig dataclass.
    """
    
    lora_rank: int = field(
        default=16,
        metadata={"help": "Rank for the LoRA adapter."}
    )
    lora_alpha: float = field(
        default=32.0,
        metadata={"help": "Alpha for the LoRA adapter."}
    )
    lora_bias: bool = field(
        default=False,
        metadata={"help": "Enable bias for the LoRA adapter."}
    )
    lora_use_rslora: bool = field(
        default=False,
        metadata={"help": "Use RSLoRA for the LoRA adapter."}
    )


@dataclass
class ProfilerArguments:
    """Profiler configuration arguments.
    
    These map to ProfilerConfig dataclass with some additional control flags.
    """
    
    profiling: bool = field(
        default=False,
        metadata={"help": "Enable profiling."}
    )
    model_profiling: bool = field(
        default=False,
        metadata={"help": "Enable model/module profiling."}
    )
    engine_profiling: bool = field(
        default=False,
        metadata={"help": "Enable engine profiling."}
    )
    
    prof_output_dir: str = field(
        default="profiler_out",
        metadata={"help": "Output directory for profiler results."}
    )
    prof_run_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name for the profiler run (auto-generated if None)."}
    )
    prof_warmup_runs: int = field(
        default=1,
        metadata={"help": "Number of warmup runs for profiling."}
    )
    prof_strict_sync: bool = field(
        default=True,
        metadata={"help": "Enable strict CUDA synchronization at step boundaries."}
    )
    prof_dist_barrier: bool = field(
        default=False,
        metadata={"help": "Enable distributed barrier at step boundaries."}
    )
    prof_kv_bins: List[int] = field(
        default_factory=lambda: [0, 512, 1024, 2048, 4096, 8192, 16384],
        metadata={"help": "KV length bins for profiling."}
    )
    prof_kv_len_reduce: Literal["mean", "max", "p50", "p90", "sum"] = field(
        default="mean",
        metadata={"help": "How to reduce per-batch KV lengths to scalar for binning."}
    )


@dataclass
class DistributedArguments:
    """Distributed inference arguments."""
    
    rank_group: Optional[List[int]] = field(
        default=None,
        metadata={"help": "Tensor parallel group ranks for the target model."}
    )


# ============================================================================
# Parsing Utilities
# ============================================================================

def _merge_to_namespace(*objs) -> SimpleNamespace:
    """Merge multiple dataclass instances into a single namespace."""
    merged = {}
    for obj in objs:
        merged.update(vars(obj))
    return SimpleNamespace(**merged)


def _postprocess_args(args: SimpleNamespace) -> SimpleNamespace:
    """Postprocess parsed arguments (convert paths, dtypes, etc.)."""
    # Convert string paths to Path objects
    args.checkpoint_path = Path(args.checkpoint_path)
    args.tokenizer_path = Path(args.tokenizer_path)
    
    if args.lora_checkpoint_path:
        args.lora_checkpoint_path = Path(args.lora_checkpoint_path)
    
    # Convert dtype string to torch dtype
    if torch.cuda.is_available():
        args.device = 'cuda'
        dtype_map = {
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
            'float32': torch.float32,
        }
        args.dtype = dtype_map.get(args.dtype, torch.bfloat16)
    else:
        args.device = 'cpu'
        args.dtype = torch.bfloat16
    
    return args


def parse_args() -> SimpleNamespace:
    """Parse arguments for run.py"""
    parser = HfArgumentParser((
        CommonArguments,
        BenchmarkArguments,
        SamplingArguments,
        SpecDecArguments,
        EAGLEArguments,
        MTPArguments,
        LoRAArguments,
        ProfilerArguments,
        DistributedArguments,
    ))
    
    parsed = parser.parse_args_into_dataclasses()
    (common, benchmark, sampling, specdec, eagle, mtp, lora, profiler, distributed) = parsed
    
    merged = _merge_to_namespace(common, benchmark, sampling, specdec, eagle, mtp, lora, profiler, distributed)
    merged = _postprocess_args(merged)
    
    return merged
