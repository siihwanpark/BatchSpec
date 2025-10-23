# Profiler Module

The `profiler` package provides a minimal‑overhead, CUDA‑event–based profiler tailored for speculative decoding and large‑batch LLM inference. It measures per‑step latency, accepted‑token throughput (tok/s), and (optionally) model/engine bucket breakdowns. Results are aggregated per run and across runs, and saved as JSON and Markdown reports.

> This README focuses on structure, responsibilities, and usage of the `profiler` package. General contribution/contact information should live in the repository’s top‑level README.

---

## Directory Structure

```
profiler/
├── __init__.py   # Public exports (Profiler, ProfilerConfig, timers, registry helpers)
├── config.py     # ProfilerConfig and canonical bucket orders
├── core.py       # Profiler class: run lifecycle, aggregation, and report writers
├── hooks.py      # Attachment points for model/engine instrumentation
├── timers.py     # Timer contexts (CUDA/CPU) and global active-profiler registry
└── utils.py      # Stats helpers, bucket utils, rank/world, run-name generation
```

### Public API (via `profiler/__init__.py`)

- `Profiler`, `ProfilerConfig`
- `generate_run_name`
- `get_active_profiler`, `register_active_profiler`, `release_active_profiler`
- `attention_compute_timer`, `rope_compute_timer`, `cuda_bucket_timer`, `cpu_bucket_timer`

---

## Key Concepts

### Runs, Steps, and Warmup
- A **run** groups multiple **decode steps**. Use `begin_run(bsz=..., label=...)` to start and `end_run()` to finish.
- Warmup runs (controlled by `cfg.warmup_runs`) are **skipped** from aggregation.
- Wrap each decode step with `with profiler.step_timing_ctx(): ...` to record per‑step elapsed time. Inside the step, you may nest bucket timers.

### Tokens and Sequence Length
- Record the number of **accepted tokens** for the step via `set_step_tokens(int)`.
- Record the **KV‑cache length** at the start of the step via `set_step_seq_len(x)`, where `x` can be an `int`, list/array, or tensor. Reduction across the batch is controlled by `cfg.kv_len_reduce` (`"mean"`, `"max"`, `"p50"`, `"p90"`, `"sum"`).

### Buckets (Model / Engine)
- Timers are labeled by **bucket** names. The profiler canonicalizes and aggregates these by **domain** (`model`/`engine`) and compacts unknown keys into `others` while preserving order.
- Canonical bucket orders are defined in `config.py`:
  - `MODEL_BUCKET_ORDER` (e.g., `embed`, `attn.qkv_proj`, `attn.qkv_proj.lora`, `attn.qk_norm`, `attn.compute`, …)
  - `ENGINE_BUCKET_ORDER` (e.g., `prefill`, `decode`, `sampler`, `postproc`, communication buckets, KV ops, …)

### Decode‑Length Bins
- Steps are also aggregated by **KV length** bins. `cfg.kv_bins` holds the bin edges (default: `[0, 512, 1024, 2048, 4096, 8192, 16384]`); the final bin is open‑ended.

---

## Configuration (`ProfilerConfig`)

Default fields (see `config.py`):
- `output_dir: str = "profiler_out"` — base directory for reports
- `collect_on_rank0_only: bool = True` — only rank 0 collects/saves
- `strict_sync: bool = True` — synchronize CUDA at step boundaries
- `dist_barrier: bool = False` — optional `torch.distributed.barrier()` (outside timing)
- `model_profiling: bool = False` — enable model bucket breakdown
- `engine_profiling: bool = False` — enable engine bucket breakdown
- `num_total_runs: int = 10` — total runs (for reporting)
- `warmup_runs: int = 1` — initial runs excluded from aggregation
- `print_per_run: bool = True` — print per‑run summary to console
- `run_name: Optional[str] = None` — override output subdirectory name
- `kv_len_reduce: str = "mean"` — reduction for sequence length across batch
- `kv_bins: List[int] = [0, 512, 1024, 2048, 4096, 8192, 16384]` — length bin edges

`ProfilerConfig.from_args(args, prefix="prof_")` constructs a config from a namespace, accepting either **prefixed** (e.g., `prof_model_profiling`) or **unprefixed** fields. For convenience, `backend_profiling` maps to `engine_profiling` if present.

**Rank gating:** When `collect_on_rank0_only=True` and `rank != 0`, the profiler is disabled; calls are safe no‑ops and no files are written.

---

## Timers and Global Registry

### Active Profiler Registry
- `register_active_profiler(prof)` — set the current active profiler
- `get_active_profiler()` — fetch the active profiler (or a safe `NullProfiler` when disabled/not set)
- `release_active_profiler()` — clear the active profiler

### Timer Contexts
- `attention_compute_timer()` — CUDA timer for attention compute
- `rope_compute_timer()` — CUDA timer for RoPE compute
- `cuda_bucket_timer(bucket: str)` — generic CUDA timer (model domain)
- `cpu_bucket_timer(bucket: str)` — generic CPU wall‑clock timer (engine domain)

Timers are **gated** by configuration and run state (e.g., disabled ranks, warmup runs, and `model_profiling`/`engine_profiling` flags).

---

## Instrumentation Hooks (`hooks.py`)

### Model Hooks (`Profiler.attach_model(model, use_gated_lora=False)`)
- Wraps module forwards to attribute elapsed time to model buckets such as
  `attn.qkv_proj`, `attn.compute`, `attn.rope`, `mlp.gate_up_proj`, `mlp.down_proj`, norms, embeddings, etc.
- When `use_gated_lora=True`, LoRA‑specific buckets (e.g., `*.lora`) are recorded alongside their base ops.
- Also wraps selected `torch.distributed` collectives (`all_reduce`, `reduce_scatter_tensor`, `all_gather`, `broadcast`) into dedicated communication buckets.

### Engine Hooks (`Profiler.attach_engine(engine_obj)`)
- Attaches timers to engine‑level phases such as `prefill`, `decode`, `sampler`, `postproc`, KV cache operations, etc.
- Intended for integration with your inference engine’s public methods.

---

## Execution Flow

1. **Initialize**
   ```python
   from types import SimpleNamespace
   from profiler import Profiler

   args = SimpleNamespace(
       run_name="demo",                # avoid relying on auto run-name
       model_profiling=True,
       engine_profiling=True,
       warmup_runs=1,
       num_total_runs=3,
   )
   prof = Profiler(args)
   ```

2. **(Optional) Register globally** — to enable timer helpers without passing `prof` around:
   ```python
   from profiler import register_active_profiler
   register_active_profiler(prof)
   ```

3. **(Optional) Attach hooks** — if your model/engine supports it:
   ```python
   # prof.attach_model(model, use_gated_lora=True)
   # prof.attach_engine(engine)
   ```

4. **Start a run**:
   ```python
   prof.begin_run(bsz=batch_size, label="decode")
   ```

5. **Time each decode step**:
   ```python
   from profiler import attention_compute_timer, cuda_bucket_timer, cpu_bucket_timer

   for _ in range(num_steps):
       with prof.step_timing_ctx():
           # Example sub-regions (optional)
           with attention_compute_timer():
               pass  # attention forward
           with cuda_bucket_timer("attn.rope"):
               pass  # RoPE computation
           with cpu_bucket_timer("engine.sampling"):
               pass  # host-side sampling/postprocess

           prof.set_step_seq_len(kv_len)       # int / list / tensor
           prof.set_step_tokens(n_accepted)    # int
   ```

6. **Finish the run**:
   ```python
   prof.end_run()
   ```

7. **Persist results**:
   ```python
   prof.save_config()  # writes config + runner args
   prof.save_all()     # writes summary.json and report.md
   ```

8. **(Optional) Release registry**:
   ```python
   from profiler import release_active_profiler
   release_active_profiler()
   ```

---

## Outputs

### `summary.json`
Top‑level fields include:
- `meta` — rank/world, configuration, output directory, start time
- `stats` — totals and percentiles: steps, mean/p50/p90/p99 (ms), tok/s, tokens, time (ms)
- `buckets_model_avg_ms`, `buckets_engine_avg_ms` — global avg per step
- `runs` — per‑run `meta`, `stats`, bucket averages, and per‑run decode‑length buckets
- `runs_aggregate` — mean ± std across runs for key metrics and buckets
- `decode_length_buckets` — global stats per KV‑length bin

### `report.md`
A human‑readable Markdown report containing:
- **Global summary** table (latency percentiles, throughput, totals)
- **Model/Engine bucket breakdowns** (avg ms/step and share)
- **Per‑run overview** (count, tokens, mean latency, tok/s)
- **Across‑run mean ± std** tables
- **Decode‑length buckets** section (bin ranges, steps, latency stats, tok/s)

---

## Notes and Behavior

- When rank‑gated (`collect_on_rank0_only=True` and `rank!=0`), the profiler is disabled; calls become safe no‑ops and no files are written.
- When `strict_sync=True`, CUDA is synchronized at step boundaries (and for CPU timers) for accuracy.
- When `dist_barrier=True` and `torch.distributed` is initialized, a barrier is inserted **outside** timing at step boundaries.

---

## Import Path

Assuming this folder is used as a Python package named `profiler`, import as:
```python
from profiler import Profiler, ProfilerConfig
from profiler import (
    get_active_profiler, register_active_profiler, release_active_profiler,
    attention_compute_timer, rope_compute_timer, cuda_bucket_timer, cpu_bucket_timer,
)
```

If your project mounts it under a different namespace, adjust the import path accordingly.

---

## Minimal Example (End‑to‑End)

```python
from types import SimpleNamespace
from profiler import Profiler, register_active_profiler, release_active_profiler
from profiler import attention_compute_timer, cuda_bucket_timer, cpu_bucket_timer

# 1) Init
args = SimpleNamespace(run_name="demo", model_profiling=True, engine_profiling=True,
                       warmup_runs=1, num_total_runs=2)
prof = Profiler(args)
register_active_profiler(prof)

# 2) Run
prof.begin_run(bsz=32, label="decode")
for _ in range(10):
    with prof.step_timing_ctx():
        with attention_compute_timer():
            pass
        with cuda_bucket_timer("attn.rope"):
            pass
        with cpu_bucket_timer("engine.sampling"):
            pass
        prof.set_step_seq_len(4096)
        prof.set_step_tokens(1)
prof.end_run()

# 3) Save
prof.save_config()
prof.save_all()
release_active_profiler()
```

This produces `summary.json` and `report.md` under `profiler_out/<run_name>/` by default.
