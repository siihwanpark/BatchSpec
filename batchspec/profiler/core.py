"""Main Profiler class and output generation."""

import json
import math
import os
from dataclasses import asdict
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist

from .config import ProfilerConfig, MODEL_BUCKET_ORDER, ENGINE_BUCKET_ORDER
from .hooks import attach_model_hooks, attach_engine_hooks
from .timers import register_active_profiler
from .utils import (
    mean_std, percentile, rank_world, canon_bucket,
    order_and_compact, fmt, now_s, dist_ready, generate_run_name
)


# ============================================================================
# Main Profiler Class
# ============================================================================

class Profiler:
    """
    Minimal-overhead CUDA-event profiler for speculative decoding.
    - Per-step latency and accepted-token throughput (tok/s).
    - Optional model/LoRA/engine/comms breakdown.
    - Rank0 aggregation by default; TP-aware.
    - Global aggregation across multiple calls.
    - Clean JSON/CSV/Markdown outputs.
    """

    def __init__(self, runner_args: SimpleNamespace):
        self.cfg = ProfilerConfig.from_args(runner_args)
        self.rank, self.world = rank_world()
        self.disabled = (self.cfg.collect_on_rank0_only and self.rank != 0)
        self.out_dir = os.path.join(
            self.cfg.output_dir,
            self.cfg.run_name or generate_run_name(runner_args)
        )
        if not self.disabled:
            os.makedirs(self.out_dir, exist_ok=True)
            print(f"[Profiler] Output dir: {self.out_dir}")
        
        self.runner_args = vars(runner_args) if hasattr(runner_args, '__dict__') else runner_args

        # Per-run State
        self._active_measure: bool = False
        self._iter_idx: int = 0
        self._current_meta: Dict[str, Any] = {}
        self._iters_elapsed_ms: List[float] = []
        self._iter_tokens: List[int] = []  # accepted tokens per step
        self._iter_kv_lens: List[int] = []  # KV length at start of step
        self._iter_events: List[tuple[str, Any, Any, str]] = []  # (etype,start,end,bucket)
        self._iters_model_sums: List[Dict[str, float]] = []
        self._iters_engine_sums: List[Dict[str, float]] = []
        self._pending_step_tokens: int = 0
        self._pending_step_kvlen: Optional[int] = None

        # Per-run Results
        self._runs: List[Dict[str, Any]] = []

        # Global Aggregation (across all runs)
        self._g_lat_ms: List[float] = []
        self._g_tokens: int = 0
        self._g_steps: int = 0
        self._g_model_bucket_sum_ms: Dict[str, float] = {}
        self._g_engine_bucket_sum_ms: Dict[str, float] = {}
        
        # Global decode-length bucket aggregation
        self._g_len_bins_meta: List[tuple[Optional[int], Optional[int], str]] = self._build_len_bins()
        self._g_len_bins_data: Dict[str, Dict[str, Any]] = {
            meta[2]: {"lat_ms": [], "tokens": 0, "time_ms": 0.0, "steps": 0}
            for meta in self._g_len_bins_meta
        }


    # ========================================================================
    # Attach / instrument
    # ========================================================================

    def attach_model(self, model: Any, use_gated_lora: bool = False) -> None:
        """Attach profiling hooks to model modules."""
        attach_model_hooks(self, model, use_gated_lora)

    def attach_engine(self, engine_obj: Any) -> None:
        """Attach profiling hooks to engine methods."""
        attach_engine_hooks(self, engine_obj)

    # ========================================================================
    # Length-bucket helpers
    # ========================================================================

    def _reduce_kv_len(self, x: Union[int, List[int], torch.Tensor]) -> int:
        """
        Reduce a batch 1D tensor/list of lengths to a single scalar
        according to cfg.kv_len_reduce.
        """
        if isinstance(x, int):
            return int(x)
        if isinstance(x, (list, tuple)):
            t = torch.as_tensor(x)
        elif torch.is_tensor(x):
            t = x
        else:
            try:
                return int(x)
            except Exception:
                return 0
        
        if t.numel() == 0:
            return 0
        if t.is_cuda:
            t = t.detach().to("cpu", non_blocking=True)
        t = t.to(torch.float32)

        mode = (self.cfg.kv_len_reduce or "mean").lower()
        if mode == "mean":
            v = t.mean()
        elif mode == "max":
            v = t.max()
        elif mode in ("p50", "median"):
            v = torch.quantile(t, 0.5)
        elif mode == "p90":
            v = torch.quantile(t, 0.9)
        elif mode == "sum":
            v = t.sum()
        else:
            v = t.mean()
        return int(v.item())

    def _build_len_bins(self) -> List[tuple[Optional[int], Optional[int], str]]:
        """Build length bucket metadata."""
        edges = sorted(int(x) for x in self.cfg.kv_bins)
        metas: List[tuple[Optional[int], Optional[int], str]] = []
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            metas.append((lo, hi, f"[{lo},{hi})"))
        # last bin: [last_edge, +inf)
        last = edges[-1] if edges else 0
        metas.append((last, None, f"[{last},∞)"))
        return metas

    def _bin_key_for_len(self, L: int) -> str:
        """Find which bin a length L belongs to."""
        for lo, hi, key in self._g_len_bins_meta:
            if hi is None:
                if L >= int(lo or 0):
                    return key
            else:
                if int(lo or 0) <= L < int(hi):
                    return key
        return "unbinned"

    # ========================================================================
    # Run lifecycle
    # ========================================================================

    def begin_run(self, *, bsz: int, label: str = "decode", prefix_len: int) -> None:
        """Begin a profiling run."""
        if self.disabled:
            return
        
        self._current_meta = {
            "label": label,
            "bsz": int(bsz),
            "prefix_len": (int(prefix_len) if prefix_len is not None else None),
            "rank": self.rank,
            "world": self.world,
            "started_at": now_s(),
            "cfg": asdict(self.cfg),
        }

        self._iters_elapsed_ms.clear()
        self._iter_tokens.clear()
        self._iter_events.clear()
        self._iters_model_sums.clear()
        self._iters_engine_sums.clear()
        self._iter_idx = 0
        self._iter_kv_lens.clear()
        
    def step_timing_ctx(self):
        """Context manager for timing a single step."""
        class _StepCtx:
            __slots__ = ("prof", "s", "e")
            
            def __init__(self, prof: "Profiler"):
                self.prof = prof
                self.s = torch.cuda.Event(enable_timing=True)
                self.e = torch.cuda.Event(enable_timing=True)
            
            def __enter__(self):
                if self.prof.cfg.strict_sync:
                    torch.cuda.synchronize()
                if self.prof.cfg.dist_barrier and dist_ready():
                    dist.barrier()  # outside timing
                
                self.prof._active_measure = True
                self.prof._iter_events.clear()
                self.prof._iter_idx += 1
                self.s.record()
                return None
            
            def __exit__(self, exc_type, exc, tb):
                self.e.record()
                self.prof._active_measure = False
                
                if self.prof.cfg.strict_sync:
                    torch.cuda.synchronize()
                self.e.synchronize()
                if self.prof.cfg.dist_barrier and dist_ready():
                    dist.barrier()  # outside timing

                step_ms = float(self.s.elapsed_time(self.e))
                self.prof._iters_elapsed_ms.append(step_ms)
                tok = getattr(self.prof, "_pending_step_tokens", 0)
                self.prof._iter_tokens.append(int(tok))
                self.prof._pending_step_tokens = 0

                # record KV length
                kvlen = getattr(self.prof, "_pending_step_kvlen", None)
                self.prof._iter_kv_lens.append(int(kvlen if kvlen is not None else 0))
                self.prof._pending_step_kvlen = None

                # Process events
                if ((self.prof.cfg.model_profiling or self.prof.cfg.engine_profiling) 
                    and self.prof._iter_events):
                    lm: Dict[str, float] = {}
                    lb: Dict[str, float] = {}
                    for etype, a, b, bucket in self.prof._iter_events:
                        dt = float(a.elapsed_time(b)) if etype == "cuda" else float(a)
                        key, domain = canon_bucket(bucket)
                        if domain == "engine":
                            lb[key] = lb.get(key, 0.0) + dt
                        else:
                            lm[key] = lm.get(key, 0.0) + dt
                    if lm:
                        self.prof._iters_model_sums.append(lm)
                    if lb:
                        self.prof._iters_engine_sums.append(lb)

                self.prof._iter_events.clear()
                return False
        
        return _StepCtx(self)

    def set_step_tokens(self, n: int):
        """Record the number of tokens accepted in this step."""
        if self.disabled:
            return
        try:
            self._pending_step_tokens = int(n)
        except Exception:
            self._pending_step_tokens = 0

    def set_step_seq_len(self, kv_len: Union[int, List[int], torch.Tensor]):
        """
        Record the KV-cache length (sequence length) *at the start of the step*.
        Call this once per decode step before the step finishes.
        """
        if self.disabled:
            return
        try:
            self._pending_step_kvlen = int(self._reduce_kv_len(kv_len))
        except Exception:
            self._pending_step_kvlen = 0

    def end_run(self) -> None:
        """End the current profiling run and aggregate stats."""
        if self.disabled:
            return

        # Per-run stats
        vals = sorted(self._iters_elapsed_ms)
        n = len(vals)
        stats = {"count": n}
        mean_val = sum(vals) / n if n > 0 else 0.0
        secs_total = sum(self._iters_elapsed_ms) / 1000.0
        tok_total = int(sum(self._iter_tokens))
        tp = (tok_total / secs_total) if secs_total > 0 else float("nan")
        
        if n > 0:
            stats.update({
                "mean_ms": mean_val,
                "min_ms": vals[0],
                "max_ms": vals[-1],
                "p50_ms": percentile(vals, 50),
                "p90_ms": percentile(vals, 90),
                "p95_ms": percentile(vals, 95),
                "p99_ms": percentile(vals, 99),
                "throughput_tok_s": tp,
                "tokens_total": tok_total,
                "time_total_ms": sum(self._iters_elapsed_ms),
            })

        # Per-run buckets
        buckets_model_avg: Dict[str, float] = {}
        buckets_engine_avg: Dict[str, float] = {}

        if n > 0 and self._iters_model_sums:
            agg_m: Dict[str, float] = {}
            for d in self._iters_model_sums:
                for k, v in d.items():
                    agg_m[k] = agg_m.get(k, 0.0) + v
            avg_m = {k: (v / n) for k, v in agg_m.items()}
            buckets_model_avg = order_and_compact(avg_m, MODEL_BUCKET_ORDER)

        if n > 0 and self._iters_engine_sums:
            agg_b: Dict[str, float] = {}
            for d in self._iters_engine_sums:
                for k, v in d.items():
                    agg_b[k] = agg_b.get(k, 0.0) + v
            avg_b = {k: (v / n) for k, v in agg_b.items()}
            buckets_engine_avg = order_and_compact(avg_b, ENGINE_BUCKET_ORDER)

        # Per-run decode-length buckets
        len_bucket_stats: Dict[str, Dict[str, Any]] = {}
        if n > 0 and len(self._iter_kv_lens) == n:
            per_bin_lat: Dict[str, List[float]] = {meta[2]: [] for meta in self._g_len_bins_meta}
            per_bin_tok: Dict[str, int] = {meta[2]: 0 for meta in self._g_len_bins_meta}
            per_bin_time_ms: Dict[str, float] = {meta[2]: 0.0 for meta in self._g_len_bins_meta}
            per_bin_steps: Dict[str, int] = {meta[2]: 0 for meta in self._g_len_bins_meta}
            
            for step_ms, step_tok, L in zip(self._iters_elapsed_ms, self._iter_tokens, self._iter_kv_lens):
                key = self._bin_key_for_len(int(L))
                per_bin_lat[key].append(float(step_ms))
                per_bin_tok[key] += int(step_tok)
                per_bin_time_ms[key] += float(step_ms)
                per_bin_steps[key] += 1

            def _mk_stat(ls: List[float], tok: int, tms: float, steps: int) -> Dict[str, Any]:
                ls_sorted = sorted(ls)
                m = (sum(ls_sorted) / steps) if steps > 0 else float("nan")
                return {
                    "steps": steps,
                    "mean_ms": m,
                    "p50_ms": percentile(ls_sorted, 50),
                    "p90_ms": percentile(ls_sorted, 90),
                    "p95_ms": percentile(ls_sorted, 95),
                    "p99_ms": percentile(ls_sorted, 99),
                    "time_total_ms": tms,
                    "tokens_total": tok,
                    "throughput_tok_s": (tok / (tms / 1000.0)) if tms > 0 else float("nan"),
                }

            for lo, hi, key in self._g_len_bins_meta:
                len_bucket_stats[key] = _mk_stat(
                    per_bin_lat[key], per_bin_tok[key], per_bin_time_ms[key], per_bin_steps[key]
                )
                # Global accumulation
                gb = self._g_len_bins_data[key]
                gb["lat_ms"].extend(per_bin_lat[key])
                gb["tokens"] += per_bin_tok[key]
                gb["time_ms"] += per_bin_time_ms[key]
                gb["steps"] += per_bin_steps[key]

        pack = {
            "meta": dict(self._current_meta),
            "stats": stats,
            "buckets_model_avg_ms": buckets_model_avg,
            "buckets_engine_avg_ms": buckets_engine_avg,
            "decode_length_buckets": {
                "bins": [
                    {"key": key, "range": [lo, hi] if hi is not None else [lo, None]}
                    for lo, hi, key in self._g_len_bins_meta
                ],
                "stats": len_bucket_stats,
            }
        }

        self._runs.append(pack)
        if self.cfg.print_per_run:
            self._print_run_summary(pack)

        # Global accumulation
        self._g_lat_ms.extend(self._iters_elapsed_ms)
        self._g_tokens += tok_total
        self._g_steps += n
        if self._iters_model_sums:
            for d in self._iters_model_sums:
                for k, v in d.items():
                    self._g_model_bucket_sum_ms[k] = self._g_model_bucket_sum_ms.get(k, 0.0) + float(v)
        if self._iters_engine_sums:
            for d in self._iters_engine_sums:
                for k, v in d.items():
                    self._g_engine_bucket_sum_ms[k] = self._g_engine_bucket_sum_ms.get(k, 0.0) + float(v)

    # ========================================================================
    # Save outputs
    # ========================================================================

    def _group_runs_by_prefix(self):
        groups = defaultdict(list)
        for r in self._runs:
            pl = r.get("meta", {}).get("prefix_len", None)
            groups[pl].append(r)
        return groups

    def save_all(self) -> None:
        """Save all profiling results to JSON and Markdown."""
        if self.disabled:
            if self.rank == 0:
                print("[Profiler] Disabled (rank gated). Nothing to save.")
            return

        # Global summary
        g_vals = sorted(self._g_lat_ms)
        g_n = len(g_vals)
        g_mean = sum(g_vals) / g_n if g_n > 0 else 0.0
        g_secs_total = sum(self._g_lat_ms) / 1000.0
        g_tp = (self._g_tokens / g_secs_total) if g_secs_total > 0 else float("nan")

        # Global bucket averages
        g_buckets_model_avg: Dict[str, float] = {}
        g_buckets_engine_avg: Dict[str, float] = {}

        if g_n > 0 and self._g_model_bucket_sum_ms:
            avgm = {k: (v / g_n) for k, v in self._g_model_bucket_sum_ms.items()}
            g_buckets_model_avg = order_and_compact(avgm, MODEL_BUCKET_ORDER)

        if g_n > 0 and self._g_engine_bucket_sum_ms:
            avgb = {k: (v / g_n) for k, v in self._g_engine_bucket_sum_ms.items()}
            g_buckets_engine_avg = order_and_compact(avgb, ENGINE_BUCKET_ORDER)

        # Across-run mean±std
        run_lat_means = [r["stats"].get("mean_ms") for r in self._runs if "mean_ms" in r["stats"]]
        run_tp_means = [r["stats"].get("throughput_tok_s") for r in self._runs if "throughput_tok_s" in r["stats"]]

        lat_mean, lat_std, lat_n = mean_std(run_lat_means)
        tp_mean, tp_std, tp_n = mean_std(run_tp_means)

        # Buckets: collect per-run averages
        bucket_samples_model: Dict[str, List[float]] = {}
        bucket_samples_engine: Dict[str, List[float]] = {}

        for r in self._runs:
            bm = r.get("buckets_model_avg_ms", {}) or {}
            for k, v in bm.items():
                bucket_samples_model.setdefault(k, []).append(float(v))
            bb = r.get("buckets_engine_avg_ms", {}) or {}
            for k, v in bb.items():
                bucket_samples_engine.setdefault(k, []).append(float(v))

        def _stats_map(d: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
            out = {}
            for k, lst in d.items():
                m, s, n = mean_std(lst)
                out[k] = {"mean": m, "std": s, "n_runs": n}
            return out

        buckets_runs_stats_model = _stats_map(bucket_samples_model)
        buckets_runs_stats_engine = _stats_map(bucket_samples_engine)

        # Global decode-length buckets
        def _mk_global_len_stats() -> Dict[str, Any]:
            out = {}
            for lo, hi, key in self._g_len_bins_meta:
                d = self._g_len_bins_data[key]
                ls_sorted = sorted(d["lat_ms"])
                steps = int(d["steps"])
                tms = float(d["time_ms"])
                tok = int(d["tokens"])
                mean_ms = (sum(ls_sorted) / steps) if steps > 0 else float("nan")
                out[key] = {
                    "steps": steps,
                    "mean_ms": mean_ms,
                    "p50_ms": percentile(ls_sorted, 50),
                    "p90_ms": percentile(ls_sorted, 90),
                    "p95_ms": percentile(ls_sorted, 95),
                    "p99_ms": percentile(ls_sorted, 99),
                    "time_total_ms": tms,
                    "tokens_total": tok,
                    "throughput_tok_s": (tok / (tms / 1000.0)) if tms > 0 else float("nan"),
                    "range": [lo, hi] if hi is not None else [lo, None],
                }
            return out

        global_summary = {
            "meta": {
                "rank": self.rank,
                "world": self.world,
                "started_at": self._runs[0]["meta"]["started_at"] if self._runs else now_s(),
                "cfg": asdict(self.cfg),
                "output_dir": self.out_dir,
            },
            "stats": {
                "steps_total": g_n,
                "time_total_ms": sum(self._g_lat_ms),
                "tokens_total": self._g_tokens,
                "mean_ms": g_mean,
                "p50_ms": percentile(g_vals, 50),
                "p90_ms": percentile(g_vals, 90),
                "p95_ms": percentile(g_vals, 95),
                "p99_ms": percentile(g_vals, 99),
                "throughput_tok_s": g_tp,
            },
            "buckets_model_avg_ms": g_buckets_model_avg,
            "buckets_engine_avg_ms": g_buckets_engine_avg,
            "runs": self._runs,
            "runs_aggregate": {
                "latency_ms": {"mean": lat_mean, "std": lat_std, "n_runs": lat_n},
                "throughput_tok_s": {"mean": tp_mean, "std": tp_std, "n_runs": tp_n},
                "buckets_model_avg_ms": buckets_runs_stats_model,
                "buckets_engine_avg_ms": buckets_runs_stats_engine,
            },
            "decode_length_buckets": {
                "bins": [
                    {"key": key, "range": [lo, hi] if hi is not None else [lo, None]}
                    for lo, hi, key in self._g_len_bins_meta
                ],
                "stats": _mk_global_len_stats(),
            }
        }

        groups = self._group_runs_by_prefix()
        prefix_groups_summary = {}
        for prefix_len, runs in groups.items():
            run_lat_means = [rr["stats"].get("mean_ms") for rr in runs if "mean_ms" in rr["stats"]]
            run_tp_means  = [rr["stats"].get("throughput_tok_s") for rr in runs if "throughput_tok_s" in rr["stats"]]

            lat_mean, lat_std, lat_n = mean_std(run_lat_means)
            tp_mean, tp_std, tp_n = mean_std(run_tp_means)

            run_p95s = [rr["stats"].get("p95_ms") for rr in runs if "p95_ms" in rr["stats"]]
            p95_mean, p95_std, p95_n = mean_std(run_p95s)

            all_step_lat = []
            total_tokens = 0
            total_time_ms = 0.0
            total_steps = 0
            for rr in runs:
                s = rr.get("stats", {})
                total_tokens += int(s.get("tokens_total", 0) or 0)
                total_time_ms += float(s.get("time_total_ms", 0.0) or 0.0)
                total_steps += int(s.get("count", 0) or 0)

            throughput_tok_s = (total_tokens / (total_time_ms / 1000.0)) if total_time_ms > 0 else float("nan")

            prefix_groups_summary[str(prefix_len)] = {
                "prefix_len": prefix_len,
                "n_runs": len(runs),
                "runs": runs,
                "over_runs_unweighted": {
                    "latency_mean_ms": {"mean": lat_mean, "std": lat_std, "n_runs": lat_n},
                    "p95_ms": {"mean": p95_mean, "std": p95_std, "n_runs": p95_n},
                    "throughput_tok_s": {"mean": tp_mean, "std": tp_std, "n_runs": tp_n},
                },
                "over_runs_weighted": {
                    "steps_total": total_steps,
                    "tokens_total": total_tokens,
                    "time_total_ms": total_time_ms,
                    "throughput_tok_s": throughput_tok_s,
                }
            }
        global_summary["prefix_len_groups"] = prefix_groups_summary

        # Write JSON
        jpath = os.path.join(self.out_dir, "summary.json")
        with open(jpath, "w") as f:
            json.dump(global_summary, f, indent=2)

        # Write Markdown report
        mpath = os.path.join(self.out_dir, "report.md")
        self._write_markdown_report(mpath, global_summary)

        if self.rank == 0:
            print(f"[Profiler] Saved:\n  {jpath}\n  {mpath}")
            print(f"[Profiler] Global: steps={g_n} mean={fmt(g_mean)}ms tok/s={fmt(g_tp)}")

    def save_config(self, filename: str = "config.json", extra: Optional[Dict[str, Any]] = None) -> None:
        """Save configuration to JSON."""
        if self.disabled:
            return
        
        cfg = asdict(self.cfg)
        cfg["rank"] = self.rank
        cfg["world"] = self.world
        
        def _to_jsonable(v):
            if isinstance(v, (str, int, float, bool)) or v is None:
                return v
            if isinstance(v, (list, tuple)):
                return [_to_jsonable(x) for x in v]
            return str(v)
        
        if hasattr(self.runner_args, "__dict__"):
            args_dict = {
                k: _to_jsonable(v) for k, v in vars(self.runner_args).items()
                if not callable(v) and not k.startswith("_")
            }
        elif isinstance(self.runner_args, dict):
            args_dict = {
                k: _to_jsonable(v) for k, v in self.runner_args.items()
                if not callable(v) and not k.startswith("_")
            }
        else:
            args_dict = {"__raw__": _to_jsonable(self.runner_args)}
        
        payload = {"runner_args": args_dict, "profiler_cfg": cfg}
        if extra:
            payload["extra"] = {k: _to_jsonable(v) for k, v in extra.items()}
        
        jpath = os.path.join(self.out_dir, filename)
        with open(jpath, "w") as f:
            json.dump(payload, f, indent=2)
        
        if self.rank == 0:
            print(f"[Profiler] Saved config: {jpath}")

    # ========================================================================
    # Internal helpers
    # ========================================================================

    def _print_run_summary(self, pack: Dict[str, Any]) -> None:
        """Print a brief run summary to console."""
        m = pack.get("meta", {}) or {}
        s = pack.get("stats", {}) or {}
        bm = pack.get("buckets_model_avg_ms") or {}
        bb = pack.get("buckets_engine_avg_ms") or {}

        steps = int(s.get("count") or 0)
        mean = s.get("mean_ms", float("nan"))
        tok_s = s.get("throughput_tok_s", float("nan"))

        print(f"[Profiler] run (bsz={m.get('bsz','?')} num decoding steps={steps}): "
              f"mean={fmt(mean)} ms, tok/s={fmt(tok_s)}")

        if not bm and not bb:
            print("[Profiler] (no bucket breakdown enabled)")
            return

        # Print top buckets
        denom = float(mean) if isinstance(mean, (int, float)) and math.isfinite(float(mean)) and float(mean) > 0 else None

        def _top_brief(d: Dict[str, float], label: str):
            if not d:
                return
            total = denom if denom is not None else (sum(float(v) for v in d.values()) or 1.0)
            top = sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:5]
            brief = ", ".join([f"{k}:{float(v):.2f}ms({(float(v)/total*100.0):.0f}%)" for k, v in top])
            print(f"[Profiler] {label} top: {brief}")

        _top_brief(bm, "model")
        _top_brief(bb, "engine")

    def _write_markdown_report(self, mpath: str, global_summary: Dict[str, Any]) -> None:
        """Write Markdown report to file."""
        with open(mpath, "w") as f:
            f.write("# Speculative Decoding — Global Profile\n\n")
            f.write(f"- Output dir: `{self.out_dir}`\n")
            cfg = asdict(self.cfg)
            f.write(
                f"- Config: `model_profiling={cfg['model_profiling']}`  "
                f"`engine_profiling={cfg['engine_profiling']}`  "
                f"`num_runs(total)={cfg['num_total_runs']}`  "
                f"`strict_sync={cfg['strict_sync']}`  `dist_barrier={cfg['dist_barrier']}`  "
                f"`kv_bins={cfg['kv_bins']}`  `kv_len_reduce={cfg['kv_len_reduce']}`\n\n"
            )

            # Global summary table
            bsz = self._runs[0]["meta"]["bsz"] if self._runs else 1
            s = global_summary["stats"]
            f.write("## Global Summary\n\n")
            f.write("| num decoding steps | mean (ms) | p50 | p90 | p99 | tok/s | tokens | mean generated tokens | time (s) |\n")
            f.write("|--:|--:|--:|--:|--:|--:|--:|--:|--:|\n")
            f.write(
                f"| {s['steps_total']} | {fmt(s['mean_ms'])} | {fmt(s['p50_ms'])} | "
                f"{fmt(s['p90_ms'])} | {fmt(s['p99_ms'])} | {fmt(s['throughput_tok_s'])} | "
                f"{int(s['tokens_total'])} | {fmt(s['tokens_total'] / (s['steps_total'] * bsz))} | "
                f"{fmt(s['time_total_ms']/1000.0)} |\n\n"
            )

            # # Model buckets
            # g_buckets_model_avg = global_summary["buckets_model_avg_ms"]
            # if g_buckets_model_avg:
            #     f.write("## Model Bucket Breakdown (avg (ms) / step)\n\n")
            #     f.write("| bucket | avg (ms) | share |\n|--:|--:|--:|\n")
            #     for k in MODEL_BUCKET_ORDER + (["others"] if "others" in g_buckets_model_avg else []):
            #         if k in g_buckets_model_avg:
            #             v = g_buckets_model_avg[k]
            #             pct = (v / float(s["mean_ms"] or 1e-9)) * 100.0 if s["mean_ms"] else 0.0
            #             f.write(f"| `{k}` | {float(v):.3f} | {pct:.1f}% |\n")
            #     f.write("\n")

            # # Engine buckets
            # g_buckets_engine_avg = global_summary["buckets_engine_avg_ms"]
            # if g_buckets_engine_avg:
            #     f.write("## Engine Bucket Breakdown (avg (ms) / step)\n\n")
            #     f.write("| bucket | avg (ms) | share |\n|--:|--:|--:|\n")
            #     for k in ENGINE_BUCKET_ORDER + (["others"] if "others" in g_buckets_engine_avg else []):
            #         if k in g_buckets_engine_avg:
            #             v = g_buckets_engine_avg[k]
            #             pct = (v / float(s["mean_ms"] or 1e-9)) * 100.0 if s["mean_ms"] else 0.0
            #             f.write(f"| `{k}` | {float(v):.3f} | {pct:.1f}% |\n")
            #     f.write("\n")

            # # Per-run quick view
            # if self._runs:
            #     f.write("## Runs\n\n")
            #     f.write("| run | bsz | num decoding steps | total generated tokens | mean (ms) | tok/s | mean generated tokens |\n")
            #     f.write("|--:|--:|--:|--:|--:|--:|--:|\n")
            #     for i, r in enumerate(self._runs):
            #         rm, rs = r["meta"], r["stats"]
            #         f.write(
            #             f"| {i} | {rm.get('bsz','?')} | {rs.get('count',0)} | "
            #             f"{int(rs.get('tokens_total', 0))} | {fmt(rs.get('mean_ms'))} | "
            #             f"{fmt(rs.get('throughput_tok_s'))} | "
            #             f"{fmt(rs.get('tokens_total', 0) / (rs.get('count', 1) * bsz))} |\n"
            #         )
            #     f.write("\n")

            # # Runs aggregate section
            # ra = global_summary.get("runs_aggregate", {})
            # if ra:
            #     f.write("## Across-run (unweighted) — mean ± std\n\n")
            #     lat = ra.get("latency_ms", {})
            #     tp = ra.get("throughput_tok_s", {})
            #     f.write("| metric | mean | std | n_runs |\n|--|--:|--:|--:|\n")
            #     f.write(
            #         f"| latency (ms/num decoding step) | {fmt(lat.get('mean'))} | "
            #         f"{fmt(lat.get('std'))} | {lat.get('n_runs', 0)} |\n"
            #     )
            #     f.write(
            #         f"| throughput (tok/s) | {fmt(tp.get('mean'))} | "
            #         f"{fmt(tp.get('std'))} | {tp.get('n_runs', 0)} |\n\n"
            #     )

            #     bstats_m = ra.get("buckets_model_avg_ms", {})
            #     if bstats_m:
            #         f.write("### Model buckets — mean ± std across runs\n\n")
            #         f.write("| bucket | mean (ms) | std (ms) | n_runs |\n|--:|--:|--:|--:|\n")
            #         for k in MODEL_BUCKET_ORDER + sorted([x for x in bstats_m.keys() if x not in MODEL_BUCKET_ORDER and x != "others"]):
            #             if k in bstats_m:
            #                 d = bstats_m[k]
            #                 f.write(f"| `{k}` | {fmt(d.get('mean'))} | {fmt(d.get('std'))} | {d.get('n_runs', 0)} |\n")
            #         if "others" in bstats_m:
            #             d = bstats_m["others"]
            #             f.write(f"| `others` | {fmt(d.get('mean'))} | {fmt(d.get('std'))} | {d.get('n_runs', 0)} |\n")
            #         f.write("\n")

            #     bstats_b = ra.get("buckets_engine_avg_ms", {})
            #     if bstats_b:
            #         f.write("### Engine buckets — mean ± std across runs\n\n")
            #         f.write("| bucket | mean (ms) | std (ms) | n_runs |\n|--:|--:|--:|--:|\n")
            #         for k in ENGINE_BUCKET_ORDER + sorted([x for x in bstats_b.keys() if x not in ENGINE_BUCKET_ORDER and x != "others"]):
            #             if k in bstats_b:
            #                 d = bstats_b[k]
            #                 f.write(f"| `{k}` | {fmt(d.get('mean'))} | {fmt(d.get('std'))} | {d.get('n_runs', 0)} |\n")
            #         if "others" in bstats_b:
            #             d = bstats_b["others"]
            #             f.write(f"| `others` | {fmt(d.get('mean'))} | {fmt(d.get('std'))} | {d.get('n_runs', 0)} |\n")
            #         f.write("\n")

            # # Decode-length buckets (global)
            # if "decode_length_buckets" in global_summary:
            #     f.write("## Decode-length Buckets (KV length at step start)\n\n")
            #     f.write("| bin | range | steps | mean (ms) | p50 (ms) | p90 (ms) | p99 (ms) | tok/s | tokens | mean generated tokens | time (s) |\n")
            #     f.write("|--:|--|--:|--:|--:|--:|--:|--:|--:|--:|--:|\n")
            #     gdb = global_summary["decode_length_buckets"]["stats"]
            #     for lo, hi, key in self._g_len_bins_meta:
            #         s = gdb.get(key, {})
            #         rng = f"[{lo},{hi})" if hi is not None else f"[{lo},∞)"
            #         f.write(
            #             f"| {key} | {rng} | {int(s.get('steps',0))} | {fmt(s.get('mean_ms'))} | "
            #             f"{fmt(s.get('p50_ms'))} | {fmt(s.get('p90_ms'))} | {fmt(s.get('p99_ms'))} | "
            #             f"{fmt(s.get('throughput_tok_s'))} | {int(s.get('tokens_total',0))} | "
            #             f"{fmt(s.get('tokens_total',0) / (1 if s.get('steps', 0) == 0 else s.get('steps',0) * bsz))} | "
            #             f"{fmt((s.get('time_total_ms',0.0))/1000.0)} |\n"
            #         )
            #     f.write("\n")

            pg = global_summary.get("prefix_len_groups", {})
            if pg:
                f.write("## Prefix-length Groups\n\n")
                def _key(k):
                    try:
                        return -1 if k == "None" else int(k)
                    except:
                        return 10**18
                for pl_key in sorted(pg.keys(), key=_key):
                    g = pg[pl_key]
                    f.write(f"### prefix_len = {pl_key}\n\n")

                    # Over-runs mean±std
                    oor = g.get("over_runs_unweighted", {})
                    orw = g.get("over_runs_weighted", {})
                    lat = oor.get("latency_mean_ms", {})
                    p95 = oor.get("p95_ms", {})
                    tp = oor.get("throughput_tok_s", {})
                    f.write("| metric | mean | std | n_runs |\n|--|--:|--:|--:|\n")
                    f.write(f"| latency mean (ms/step) | {fmt(lat.get('mean'))} | {fmt(lat.get('std'))} | {lat.get('n_runs',0)} |\n")
                    f.write(f"| latency p95 (ms/step) | {fmt(p95.get('mean'))} | {fmt(p95.get('std'))} | {p95.get('n_runs',0)} |\n")
                    f.write(f"| throughput (tok/s) | {fmt(tp.get('mean'))} | {fmt(tp.get('std'))} | {tp.get('n_runs',0)} |\n")
                    f.write(f"| mean generated tokens | {fmt(int(orw.get('tokens_total', 0)) / (int(orw.get('steps_total', 0)) * bsz))} |\n\n")

                    # Runs table
                    f.write("| run_idx | bsz | steps | tokens | mean (ms) | p95 (ms) | tok/s | mean generated tokens |\n")
                    f.write("|--:|--:|--:|--:|--:|--:|--:|\n")
                    for i, r in enumerate(g.get("runs", [])):
                        rm, rs = r.get("meta", {}), r.get("stats", {})
                        f.write(
                            f"| {i} | {rm.get('bsz','?')} | {rs.get('count',0)} | "
                            f"{int(rs.get('tokens_total',0) or 0)} | {fmt(rs.get('mean_ms'))} | "
                            f"{fmt(rs.get('p95_ms'))} | {fmt(rs.get('throughput_tok_s'))} | {fmt(int(rs.get('tokens_total', 0)) / (int(rs.get('count', 1)) * int(rm.get('bsz', 1))))} |\n"
                        )
                    f.write("\n")
