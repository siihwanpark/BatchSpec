"""Main Profiler class and output generation."""

import json
import math
import os
from dataclasses import asdict
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist

from .config import ProfilerConfig, MODEL_BUCKET_ORDER, ENGINE_BUCKET_ORDER
from .hooks import attach_model_hooks, attach_engine_hooks
from .utils import (
    mean_std, percentile, rank_world, canon_bucket,
    order_and_compact, fmt, now_s, dist_ready, generate_run_name
)


class Profiler:
    """
    CUDA-event profiler for speculative decoding benchmarks.
    
    Features:
    - Per-step latency and accepted-token throughput (tok/s)
    - Optional model/engine latency breakdown
    - Prefix-length grouped statistics
    - JSON/Markdown report outputs
    
    Usage:
        prof = Profiler(args)
        prof.attach_model(model)
        prof.attach_engine(engine)
        
        for prefix_len in prefix_lens:
            prof.begin_run(bsz=batch_size, prefix_len=prefix_len)
            for step in decode_steps:
                with prof.step_timing_ctx():
                    # decode step
                prof.set_step_tokens(num_accepted)
            prof.end_run()
        
        prof.save_all()
    """

    def __init__(self, runner_args: SimpleNamespace):
        # Configuration
        self.cfg = ProfilerConfig.from_args(runner_args)
        self.rank, self.world = rank_world()
        self.disabled = (self.cfg.collect_on_rank0_only and self.rank != 0)
        
        # Output directory
        self.out_dir = os.path.join(
            self.cfg.output_dir,
            self.cfg.run_name or generate_run_name(runner_args)
        )
        if not self.disabled:
            os.makedirs(self.out_dir, exist_ok=True)
            print(f"[Profiler] Output dir: {self.out_dir}")
        
        # Store runner args for config export
        self.runner_args = vars(runner_args) if hasattr(runner_args, '__dict__') else runner_args

        # === Per-run state (reset each begin_run) ===
        self._is_measuring: bool = False
        self._step_idx: int = 0
        self._run_meta: Dict[str, Any] = {}
        
        # Step-level data within current run
        self._step_latencies_ms: List[float] = []
        self._step_tokens: List[int] = []
        self._step_events: List[tuple[str, Any, Any, str]] = []  # (type, start, end, bucket)
        self._step_model_buckets: List[Dict[str, float]] = []
        self._step_engine_buckets: List[Dict[str, float]] = []
        self._pending_tokens: int = 0

        # === Accumulated results ===
        self._runs: List[Dict[str, Any]] = []
        
        # Global aggregation (across all runs)
        self._global_latencies_ms: List[float] = []
        self._global_tokens: int = 0
        self._global_steps: int = 0

    # ========================================================================
    # Attach hooks
    # ========================================================================

    def attach_model(self, model: Any, use_gated_lora: bool = False) -> None:
        """Attach profiling hooks to model modules."""
        attach_model_hooks(self, model, use_gated_lora)

    def attach_engine(self, engine_obj: Any) -> None:
        """Attach profiling hooks to engine methods."""
        attach_engine_hooks(self, engine_obj)

    # ========================================================================
    # Run lifecycle
    # ========================================================================

    def begin_run(self, *, bsz: int, prefix_len: int, label: str = "decode") -> None:
        """
        Begin a profiling run for a specific prefix length.
        
        Args:
            bsz: Batch size
            prefix_len: Current prefix length being benchmarked
            label: Run label (default: "decode")
        """
        if self.disabled:
            return
        
        self._run_meta = {
            "label": label,
            "bsz": int(bsz),
            "prefix_len": int(prefix_len) if prefix_len is not None else None,
            "rank": self.rank,
            "world": self.world,
            "started_at": now_s(),
            "cfg": asdict(self.cfg),
        }

        # Reset per-run state
        self._step_latencies_ms.clear()
        self._step_tokens.clear()
        self._step_events.clear()
        self._step_model_buckets.clear()
        self._step_engine_buckets.clear()
        self._step_idx = 0
        
    def step_timing_ctx(self):
        """
        Context manager for timing a single decode step.
        
        Usage:
            with prof.step_timing_ctx():
                # perform decode step
        """
        profiler = self
        
        class StepTimingContext:
            __slots__ = ("start_event", "end_event")
            
            def __init__(self):
                self.start_event = torch.cuda.Event(enable_timing=True)
                self.end_event = torch.cuda.Event(enable_timing=True)
            
            def __enter__(self):
                if profiler.cfg.strict_sync:
                    torch.cuda.synchronize()
                if profiler.cfg.dist_barrier and dist_ready():
                    dist.barrier()
                
                profiler._is_measuring = True
                profiler._step_events.clear()
                profiler._step_idx += 1
                self.start_event.record()
                return None
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.end_event.record()
                profiler._is_measuring = False
                
                if profiler.cfg.strict_sync:
                    torch.cuda.synchronize()
                self.end_event.synchronize()
                if profiler.cfg.dist_barrier and dist_ready():
                    dist.barrier()

                # Record step latency
                step_ms = float(self.start_event.elapsed_time(self.end_event))
                profiler._step_latencies_ms.append(step_ms)
                
                # Record tokens
                tokens = profiler._pending_tokens
                profiler._step_tokens.append(int(tokens))
                profiler._pending_tokens = 0

                # Process bucket events
                if profiler._step_events and (profiler.cfg.model_profiling or profiler.cfg.engine_profiling):
                    model_buckets: Dict[str, float] = {}
                    engine_buckets: Dict[str, float] = {}
                    
                    for event_type, start, end, bucket in profiler._step_events:
                        elapsed = float(start.elapsed_time(end)) if event_type == "cuda" else float(start)
                        key, domain = canon_bucket(bucket)
                        
                        if domain == "engine":
                            engine_buckets[key] = engine_buckets.get(key, 0.0) + elapsed
                        else:
                            model_buckets[key] = model_buckets.get(key, 0.0) + elapsed
                    
                    if model_buckets:
                        profiler._step_model_buckets.append(model_buckets)
                    if engine_buckets:
                        profiler._step_engine_buckets.append(engine_buckets)

                profiler._step_events.clear()
                return False
        
        return StepTimingContext()

    def set_step_tokens(self, n: int) -> None:
        """Record the number of tokens accepted in this step."""
        if self.disabled:
            return
        try:
            self._pending_tokens = int(n)
        except Exception:
            self._pending_tokens = 0

    def end_run(self) -> None:
        """End the current profiling run and aggregate statistics."""
        if self.disabled:
            return

        num_steps = len(self._step_latencies_ms)
        sorted_latencies = sorted(self._step_latencies_ms)
        
        # Compute run statistics
        stats = {"count": num_steps}
        
        if num_steps > 0:
            total_time_ms = sum(self._step_latencies_ms)
            total_tokens = sum(self._step_tokens)
            mean_latency = total_time_ms / num_steps
            throughput = (total_tokens / (total_time_ms / 1000.0)) if total_time_ms > 0 else float("nan")
            
            stats.update({
                "mean_ms": mean_latency,
                "min_ms": sorted_latencies[0],
                "max_ms": sorted_latencies[-1],
                "p50_ms": percentile(sorted_latencies, 50),
                "p90_ms": percentile(sorted_latencies, 90),
                "p95_ms": percentile(sorted_latencies, 95),
                "p99_ms": percentile(sorted_latencies, 99),
                "throughput_tok_s": throughput,
                "tokens_total": total_tokens,
                "time_total_ms": total_time_ms,
            })

        # Compute bucket averages for this run (with "other" calculation for model buckets)
        model_bucket_avg = self._compute_model_bucket_averages_with_other(
            self._step_model_buckets, self._step_latencies_ms, num_steps
        )
        engine_bucket_avg = self._compute_bucket_averages(
            self._step_engine_buckets, num_steps, ENGINE_BUCKET_ORDER
        )

        # Package run results
        run_result = {
            "meta": dict(self._run_meta),
            "stats": stats,
            "buckets_model_avg_ms": model_bucket_avg,
            "buckets_engine_avg_ms": engine_bucket_avg,
        }

        self._runs.append(run_result)
        
        if self.cfg.print_per_run:
            self._print_run_summary(run_result)

        # Global accumulation
        self._global_latencies_ms.extend(self._step_latencies_ms)
        self._global_tokens += stats.get("tokens_total", 0)
        self._global_steps += num_steps

    def _compute_model_bucket_averages_with_other(
        self,
        step_buckets: List[Dict[str, float]],
        step_latencies: List[float],
        num_steps: int
    ) -> Dict[str, float]:
        """
        Compute average model bucket times with "other" (untracked time).
        
        "other" = total step latency - sum of all tracked model buckets
        """
        if num_steps == 0 or not step_buckets:
            return {}
        
        # Sum across all steps
        bucket_sums: Dict[str, float] = {}
        total_tracked_time = 0.0
        
        for step_data in step_buckets:
            step_tracked = 0.0
            for key, value in step_data.items():
                bucket_sums[key] = bucket_sums.get(key, 0.0) + value
                step_tracked += value
            total_tracked_time += step_tracked
        
        # Compute averages
        bucket_avgs = {key: (total / num_steps) for key, total in bucket_sums.items()}
        
        # Compute "other" (untracked time)
        total_latency = sum(step_latencies)
        other_total = total_latency - total_tracked_time
        if other_total > 0:
            bucket_avgs["other"] = other_total / num_steps
        
        return order_and_compact(bucket_avgs, MODEL_BUCKET_ORDER)

    def _compute_bucket_averages(
        self, 
        step_buckets: List[Dict[str, float]], 
        num_steps: int,
        bucket_order: List[str]
    ) -> Dict[str, float]:
        """Compute average bucket times across all steps in a run."""
        if num_steps == 0 or not step_buckets:
            return {}
        
        # Sum across all steps
        bucket_sums: Dict[str, float] = {}
        for step_data in step_buckets:
            for key, value in step_data.items():
                bucket_sums[key] = bucket_sums.get(key, 0.0) + value
        
        # Compute averages
        bucket_avgs = {key: (total / num_steps) for key, total in bucket_sums.items()}
        
        return order_and_compact(bucket_avgs, bucket_order)

    # ========================================================================
    # Save outputs
    # ========================================================================

    def save_all(self) -> None:
        """Save all profiling results to JSON and Markdown."""
        if self.disabled:
            if self.rank == 0:
                print("[Profiler] Disabled (rank gated). Nothing to save.")
            return

        # Compute global statistics
        global_stats = self._compute_global_stats()
        
        # Group runs by prefix_len and compute per-group statistics
        prefix_groups = self._compute_prefix_group_stats()

        # Build summary
        global_summary = {
            "meta": {
                "rank": self.rank,
                "world": self.world,
                "started_at": self._runs[0]["meta"]["started_at"] if self._runs else now_s(),
                "cfg": asdict(self.cfg),
                "output_dir": self.out_dir,
            },
            "stats": global_stats,
            "prefix_len_groups": prefix_groups,
            "runs": self._runs,
        }

        # Write JSON
        json_path = os.path.join(self.out_dir, "summary.json")
        with open(json_path, "w") as f:
            json.dump(global_summary, f, indent=2)

        # Write Markdown report
        md_path = os.path.join(self.out_dir, "report.md")
        self._write_markdown_report(md_path, global_summary)

        if self.rank == 0:
            print(f"[Profiler] Saved:\n  {json_path}\n  {md_path}")
            print(f"[Profiler] Global: steps={global_stats['steps_total']} "
                  f"mean={fmt(global_stats['mean_ms'])}ms tok/s={fmt(global_stats['throughput_tok_s'])}")

    def _compute_global_stats(self) -> Dict[str, Any]:
        """Compute global statistics across all runs."""
        sorted_latencies = sorted(self._global_latencies_ms)
        num_steps = len(sorted_latencies)
        
        if num_steps == 0:
            return {"steps_total": 0, "time_total_ms": 0, "tokens_total": 0}
        
        total_time_ms = sum(self._global_latencies_ms)
        mean_latency = total_time_ms / num_steps
        throughput = (self._global_tokens / (total_time_ms / 1000.0)) if total_time_ms > 0 else float("nan")
        
        return {
            "steps_total": num_steps,
            "time_total_ms": total_time_ms,
            "tokens_total": self._global_tokens,
            "mean_ms": mean_latency,
            "p50_ms": percentile(sorted_latencies, 50),
            "p90_ms": percentile(sorted_latencies, 90),
            "p95_ms": percentile(sorted_latencies, 95),
            "p99_ms": percentile(sorted_latencies, 99),
            "throughput_tok_s": throughput,
        }

    def _compute_prefix_group_stats(self) -> Dict[str, Dict[str, Any]]:
        """Group runs by prefix_len and compute statistics for each group."""
        # Group runs by prefix_len
        groups: Dict[Any, List[Dict]] = defaultdict(list)
        for run in self._runs:
            prefix_len = run.get("meta", {}).get("prefix_len", None)
            groups[prefix_len].append(run)
        
        result = {}
        for prefix_len, runs in groups.items():
            result[str(prefix_len)] = self._compute_single_group_stats(prefix_len, runs)
        
        return result

    def _compute_single_group_stats(self, prefix_len: Any, runs: List[Dict]) -> Dict[str, Any]:
        """Compute statistics for a single prefix_len group."""
        # Extract per-run metrics
        run_mean_latencies = [r["stats"]["mean_ms"] for r in runs if "mean_ms" in r["stats"]]
        run_p95_latencies = [r["stats"]["p95_ms"] for r in runs if "p95_ms" in r["stats"]]
        run_throughputs = [r["stats"]["throughput_tok_s"] for r in runs if "throughput_tok_s" in r["stats"]]
        
        # Compute mean ± std across runs
        lat_mean, lat_std, lat_n = mean_std(run_mean_latencies)
        p95_mean, p95_std, p95_n = mean_std(run_p95_latencies)
        tp_mean, tp_std, tp_n = mean_std(run_throughputs)

        # Compute weighted totals
        total_tokens = sum(r["stats"].get("tokens_total", 0) or 0 for r in runs)
        total_time_ms = sum(r["stats"].get("time_total_ms", 0.0) or 0.0 for r in runs)
        total_steps = sum(r["stats"].get("count", 0) or 0 for r in runs)
        weighted_throughput = (total_tokens / (total_time_ms / 1000.0)) if total_time_ms > 0 else float("nan")

        # Aggregate bucket statistics across runs
        model_bucket_stats = self._aggregate_bucket_stats_across_runs(runs, "buckets_model_avg_ms")
        engine_bucket_stats = self._aggregate_bucket_stats_across_runs(runs, "buckets_engine_avg_ms")

        return {
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
                "throughput_tok_s": weighted_throughput,
            },
            "buckets_model_avg_ms": model_bucket_stats,
            "buckets_engine_avg_ms": engine_bucket_stats,
        }

    def _aggregate_bucket_stats_across_runs(
        self, 
        runs: List[Dict], 
        bucket_key: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate bucket statistics across multiple runs.
        
        Returns: {bucket_name: {"mean": ..., "std": ..., "n_runs": ...}}
        """
        # Collect per-run bucket values
        bucket_samples: Dict[str, List[float]] = {}
        for run in runs:
            buckets = run.get(bucket_key, {}) or {}
            for key, value in buckets.items():
                bucket_samples.setdefault(key, []).append(float(value))
        
        # Compute mean ± std for each bucket
        result = {}
        for key, values in bucket_samples.items():
            m, s, n = mean_std(values)
            result[key] = {"mean": m, "std": s, "n_runs": n}
        
        return result

    def save_config(self, filename: str = "config.json", extra: Optional[Dict[str, Any]] = None) -> None:
        """Save configuration to JSON."""
        if self.disabled:
            return
        
        def to_jsonable(value):
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            if isinstance(value, (list, tuple)):
                return [to_jsonable(x) for x in value]
            return str(value)
        
        # Convert runner args
        if isinstance(self.runner_args, dict):
            args_dict = {
                k: to_jsonable(v) for k, v in self.runner_args.items()
                if not callable(v) and not k.startswith("_")
            }
        else:
            args_dict = {"__raw__": to_jsonable(self.runner_args)}
        
        cfg_dict = asdict(self.cfg)
        cfg_dict["rank"] = self.rank
        cfg_dict["world"] = self.world
        
        payload = {"runner_args": args_dict, "profiler_cfg": cfg_dict}
        if extra:
            payload["extra"] = {k: to_jsonable(v) for k, v in extra.items()}
        
        config_path = os.path.join(self.out_dir, filename)
        with open(config_path, "w") as f:
            json.dump(payload, f, indent=2)
        
        if self.rank == 0:
            print(f"[Profiler] Saved config: {config_path}")

    # ========================================================================
    # Internal helpers
    # ========================================================================

    def _print_run_summary(self, run_result: Dict[str, Any]) -> None:
        """Print a brief run summary to console."""
        meta = run_result.get("meta", {})
        stats = run_result.get("stats", {})
        model_buckets = run_result.get("buckets_model_avg_ms", {})
        engine_buckets = run_result.get("buckets_engine_avg_ms", {})

        num_steps = stats.get("count", 0)
        mean_latency = stats.get("mean_ms", float("nan"))
        throughput = stats.get("throughput_tok_s", float("nan"))

        print(f"[Profiler] run (bsz={meta.get('bsz', '?')} prefix_len={meta.get('prefix_len', '?')} "
              f"steps={num_steps}): mean={fmt(mean_latency)}ms, tok/s={fmt(throughput)}")

        if not model_buckets and not engine_buckets:
            return

        # Print bucket breakdown
        denom = float(mean_latency) if math.isfinite(float(mean_latency)) and mean_latency > 0 else None

        def print_top_buckets(buckets: Dict[str, float], label: str):
            if not buckets:
                return
            total = denom if denom else sum(buckets.values()) or 1.0
            sorted_buckets = sorted(buckets.items(), key=lambda kv: kv[1], reverse=True)[:5]
            breakdown = ", ".join([f"{k}:{v:.2f}ms({v/total*100:.0f}%)" for k, v in sorted_buckets])
            print(f"[Profiler]   {label}: {breakdown}")

        print_top_buckets(model_buckets, "model")
        print_top_buckets(engine_buckets, "engine")

    def _write_markdown_report(self, path: str, global_summary: Dict[str, Any]) -> None:
        """Write Markdown report to file."""
        cfg = asdict(self.cfg)
        stats = global_summary["stats"]
        bsz = self._runs[0]["meta"]["bsz"] if self._runs else 1

        with open(path, "w") as f:
            # Header
            f.write("# Speculative Decoding — Profile Report\n\n")
            f.write(f"- Output dir: `{self.out_dir}`\n")
            f.write(f"- Config: `model_profiling={cfg['model_profiling']}` "
                    f"`engine_profiling={cfg['engine_profiling']}` "
                    f"`num_runs={cfg['num_total_runs']}` "
                    f"`strict_sync={cfg['strict_sync']}` "
                    f"`dist_barrier={cfg['dist_barrier']}`\n\n")

            # Global summary table
            f.write("## Global Summary\n\n")
            f.write("| steps | mean (ms) | p50 | p90 | p99 | tok/s | tokens | mean tokens/step | time (s) |\n")
            f.write("|--:|--:|--:|--:|--:|--:|--:|--:|--:|\n")
            
            mean_tokens_per_step = stats['tokens_total'] / (stats['steps_total'] * bsz) if stats['steps_total'] > 0 else 0
            f.write(f"| {stats['steps_total']} | {fmt(stats['mean_ms'])} | {fmt(stats['p50_ms'])} | "
                    f"{fmt(stats['p90_ms'])} | {fmt(stats['p99_ms'])} | {fmt(stats['throughput_tok_s'])} | "
                    f"{stats['tokens_total']} | {fmt(mean_tokens_per_step)} | "
                    f"{fmt(stats['time_total_ms']/1000.0)} |\n\n")

            # Prefix-length groups
            prefix_groups = global_summary.get("prefix_len_groups", {})
            if prefix_groups:
                f.write("## Results by Prefix Length\n\n")
                
                # Sort by prefix_len numerically
                def sort_key(k):
                    try:
                        return -1 if k == "None" else int(k)
                    except:
                        return float('inf')
                
                for prefix_key in sorted(prefix_groups.keys(), key=sort_key):
                    group = prefix_groups[prefix_key]
                    self._write_prefix_group_section(f, prefix_key, group, bsz)

    def _write_prefix_group_section(self, f, prefix_key: str, group: Dict, bsz: int) -> None:
        """Write a single prefix_len group section to the Markdown file."""
        f.write(f"### prefix_len = {prefix_key}\n\n")

        # Unweighted statistics (mean ± std across runs)
        unweighted = group.get("over_runs_unweighted", {})
        weighted = group.get("over_runs_weighted", {})
        
        lat = unweighted.get("latency_mean_ms", {})
        p95 = unweighted.get("p95_ms", {})
        tp = unweighted.get("throughput_tok_s", {})
        
        total_steps = weighted.get("steps_total", 1)
        total_tokens = weighted.get("tokens_total", 0)
        mean_tokens_per_step = total_tokens / (total_steps * bsz) if total_steps > 0 else 0

        f.write("| metric | mean | std | n_runs |\n|--|--:|--:|--:|\n")
        f.write(f"| latency mean (ms/step) | {fmt(lat.get('mean'))} | {fmt(lat.get('std'))} | {lat.get('n_runs', 0)} |\n")
        f.write(f"| latency p95 (ms/step) | {fmt(p95.get('mean'))} | {fmt(p95.get('std'))} | {p95.get('n_runs', 0)} |\n")
        f.write(f"| throughput (tok/s) | {fmt(tp.get('mean'))} | {fmt(tp.get('std'))} | {tp.get('n_runs', 0)} |\n")
        f.write(f"| mean tokens/step | {fmt(mean_tokens_per_step)} | | |\n\n")

        # Per-run table
        runs = group.get("runs", [])
        if runs:
            f.write("| run | bsz | steps | tokens | mean (ms) | p95 (ms) | tok/s | tokens/step |\n")
            f.write("|--:|--:|--:|--:|--:|--:|--:|--:|\n")
            for i, run in enumerate(runs):
                meta = run.get("meta", {})
                stats = run.get("stats", {})
                run_steps = stats.get("count", 1)
                run_tokens = stats.get("tokens_total", 0)
                run_bsz = meta.get("bsz", 1)
                tokens_per_step = run_tokens / (run_steps * run_bsz) if run_steps > 0 else 0
                
                f.write(f"| {i} | {run_bsz} | {run_steps} | {run_tokens} | "
                        f"{fmt(stats.get('mean_ms'))} | {fmt(stats.get('p95_ms'))} | "
                        f"{fmt(stats.get('throughput_tok_s'))} | {fmt(tokens_per_step)} |\n")
            f.write("\n")

        # Model bucket breakdown (with % of total latency)
        model_stats = group.get("buckets_model_avg_ms", {})
        if model_stats:
            mean_latency = lat.get("mean", 0) or 0
            f.write("#### Model Latency Breakdown (mean ± std across runs)\n\n")
            self._write_bucket_table(f, model_stats, MODEL_BUCKET_ORDER, mean_latency)

        # Engine bucket breakdown
        engine_stats = group.get("buckets_engine_avg_ms", {})
        if engine_stats:
            f.write("#### Engine Latency Breakdown (mean ± std across runs)\n\n")
            self._write_bucket_table(f, engine_stats, ENGINE_BUCKET_ORDER, None)

    def _write_bucket_table(
        self, 
        f, 
        bucket_stats: Dict[str, Dict], 
        bucket_order: List[str],
        total_latency_ms: Optional[float] = None
    ) -> None:
        """
        Write a bucket breakdown table to the Markdown file.
        
        Args:
            f: File handle
            bucket_stats: {bucket_name: {"mean": ..., "std": ..., "n_runs": ...}}
            bucket_order: Preferred ordering of buckets
            total_latency_ms: Total step latency for % calculation (None to skip %)
        """
        # Header with or without % column
        if total_latency_ms and total_latency_ms > 0:
            f.write("| bucket | mean (ms) ± std (ms) (%) | n_runs |\n|--:|--:|--:|\n")
        else:
            f.write("| bucket | mean (ms) ± std (ms) | n_runs |\n|--:|--:|--:|\n")
        
        def write_row(key: str, d: Dict):
            mean_val = d.get('mean', 0) or 0
            std_val = d.get('std', 0) or 0
            n_runs = d.get('n_runs', 0)
            
            if total_latency_ms and total_latency_ms > 0:
                pct = (mean_val / total_latency_ms) * 100
                f.write(f"| `{key}` | {fmt(mean_val)} ± {fmt(std_val)} ({pct:.1f}%) | {n_runs} |\n")
            else:
                f.write(f"| `{key}` | {fmt(mean_val)} ± {fmt(std_val)} | {n_runs} |\n")
        
        # Write ordered buckets first
        for key in bucket_order:
            if key in bucket_stats:
                write_row(key, bucket_stats[key])
        
        # Write remaining buckets (not in order, excluding "other")
        remaining = sorted([k for k in bucket_stats.keys() if k not in bucket_order and k != "other"])
        for key in remaining:
            write_row(key, bucket_stats[key])
        
        # Write "other" last
        if "other" in bucket_stats:
            write_row("other", bucket_stats["other"])
        
        f.write("\n")
