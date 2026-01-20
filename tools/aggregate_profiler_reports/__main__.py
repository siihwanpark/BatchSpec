#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .common import (
    ExpKey, PrefixSection,
    METHOD_ORDER,
    KSelect, SummaryMode,
    fmt_plain, fmt_with_ratio,
    forced_targets_for_group, method_max_target,
)
from .parse import discover_reports, parse_report_md
from .aggregate import (
    compute_cells_fixed_k,
    compute_cells_optimal_k,
    compute_cells_optimal_k_late_spec,
)

Row = Tuple[str, str, str, str, str]  # (name, lm, lp, ts, acc_len)


def write_markdown(
    out_path: Path,
    title: str,
    tables: Dict[Tuple[int, int, str, int], List[Row]],
) -> None:
    """Write summary tables to markdown file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = [f"# {title}\n"]

    keys = sorted(tables.keys(), key=lambda x: (x[0], x[1], x[2], x[3]))
    last_group = None
    for bsz, tp, mode, target in keys:
        group = (bsz, tp, mode)
        if group != last_group:
            lines.append(f"\n## bsz={bsz}, tp={tp}, mode={mode}\n")
            last_group = group

        lines.append(f"\n### up to {target}\n")
        lines.append("| Method | Latency (mean, ms/step) | Latency (p95, ms/step) | Throughput (tok/s) | Mean accepted length |")
        lines.append("|---|---:|---:|---:|---:|")
        for row_name, lm, lp, ts, al in tables[(bsz, tp, mode, target)]:
            lines.append(f"| {row_name} | {lm} | {lp} | {ts} | {al} |")

    out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def load_experiments(input_root: Path, debug_parse: bool) -> List[Tuple[ExpKey, Dict[int, PrefixSection]]]:
    """Discover and parse all report.md under input_root."""
    exps: List[Tuple[ExpKey, Dict[int, PrefixSection]]] = []
    for ek, rp in discover_reports(input_root):
        sec = parse_report_md(rp, debug_parse=debug_parse)
        if sec:
            exps.append((ek, sec))
        elif debug_parse:
            print(f"[debug_parse] Parsed EMPTY (skipped): {rp}", file=sys.stderr)
    return exps


def group_by_model_dataset(
    exps: List[Tuple[ExpKey, Dict[int, PrefixSection]]]
) -> Dict[Tuple[str, str], List[Tuple[ExpKey, Dict[int, PrefixSection]]]]:
    out: Dict[Tuple[str, str], List[Tuple[ExpKey, Dict[int, PrefixSection]]]] = {}
    for ek, sec in exps:
        out.setdefault((ek.model, ek.dataset), []).append((ek, sec))
    return out


def build_standard_cache_from_reference(
    grouped: Dict[Tuple[str, str], List[Tuple[ExpKey, Dict[int, PrefixSection]]]],
    reference_dataset_name: str,
) -> Dict[Tuple[str, int, int, str], Dict[int, PrefixSection]]:
    """
    Standard is dataset-invariant:
    cache standard sections from reference dataset and reuse for all datasets.
    """
    cache: Dict[Tuple[str, int, int, str], Dict[int, PrefixSection]] = {}
    for (model, dataset), exp_list in grouped.items():
        if dataset != reference_dataset_name:
            continue
        for ek, sec in exp_list:
            if ek.method_base == "Standard":
                cache[(model, ek.bsz, ek.tp, ek.mode)] = sec
    return cache


def summarize_one_model_dataset(
    model: str,
    dataset: str,
    exp_list: List[Tuple[ExpKey, Dict[int, PrefixSection]]],
    standard_reference_cache: Dict[Tuple[str, int, int, str], Dict[int, PrefixSection]],
    output_root: Path,
    summary_mode: SummaryMode,
    k_select: KSelect,
    details_include_policies: bool,
) -> None:
    """
    Build summary tables for one (model,dataset) and write summary.md (and optionally details.md).
    """
    # (bsz,tp,mode) -> method_base -> k -> (ExpKey, sections)
    grouped: Dict[Tuple[int, int, str], Dict[str, Dict[int, Tuple[ExpKey, Dict[int, PrefixSection]]]]] = {}

    for ek, sec in exp_list:
        g = (ek.bsz, ek.tp, ek.mode)
        if ek.method_base == "Standard":
            grouped.setdefault(g, {}).setdefault("Standard", {})[0] = (ek, sec)
        else:
            if ek.k is None:
                continue
            grouped.setdefault(g, {}).setdefault(ek.method_base, {})[ek.k] = (ek, sec)

    summary_tables: Dict[Tuple[int, int, str, int], List[Row]] = {}
    detail_tables: Dict[Tuple[int, int, str, int], List[Row]] = {}

    for (bsz, tp, mode), by_method in grouped.items():
        std_sec = standard_reference_cache.get((model, bsz, tp, mode))

        # targets: union across methods, each method contributes forced targets capped by its endpoint.
        targets_union: Set[int] = set()

        if std_sec is not None:
            targets_union |= set(forced_targets_for_group(sorted(std_sec.keys()), "Standard", bsz))

        for method_base, by_k in by_method.items():
            if method_base == "Standard":
                continue
            any_sec = next(iter(by_k.values()))[1]
            targets_union |= set(forced_targets_for_group(sorted(any_sec.keys()), method_base, bsz))

        targets = sorted(targets_union)

        # Standard aggregated cells per target (for ratios)
        std_cells = compute_cells_fixed_k(std_sec, targets) if (std_sec is not None and targets) else {}

        for target in targets:
            rows_full: List[Row] = []
            rows_compact: List[Row] = []

            # Standard row
            if target in std_cells:
                (lm_m, lm_s), (lp_m, lp_s), (ts_m, ts_s), (al_m, al_s) = std_cells[target]
                std_lm_mean = lm_m
                std_lp_mean = lp_m
                std_ts_mean = ts_m
                std_row = (
                    "Standard",
                    fmt_with_ratio(lm_m, lm_s, std_lm_mean),
                    fmt_with_ratio(lp_m, lp_s, std_lp_mean),
                    fmt_with_ratio(ts_m, ts_s, std_ts_mean),
                    fmt_plain(al_m, al_s),
                )
            else:
                std_lm_mean = None
                std_lp_mean = None
                std_ts_mean = None
                std_row = ("Standard", "", "", "", "")

            rows_full.append(std_row)
            rows_compact.append(std_row)
            detail_tables.setdefault((bsz, tp, mode, target), []).append(std_row)

            # Non-standard methods
            for method_base in [m for m in METHOD_ORDER if m != "Standard"]:
                if method_base not in by_method:
                    continue

                by_k = by_method[method_base]
                ks = sorted(by_k.keys())

                # Fixed-k rows
                for k in ks:
                    ek_k, sec_k = by_k[k]
                    max_t = method_max_target(sorted(sec_k.keys()), method_base, bsz)

                    tmp = {}
                    if target <= max_t:
                        tmp = compute_cells_fixed_k(sec_k, [target])

                    if target in tmp:
                        (lm_m, lm_s), (lp_m, lp_s), (ts_m, ts_s), (al_m, al_s) = tmp[target]
                        row = (
                            f"{method_base} ($k={k}$)",
                            fmt_with_ratio(lm_m, lm_s, std_lm_mean),
                            fmt_with_ratio(lp_m, lp_s, std_lp_mean),
                            fmt_with_ratio(ts_m, ts_s, std_ts_mean),
                            fmt_plain(al_m, al_s),
                        )
                    else:
                        row = (f"{method_base} ($k={k}$)", "", "", "", "")

                    rows_full.append(row)
                    detail_tables.setdefault((bsz, tp, mode, target), []).append(row)

                # Policy rows (optimal-k / late-spec)
                k_sections: Dict[int, Dict[int, PrefixSection]] = {k: by_k[k][1] for k in ks}
                method_prefix_lens = sorted({pl for sec in k_sections.values() for pl in sec.keys()})
                max_t = method_max_target(method_prefix_lens, method_base, bsz)

                if target > max_t:
                    opt_row = (f"{method_base} (optimal $k$)", "", "", "", "")
                    late_row = (f"{method_base} (optimal $k$ + late spec)", "", "", "", "")
                else:
                    opt_map = compute_cells_optimal_k(k_sections, [target], k_select)
                    if target in opt_map:
                        (lm_m, lm_s), (lp_m, lp_s), (ts_m, ts_s), (al_m, al_s) = opt_map[target]
                        opt_row = (
                            f"{method_base} (optimal $k$)",
                            fmt_with_ratio(lm_m, lm_s, std_lm_mean),
                            fmt_with_ratio(lp_m, lp_s, std_lp_mean),
                            fmt_with_ratio(ts_m, ts_s, std_ts_mean),
                            fmt_plain(al_m, al_s),
                        )
                    else:
                        opt_row = (f"{method_base} (optimal $k$)", "", "", "", "")

                    late_row = (f"{method_base} (optimal $k$ + late spec)", "", "", "", "")
                    if std_sec is not None:
                        late_map = compute_cells_optimal_k_late_spec(k_sections, std_sec, [target], k_select)
                        if target in late_map:
                            (lm_m, lm_s), (lp_m, lp_s), (ts_m, ts_s), (al_m, al_s) = late_map[target]
                            late_row = (
                                f"{method_base} (optimal $k$ + late spec)",
                                fmt_with_ratio(lm_m, lm_s, std_lm_mean),
                                fmt_with_ratio(lp_m, lp_s, std_lp_mean),
                                fmt_with_ratio(ts_m, ts_s, std_ts_mean),
                                fmt_plain(al_m, al_s),
                            )

                rows_full.extend([opt_row, late_row])
                rows_compact.extend([opt_row, late_row])

                if summary_mode == "compact" and details_include_policies:
                    detail_tables.setdefault((bsz, tp, mode, target), []).extend([opt_row, late_row])

            summary_tables[(bsz, tp, mode, target)] = rows_full if summary_mode == "full" else rows_compact

    # Write outputs
    sum_path = output_root / model / dataset / "summary.md"
    write_markdown(
        sum_path,
        title=f"Profiling Summary: {model} / {dataset} (mode={summary_mode}, k_select={k_select})",
        tables=summary_tables,
    )

    if summary_mode == "compact":
        det_path = output_root / model / dataset / "details.md"
        write_markdown(
            det_path,
            title=f"Profiling Details: {model} / {dataset} (k_select={k_select}, include_policies={details_include_policies})",
            tables=detail_tables,
        )
        print(f"[OK] Wrote {sum_path} and {det_path}")
    else:
        print(f"[OK] Wrote {sum_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", type=str, default="profiler_out")
    ap.add_argument("--output_root", type=str, default="profiler_summary")
    ap.add_argument("--reference_dataset_name", type=str, default="AIME2025")
    ap.add_argument("--summary_mode", type=str, choices=["full", "compact"], default="full")
    ap.add_argument("--k_select", type=str, choices=["throughput", "lat_mean", "lat_p95"], default="throughput")
    ap.add_argument("--details_include_policies", action="store_true")
    ap.add_argument("--debug_parse", action="store_true")
    args = ap.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    summary_mode: SummaryMode = args.summary_mode  # type: ignore
    k_select: KSelect = args.k_select  # type: ignore

    exps = load_experiments(input_root, debug_parse=args.debug_parse)
    grouped = group_by_model_dataset(exps)
    standard_cache = build_standard_cache_from_reference(grouped, args.reference_dataset_name)

    for (model, dataset), exp_list in grouped.items():
        summarize_one_model_dataset(
            model=model,
            dataset=dataset,
            exp_list=exp_list,
            standard_reference_cache=standard_cache,
            output_root=output_root,
            summary_mode=summary_mode,
            k_select=k_select,
            details_include_policies=args.details_include_policies,
        )

    print("[DONE]")


if __name__ == "__main__":
    main()
