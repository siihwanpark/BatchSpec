from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from .common import (
    PrefixSection, RunStats,
    build_chain_starts, mean_std, score_of, spec_beats_standard, KSelect,
)

AggCell = Tuple[
    Tuple[float, float],  # lat_mean (mean,std)
    Tuple[float, float],  # lat_p95  (mean,std)
    Tuple[float, float],  # tok/s    (mean,std)
    Tuple[float, float],  # accepted_len (mean,std)
]


def aggregate_one_run(
    chain_starts: List[int],
    target: int,
    chosen_by_prefix: Dict[int, RunStats],
) -> Optional[Tuple[float, float, float, float]]:
    """
    Weighted aggregation over [1024..target].

    Returns:
      (lat_mean_ms, lat_p95_ms, throughput_tok_s, accepted_len_tok_per_step)
    """
    total_tokens = 0.0
    total_time_s = 0.0

    sum_steps = 0.0
    sum_lat_mean = 0.0
    sum_lat_p95 = 0.0
    sum_acc_len = 0.0

    points = chain_starts + [target]
    for a, b in zip(points[:-1], points[1:]):
        seg_tokens = float(b - a)
        if seg_tokens <= 0:
            return None

        rs = chosen_by_prefix.get(a)
        if rs is None or rs.tok_per_step <= 0 or rs.tok_s <= 0:
            return None

        seg_steps = seg_tokens / rs.tok_per_step
        seg_time_s = seg_tokens / rs.tok_s

        total_tokens += seg_tokens
        total_time_s += seg_time_s

        sum_steps += seg_steps
        sum_lat_mean += rs.latency_mean_ms * seg_steps
        sum_lat_p95 += rs.latency_p95_ms * seg_steps
        sum_acc_len += rs.tok_per_step * seg_steps

    if sum_steps <= 0 or total_time_s <= 0:
        return None

    lat_mean = sum_lat_mean / sum_steps
    lat_p95 = sum_lat_p95 / sum_steps
    tok_s = total_tokens / total_time_s
    acc_len = sum_acc_len / sum_steps
    return lat_mean, lat_p95, tok_s, acc_len


def compute_cells_fixed_k(
    sections: Dict[int, PrefixSection],
    targets: List[int],
) -> Dict[int, AggCell]:
    """
    Aggregate metrics for a fixed experiment (i.e., fixed k).

    Returns:
      {target: ((lm_m,lm_s),(lp_m,lp_s),(ts_m,ts_s),(al_m,al_s))}
    """
    prefix_lens = sorted(sections.keys())
    out: Dict[int, AggCell] = {}

    for target in targets:
        chain_starts = build_chain_starts(prefix_lens, target)
        if chain_starts is None:
            continue

        run_sets: List[Set[int]] = []
        for pl in chain_starts:
            ids = set(sections[pl].runs.keys())
            if pl == 1024 and 0 in ids:  # ignore compile run
                ids.remove(0)
            run_sets.append(ids)

        eligible = set.intersection(*run_sets) if run_sets else set()
        if not eligible:
            continue

        lm, lp, ts, al = [], [], [], []
        for rid in sorted(eligible):
            chosen = {pl: sections[pl].runs[rid] for pl in chain_starts}
            agg = aggregate_one_run(chain_starts, target, chosen)
            if agg is None:
                continue
            a, b, c, d = agg
            lm.append(a); lp.append(b); ts.append(c); al.append(d)

        ms = mean_std(lm); ps = mean_std(lp); th = mean_std(ts); ac = mean_std(al)
        if ms and ps and th and ac:
            out[target] = (ms, ps, th, ac)

    return out


def compute_cells_optimal_k(
    k_sections: Dict[int, Dict[int, PrefixSection]],  # k -> {prefix_len -> PrefixSection}
    targets: List[int],
    k_select: KSelect,
) -> Dict[int, AggCell]:
    """
    Optimal-k selection per prefix_len by maximizing k_select.
    """
    all_prefix = sorted({pl for sec in k_sections.values() for pl in sec.keys()})
    out: Dict[int, AggCell] = {}

    for target in targets:
        chain_starts = build_chain_starts(all_prefix, target)
        if chain_starts is None:
            continue

        # eligible runs: union across k at each pl, then intersection across pl
        eligible: Optional[Set[int]] = None
        for pl in chain_starts:
            ids_pl: Set[int] = set()
            for k, sec in k_sections.items():
                ps = sec.get(pl)
                if ps is None:
                    continue
                ids_pl |= set(ps.runs.keys())
            if pl == 1024 and 0 in ids_pl:
                ids_pl.remove(0)
            eligible = ids_pl if eligible is None else (eligible & ids_pl)

        if not eligible:
            continue

        lm, lp, ts, al = [], [], [], []
        for rid in sorted(eligible):
            chosen_by_prefix: Dict[int, RunStats] = {}
            ok = True
            for pl in chain_starts:
                best_rs = None
                best_sc = -1e300
                for k, sec in k_sections.items():
                    ps = sec.get(pl)
                    if ps is None:
                        continue
                    rs = ps.runs.get(rid)
                    if rs is None or (pl == 1024 and rs.run_id == 0):
                        continue
                    sc = score_of(rs, k_select)
                    if sc > best_sc:
                        best_sc = sc
                        best_rs = rs
                if best_rs is None:
                    ok = False
                    break
                chosen_by_prefix[pl] = best_rs

            if not ok:
                continue

            agg = aggregate_one_run(chain_starts, target, chosen_by_prefix)
            if agg is None:
                continue
            a, b, c, d = agg
            lm.append(a); lp.append(b); ts.append(c); al.append(d)

        ms = mean_std(lm); ps = mean_std(lp); th = mean_std(ts); ac = mean_std(al)
        if ms and ps and th and ac:
            out[target] = (ms, ps, th, ac)

    return out


def compute_cells_optimal_k_late_spec(
    k_sections: Dict[int, Dict[int, PrefixSection]],
    standard_sections: Dict[int, PrefixSection],
    targets: List[int],
    k_select: KSelect,
) -> Dict[int, AggCell]:
    """
    Late speculation:
      - for each prefix_len, pick best spec-k
      - stay on Standard until spec beats standard (by k_select)
      - once switched, never go back
    Uses floor-mapping from spec prefix_len to closest standard prefix_len <= spec prefix_len.
    """
    all_prefix = sorted({pl for sec in k_sections.values() for pl in sec.keys()})
    std_prefix_sorted = sorted(standard_sections.keys())

    def map_to_std_floor(pl: int) -> Optional[int]:
        lo, hi = 0, len(std_prefix_sorted) - 1
        ans = None
        while lo <= hi:
            mid = (lo + hi) // 2
            if std_prefix_sorted[mid] <= pl:
                ans = std_prefix_sorted[mid]
                lo = mid + 1
            else:
                hi = mid - 1
        return ans

    out: Dict[int, AggCell] = {}
    for target in targets:
        chain_starts = build_chain_starts(all_prefix, target)
        if chain_starts is None:
            continue

        # eligible runs must exist in BOTH standard(std_pl) and spec(pl) for each segment
        eligible: Optional[Set[int]] = None
        for pl in chain_starts:
            std_pl = map_to_std_floor(pl)
            if std_pl is None:
                eligible = set()
                break

            std_ps = standard_sections.get(std_pl)
            if std_ps is None:
                eligible = set()
                break

            std_ids = set(std_ps.runs.keys())
            if std_pl == 1024 and 0 in std_ids:
                std_ids.remove(0)

            spec_ids: Set[int] = set()
            for k, sec in k_sections.items():
                ps = sec.get(pl)
                if ps is None:
                    continue
                spec_ids |= set(ps.runs.keys())
            if pl == 1024 and 0 in spec_ids:
                spec_ids.remove(0)

            ids_pl = std_ids & spec_ids
            eligible = ids_pl if eligible is None else (eligible & ids_pl)

        if not eligible:
            continue

        lm, lp, ts, al = [], [], [], []
        for rid in sorted(eligible):
            chosen_by_prefix: Dict[int, RunStats] = {}
            switched = False
            ok = True

            for pl in chain_starts:
                std_pl = map_to_std_floor(pl)
                if std_pl is None:
                    ok = False
                    break

                std_rs = standard_sections.get(std_pl, PrefixSection(std_pl, {})).runs.get(rid)
                if std_rs is None or (std_pl == 1024 and std_rs.run_id == 0):
                    ok = False
                    break

                best_spec = None
                best_sc = -1e300
                for k, sec in k_sections.items():
                    ps = sec.get(pl)
                    if ps is None:
                        continue
                    rs = ps.runs.get(rid)
                    if rs is None or (pl == 1024 and rs.run_id == 0):
                        continue
                    sc = score_of(rs, k_select)
                    if sc > best_sc:
                        best_sc = sc
                        best_spec = rs

                if best_spec is None:
                    ok = False
                    break

                if not switched:
                    if spec_beats_standard(best_spec, std_rs, k_select):
                        switched = True
                        chosen_by_prefix[pl] = best_spec
                    else:
                        chosen_by_prefix[pl] = std_rs
                else:
                    chosen_by_prefix[pl] = best_spec

            if not ok:
                continue

            agg = aggregate_one_run(chain_starts, target, chosen_by_prefix)
            if agg is None:
                continue
            a, b, c, d = agg
            lm.append(a); lp.append(b); ts.append(c); al.append(d)

        ms = mean_std(lm); ps = mean_std(lp); th = mean_std(ts); ac = mean_std(al)
        if ms and ps and th and ac:
            out[target] = (ms, ps, th, ac)

    return out
