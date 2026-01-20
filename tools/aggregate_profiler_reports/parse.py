from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .common import (
    ExpKey, PrefixSection, RunStats,
    METHOD_BASE, EXP_RE, PREFIX_HDR_RE, split_md_row,
)

# -------------------------
# Parsing helpers
# -------------------------

def is_run_table_header(line: str) -> bool:
    """
    Detect per-run markdown table header across old/new formats.

    Expected signals:
      - first column ~ run/run_idx/...
      - contains steps, tokens
      - contains mean(ms), p95, tok/s (or throughput)
    """
    s = line.strip()
    if not s.startswith("|"):
        return False

    cols = [c.strip().lower() for c in split_md_row(s)]
    if len(cols) < 6:
        return False

    first = cols[0]
    if not (first.startswith("run") or first in {"idx", "runid", "run_id"}):
        return False

    if not {"steps", "tokens"}.issubset(set(cols)):
        return False

    has_toks = any(("tok/s" in c) or ("throughput" in c) for c in cols)
    has_p95 = any("p95" in c for c in cols)
    has_mean_ms = any(("mean" in c and "ms" in c) for c in cols)
    return has_toks and has_p95 and has_mean_ms


def parse_report_md(report_path: Path, debug_parse: bool = False) -> Dict[int, PrefixSection]:
    """
    Parse report.md into {prefix_len: PrefixSection}, starting from first prefix_len=1024.

    Returns empty dict on failure / missing required section.
    """
    try:
        text = report_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        if debug_parse:
            print(f"[debug_parse] FAIL read: {report_path}: {e}", file=sys.stderr)
        return {}

    lines = text.splitlines()

    # Find first prefix_len=1024
    start_idx = None
    for i, ln in enumerate(lines):
        m = PREFIX_HDR_RE.match(ln.strip())
        if m and int(m.group(1)) == 1024:
            start_idx = i
            break
    if start_idx is None:
        if debug_parse:
            print(f"[debug_parse] NO prefix_len=1024 section: {report_path}", file=sys.stderr)
        return {}

    sections: Dict[int, PrefixSection] = {}
    i = start_idx
    seen_any_run_table = False

    while i < len(lines):
        m = PREFIX_HDR_RE.match(lines[i].strip())
        if not m:
            i += 1
            continue

        prefix_len = int(m.group(1))
        i += 1

        # advance to run-table header or next prefix
        while i < len(lines) and (not is_run_table_header(lines[i])) and (not PREFIX_HDR_RE.match(lines[i].strip())):
            i += 1

        runs: Dict[int, RunStats] = {}

        if i < len(lines) and is_run_table_header(lines[i]):
            seen_any_run_table = True

            header_cols = [c.strip().lower() for c in split_md_row(lines[i])]
            i += 1

            # skip separator row
            if i < len(lines):
                sep = lines[i].strip()
                if sep.startswith("|") and set(sep) <= set("|-: "):
                    i += 1

            def find_col(pred):
                for idx, c in enumerate(header_cols):
                    if pred(c):
                        return idx
                return None

            idx_run = find_col(lambda c: c.startswith("run"))
            idx_bsz = find_col(lambda c: c == "bsz")
            idx_steps = find_col(lambda c: c == "steps")
            idx_tokens = find_col(lambda c: c == "tokens")
            idx_mean = find_col(lambda c: ("mean" in c and "ms" in c))
            idx_p95 = find_col(lambda c: ("p95" in c and "ms" in c) or c.startswith("p95"))
            idx_toks = find_col(lambda c: ("tok/s" in c) or ("throughput" in c))
            idx_tps = find_col(lambda c: ("tokens/step" in c) or ("mean generated tokens" in c))

            while i < len(lines):
                row = lines[i].strip()
                if PREFIX_HDR_RE.match(row) or (not row.startswith("|")):
                    break

                parts = split_md_row(row)
                try:
                    if not parts or not parts[0].isdigit():
                        i += 1
                        continue

                    run_id = int(parts[idx_run] if idx_run is not None else parts[0])
                    bsz = int(parts[idx_bsz] if idx_bsz is not None else parts[1])
                    steps = int(parts[idx_steps] if idx_steps is not None else parts[2])
                    tokens = int(parts[idx_tokens] if idx_tokens is not None else parts[3])
                    lat_mean = float(parts[idx_mean] if idx_mean is not None else parts[4])
                    lat_p95 = float(parts[idx_p95] if idx_p95 is not None else parts[5])
                    tok_s = float(parts[idx_toks] if idx_toks is not None else parts[6])

                    tok_per_step = float(parts[idx_tps]) if (idx_tps is not None and idx_tps < len(parts)) else float(parts[-1])

                    runs[run_id] = RunStats(
                        run_id=run_id,
                        bsz=bsz,
                        steps=steps,
                        tokens=tokens,
                        latency_mean_ms=lat_mean,
                        latency_p95_ms=lat_p95,
                        tok_s=tok_s,
                        tok_per_step=tok_per_step,
                    )
                except Exception:
                    # ignore malformed lines
                    pass

                i += 1

        sections[prefix_len] = PrefixSection(prefix_len=prefix_len, runs=runs)

    if debug_parse:
        pls = sorted(sections.keys())
        print(
            f"[debug_parse] Parsed {report_path}: prefix_lens={pls}, "
            f"seen_any_run_table={seen_any_run_table}, "
            f"nonempty_sections={sum(1 for pl in pls if sections[pl].runs)}",
            file=sys.stderr,
        )

    return sections


# -------------------------
# Discovery (path -> ExpKey)
# -------------------------

def infer_expkey_from_path(report_path: Path) -> Optional[ExpKey]:
    """
    Expected layout:
      .../{model}/{method}/{dataset}/{exp_dir}/report.md
    exp_dir example:
      bsz128-tp1-k4-sampling
    """
    try:
        exp_dir = report_path.parent
        dataset_dir = exp_dir.parent
        method_dir = dataset_dir.parent
        model_dir = method_dir.parent
        exp_name = exp_dir.name
        dataset = dataset_dir.name
        method_folder = method_dir.name
        model = model_dir.name
    except Exception:
        return None

    m = EXP_RE.match(exp_name)
    if not m:
        return None

    method_base = METHOD_BASE.get(method_folder)
    if method_base is None:
        return None

    bsz = int(m.group("bsz"))
    tp = int(m.group("tp"))
    k = int(m.group("k")) if m.group("k") is not None else None
    mode = m.group("mode")

    # standard has no k; all other methods must have k
    if method_base != "Standard" and k is None:
        return None

    return ExpKey(model, dataset, method_folder, method_base, bsz, tp, k, mode)


def discover_reports(root: Path) -> List[Tuple[ExpKey, Path]]:
    """Find all report.md under root that matches expected experiment layout."""
    out: List[Tuple[ExpKey, Path]] = []
    for rp in root.rglob("report.md"):
        ek = infer_expkey_from_path(rp)
        if ek is not None:
            out.append((ek, rp))
    return out
