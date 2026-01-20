from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal

import numpy as np

# -------------------------
# Experiment naming / constants
# -------------------------

METHOD_BASE: Dict[str, str] = {
    "standard": "Standard",
    "standalone": "Standalone",
    "ngram": "PLD",
    "eagle": "EAGLE",
    "magicdec": "MagicDec",
    "mtp": "MTP",
}

METHOD_ORDER = ["Standard", "Standalone", "PLD", "EAGLE", "MagicDec", "MTP"]

# exp dir name example: bsz128-tp1-k4-sampling (standard has no -k)
EXP_RE = re.compile(r"^bsz(?P<bsz>\d+)-tp(?P<tp>\d+)(?:-k(?P<k>\d+))?-(?P<mode>sampling)$")

# allow ##/###/... and "prefix_len = 1024" or "prefix_len: 1024"
PREFIX_HDR_RE = re.compile(r"^#{2,6}\s*prefix_len\s*(?:[:=]|\s)\s*(\d+)\s*$", re.IGNORECASE)

# Forced "up to" list, capped by endpoint logic (8192/16384/32768) and per-method constraints
UP_TO_LIST = [2048, 4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768]

KSelect = Literal["throughput", "lat_mean", "lat_p95"]
SummaryMode = Literal["full", "compact"]


# -------------------------
# Data structures
# -------------------------

def split_md_row(line: str) -> List[str]:
    return [p.strip() for p in line.strip().strip("|").split("|")]


@dataclass(frozen=True)
class RunStats:
    """Single run stats at a given prefix_len."""
    run_id: int
    bsz: int
    steps: int
    tokens: int
    latency_mean_ms: float
    latency_p95_ms: float
    tok_s: float
    tok_per_step: float


@dataclass
class PrefixSection:
    """All runs at a single prefix_len."""
    prefix_len: int
    runs: Dict[int, RunStats]


@dataclass(frozen=True)
class ExpKey:
    """
    Identifies one experiment instance:
      model / dataset / method / (bsz,tp,k,mode)
    """
    model: str
    dataset: str
    method_folder: str
    method_base: str
    bsz: int
    tp: int
    k: Optional[int]
    mode: str


# -------------------------
# Small helpers (math / formatting / policies)
# -------------------------

def mean_std(vals: List[float]) -> Optional[Tuple[float, float]]:
    """Population mean/std over vals. Returns None if empty."""
    if not vals:
        return None
    arr = np.asarray(vals, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr, ddof=0))


def fmt_with_ratio(mean: float, std: float, std_mean: Optional[float], fmt: str = "{:.3f}") -> str:
    """Format 'mean ± std (×ratio)' where ratio is against std_mean."""
    if std_mean is None or std_mean <= 0:
        return f"{fmt.format(mean)} ± {fmt.format(std)}"
    ratio = mean / std_mean
    return f"{fmt.format(mean)} ± {fmt.format(std)} (×{ratio:.2f})"


def fmt_plain(mean: float, std: float, fmt: str = "{:.3f}") -> str:
    """Format 'mean ± std'."""
    return f"{fmt.format(mean)} ± {fmt.format(std)}"


def choose_final_endpoint(max_prefix: int) -> int:
    """Round UP max_prefix to nearest among {8192, 16384, 32768}."""
    for t in (8192, 16384, 32768):
        if max_prefix <= t:
            return t
    return 32768


def standalone_maxlen_limit(bsz: int) -> int:
    """Standalone has shorter coverage by design."""
    if bsz <= 128:
        return 16384
    if bsz == 256:
        return 8192
    return 0


def method_max_target(prefix_lens: List[int], method_base: str, bsz: int) -> int:
    """
    Max target allowed for a method, based on measured prefix coverage.
    Prevents extrapolating beyond the intended endpoint.
    """
    if not prefix_lens:
        return 0
    final = choose_final_endpoint(max(prefix_lens))
    if method_base == "Standalone":
        lim = standalone_maxlen_limit(bsz)
        if lim > 0:
            final = min(final, lim)
    return final


def forced_targets_for_group(prefix_lens: List[int], method_base: str, bsz: int) -> List[int]:
    """
    Forced target list (UP_TO_LIST) capped by endpoint logic and method constraints.
    Returns empty if missing prefix_len=1024.
    """
    if not prefix_lens or 1024 not in prefix_lens:
        return []
    final = method_max_target(prefix_lens, method_base, bsz)
    return [t for t in UP_TO_LIST if t <= final]


def build_chain_starts(prefix_lens: List[int], target: int) -> Optional[List[int]]:
    """
    Segment start points for [1024..target] using available prefix lens.
    We synthesize the last segment to 'target' (i.e., last start -> target).
    """
    starts = sorted([pl for pl in prefix_lens if 1024 <= pl < target])
    if not starts or starts[0] != 1024:
        return None
    return starts


def score_of(rs: RunStats, k_select: KSelect) -> float:
    """Larger is better score for selecting k per prefix_len."""
    if k_select == "throughput":
        return rs.tok_s
    if k_select == "lat_mean":
        return -rs.latency_mean_ms
    if k_select == "lat_p95":
        return -rs.latency_p95_ms
    raise ValueError(k_select)


def spec_beats_standard(spec_rs: RunStats, std_rs: RunStats, k_select: KSelect) -> bool:
    """Decision rule for late speculation switch."""
    if k_select == "throughput":
        return spec_rs.tok_s > std_rs.tok_s
    if k_select == "lat_mean":
        return spec_rs.latency_mean_ms < std_rs.latency_mean_ms
    if k_select == "lat_p95":
        return spec_rs.latency_p95_ms < std_rs.latency_p95_ms
    raise ValueError(k_select)
