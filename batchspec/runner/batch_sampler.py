"""
Batch sampling utilities for benchmarking.

Provides:
  1) BatchSampler:
     - Builds [bsz, seq_len] rows such that the cut at seq_len lies inside the *main* sample's output
       and at least `margin_before_eos` output tokens remain after the cut.
     - If needed, prepends filler context from other samples' (input+output)+EOS streams.
     - Ensures an EOS boundary between filler prefix and main sample when prefix exists.

  2) StrictPrefixBatchSampler (fixed prefix_len):
     - Builds [bsz, prefix_len] rows by taking the first `prefix_len` tokens of a SINGLE sample's
       (input+output).
     - Eligible samples: len(input+output) >= prefix_len.
     - prefix_len is fixed at initialization (simpler + faster than dynamic prefix_len).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import random

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizerBase


# ----------------------------
# shared token cache structure
# ----------------------------

@dataclass
class _TokSample:
    input_ids: List[int]
    output_ids: List[int]
    combo_ids: List[int]       # input + output
    combo_eos_ids: List[int]   # input + output + [eos]
    len_in: int
    len_out: int
    len_combo: int


def _get_eos_id(tok: PreTrainedTokenizerBase) -> int:
    eos_id = getattr(tok, "eos_token_id", None)
    if eos_id is not None:
        return int(eos_id)
    eos_tok = getattr(tok, "eos_token", None)
    if eos_tok is None:
        raise ValueError("Tokenizer must provide eos_token_id (or eos_token).")
    eos_id = tok.convert_tokens_to_ids(eos_tok)
    if eos_id is None:
        raise ValueError("Tokenizer eos_token could not be converted to an id.")
    return int(eos_id)


def _build_tok_cache(
    ds: Dataset,
    tok: PreTrainedTokenizerBase,
    eos_id: int,
) -> List[_TokSample]:
    cache: List[_TokSample] = []
    for i in range(len(ds)):
        r = ds[i]
        raw_in = r["input"]
        raw_out = r["output"]

        ids_in = tok(raw_in, add_special_tokens=False)["input_ids"]
        ids_out = tok(raw_out, add_special_tokens=False)["input_ids"]

        combo = ids_in + ids_out
        cache.append(
            _TokSample(
                input_ids=ids_in,
                output_ids=ids_out,
                combo_ids=combo,
                combo_eos_ids=combo + [eos_id],
                len_in=len(ids_in),
                len_out=len(ids_out),
                len_combo=len(combo),
            )
        )
    return cache


# ============================================================
# 1) BatchSampler: cut is inside main output + margin remaining
# ============================================================

class BatchSampler:
    """
    Build [bsz, seq_len] input_ids where the cut is inside the *current* sample's output
    and there are at least `margin_before_eos` tokens remaining after the cut.

    Main constraints:
      - A = len(input)  < seq_len     (cut never inside input)
      - B = len(output) >= margin+1   (>=1 token before cut, >=margin tokens after)

    Construction:
      L_pre = max(0, seq_len - (A + B - margin))
      row = (prefix of length L_pre, ending with EOS if L_pre>0) + (input+output)
      row = row[:seq_len]
    """

    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        seq_len: int,
        margin_before_eos: int,
        batch_size: int,
        seed: int = 0,
        pretokenize: bool = True,
    ):
        assert seq_len > 0 and margin_before_eos >= 0 and batch_size > 0
        self.ds = dataset
        self.tok = tokenizer
        self.seq_len = int(seq_len)
        self.margin = int(margin_before_eos)
        self.bsz = int(batch_size)
        self.rng = random.Random(seed)
        self.eos_id = _get_eos_id(self.tok)

        self._tok_cache: Optional[List[_TokSample]] = None
        self._eligible_main_idxs: Optional[List[int]] = None

        if pretokenize:
            self._build_cache()

    def sample_batch(self) -> torch.LongTensor:
        if self._tok_cache is None:
            self._build_cache()

        idxs = [self.rng.choice(self._eligible_main_idxs) for _ in range(self.bsz)]
        rows = [self._build_row(main_idx=i) for i in idxs]
        return torch.tensor(rows, dtype=torch.long)

    # -------------------- internals --------------------

    def _build_cache(self) -> None:
        cache = _build_tok_cache(self.ds, self.tok, self.eos_id)
        self._tok_cache = cache

        self._eligible_main_idxs = [
            i for i, s in enumerate(cache)
            if (s.len_out >= self.margin + 1) and (s.len_in < self.seq_len)
        ]
        if not self._eligible_main_idxs:
            raise RuntimeError(
                "No eligible main samples found. Need at least one with "
                f"len(output) >= margin+1 ({self.margin+1}) and len(input) < seq_len ({self.seq_len})."
            )

    def _random_filler_index(self, avoid_idx: int) -> int:
        n = len(self._tok_cache)
        if n == 1:
            return 0
        j = self.rng.randrange(n - 1)
        return j if j < avoid_idx else j + 1

    def _collect_prefix(self, need_len: int, avoid_idx: int) -> List[int]:
        if need_len <= 0:
            return []

        buf: List[int] = []
        while len(buf) < need_len:
            j = self._random_filler_index(avoid_idx)
            buf.extend(self._tok_cache[j].combo_eos_ids)

        # take the suffix => ends with EOS (since buf ends with EOS)
        return buf[-need_len:]

    def _build_row(self, main_idx: int) -> List[int]:
        main = self._tok_cache[main_idx]
        A, B, c, m = main.len_in, main.len_out, self.seq_len, self.margin

        if not (A < c and B >= m + 1):
            raise RuntimeError("Chosen sample cannot satisfy constraints; please keep pretokenize=True.")

        # L_pre = max(0, c - (A + B - m))
        L_pre = c - (A + B - m)
        if L_pre < 0:
            L_pre = 0

        prefix = self._collect_prefix(L_pre, avoid_idx=main_idx) if L_pre > 0 else []
        row = (prefix + main.combo_ids)[:c]
        return row


# ==========================================================
# 2) StrictPrefixBatchSampler: fixed prefix_len, single sample
# ==========================================================

class StrictPrefixBatchSampler:
    """
    Fixed seq_len version with optional tail margin.

    Build [bsz, seq_len] rows by:
      - selecting samples where len(input+output) >= seq_len + margin
      - randomly sampling `batch_size` of them (WITH replacement)
      - returning the first `seq_len` tokens of (input+output)

    This guarantees at least `margin` tokens remain AFTER the prefix cut.
    """

    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        seq_len: int,
        batch_size: int,
        margin_before_eos: int = 0,
        seed: int = 0,
        pretokenize: bool = True,
        verbose: bool = True,
    ):
        assert seq_len > 0 and batch_size > 0 and margin_before_eos >= 0
        self.ds = dataset
        self.tok = tokenizer
        self.seq_len = int(seq_len)
        self.margin = int(margin_before_eos)
        self.bsz = int(batch_size)
        self.rng = random.Random(seed)
        self.verbose = bool(verbose)

        self.eos_id = _get_eos_id(self.tok)  # unused here, but kept for shared cache format

        self._tok_cache: Optional[List[_TokSample]] = None
        self._eligible_idxs: Optional[List[int]] = None

        if pretokenize:
            self._build_cache()

    def sample_batch(self) -> torch.LongTensor:
        if self._tok_cache is None:
            self._build_cache()

        idxs = [self.rng.choice(self._eligible_idxs) for _ in range(self.bsz)]
        rows = [self._tok_cache[i].combo_ids[: self.seq_len] for i in idxs]
        return torch.tensor(rows, dtype=torch.long)

    def _build_cache(self) -> None:
        cache = _build_tok_cache(self.ds, self.tok, self.eos_id)
        self._tok_cache = cache

        need = self.seq_len + self.margin
        self._eligible_idxs = [i for i, s in enumerate(cache) if s.len_combo >= need]

        if not self._eligible_idxs:
            max_len = max((s.len_combo for s in cache), default=0)
            raise RuntimeError(
                f"No eligible samples found for seq_len={self.seq_len} with margin={self.margin}. "
                f"Need len(input+output) >= {need}, but max is {max_len}."
            )
        elif self.verbose:
            print(
                f"[StrictPrefixBatchSampler] Found {len(self._eligible_idxs)} eligible samples "
                f"for seq_len={self.seq_len} with margin={self.margin} (need >= {need})."
            )