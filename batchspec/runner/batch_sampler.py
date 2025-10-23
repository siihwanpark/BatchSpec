"""Batch sampling utilities for benchmarking."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import random

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizerBase


@dataclass
class _TokSample:
    input_ids: List[int]      # tokenized input
    output_ids: List[int]     # tokenized output
    combo_ids: List[int]      # input + output
    combo_eos_ids: List[int]  # input + output + [eos]
    len_in: int
    len_out: int


class BatchSampler:
    """
    Build [bsz, seq_len] input_ids where the cut is inside the *current* sample's output
    and there are at least `margin` tokens remaining after the cut.

    - If the current sample alone is too short to place the cut with `margin` tail,
      prepend other random samples' (input+output)+EOS until enough prefix length is available.
    - The EOS between prepended context and the current sample is guaranteed (when any prefix exists).
    - We enforce:
        * len(output) >= margin + 1    (at least 1 token of output before cut + margin after)
        * len(input) < seq_len         (cut never inside input)
        * L_pre = max(0, seq_len - (len(input)+len(output)-margin))
          Then final sequence = prefix(L_pre tokens, ending with EOS if L_pre>0) + (input+output)
          Ret = final_sequence[:seq_len]
          This guarantees the cut lies in current output and leaves >= margin tokens.
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

        # EOS id needed for boundaries
        self.eos_id = getattr(self.tok, "eos_token_id", None)
        if self.eos_id is None:
            # fallback if tokenizer lacks eos_token_id
            try:
                self.eos_id = self.tok.convert_tokens_to_ids(self.tok.eos_token)
            except Exception:
                raise ValueError("Tokenizer must provide eos_token_id (or eos_token).")

        # Optional: pre-tokenize for speed
        self._tok_cache: Optional[List[_TokSample]] = None
        self._eligible_idxs: Optional[List[int]] = None
        if pretokenize:
            self._build_cache()

    # -------------------- public API --------------------

    def sample_batch(self) -> torch.LongTensor:
        """Return a [bsz, seq_len] tensor of input_ids."""
        if self._tok_cache is None:
            self._build_cache()

        idxs = [self._pick_main_index() for _ in range(self.bsz)]
        rows = [self._build_row(self._tok_cache[i]) for i in idxs]
        return torch.tensor(rows, dtype=torch.long)

    # -------------------- internals --------------------

    def _build_cache(self) -> None:
        cache: List[_TokSample] = []
        for i in range(len(self.ds)):
            record = self.ds[i]
            raw_in = record["input"]
            raw_out = record["output"]
            # tokenize without adding BOS/EOS — we manage EOS ourselves
            ids_in = self.tok(raw_in, add_special_tokens=False)["input_ids"]
            ids_out = self.tok(raw_out, add_special_tokens=False)["input_ids"]
            combo = ids_in + ids_out
            combo_eos = combo + [self.eos_id]
            cache.append(
                _TokSample(
                    input_ids=ids_in,
                    output_ids=ids_out,
                    combo_ids=combo,
                    combo_eos_ids=combo_eos,
                    len_in=len(ids_in),
                    len_out=len(ids_out),
                )
            )
        self._tok_cache = cache

        # Eligible mains: output >= margin+1, input < seq_len
        self._eligible_idxs = [
            i for i, s in enumerate(cache)
            if (s.len_out >= self.margin + 1) and (s.len_in < self.seq_len)
        ]
        if not self._eligible_idxs:
            raise RuntimeError(
                "No eligible samples found. Need at least one with "
                f"len(output) >= margin+1 ({self.margin+1}) and len(input) < seq_len ({self.seq_len})."
            )

    def _pick_main_index(self) -> int:
        return self.rng.choice(self._eligible_idxs)

    def _random_filler_stream(self, avoid_idx: int):
        """
        Infinite-like generator of filler sample indices (excluding avoid_idx if possible).
        We return a *list we extend* progressively in the caller.
        """
        # if dataset has just 1 eligible filler candidate, it may equal avoid_idx; allow duplicates in that case
        while True:
            j = self.rng.randrange(len(self._tok_cache))
            if j != avoid_idx or len(self._tok_cache) == 1:
                yield j

    def _collect_prefix(self, need_len: int, avoid_idx: int) -> List[int]:
        """
        Collect a prefix of length exactly `need_len` from other samples'
        (input+output)+EOS concatenations. The resulting prefix ends with EOS (need_len>=1).
        Strategy:
          - Keep concatenating random filler samples' combo_eos_ids
          - Take the *last* `need_len` tokens (trim from the left)
          - Because each filler chunk ends with EOS, the concatenation ends with EOS,
            hence the last `need_len` tokens end with EOS.
        """
        if need_len <= 0:
            return []

        buf: List[int] = []
        stream = self._random_filler_stream(avoid_idx)
        for j in stream:
            buf.extend(self._tok_cache[j].combo_eos_ids)
            if len(buf) >= need_len:
                break
        # Trim from the left to exact length
        return buf[-need_len:]

    def _build_row(self, main: _TokSample) -> List[int]:
        """
        Build a single example of length seq_len where:
          cut is inside main.output, and there remain >= margin tokens after cut.
        """
        A = main.len_in
        B = main.len_out
        c = self.seq_len

        # Preconditions (also enforced in eligibility):
        # - A < c (cut after input)
        # - B >= margin + 1 (>=1 before cut, >=margin after cut)
        if not (A < c and B >= self.margin + 1):
            # extremely rare if user disabled pretokenize/eligibility; resample would be better,
            # but for safety we raise.
            raise RuntimeError("Chosen sample cannot satisfy constraints; please keep pretokenize=True.")

        # Required prefix length (including EOS if any prefix exists)
        # L_pre in [ c - (A + B - margin),  c - A - 1 ], choose minimal valid (>=0)
        L_pre = max(0, c - (A + B - self.margin))

        # Collect prefix from other samples if needed
        if L_pre > 0:
            # pick an arbitrary avoid index by reference equality
            avoid_idx = self._tok_cache.index(main)
            prefix = self._collect_prefix(L_pre, avoid_idx=avoid_idx)
        else:
            prefix = []

        # Final sequence = prefix + (input+output)
        full = prefix + main.combo_ids
        # Take first c tokens
        row = full[:c]

        # Sanity checks in debug (comment out for speed if needed)
        # pos_in_main = c - len(prefix)           # index within main.combo where we cut
        # assert 1 <= pos_in_main - A <= B - self.margin, "Cut must be inside output and leave margin tail."

        return row

