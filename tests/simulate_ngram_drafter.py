import json
from typing import List, Dict, Any
from tqdm import tqdm

from datasets import Dataset
from transformers import AutoTokenizer
import numpy as np

# DATASET_FILE = "/workspace/BatchSpec/benchmark_data/responses/DeepSeek-R1-Distill-Llama-8B/AIME2025_1000_sampling.json"
# TOKENIZER_NAME_OR_PATH = "/workspace/checkpoints/DeepSeek-R1-Distill-Llama-8B"

DATASET_FILE = "/workspace/BatchSpec/benchmark_data/responses/Qwen3-8B/AIME2025_1000_sampling.json"
TOKENIZER_NAME_OR_PATH = "/workspace/checkpoints/Qwen3-0.6B"


def load_results_dataset(file_path: str) -> Dataset:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data.get("results", []))


def simulate_dataset_mean_accept(
    prefix_len_list: List[int],
    draft_length: int = 10,
    max_ngram_size: int = 3,
    max_samples: int | None = None,
):
    ds = load_results_dataset(DATASET_FILE).shuffle(seed=42)
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME_OR_PATH, use_fast=True)

    prefix_len_list = sorted(prefix_len_list)
    N = len(ds) if max_samples is None else min(len(ds), max_samples)

    per_p_values = {p: [] for p in prefix_len_list}
    per_p_counts = {p: 0 for p in prefix_len_list}
    
    global_mals = []
    global_steps = []
    global_accepts = []
    
    for i in tqdm(range(N), desc="Simulating PLD...", total=N):
        ex = ds[i]
        prompt = ex["input"]
        resp = ex["output"]

        prompt_ids = tok(prompt, add_special_tokens=False).input_ids
        resp_ids = tok(resp, add_special_tokens=False).input_ids

        Lp, Lr = len(prompt_ids), len(resp_ids)
        eligible = [p for p in prefix_len_list if (Lp < p <= Lp + Lr)]
        r = simulate_oracle_acceptance(
            prompt_ids=prompt_ids,
            generated_ids=resp_ids,
            prefix_len_list=eligible,
            draft_length=draft_length,
            max_ngram_size=max_ngram_size,
        )

        if r["final_steps"] > 0:
            global_mals.append(r["final_mal"])
            global_steps.append(r["final_steps"])
            global_accepts.append(r["final_total_accepted"])

        for p in eligible:
            if p in r["mal_at"]:
                per_p_values[p].append(r["mal_at"][p])
                per_p_counts[p] += 1

    summary: Dict[str, Any] = {
        "N_total_samples": N,
        "draft_length": draft_length,
        "max_ngram_size": max_ngram_size,
        "prefix_len_list": prefix_len_list,

        "included_counts_by_prefix": per_p_counts,

        "mean_MAL_by_prefix": {},
        "p50_MAL_by_prefix": {},
        "p90_MAL_by_prefix": {},

        "N_effective_global": len(global_mals),
        "global_mean_MAL": float(np.mean(global_mals)) if global_mals else None,
        "global_median_MAL": float(np.median(global_mals)) if global_mals else None,
        "global_p90_MAL": float(np.quantile(np.array(global_mals, dtype=np.float64), 0.9)) if global_mals else None,
        "global_token_weighted_MAL": (float(sum(global_accepts) / sum(global_steps)) if sum(global_steps) > 0 else None),
    }

    for p in prefix_len_list:
        vals = per_p_values[p]
        if len(vals) == 0:
            summary["mean_MAL_by_prefix"][p] = None
            summary["p50_MAL_by_prefix"][p] = None
            summary["p90_MAL_by_prefix"][p] = None
            continue

        arr = np.array(vals, dtype=np.float64)
        summary["mean_MAL_by_prefix"][p] = float(arr.mean())
        summary["p50_MAL_by_prefix"][p] = float(np.median(arr))
        summary["p90_MAL_by_prefix"][p] = float(np.quantile(arr, 0.9))

    return summary


def draft_lookup_lastmatch(
    ctx: List[int],
    draft_length: int,
    max_ngram_size: int,
) -> List[int]:
    """
    ctx: prompt + generated tokens (no padding)
    return: draft_length length of candidate tokens (empty list if none)
    rules:
      - find suffix n-gram (n=max..1)
      - select the "most recent" (last) occurrence of suffix in prefix
      - exclude self-match (suffix itself in the last window)
      - and the position after the selected position must have draft_length tokens actually exist
    """
    L = len(ctx)
    K = draft_length
    for n in range(min(max_ngram_size, L), 0, -1):
        if L < n + K:
            continue  # suffix is possible, but cannot draw K continuation tokens

        suffix = ctx[L - n : L]  # last n tokens

        # "exclude self-match" to allow window start i from 0..(L-n-1)
        last_i = -1
        for i in range(0, L - n):
            if ctx[i : i + n] == suffix:
                # continuation must exist: i+n+K <= L
                if i + n + K <= L:
                    last_i = i

        if last_i >= 0:
            start = last_i + n
            return ctx[start : start + K]

    return []


def simulate_oracle_acceptance(
    prompt_ids: List[int],
    generated_ids: List[int],
    prefix_len_list: List[int],
    draft_length: int = 10,
    max_ngram_size: int = 3,
) -> dict:
    """
    - generation starts from the first token of the response (t=0)
    - lookup is performed in the prompt+generated(prefix) context
    - accepted length is obtained by comparing generated_ids[t:] and draft
    """
    Lp = len(prompt_ids)
    Lr = len(generated_ids)

    prefix_len_list = sorted(prefix_len_list)
    mal_at: Dict[int, float] = {}

    t_resp = 0
    cum_steps = 0
    cum_accepted = 0
    cp_idx = 0

    def cur_mal() -> float:
        return (cum_accepted / cum_steps) if cum_steps > 0 else 0.0

    while t_resp < Lr:
        total_len = Lp + t_resp
        while cp_idx < len(prefix_len_list) and total_len >= prefix_len_list[cp_idx]:
            p = prefix_len_list[cp_idx]
            mal_at[p] = cur_mal()
            cp_idx += 1

        ctx = prompt_ids + generated_ids[:t_resp]
        draft = draft_lookup_lastmatch(ctx, draft_length, max_ngram_size)

        gt = generated_ids[t_resp : t_resp + draft_length]
        m = 0
        max_check = min(len(draft), len(gt))
        while m < max_check and draft[m] == gt[m]:
            m += 1

        cum_steps += 1
        cum_accepted += m + 1
        t_resp = min(Lr, t_resp + m + 1)

    return {
        "mal_at": mal_at,
        "final_mal": cur_mal(),
        "final_steps": cum_steps,
        "final_total_accepted": cum_accepted,
        "prompt_len": Lp,
        "generated_len": Lr,
    }


if __name__ == "__main__":
    prefix_len_list = [1024, 2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576, 26624, 28672]
    stats = simulate_dataset_mean_accept(prefix_len_list=prefix_len_list, draft_length=10, max_ngram_size=3, max_samples=10)
    print(stats["global_token_weighted_MAL"])
    print(stats["mean_MAL_by_prefix"])