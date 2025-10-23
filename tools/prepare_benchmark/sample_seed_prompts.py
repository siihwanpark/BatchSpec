"""
Sample seed prompts from the dataset and save to .jsonl.

Usage:
    python sample_seed_prompts.py --dataset AIME2025 --seed 42 --output_path data/seed_prompts --num_samples 1000
"""


import os
import json
import argparse
import torch
from datasets import load_from_disk, concatenate_datasets


DEFAULT_LOCAL_DATASET_PATH = {
    "AIME2025_I": "/home/mngcuser1/sihwan_workspace/datasets/AIME2025-I/test",
    "AIME2025_II": "/home/mngcuser1/sihwan_workspace/datasets/AIME2025-II/test",
    "LiveMathBench": "/home/mngcuser1/sihwan_workspace/datasets/LiveMathBench/v202505_hard_en/test",
    "LiveCodeBench": "/home/mngcuser1/sihwan_workspace/datasets/code_generation_lite/release_v1/test",
    "CodeForces": "/home/mngcuser1/sihwan_workspace/datasets/codeforces/verifiable-prompts/test",
    "GPQA-Diamond": "/home/mngcuser1/sihwan_workspace/datasets/gpqa/gpqa_diamond/train",
}


PROMPT_KEY_BY_DATASET = {
    "AIME2025": "question",
    "LiveMathBench": "question",
    "LiveCodeBench": "question_content",
    "CodeForces": "prompt",
    "GPQA-Diamond": "Question",
}


def parse_args():
    parser = argparse.ArgumentParser("Sample prompts with replacement and save to .jsonl")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["AIME2025", "LiveMathBench", "LiveCodeBench", "CodeForces", "GPQA-Diamond"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", type=str, default="data/seed_prompts")
    parser.add_argument("--num_samples", type=int, default=1000)
    return parser.parse_args()


def load_local_dataset_by_name(dataset_name):
    if dataset_name == "AIME2025":
        ds_1 = load_from_disk(DEFAULT_LOCAL_DATASET_PATH["AIME2025_I"])
        ds_2 = load_from_disk(DEFAULT_LOCAL_DATASET_PATH["AIME2025_II"])
        ds = concatenate_datasets([ds_1, ds_2])
    else:
        ds = load_from_disk(DEFAULT_LOCAL_DATASET_PATH[dataset_name])
    prompt_key = PROMPT_KEY_BY_DATASET[dataset_name]
    return ds, prompt_key


def sample_indices_with_replacement(n_total: int, n_samples: int, seed: int):
    g = torch.Generator()
    g.manual_seed(seed)
    idx = torch.randint(low=0, high=n_total, size=(n_samples,), generator=g)
    return idx.tolist()


def write_jsonl(path: str, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    args = parse_args()

    ds, prompt_key = load_local_dataset_by_name(args.dataset)
    n = len(ds)
    if n == 0:
        raise RuntimeError(f"Empty dataset for {args.dataset}")

    # 1) sample indices with replacement
    sampled_idx = sample_indices_with_replacement(n, args.num_samples, args.seed)

    # 2) select rows
    sampled = ds.select(sampled_idx)

    # 3) normalize to {"id": i, "prompt": "..."}
    def _to_prompt(example, idx):
        val = example.get(prompt_key, "")
        if not isinstance(val, str):
            val = str(val)
        return {
            "id": int(idx),
            "prompt": val.strip(),
        }

    sampled = sampled.map(_to_prompt, with_indices=True, remove_columns=sampled.column_names)
    out_file = os.path.join(args.output_path, f"{args.dataset}_{args.num_samples}.jsonl")
    write_jsonl(out_file, sampled)

    print(f"[OK] Wrote {len(sampled)} prompts to {out_file}")


if __name__ == "__main__":
    main()