"""Dataset loading and engine setup utilities."""

import json
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_from_disk


# TODO : Modify the dataset paths to the local paths and support huggingface dataset
DEFAULT_DATASET_PATH_DICT = {
    "GSM8K": "/home/mngcuser1/sihwan_workspace/datasets/gsm8k/test",
    "MATH-500": "/home/mngcuser1/sihwan_workspace/datasets/MATH-500/test",
    "AIME2025-I": "/home/mngcuser1/sihwan_workspace/datasets/AIME2025-I/test",
    "AIME2025-II": "/home/mngcuser1/sihwan_workspace/datasets/AIME2025-II/test",
    "LiveMathBench": "/home/mngcuser1/sihwan_workspace/datasets/LiveMathBench/v202505_hard_en/test",
    "LiveCodeBench": "/home/mngcuser1/sihwan_workspace/datasets/code_generation_lite/release_v1/test",
    "CodeForces": "/home/mngcuser1/sihwan_workspace/datasets/codeforces/verifiable-prompts/test",
    "GPQA-Diamond": "/home/mngcuser1/sihwan_workspace/datasets/gpqa/gpqa_diamond/train",
}

PROMPT_KEY_DICT = {
    "GSM8K": "question",
    "MATH-500": "problem",
    "AIME2025": "question",
    "LiveMathBench": "question",
    "LiveCodeBench": "question_content",
    "CodeForces": "prompt",
    "GPQA-Diamond": "Question",
}

BENCHMARK_DATASET_BASE_DIR = "/workspace/BatchSpec/benchmark_data/responses"


def load_dataset(tokenizer, dataset_name, num_samples=None, num_questions_in_prompt=1):
    """
    Load and tokenize a dataset for E2E generation.
    
    Args:
        tokenizer: HuggingFace tokenizer
        dataset_name: Name of the dataset (e.g., "GSM8K", "AIME2025")
        num_samples: Number of samples to load (None = all)
        num_questions_in_prompt: Number of questions to combine in a single prompt
        
    Returns:
        Dataset with tokenized input_ids and attention_mask
    """
    if dataset_name not in PROMPT_KEY_DICT:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    if dataset_name == "AIME2025":
        ds = load_from_disk(DEFAULT_DATASET_PATH_DICT[dataset_name+"-I"])
        ds_2 = load_from_disk(DEFAULT_DATASET_PATH_DICT[dataset_name+"-II"])
        ds = concatenate_datasets([ds, ds_2])
    else:
        ds = load_from_disk(DEFAULT_DATASET_PATH_DICT[dataset_name])
    prompt_key = PROMPT_KEY_DICT[dataset_name]
    
    def tokenize_fn(examples):
        def apply_chat_template(tokenizer, text: str) -> str:
            templated = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            return templated
        texts = [apply_chat_template(tokenizer, q) for q in examples[prompt_key]]
        return tokenizer(texts, return_attention_mask=False)
    
    if num_questions_in_prompt > 1:
        all_questions = ds[prompt_key]
        new_prompts = []
        for i in range(0, len(all_questions), num_questions_in_prompt):
            new_prompts.append("\n\n".join(
                [f"Question {j+1}: {q}" for j, q in enumerate(all_questions[i:i+num_questions_in_prompt])]
                )
            )
        ds = Dataset.from_dict({f"{prompt_key}": new_prompts})
    
    ds = ds.map(tokenize_fn, batched=True, remove_columns=[prompt_key])
    ds.set_format(type="torch", columns=["input_ids"])
    
    if num_samples is not None:
        n = len(ds)
        num_samples = int(num_samples)
        
        if num_samples == n:
            return ds
        elif num_samples < n:
            ds = ds.select(range(num_samples))
        else:
            times, rem = divmod(num_samples, n)
            parts = []
            if times > 0:
                parts.extend([ds] * times)         # repeat full copies
            if rem > 0:
                parts.append(ds.select(range(rem))) # tail
            ds = concatenate_datasets(parts)

    return ds


def load_benchmark_dataset(model_name, dataset_name, greedy=False):
    """
    Load a benchmark dataset from the responses directory.
    
    Args:
        tokenizer: HuggingFace tokenizer (unused, kept for consistency)
        dataset_name: Name of the dataset
        greedy: Whether to use greedy decoding (If set to False, use sampling)

    Returns:
        Dataset with input/output fields
    """
    filepath = Path(f"{BENCHMARK_DATASET_BASE_DIR}/{model_name}/{dataset_name}_1000_{'greedy' if greedy else 'sampling'}.json")
    with Path(filepath).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data.get("results", []))

