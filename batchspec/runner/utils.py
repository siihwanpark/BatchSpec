"""Dataset loading and runner utilities."""

import os
import ast
import json

from pathlib import Path
from typing import Union

from datasets import Dataset, load_dataset, concatenate_datasets


PROMPT_KEY_DICT = {
    "GSM8K": "question",
    "MATH-500": "problem",
    "AIME2025": "question",
    "LiveMathBench": "question",
    "LiveCodeBench": "question_content",
    "CodeForces": "prompt",
    "GPQA-Diamond": "Question",
    "MMLU-Pro": "question",
    "SuperGPQA": "question",
}

SYSTEM_PROMPT = "You are a helpful assistant."
CODE_GENERATION_PREFIX = (
    "You will be given a question (problem specification) and will generate a correct Python program"
    "that matches the specification and passes all tests.\n\nQuestion: {question}\n\n"
)
MMLU_PRO_PREFIX = (
    "You are an knowledge expert, you are supposed to answer the multi-choice question to derive your final answer as `The answer is ...`."
    "Q: {question}\n{options_str}\n"
)

BENCHMARK_DATASET_BASE_DIR = "/workspace/BatchSpec/benchmark_data/responses"

def load_hf_dataset(tokenizer, dataset_name, num_questions_in_prompt=1):
    """
    Load and tokenize a dataset for continuous generation.
    
    Args:
        tokenizer: HuggingFace tokenizer
        dataset_name: Name of the dataset (e.g., "GSM8K", "AIME2025")
        num_questions_in_prompt: Number of questions to combine in a single prompt
        
    Returns:
        Dataset with tokenized input_ids and attention_mask
    """
    if dataset_name not in PROMPT_KEY_DICT:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    if dataset_name == "GSM8K":
        ds = load_dataset("openai/gsm8k", split="test")
    elif dataset_name == "MATH-500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    elif dataset_name == "AIME2025":
        ds_1 = load_dataset("opencompass/AIME2025", "AIME2025-I", split="test")
        ds_2 = load_dataset("opencompass/AIME2025", "AIME2025-II", split="test")
        ds = concatenate_datasets([ds_1, ds_2])
    elif dataset_name == "LiveMathBench":
        ds = load_dataset("opencompass/LiveMathBench", "v202505_hard_en", split="test")
    elif dataset_name == "LiveCodeBench":
        ds = load_dataset("livecodebench/code_generation_lite", revision="release_v1", split="test")
    elif dataset_name == "CodeForces":
        ds = load_dataset("open-r1/codeforces", "verifiable-prompts", split="test")
    elif dataset_name == "GPQA-Diamond":
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    elif dataset_name == "MMLU-Pro":
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    elif dataset_name == "SuperGPQA":
        ds = load_dataset("m-a-p/SuperGPQA", split="train")
    else:
        # Unreachable
        raise ValueError(f"Unsupported dataset: {dataset_name}")
        
    prompt_key = PROMPT_KEY_DICT[dataset_name]

    general = ("MMLU" in dataset_name) or ("SuperGPQA" in dataset_name)
    code = "code" in dataset_name.lower()

    def form_options(options: Union[str, list]):
        if isinstance(options, str):
            options = ast.literal_eval(options)
        option_str = 'Options are:\n'
        opts = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        for opt, o in zip(options, opts):
            option_str += f'({o}): {opt}' + '\n'
        return option_str
    
    def tokenize_fn(examples):
        def apply_chat_template(tokenizer, text: str) -> str:
            templated = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text}
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            return templated
        
        if code:
            texts = [apply_chat_template(tokenizer, CODE_GENERATION_PREFIX.format(question=q)) for q in examples[prompt_key]]
        elif general:
            texts = [apply_chat_template(tokenizer, MMLU_PRO_PREFIX.format(question=q, options_str=form_options(opt))) for q, opt in zip(examples[prompt_key], examples["options"])]
        else:
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


def check_path_and_broadcast(file_path: str, rank: int, src_rank: int = 0) -> bool:
    """
    Check if the `file_path` exists, and synchronize the decision across all ranks when TP is enabled.
    
    - If TP is enabled (dist initialized), only `src_rank` checks the filesystem and broadcasts the decision.
    - If TP is disabled, each process checks locally.
    """
    import torch.distributed as dist
    if not dist.is_initialized():
        exists = os.path.exists(file_path)
        if exists:
            print(f"[Rank {rank}] Profiler report already exists: {file_path}", flush=True)
        return exists

    exists = rank == src_rank and os.path.exists(file_path)
    obj = [exists]
    dist.broadcast_object_list(obj, src=src_rank)
    exists = obj[0]

    if exists and rank == src_rank:
        print(f"[Rank {rank}] Profiler report already exists: {file_path}", flush=True)
    return exists