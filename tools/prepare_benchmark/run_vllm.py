"""
Run vLLM inference on the seed prompts.

Usage:
    python run_vllm.py --model Qwen/Qwen3-8B --tokenizer Qwen/Qwen3-8B \
        --tp_size 8 --gpu_mem_util 0.9 \
        --input_jsonl data/seed_prompts/AIME2025_1000.jsonl --num_samples 1000 \
        --seed 42 --system_prompt "You are a helpful assistant." --prefix_len 1024 \
        --max_model_len 32768 --max_gen_len 30720 --temperature 0.6 --top_p 0.95 --top_k 20 \
        --output_dir data/responses --outfile_suffix responses
"""


import os
import ast
import sys
import time
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


CODE_GENERATION_PREFIX = (
    "You will be given a question (problem specification) and will generate a correct Python program"
    "that matches the specification and passes all tests.\n\nQuestion: {question}\n\n"
)

MMLU_PRO_PREFIX = (
    "You are an knowledge expert, you are supposed to answer the multi-choice question to derive your final answer as `The answer is ...`."
    "Q: {question}\n{options_str}\n"
)


def parse_args():
    p = argparse.ArgumentParser("vLLM JSONL Inference (multi-GPU, chat-template)")

    # Model / tokenizer
    p.add_argument("--model", type=str, required=True,
                   help="HuggingFace model name or local path (e.g., Qwen/Qwen3-8B).")
    p.add_argument("--tokenizer", type=str, default=None,
                   help="Tokenizer name/path. Defaults to --model.")
    p.add_argument("--tp_size", type=int, default=8,
                   help="vLLM tensor_parallel_size. Set to number of GPUs to use.")
    p.add_argument("--gpu_mem_util", type=float, default=0.9,
                   help="vLLM gpu_memory_utilization (0~1).")

    # Data
    p.add_argument("--input_jsonl", type=str, default="data/seed_prompts/AIME2025_1000.jsonl",
                   help="JSONL file with lines like {'prompt': '...'}")
    p.add_argument("--num_samples", type=int, default=None,
                   help="Limit number of prompts (None = all).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for sampling `num_samples`.")

    # Prompt shaping
    p.add_argument("--system_prompt", type=str, default=None,
                   help="Optional system prompt when using chat template.")
    p.add_argument("--prefix_len", type=int, default=None,
                   help="If set, truncate prompt to this many tokens before generation.")

    # Generation
    p.add_argument("--max_model_len", type=int, default=32768,
                   help="vLLM max model length (KV cache bound).")
    p.add_argument("--max_gen_len", type=int, default=30720,
                   help="Max new tokens.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=-1)

    # Output
    p.add_argument("--output_dir", type=str, default="data/responses")
    p.add_argument("--outfile_suffix", type=str, default="responses")

    args = p.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    return args


def read_prompts(
    jsonl_path: str,
    num_samples: Optional[int] = None,
    seed: Optional[int] = None,
    streaming_sample: bool = False,
) -> List[Dict[str, Any]]:
    """
    Read prompts from a JSONL file.
    - If num_samples is None: return all prompts (no sampling).
    - If num_samples is set:
        * streaming_sample=True  -> reservoir sampling (single pass, low memory)
        * streaming_sample=False -> load all then random.sample

    Sampling is reproducible with `seed`.
    """
    rng = random.Random(seed) if seed is not None else random

    def _normalize(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if "prompt" not in obj:
            if "input" in obj and isinstance(obj["input"], str):
                obj["prompt"] = obj["input"]
            else:
                return None
        return obj

    # Case 1: no sampling -> return all
    if num_samples is None:
        prompts: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                obj = _normalize(obj)
                if obj is None:
                    continue
                prompts.append(obj)
        return prompts

    # Case 2: sampling enabled
    if not streaming_sample:
        # Load-all + sample
        pool: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                obj = _normalize(obj)
                if obj is None:
                    continue
                pool.append(obj)
        k = min(num_samples, len(pool))
        return rng.sample(pool, k)

    # Case 3: reservoir sampling (single pass)
    reservoir: List[Dict[str, Any]] = []
    seen = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            obj = _normalize(obj)
            if obj is None:
                continue

            seen += 1
            if len(reservoir) < num_samples:
                reservoir.append(obj)
            else:
                # pick a random index in [0, seen-1]; if it's inside reservoir, replace
                j = rng.randint(0, seen - 1)
                if j < num_samples:
                    reservoir[j] = obj

    return reservoir


def form_options(options: str):
    options = ast.literal_eval(options)
    option_str = 'Options are:\n'
    opts = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    for opt, o in zip(options, opts):
        option_str += f'({o}): {opt}' + '\n'
    return option_str


def apply_chat_template(tokenizer, text: str, options: Optional[list], system_prompt: Optional[str], enable_thinking: bool, code_generation: bool=False, general_task: bool=False) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    if code_generation:
        messages.append({"role": "user", "content": CODE_GENERATION_PREFIX.format(question=text)})
    elif general_task:
        messages.append({"role": "user", "content": MMLU_PRO_PREFIX.format(question=text, options_str=options)})
    else:
        messages.append({"role": "user", "content": text})

    # Qwen3 supports enable_thinking flag; others will just ignore extra kw.
    # Ref: Qwen docs about apply_chat_template & thinking mode.
    try:
        templated = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        # Some tokenizers don't accept enable_thinking kwarg
        templated = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return templated


def truncate_by_tokens(tokenizer, text: str, max_tokens: int) -> str:
    if max_tokens is None:
        return text
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    ids = ids[:max_tokens]
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


class VLLMRunner:
    def __init__(self, args):
        self.args = args
        print(f"[Init] Loading tokenizer: {args.tokenizer}")
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, padding_side="right", trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[Init] Loading vLLM model on {args.tp_size} GPUs: {args.model}")
        self.llm = LLM(
            model=args.model,
            tokenizer=args.tokenizer,
            tensor_parallel_size=args.tp_size,          # multi-GPU (tensor parallel)
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_mem_util,  # KV cache sizing
            trust_remote_code=True,
        )

        self.sampling = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_gen_len,
        )

    def build_inputs(self, records: List[Dict[str, Any]]) -> List[str]:
        general = ("MMLU" in self.args.input_jsonl) or ("SuperGPQA" in self.args.input_jsonl)
        code = "code" in self.args.input_jsonl.lower()

        inputs = []
        for rec in records:
            prompt = apply_chat_template(
                tokenizer=self.tokenizer,
                text=rec["prompt"],
                options=form_options(rec["options"]) if general else None,
                system_prompt=self.args.system_prompt,
                enable_thinking=True,
                code_generation=code,
                general_task=general,
            )
            if self.args.prefix_len:
                prompt = truncate_by_tokens(self.tokenizer, prompt, self.args.prefix_len)
            inputs.append(prompt)
        return inputs

    def run(self):
        t0 = time.time()
        records = read_prompts(self.args.input_jsonl, self.args.num_samples, self.args.seed)
        print(f"[Data] Loaded {len(records)} prompts from {self.args.input_jsonl}")

        inputs = self.build_inputs(records)
        print(f"[Gen] Generating with vLLM... (batching handled internally)")

        # vLLM generate with continuous batching
        outputs = self.llm.generate(inputs, self.sampling)

        results = []
        total_prompt_tokens = 0
        total_output_tokens = 0

        for i, out in enumerate(tqdm(outputs, desc="Collecting outputs")):
            # vLLM returns one or more candidates per prompt; we take the first
            cand = out.outputs[0]
            input_tokens = len(out.prompt_token_ids)
            output_tokens = len(cand.token_ids)
            total_prompt_tokens += input_tokens
            total_output_tokens += output_tokens

            item = {
                "input": inputs[i],
                "output": cand.text,
            }
            results.append(item)

        t1 = time.time()
        self.save(results, t1 - t0, total_prompt_tokens, total_output_tokens)

    def save(self, results: List[Dict[str, Any]], elapsed: float, ptoks: int, otoks: int):
        Path(self.args.output_dir).mkdir(parents=True, exist_ok=True)
        output_filename = f"{self.args.input_jsonl.split('/')[-1].split('.')[0]}_{self.args.outfile_suffix}.json"
        outpath = Path(self.args.output_dir) / output_filename

        payload = {
            "config": {
                "model": self.args.model,
                "tokenizer": self.args.tokenizer,
                "tp_size": self.args.tp_size,
                "gpu_memory_util": self.args.gpu_mem_util,
                "max_model_len": self.args.max_model_len,
                "prefix_len": self.args.prefix_len,
                "max_gen_len": self.args.max_gen_len,
                "temperature": self.args.temperature,
                "top_p": self.args.top_p,
                "top_k": self.args.top_k,
                "use_chat_template": True,
                "enable_thinking": True,
                "system_prompt": self.args.system_prompt,
                "input_jsonl": self.args.input_jsonl,
                "num_samples": self.args.num_samples,
            },
            "runtime": {
                "elapsed_sec": elapsed,
                "num_samples": len(results),
                "prompt_tokens": ptoks,
                "output_tokens": otoks,
                "total_tokens": ptoks + otoks,
                "samples_per_sec": (len(results) / elapsed) if elapsed > 0 else None,
                "tok_per_sec": ((ptoks + otoks) / elapsed) if elapsed > 0 else None,
            },
            "results": results,
        }

        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        rt = payload["runtime"]
        print("\n=== Summary ===")
        print(f"Saved: {outpath}")
        print(f"Samples: {rt['num_samples']}, time: {rt['elapsed_sec']:.2f}s")
        print(f"Tokens: {rt['total_tokens']} (prompt {rt['prompt_tokens']}, output {rt['output_tokens']})")
        print(f"Throughput: {rt['samples_per_sec']:.2f} samples/s, {rt['tok_per_sec']:.2f} tok/s")


def main():
    args = parse_args()
    runner = VLLMRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
