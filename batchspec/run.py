import os
from tqdm import tqdm

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from batchspec.backends.utils import init_dist, setup_seed
from batchspec.backends import (
    StandardEngine, StandaloneEngine, NGramDraftEngine,
    EAGLEChainEngine, MagicDecEngine, MTPEngine
)
from batchspec.profiler import Profiler, register_active_profiler, release_active_profiler
from batchspec.runner import Runner, BatchSampler, StrictPrefixBatchSampler, load_benchmark_dataset
from batchspec.args import parse_args

   
def check_path_and_broadcast(file_path: str, rank: int, src_rank: int = 0) -> bool:
    """
    Check if the `file_path` exists, and synchronize the decision across all ranks when TP is enabled.
    
    - If TP is enabled (dist initialized), only `src_rank` checks the filesystem and broadcasts the decision.
    - If TP is disabled, each process checks locally.
    """
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


def main():
    args = parse_args()
    
    rank, process_group = 0, None
    use_tp = len(args.rank_group) > 1 if args.rank_group else False
    if use_tp:
        rank, process_group = init_dist()
        
        if rank != args.rank_group[0]:
            import sys
            os.makedirs("logs/system_logs", exist_ok=True)
            log_file = open(f"logs/system_logs/rank_{rank}.log", 'w', buffering=1)
            sys.stdout = log_file
            sys.stderr = log_file
    
    try:
        setup_seed(args.seed)
        print(f"Initializing benchmark ...")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, padding_side="right", local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Prepare the dataset
        dataset = load_benchmark_dataset(
            model_name=args.model_name,
            dataset_name=args.dataset,
            greedy=args.temperature == 0.0,
        )

        # Initialize the engine
        if args.backend == "standard":
            engine = StandardEngine(tokenizer=tokenizer, dtype=args.dtype, device=args.device)
        elif args.backend == "standalone":
            engine = StandaloneEngine(tokenizer=tokenizer, dtype=args.dtype, device=args.device, draft_length=args.draft_length)
        elif args.backend == "ngram":
            engine = NGramDraftEngine(tokenizer=tokenizer, dtype=args.dtype, device=args.device, draft_length=args.draft_length, max_ngram_size=args.max_ngram_size)
        elif args.backend == "eagle":
            engine = EAGLEChainEngine(tokenizer=tokenizer, dtype=args.dtype, device=args.device, draft_length=args.draft_length)
        elif args.backend == "magicdec":
            engine = MagicDecEngine(tokenizer=tokenizer, dtype=args.dtype, device=args.device, draft_length=args.draft_length)
        elif args.backend == "mtp":
            engine = MTPEngine(tokenizer=tokenizer, dtype=args.dtype, device=args.device, draft_length=args.draft_length)
        else:
            raise ValueError(f"Unsupported backend: {args.backend}")

        if args.profiling:
            # Initialize the profiler
            prof = Profiler(runner_args=args)

            # Skip if the profiler report already exists
            src_rank = args.rank_group[0] if (args.rank_group and len(args.rank_group) > 0) else 0
            if check_path_and_broadcast(os.path.join(prof.out_dir, "report.md"), rank, src_rank):
                return

        # Initialize the caches with the maximum prefix length
        max_gen_per_step = 1 if args.backend == "standard" else args.draft_length + 1
        args.max_len = max(args.prefix_len_list) + (args.max_gen_len * max_gen_per_step)  # For the initialization of caches
        print(f"Initializing the Caches with the maximum prefix length: {max(args.prefix_len_list)} + upper bound of the generation length: {args.max_gen_len * max_gen_per_step}")
        
        batch_sampler = BatchSampler(dataset=dataset, tokenizer=tokenizer, batch_size=args.batch_size, 
            seq_len=max(args.prefix_len_list), margin_before_eos=args.max_gen_len * max_gen_per_step, pretokenize=True, seed=args.seed)
        # batch_sampler = StrictPrefixBatchSampler(dataset=dataset, tokenizer=tokenizer, batch_size=args.batch_size, 
        #     seq_len=max(args.prefix_len_list), margin_before_eos=3 * args.max_gen_len, pretokenize=True, seed=args.seed)
        
        runner = Runner(args, engine, tokenizer, batch_sampler=batch_sampler)
        runner.setup(process_group)

        if args.profiling:
            register_active_profiler(prof)
            prof.attach_model(engine.model, use_gated_lora=args.backend == "mtp")
            prof.attach_engine(engine)

        for run_idx in tqdm(range(args.num_total_runs), total=args.num_total_runs, desc="Running benchmark"):
            print("\n" + "="*50 + f" Run {run_idx} Start " + "="*50)
            runner.run()
        
        if args.profiling:
            # Save the profiling results and unregister the profiler
            prof.save_config()
            prof.save_all()
            release_active_profiler()

    except Exception as e:
        print(f"[Rank {rank}] Exception occurred: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if use_tp and dist.is_initialized():
            try:
                dist.barrier()
            finally:
                print(f"[Rank {rank}] Cleaning up distributed process group ...", flush=True)
                dist.destroy_process_group()


if __name__ == "__main__":
    main()
