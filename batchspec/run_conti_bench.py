import os
import sys

from tqdm import tqdm

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from batchspec.backends.utils import init_dist, setup_seed
from batchspec.continuous import (
    StandardContinuousEngine, MTPContinuousEngine,
)
from batchspec.profiler import Profiler, register_active_profiler, release_active_profiler
from batchspec.runner import Runner, BatchSampler, load_benchmark_dataset, check_path_and_broadcast
from batchspec.args import parse_continuous_benchmark_args


def main():
    args = parse_continuous_benchmark_args()
    
    rank, process_group = 0, None
    use_tp = len(args.rank_group) > 1 if args.rank_group else False
    if use_tp:
        rank, process_group = init_dist()
        if rank != args.rank_group[0]:
            os.makedirs("logs/system_logs", exist_ok=True)
            log_file = open(f"logs/system_logs/rank_{rank}.log", 'w', buffering=1)
            sys.stdout = log_file; sys.stderr = log_file
    
    try:
        setup_seed(args.seed)
        print(f"Initializing generation with continuous batching...")
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
            engine = StandardContinuousEngine(tokenizer=tokenizer, dtype=args.dtype, device=args.device)
        elif args.backend == "mtp":
            engine = MTPContinuousEngine(tokenizer=tokenizer, dtype=args.dtype, device=args.device, draft_length=args.draft_length)
        else:
            raise ValueError(f"Unsupported backend: {args.backend}")

        if args.profiling:
            prof = Profiler(runner_args=args)
            if check_path_and_broadcast(os.path.join(prof.out_dir, "report.md"), rank):
                return
        
        # Initialize the batch sampler
        batch_sampler = BatchSampler(dataset=dataset, tokenizer=tokenizer, batch_size=args.batch_size, 
            seq_len=args.max_seq_len, margin_before_eos=1024, pretokenize=True, seed=args.seed)

        # Initialize the runner
        runner = Runner(args, engine, tokenizer, batch_sampler=batch_sampler)
        runner.setup(process_group)
        
        if args.profiling:
            register_active_profiler(prof)
            prof.attach_model(engine.model, use_gated_lora=args.backend == "mtp")
            prof.attach_engine(engine)

        # Setup the experiment configuration
        exp_config = {
            'short_ratio': args.short_ratio,
            'short_target_len': args.short_target_len,
            'long_target_len': args.long_target_len,
        }

        # Run the benchmark
        for run_idx in tqdm(range(args.num_total_runs), total=args.num_total_runs, desc="Running benchmark"):
            print("\n" + "="*50 + f" Run {run_idx} Start " + "="*50)
            runner.run_continuous_benchmark(exp_config=exp_config)
        
        if args.profiling:
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
