import os
from tqdm import tqdm

import torch.distributed as dist
from transformers import AutoTokenizer

from batchspec.backends.utils import init_dist, setup_seed
from batchspec.backends import (
    StandardEngine, StandaloneEngine, EAGLEChainEngine, MagicDecEngine, MTPEngine
)
from batchspec.profiler import Profiler, register_active_profiler, release_active_profiler
from batchspec.runner import Runner, BatchSampler, load_benchmark_dataset
from batchspec.args import parse_args


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
        dataset = load_benchmark_dataset(model_name=args.model_name, dataset_name=args.dataset)

        # Initialize the engine
        if args.backend == "standard":
            engine = StandardEngine(tokenizer=tokenizer, dtype=args.dtype, device=args.device)
        elif args.backend == "standalone":
            engine = StandaloneEngine(tokenizer=tokenizer, dtype=args.dtype, device=args.device, draft_length=args.draft_length)
        elif args.backend == "eagle":
            engine = EAGLEChainEngine(tokenizer=tokenizer, dtype=args.dtype, device=args.device, draft_length=args.draft_length)
        elif args.backend == "magicdec":
            engine = MagicDecEngine(tokenizer=tokenizer, dtype=args.dtype, device=args.device, draft_length=args.draft_length)
        elif args.backend == "mtp":
            engine = MTPEngine(tokenizer=tokenizer, dtype=args.dtype, device=args.device, draft_length=args.draft_length)
        else:
            raise ValueError(f"Unsupported backend: {args.backend}")

        # Initialize the caches with the maximum prefix length
        print(f"Initializing the Caches with the maximum prefix length: {max(args.prefix_len_list)}")
        args.max_len = max(args.prefix_len_list) + (args.max_gen_len * 3)  # For the initialization of caches

        if args.profiling:
            # Initialize the profiler
            prof = Profiler(runner_args=args)
            register_active_profiler(prof)
            prof.attach_model(engine.model, use_gated_lora=args.backend == "mtp")
            prof.attach_engine(engine)

        batch_sampler = BatchSampler(dataset=dataset, tokenizer=tokenizer, batch_size=args.batch_size, 
            seq_len=max(args.prefix_len_list), margin_before_eos=5 * args.max_gen_len, pretokenize=True, seed=args.seed)
        runner = Runner(args, engine, tokenizer, batch_sampler=batch_sampler)
        runner.setup(process_group)

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
            print(f"[Rank {rank}] Cleaning up distributed process group ...")
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
