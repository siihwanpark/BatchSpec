"""Runner for E2E and benchmark execution."""

import torch
import torch.distributed as dist
from tqdm import tqdm

from batchspec.models import LoRAConfig


class Runner:
    """
    Runner for executing generation loops.
    
    Handles:
    - Batch loading/sampling
    - Calling engine.generate()
    - Output processing and statistics
    """
    
    def __init__(self, args, engine, tokenizer, dataloader=None, batch_sampler=None):
        """
        Initialize the runner.
        
        Args:
            args: Command-line arguments
            engine: Backend engine (StandardEngine or MTPEngine)
            tokenizer: HuggingFace tokenizer
            dataloader: DataLoader for E2E mode (mutually exclusive with batch_sampler)
            batch_sampler: BatchSampler for benchmark mode (mutually exclusive with dataloader)
        """
        self.args = args
        self.engine = engine
        self.dataloader = dataloader
        self.batch_sampler = batch_sampler
        self.device = engine.device
        
        if dataloader is not None and batch_sampler is not None:
            raise ValueError("Provide either dataloader or batch_sampler, not both.")
        self.mode = "e2e" if dataloader is not None else "benchmark"

    def setup(self, process_group):
        """
        Setup the engine with model, caches, and sampling parameters.
        
        Args:
            process_group: Distributed process group for tensor parallelism
        """
        load_params = {
            'model_name': self.args.model_name, 
            'checkpoint_path': self.args.checkpoint_path,
            'use_tp': len(self.args.rank_group) > 1 if self.args.rank_group else False,
            'rank_group': self.args.rank_group, 
            'group': process_group
        }

        if self.args.backend == "mtp":
            assert self.args.lora_checkpoint_path is not None, "LoRA checkpoint path is required for MTP backend"
            load_params.update({
                'lora_checkpoint_path': self.args.lora_checkpoint_path,
                'lora_config': LoRAConfig.from_args(self.args)
            })
        
        self.engine.load_model(**load_params)

        if self.args.compile:
            self.engine.compile()

        prefill_chunk_size = {
            2: 128, 4: 128, 8: 128, 16: 128, 
            32: 128, 64: 64, 128: 16, 256: 8
        }
        cache_params = {
            'max_batch_size': self.args.batch_size, 
            'max_seq_length': self.args.max_len, 
            'page_size': 16,
            'prefill_chunk_size': prefill_chunk_size.get(self.args.batch_size, 128)
        }
        self.engine.setup_caches(**cache_params)
        self.engine.setup_sampling_params(
            temperature=self.args.temperature, 
            top_p=self.args.top_p, 
            top_k=self.args.top_k,
            force_budget=self.args.force_budget,
        )
        self.engine.setup_special_tokens()
        
    def run(self):
        """
        Main execution loop.
        
        Iterates through batches, calls engine.generate(), and prints statistics.
        """
        total_gen_tokens = 0
        total_model_steps = 0
        
        num_total_runs = self.args.num_total_runs
        
        if self.mode == "e2e":
            num_total_runs = min(num_total_runs, len(self.dataloader))
            iterator = enumerate(self.dataloader)
        else:
            iterator = enumerate(range(num_total_runs))
        
        for run_idx, batch_or_idx in tqdm(iterator, total=num_total_runs):
            if run_idx >= num_total_runs:
                break
            
            # Load or sample batch
            if self.mode == "e2e":
                input_ids, query_lens = self._load_batch(batch_or_idx)
            else:
                input_ids, query_lens = self._sample_batch()
            
            # Call engine.generate()
            output, num_generated_tokens, num_total_tokens, model_steps = self.engine.generate_batch(input_ids=input_ids, query_lens=query_lens)
            
            # Print output if requested
            if self.args.printoutput:
                self._print_output(run_idx, output, query_lens, num_generated_tokens, num_total_tokens, model_steps)
            
            # Accumulate statistics
            total_gen_tokens += num_generated_tokens.sum().item()
            total_model_steps += model_steps
            
            # Distributed barrier if needed
            if self.args.rank_group and len(self.args.rank_group) > 1:
                dist.barrier()
        
        # Print final statistics
        bsz = input_ids.shape[0]
        print(f"Total generated tokens: {total_gen_tokens}")
        print(f"Total model steps (batch_size): {total_model_steps} (batch_size: {bsz})")
        print(f"➡️  Mean generated tokens: {total_gen_tokens / (total_model_steps * bsz):.2f}")
    
    def _load_batch(self, batch):
        """Load batch from dataloader (E2E mode)."""
        input_ids = batch['input_ids'].to(self.device)
        query_lens = batch['attention_mask'].to(self.device).sum(dim=-1).to(torch.int32)
        return input_ids, query_lens
    
    def _sample_batch(self):
        """Sample batch from batch_sampler (benchmark mode)."""
        input_ids = self.batch_sampler.sample_batch().to(self.device)
        bsz = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        query_lens = torch.ones(bsz, device=self.device, dtype=torch.int32) * seq_len
        return input_ids, query_lens
    
    def _print_output(self, run_idx, output, query_lens, num_generated_tokens, num_total_tokens, model_steps):
        """Print generated output for each sequence."""
        bsz = output.shape[0]
        print("\n" + "="*50 + f" Run {run_idx} Output " + "="*50)
        for i in range(bsz):
            mean_tokens_per_step = (num_generated_tokens[i] / model_steps).item()
            print(f"########## Sequence {i} ########## "
                  f"(Total generated tokens: {num_generated_tokens[i]}, "
                  f"mean generated tokens per step: {mean_tokens_per_step:.2f})")
            decoded = self.engine.tokenizer.decode(
                output[i, query_lens[i]:num_total_tokens[i]], 
                skip_special_tokens=True
            )
            print(decoded)

