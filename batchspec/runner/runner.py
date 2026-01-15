"""Runner for E2E and benchmark execution."""

import torch
import torch.distributed as dist

from batchspec.models import LoRAConfig


class Runner:
    """
    Runner for executing generation loops.
    
    Handles:
    - Batch loading/sampling
    - Calling engine.generate()
    - Output processing and statistics
    """
    
    def __init__(self, args, engine, tokenizer, batch_sampler=None):
        """
        Initialize the runner.
        
        Args:
            args: Command-line arguments
            engine: Backend engine (StandardEngine or MTPEngine)
            tokenizer: HuggingFace tokenizer
            batch_sampler: BatchSampler for benchmark dataset
        """
        self.args = args
        self.engine = engine
        self.batch_sampler = batch_sampler
        self.device = engine.device
        
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
            32: 128, 64: 128, 128: 64, 256: 32
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
        Main execution loop for benchmarking.
        
        Iterates through batches, calls engine.generate_batch(), and prints statistics.
        """
        total_gen_tokens = 0
        total_model_steps = 0
        input_ids = self.batch_sampler.sample_batch().to(self.device)
        
        # Clear the KV cache for the first run
        self.engine.kv_page_table.clear_kv(self.engine.model)

        # Run the benchmark
        for i, prefix_len in enumerate(self.args.prefix_len_list):
            self.args.prefix_len = prefix_len
            if self.args.backend == "mtp":
                self.args.max_total_tokens = self.args.batch_size * self.args.max_gen_len
                self.args.max_len = self.args.prefix_len + (self.args.max_gen_len * 3) # prepare some margin for the fastest sequence
            else:
                self.args.max_len = self.args.prefix_len + (self.args.max_gen_len * 3) # prepare some margin for the fastest sequence
            
            start_idx = self.args.prefix_len_list[i-1] if i > 0 else 0
            end_idx = self.args.prefix_len_list[i]
            current_input_ids = input_ids[:, start_idx:end_idx]
            
            # Call engine.generate()
            output, num_generated_tokens, model_steps = self.engine.generate_batch(input_ids=current_input_ids, max_gen_len=self.args.max_gen_len, prefix_len=prefix_len)
            
            # Print output if requested
            if self.args.printoutput:
                self._print_output(prefix_len, output, num_generated_tokens, model_steps)
            
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
    
    def _print_output(self, prefix_len, output, num_generated_tokens, model_steps):
        """Print generated output for each sequence."""
        bsz = output.shape[0]
        print("\n" + "="*50 + f" Prefix Length {prefix_len} Output " + "="*50)
        mean_tokens_per_step = (num_generated_tokens[0] / model_steps).item()
        print(f"########## Sequence 0 ########## "
                f"(Total generated tokens: {num_generated_tokens[0]}, "
                f"mean generated tokens per step: {mean_tokens_per_step:.2f})")
        decoded = self.engine.tokenizer.decode(
            output[0, :num_generated_tokens[0]+1], 
            skip_special_tokens=True
        )
        print(decoded)

