"""Runner for benchmark execution."""

import time
import numpy as np
import torch
import torch.distributed as dist

from batchspec.models import LoRAConfig
from batchspec.continuous.base import Sequence, Status


class Runner:
    """
    Runner for executing generation loops.
    
    Handles:
    - Batch loading/sampling
    - Calling engine.generate()
    - Output processing and statistics
    """
    
    def __init__(self, args, engine, tokenizer, batch_sampler=None, dataset=None):
        """
        Initialize the runner.
        
        Args:
            args: Command-line arguments
            engine: Backend engine (StandardEngine or MTPEngine)
            tokenizer: HuggingFace tokenizer
            batch_sampler: BatchSampler for benchmark dataset
            dataset: Dataset for continuous generation
        """
        self.args = args
        self.engine = engine
        self.batch_sampler = batch_sampler
        self.dataset = dataset
        self.device = engine.device
        self.run_cnt = 0
        
        if (dataset is None) == (batch_sampler is None):
            raise ValueError("Provide exactly one of dataset or batch_sampler, not both or neither.")
        
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

        if self.args.backend == "standalone":
            assert self.args.drafter_name is not None and self.args.drafter_checkpoint_path is not None, "Standalone drafter name and checkpoint path are required for Standalone backend"
            load_params.update({
                'drafter_name': self.args.drafter_name,
                'drafter_checkpoint_path': self.args.drafter_checkpoint_path,
                'use_drafter_tp': self.args.use_drafter_tp,
            })
        elif self.args.backend == "eagle":
            assert self.args.eagle_name is not None and self.args.eagle_checkpoint_path is not None, "EAGLE drafter name and checkpoint path are required for EAGLE backend"
            load_params.update({
                'eagle_name': self.args.eagle_name,
                'eagle_checkpoint_path': self.args.eagle_checkpoint_path,
                'use_eagle_tp': self.args.use_eagle_tp,
            })
        elif self.args.backend == "mtp":
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
            32: 128, 64: 128, 128: 128, 256: 128
        }
        cache_params = {
            'batch_size': self.args.batch_size, 
            'max_seq_length': self.args.max_seq_len,
            'page_size': 16,
            'prefill_chunk_size': prefill_chunk_size.get(self.args.batch_size, 128),
            'attn_buffer_size_mb': self.args.attn_buffer_size_mb,
        }

        if self.args.backend == "magicdec":
            assert self.args.num_sink_tokens is not None and self.args.stream_budget is not None, "Number of sink tokens and stream budget are required for MagicDec backend"
            cache_params.update({
                'num_sink_tokens': self.args.num_sink_tokens,
                'stream_budget': self.args.stream_budget,
            })
        
        self.engine.setup_caches(**cache_params)
        self.engine.setup_sampling_params(
            temperature=self.args.temperature, 
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            force_budget=self.args.force_budget,
        )
        self.engine.setup_special_tokens()


    def run_benchmark(self):
        """
        Main execution loop for benchmarking.
        
        Iterates through batches, calls engine.generate_batch(), and prints statistics.
        """
        self.run_cnt += 1

        total_gen_tokens = 0
        total_model_steps = 0
        input_ids = self.batch_sampler.sample_batch().to(self.device)
        
        # Clear the KV cache at the beginning of the run
        self.engine.init_cache()

        # Run the benchmark
        for i, prefix_len in enumerate(self.args.prefix_len_list):
            start_idx = self.args.prefix_len_list[i-1] if i > 0 else 0
            end_idx = self.args.prefix_len_list[i]
            current_input_ids = input_ids[:, start_idx:end_idx]
            
            # Call engine.generate()
            generate_params = {'input_ids': current_input_ids, 'max_gen_len': self.args.max_gen_len, 'prefix_len': prefix_len}
            if self.args.backend == "ngram":
                generate_params['full_input_ids'] = input_ids[:, :end_idx]
            
            output, num_generated_tokens, model_steps = self.engine.generate_batch(**generate_params)
            
            # Print output if requested
            if self.args.printoutput:
                self._print_batch_output(prefix_len, output, num_generated_tokens, model_steps)
            
            # Accumulate statistics
            total_gen_tokens += num_generated_tokens.sum().item()
            total_model_steps += model_steps
            
            # Distributed barrier if needed
            if self.args.rank_group and len(self.args.rank_group) > 1:
                dist.barrier()
            
            if self.run_cnt == 1:
                print("[Runner] Early exit for the 1st run, which is used only for compilation and excluded from statistics.")
                break
        
        # Print final statistics
        bsz = input_ids.shape[0]
        print(f"Total generated tokens: {total_gen_tokens}")
        print(f"Total model steps (batch_size): {total_model_steps} (batch_size: {bsz})")
        print(f"➡️  Mean generated tokens: {total_gen_tokens / (total_model_steps * bsz):.3f}")
        
        return


    def run_continuous(self):
        """
        Main execution loop for generation with continuous batching.
        
        Iterates through batches, calls engine.generate(), and prints statistics.
        """
        self.engine.set_stop_on_tail(self.args.stop_on_tail)
        
        total_gen_tokens = 0
        total_model_steps = 0
        
        # Wrapping dataloader with List[Sequence]
        sampled_batch = self.sample_from_dataset(self.args.num_samples)
        sequences = [
            Sequence(seq_id=seq_id, prompt_ids=input_ids, prompt_len=len(input_ids), max_seq_len=self.args.max_seq_len, max_gen_len=self.args.max_gen_len)
            for seq_id, input_ids in enumerate(sampled_batch['input_ids'])
        ]
        
        time_start = time.perf_counter()
        model_steps = self.engine.generate(sequences)
        time_end = time.perf_counter()
        
        # Print output if requested
        if self.args.printoutput:
            self._print_sequence_output(sequences[0], model_steps)
        
        # Accumulate statistics
        total_prefilled_tokens = sum([seq.prompt_len for seq in sequences])
        total_generated_tokens = sum([seq.num_generated_tokens for seq in sequences])

        # Distributed barrier if needed
        if self.args.rank_group and len(self.args.rank_group) > 1:
            dist.barrier()
        
        # Print final statistics
        print("="*20 + " Statistics (batch_size: " + str(self.args.batch_size) + ")" + "="*20)
        print(f"Total prefilled tokens: {total_prefilled_tokens} tokens")
        print(f"Total generated tokens: {total_generated_tokens} tokens")
        print(f"Total model steps: {model_steps}")
        print(f"➡️  Mean prefilled tokens: {total_prefilled_tokens / self.args.batch_size:.2f} tokens/seq")
        print(f"➡️  Mean generated tokens: {total_generated_tokens / (model_steps * self.args.batch_size):.2f} tokens/seq")
        print(f"📈 Mean latency: {(time_end - time_start)*1000 / model_steps:.2f} ms/step")
        print(f"📈 Throughput: {total_generated_tokens / (time_end - time_start):.2f} tokens/s")


    def run_continuous_benchmark(self, exp_config: dict):
        """
        Main execution loop for generation with continuous batching.
        
        Iterates through batches, calls engine.generate(), and prints statistics.
        """
        total_gen_tokens = 0
        total_model_steps = 0
        input_ids = self.batch_sampler.sample_batch().to(self.device)

        if exp_config['type'] == 'prefill_decode':
            prefill_ratio = exp_config['prefill_ratio']
            decode_start_len = exp_config['decode_start_len']
            assert 0 <= prefill_ratio < 1, "Prefill ratio must be between 0 and 1"
            decode_ratio = 1 - prefill_ratio
            
            num_prefill = int(prefill_ratio * self.args.batch_size)
            perm = torch.randperm(self.args.batch_size).to(self.device)
            prefill_indices = perm[:num_prefill]
            decode_indices = perm[num_prefill:]

            # For decode sequences, we need to prefill before generate
            sequences = [
                Sequence(seq_id=seq_id, prompt_ids=input_ids[seq_id, :decode_start_len], prompt_len=len(input_ids[seq_id, :decode_start_len]), max_seq_len=self.args.max_seq_len, max_gen_len=self.args.max_gen_len)
                for seq_id in range(self.args.batch_size)
            ]
            prefill_scheduler = self.engine.prefill(sequences)

            # Migrate the state of the prefill scheduler to the main scheduler
            self.engine.scheduler.slots = prefill_scheduler.slots
            self.engine.scheduler.free_slots = prefill_scheduler.free_slots

            # Force to continue the prefill
            for seq_id in prefill_indices.cpu().tolist():
                seq = sequences[seq_id]
                seq.prompt_ids = input_ids[seq_id, :decode_start_len * 2]
                seq.prompt_len = len(input_ids[seq_id, :decode_start_len * 2])
                seq.status = Status.PREFILL
                self.engine.scheduler.completed.append(seq) # Mark the sequence as completed to stop when decode is done.

            # Run the benchmark
            time_start = time.perf_counter()
            model_steps = self.engine.generate_with_repeated_prefill(sequences)
            time_end = time.perf_counter()

        elif exp_config['type'] == 'hetero_decode':
            short_ratio = exp_config['short_ratio']
            assert 0 <= short_ratio < 1, "Short ratio must be between 0 and 1"
            long_ratio = 1 - short_ratio

            short_target_len = exp_config['short_target_len']
            long_target_len = exp_config['long_target_len']

            num_short = int(short_ratio * self.args.batch_size)
            perm = torch.randperm(self.args.batch_size).to(self.device)
            short_indices = perm[:num_short]
            long_indices = perm[num_short:]

            # For short sequences, we need to prefill before generate
            sequences = [
                Sequence(seq_id=seq_id, prompt_ids=input_ids[seq_id, :short_target_len], prompt_len=len(input_ids[seq_id, :short_target_len]), max_seq_len=self.args.max_seq_len, max_gen_len=self.args.max_gen_len)
                for seq_id in short_indices.cpu().tolist()
            ]
            sequences.extend([
                Sequence(seq_id=seq_id, prompt_ids=input_ids[seq_id, :long_target_len], prompt_len=len(input_ids[seq_id, :long_target_len]), max_seq_len=self.args.max_seq_len, max_gen_len=self.args.max_gen_len)
                for seq_id in long_indices.cpu().tolist()
            ])
            prefill_scheduler = self.engine.prefill(sequences)
            
            # Migrate the state of the prefill scheduler to the main scheduler
            self.engine.scheduler.slots = prefill_scheduler.slots
            self.engine.scheduler.free_slots = prefill_scheduler.free_slots

            # Run the benchmark
            time_start = time.perf_counter()
            model_steps = self.engine.generate(sequences)
            time_end = time.perf_counter()

        else:
            raise ValueError(f"Unsupported experiment type: {exp_config['type']}")
        

        # Print output if requested
        if self.args.printoutput:
            self._print_sequence_output(sequences[0], model_steps)
        
        # Accumulate statistics
        total_prefilled_tokens = sum([seq.prompt_len for seq in sequences])
        total_generated_tokens = sum([seq.num_generated_tokens for seq in sequences])

        # Distributed barrier if needed
        if self.args.rank_group and len(self.args.rank_group) > 1:
            dist.barrier()
        
        # Print final statistics
        print("="*20 + " Statistics (batch_size: " + str(self.args.batch_size) + ")" + "="*20)
        print(f"Total prefilled tokens: {total_prefilled_tokens} tokens")
        print(f"Total generated tokens: {total_generated_tokens} tokens")
        print(f"Total model steps: {model_steps}")
        print(f"➡️  Mean prefilled tokens: {total_prefilled_tokens / self.args.batch_size:.2f} tokens/seq")
        print(f"➡️  Mean generated tokens: {total_generated_tokens / (model_steps * self.args.batch_size):.2f} tokens/seq")
        print(f"📈 Mean latency: {(time_end - time_start)*1000 / model_steps:.2f} ms/step")
        print(f"📈 Throughput: {total_generated_tokens / (time_end - time_start):.2f} tokens/s")

        return
    
    
    # ============================================
    # Helper functions
    # ============================================ 
    def sample_from_dataset(self, num_samples: int):
        """Sample a batch from the dataset with replacement."""
        ds_len = len(self.dataset)
        rng = np.random.default_rng(self.args.seed)
        idx = rng.integers(0, ds_len, size=num_samples)
        return self.dataset.select(idx.tolist())

        
    def _print_batch_output(self, prefix_len, output, num_generated_tokens, model_steps):
        """Print generated output for each sequence."""
        bsz = output.shape[0]
        print("\n" + "="*50 + f" Prefix Length {prefix_len} Output " + "="*50)
        mean_tokens_per_step = (num_generated_tokens[0] / model_steps).item()
        print(f"########## Sequence 0 ########## "
                f"(Total generated tokens: {num_generated_tokens[0]}, "
                f"mean generated tokens per step: {mean_tokens_per_step:.2f})")
        print(self.engine.tokenizer.decode(
            output[0, :num_generated_tokens[0]], 
            skip_special_tokens=True
        ))

    
    def _print_sequence_output(self, seq: Sequence, model_steps: int) -> None:
        """Print output for a single sequence."""
        print(f"########## Sequence {seq.seq_id} ########## "
                f"(Total generated tokens: {seq.num_generated_tokens}, "
                f"mean generated tokens per step: {seq.num_generated_tokens / model_steps:.2f})")
        print(self.engine.tokenizer.decode(seq.content[:seq.cur_pos+1], skip_special_tokens=True))

