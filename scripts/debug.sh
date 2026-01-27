#!/bin/bash

# =============================================================================
# Debug Script for Speculative Decoding Benchmark with Continuous Batching
# =============================================================================

# Import utils.sh
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils.sh"

get_extra_args() {
    local backend="$1"

    case "$backend" in
        standard)
            echo ""
            ;;
        standalone)
            echo "--drafter_name Qwen3-0.6B --drafter_checkpoint_path $base_ckpt_dir/Qwen3-0.6B/model.pth --draft_length 4"
            ;;
        eagle)
            echo "--eagle_name EAGLE3-DeepSeek-R1-Distill-LLaMA-8B --eagle_checkpoint_path $base_ckpt_dir/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B/model.pth --draft_length 4"
            ;;
        magicdec)
            echo "--num_sink_tokens 16 --stream_budget 512 --draft_length 4"
            ;;
        mtp)
            echo "--lora_checkpoint_path $base_ckpt_dir/MTP_adapters/${model_name}/model.pth --lora_rank 16 --lora_alpha 32 --draft_length 4"
            ;;
        *)
            echo "Unknown backend: $backend" >&2
            return 1
            ;;
    esac
}

# Set environment variables
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning,ignore::DeprecationWarning"
export FLASHINFER_JIT_VERBOSE=1

# Set CUDA architecture list
set_torch_cuda_arch_list 0
log "Auto-detected CUDA compute capability: ${TORCH_CUDA_ARCH_LIST}"

# Model Configuration
base_ckpt_dir=/workspace/checkpoints

model_name=Qwen3-8B
model_path=$base_ckpt_dir/Qwen3-8B/model.pth
tokenizer_path=$base_ckpt_dir/Qwen3-8B

nproc_per_node=1
rank_group="0"

bsz=128
for backend in mtp; do
	extra_args=$(get_extra_args $backend)
	torchrun --standalone --nproc_per_node=$nproc_per_node -m batchspec.run_continuous\
		--backend $backend\
		--checkpoint_path $model_path\
		--tokenizer_path $tokenizer_path\
		--model_name $model_name\
		--rank_group $rank_group\
		--dataset AIME2025\
		--dtype bfloat16\
		--batch_size $bsz --max_gen_len 1024 --max_seq_len 2048\
		--num_samples $bsz --num_questions_in_prompt 5\
		--temperature 0.6 --top_p 0.95 --top_k 20\
		--printoutput --stop_on_tail\
		--profiling --engine_profiling --num_total_runs 4\
		$extra_args
done