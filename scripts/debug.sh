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

nproc_per_node=8
rank_group="0 1 2 3 4 5 6 7"

bsz=128
for max_gen_len in 2048 4096 8192 16384; do
    if [ $max_gen_len -eq 2048 ]; then
        draft_length=2
    else
        draft_length=3
    fi

    for backend in standard mtp; do
        extra_args=$(get_extra_args $backend)
        torchrun --standalone --nproc_per_node=$nproc_per_node -m batchspec.run_conti\
            --backend $backend\
            --checkpoint_path $model_path\
            --tokenizer_path $tokenizer_path\
            --model_name $model_name\
            --rank_group $rank_group\
            --dataset AIME2025\
            --dtype bfloat16\
            --batch_size $bsz --max_gen_len $max_gen_len --max_seq_len $max_gen_len\
            --num_samples 256\
            --temperature 0.6 --top_p 0.95 --top_k 20\
            --printoutput --prof_output_dir num_samples_256/max_gen_${max_gen_len}\
            --profiling --engine_profiling --num_total_runs 1\
            $extra_args --draft_length $draft_length
    done
done

for short_ratio in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    for backend in mtp; do
        for draft_length in 2 3; do
            extra_args=$(get_extra_args $backend)
            torchrun --standalone --nproc_per_node=$nproc_per_node -m batchspec.run_conti_bench\
                --backend $backend\
                --checkpoint_path $model_path\
                --tokenizer_path $tokenizer_path\
                --model_name $model_name\
                --rank_group $rank_group\
                --dataset AIME2025\
                --dtype bfloat16\
                --batch_size $bsz --max_gen_len 128 --max_seq_len 16384\
                --temperature 0.6 --top_p 0.95 --top_k 20\
                --printoutput --prof_output_dir hetero_seqlen/short_ratio_${short_ratio}\
                --profiling --engine_profiling --num_total_runs 1\
                --short_ratio $short_ratio --short_target_len 1024 --long_target_len 15360\
                $extra_args --draft_length $draft_length
        done
    done
done
