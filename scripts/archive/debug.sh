#!/bin/bash

# =============================================================================
# Debug Script for Speculative Decoding Benchmark
# =============================================================================

# Import utils.sh
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils.sh"

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

bsz=64
prefix_len_list=(1024 28672)
backend=standalone
if [ $backend == "standard" ]; then
    extra_args=()
elif [ $backend == "standalone" ]; then
    extra_args=(--drafter_name Qwen3-0.6B --drafter_checkpoint_path $base_ckpt_dir/Qwen3-0.6B/model.pth --draft_length 4)
elif [ $backend == "eagle" ]; then
	if [ $model_name == "DeepSeek-R1-Distill-Llama-8B" ]; then
		extra_args=(--eagle_name EAGLE3-DeepSeek-R1-Distill-LLaMA-8B --eagle_checkpoint_path $base_ckpt_dir/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B/model.pth --draft_length 4)
	else
		extra_args=(--eagle_name ${model_name}_eagle3 --eagle_checkpoint_path $base_ckpt_dir/${model_name}_eagle3/model.pth --draft_length 4)
	fi
elif [ $backend == "magicdec" ]; then
    extra_args=(--num_sink_tokens 16 --stream_budget 512 --draft_length 4)
elif [ $backend == "mtp" ]; then
    extra_args=(--lora_checkpoint_path $base_ckpt_dir/MTP_adapters/${model_name}/model.pth --lora_rank 16 --lora_alpha 32 --draft_length 4)
fi

torchrun --standalone --nproc_per_node=$nproc_per_node -m batchspec.run\
	--backend $backend\
	--checkpoint_path $model_path\
	--tokenizer_path $tokenizer_path\
	--model_name $model_name\
	--rank_group $rank_group\
	--dataset AIME2025\
	--dtype bfloat16\
	--batch_size $bsz --prefix_len_list ${prefix_len_list[@]} --max_gen_len 128\
	--temperature 0.6 --top_p 0.95 --top_k 20 --force_budget\
	--printoutput\
	--profiling\
	--num_total_runs 2\
	"${extra_args[@]}"
exit 0

# for backend in standard standalone eagle magicdec mtp; do
#     if [ $backend == "standalone" ]; then
#         extra_args=(--drafter_name Qwen3-0.6B --drafter_checkpoint_path $base_ckpt_dir/Qwen3-0.6B/model.pth --draft_length 4)
#     elif [ $backend == "eagle" ]; then
#         extra_args=(--eagle_name Qwen3-8B_eagle3 --eagle_checkpoint_path $base_ckpt_dir/Qwen3-8B_eagle3/model.pth --draft_length 4)
#     elif [ $backend == "magicdec" ]; then
#         extra_args=(--draft_length 4 --num_sink_tokens 16 --stream_budget 512)
#     elif [ $backend == "mtp" ]; then
#         extra_args=(--lora_checkpoint_path $base_ckpt_dir/MTP_adapters/Qwen3-8B/model.pth --lora_rank 16 --lora_alpha 32 --draft_length 4)
#     fi

#     torchrun --standalone --nproc_per_node=4 -m batchspec.run\
#         --backend $backend\
#         --checkpoint_path $model_path\
#         --tokenizer_path $tokenizer_path\
#         --model_name $model_name\
#         --rank_group 0 1 2 3\
#         --dataset AIME2025\
#         --dtype bfloat16\
#         --batch_size 64 --prefix_len_list 1024 2048 --max_gen_len 128\
#         --temperature 0.0 --force_budget\
#         --printoutput\
#         --profiling\
#         --num_total_runs 3\
#         "${extra_args[@]}"

#     torchrun --standalone --nproc_per_node=1 -m batchspec.run\
#         --backend $backend\
#         --checkpoint_path $model_path\
#         --tokenizer_path $tokenizer_path\
#         --model_name $model_name\
#         --rank_group 0\
#         --dataset AIME2025\
#         --dtype bfloat16\
#         --batch_size 64 --prefix_len_list 1024 2048 --max_gen_len 128\
#         --temperature 0.6 --top_p 0.95 --top_k 20 --force_budget\
#         --printoutput\
#         --profiling\
#         --num_total_runs 3\
#         "${extra_args[@]}"
# done