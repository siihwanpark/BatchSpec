#!/bin/bash

# =============================================================================
# Speculative Decoding Benchmark Script
# =============================================================================
# Suppress Python warnings
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning,ignore::DeprecationWarning"

# Environment Setup
export ENABLE_INTRA_NODE_COMM=1
export FLASHINFER_JIT_VERBOSE=1
export TORCH_CUDA_ARCH_LIST=$(python -c "import torch; cc=torch.cuda.get_device_capability(0); print(f'{cc[0]}.{cc[1]}')")
echo "Auto-detected CUDA compute capability: ${TORCH_CUDA_ARCH_LIST}"

# Model Configuration
model_name=Qwen3-8B
model_path=/workspace/checkpoints/Qwen3-8B/model.pth
tokenizer_path=/workspace/checkpoints/Qwen3-8B

for backend in standard standalone eagle magicdec mtp; do
    if [ $backend == "standalone" ]; then
        extra_args=(--drafter_name Qwen3-0.6B --drafter_checkpoint_path /workspace/checkpoints/Qwen3-0.6B/model.pth --draft_length 4)
    elif [ $backend == "eagle" ]; then
        extra_args=(--eagle_name Qwen3-8B_eagle3 --eagle_checkpoint_path /workspace/checkpoints/Qwen3-8B_eagle3/model.pth --draft_length 4)
    elif [ $backend == "magicdec" ]; then
        extra_args=(--draft_length 4 --num_sink_tokens 16 --stream_budget 256)
    elif [ $backend == "mtp" ]; then
        extra_args=(--lora_checkpoint_path /workspace/checkpoints/MTP_adapters/model.pth --lora_rank 16 --lora_alpha 32 --draft_length 4)
    fi

    torchrun --standalone --nproc_per_node=1 -m batchspec.run\
        --backend $backend\
        --checkpoint_path $model_path\
        --tokenizer_path $tokenizer_path\
        --model_name $model_name\
        --rank_group 0\
        --dataset AIME2025\
        --dtype bfloat16\
        --batch_size 4 --prefix_len_list 1024 2048 --max_gen_len 128\
        --temperature 0.0 --force_budget\
        --printoutput\
        --profiling\
        --num_total_runs 3\
        "${extra_args[@]}"

    torchrun --standalone --nproc_per_node=1 -m batchspec.run\
        --backend $backend\
        --checkpoint_path $model_path\
        --tokenizer_path $tokenizer_path\
        --model_name $model_name\
        --rank_group 0\
        --dataset AIME2025\
        --dtype bfloat16\
        --batch_size 4 --prefix_len_list 1024 2048 --max_gen_len 128\
        --temperature 0.6 --top_p 0.95 --top_k 20 --force_budget\
        --printoutput\
        --profiling\
        --num_total_runs 3\
        "${extra_args[@]}"
done