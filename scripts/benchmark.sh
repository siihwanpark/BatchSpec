#!/bin/bash

# =============================================================================
# Speculative Decoding Script
# =============================================================================

# Environment Setup
export ENABLE_INTRA_NODE_COMM=1
export TORCH_CUDA_ARCH_LIST=8.9
export FLASHINFER_JIT_VERBOSE=1

# Model Configuration
model_name=Qwen3-8B
model_path=/workspace/checkpoints/Qwen3-8B/model.pth
tokenizer_path=/workspace/checkpoints/Qwen3-8B

export CUDA_VISIBLE_DEVICES=0
# torchrun --standalone --nproc_per_node=1 -m batchspec.run\
#     --backend standard\
#     --checkpoint_path $model_path\
#     --tokenizer_path $tokenizer_path\
#     --model_name $model_name\
#     --rank_group 0\
#     --dataset AIME2025\
#     --dtype bfloat16\
#     --batch_size 2 --prefix_len_list 1024 2048 --max_gen_len 128\
#     --temperature 0.0\
#     --printoutput\
#     --profiling\
#     --num_total_runs 3

# exit 0

# drafter=/workspace/checkpoints/MTP_adapters/model.pth
# torchrun --standalone --nproc_per_node=1 -m batchspec.run\
#     --backend mtp\
#     --checkpoint_path $model_path\
#     --tokenizer_path $tokenizer_path\
#     --model_name $model_name\
#     --rank_group 0\
#     --dataset AIME2025 --force_budget\
#     --dtype bfloat16\
#     --batch_size 4 --prefix_len_list 1024 2048 --max_gen_len 128\
#     --temperature 0.6 --top_p 0.95 --top_k 20\
#     --printoutput\
#     --profiling\
#     --num_total_runs 3\
#     --lora_checkpoint_path $drafter --lora_rank 16 --lora_alpha 32\
#     --draft_length 4

# drafter=/workspace/checkpoints/Qwen3-8B_eagle3/model.pth
# torchrun --standalone --nproc_per_node=1 -m batchspec.run\
#     --backend eagle\
#     --checkpoint_path $model_path\
#     --eagle_checkpoint_path $drafter\
#     --tokenizer_path $tokenizer_path\
#     --model_name $model_name\
#     --eagle_name Qwen3-8B_eagle3\
#     --rank_group 0\
#     --dataset AIME2025 --force_budget\
#     --dtype bfloat16\
#     --batch_size 4 --prefix_len_list 1024 2048 --max_gen_len 128\
#     --temperature 0.6 --top_p 0.95 --top_k 20\
#     --printoutput\
#     --profiling\
#     --num_total_runs 3\
#     --draft_length 4

# drafter=/workspace/checkpoints/Qwen3-0.6B/model.pth
# torchrun --standalone --nproc_per_node=1 -m batchspec.run\
#     --backend standalone\
#     --checkpoint_path $model_path\
#     --tokenizer_path $tokenizer_path\
#     --model_name $model_name\
#     --drafter_name Qwen3-0.6B\
#     --drafter_checkpoint_path $drafter\
#     --rank_group 0\
#     --dataset AIME2025 --force_budget\
#     --dtype bfloat16\
#     --batch_size 4 --prefix_len_list 1024 2048 --max_gen_len 128\
#     --temperature 0.0\
#     --printoutput\
#     --profiling\
#     --num_total_runs 3\
#     --draft_length 4

torchrun --standalone --nproc_per_node=1 -m batchspec.run\
    --backend magicdec\
    --checkpoint_path $model_path\
    --tokenizer_path $tokenizer_path\
    --model_name $model_name\
    --rank_group 0\
    --dataset AIME2025 --force_budget\
    --dtype bfloat16\
    --batch_size 2 --prefix_len_list 1024 2048 --max_gen_len 128\
    --temperature 0.6 --top_p 0.95 --top_k 20\
    --printoutput\
    --profiling\
    --num_total_runs 3\
    --draft_length 2 --num_sink_tokens 16 --stream_budget 256