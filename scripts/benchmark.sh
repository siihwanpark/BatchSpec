#!/bin/bash

# =============================================================================
# Speculative Decoding Script
# =============================================================================

# Environment Setup
export ENABLE_INTRA_NODE_COMM=1
export TORCH_CUDA_ARCH_LIST=8.6
export FLASHINFER_JIT_VERBOSE=1

# Model Configuration
model_name=Qwen3-8B
model_path=/workspace/checkpoints/Qwen3-8B/model.pth
tokenizer_path=/workspace/checkpoints/Qwen3-8B
# drafter=/workspace/checkpoints/MTP-adapter/Qwen3-8B/AM_Qwen3_Distilled_120k/rank16-lr2e-4-bsz8-SoftSCE-fused/step-35000/model.pth
drafter=/workspace/checkpoints/Qwen3-8B_eagle3/model.pth

# model_name=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
# model_path=/home/mngcuser1/sihwan_workspace/checkpoints/DeepSeek-R1-Distill-Llama-8B/model.pth
# tokenizer_path=/home/mngcuser1/sihwan_workspace/checkpoints/DeepSeek-R1-Distill-Llama-8B
# drafter=/home/mngcuser1/sihwan_workspace/checkpoints/MTP_adapters/DeepSeek-R1-Distill-Llama-8B/openthoughts-114k-rank16-lr2e-4-bsz8/step-30000/model.pth

if [ ! -f "$model_path" ] || [ ! -f "$drafter" ]; then
    echo "Model or drafter file not found! Please prepare the model first."
    exit 1
fi

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

# torchrun --standalone --nproc_per_node=1 -m batchspec.run\
#     --backend mtp\
#     --checkpoint_path $model_path\
#     --tokenizer_path $tokenizer_path\
#     --model_name $model_name\
#     --rank_group 0\
#     --dataset AIME2025 --force_budget\
#     --dtype bfloat16\
#     --batch_size 4 --prefix_len_list 1024 2048 --max_gen_len 128\
#     --temperature 0.0\
#     --printoutput\
#     --profiling\
#     --num_total_runs 3\
#     --lora_checkpoint_path $drafter --lora_rank 16 --lora_alpha 32\
#     --draft_length 4

# exit 0
export CUDA_LAUNCH_BLOCKING=1
torchrun --standalone --nproc_per_node=1 -m batchspec.run\
    --backend eagle\
    --checkpoint_path $model_path\
    --eagle_checkpoint_path $drafter\
    --tokenizer_path $tokenizer_path\
    --model_name $model_name\
    --eagle_name Qwen3-8B_eagle3\
    --rank_group 0\
    --dataset AIME2025 --force_budget\
    --dtype bfloat16\
    --batch_size 4 --prefix_len_list 1024 2048 --max_gen_len 128\
    --temperature 0.6 --top_p 0.95 --top_k 20\
    --printoutput\
    --profiling\
    --num_total_runs 3\
    --draft_length 4

exit 0

OPTION=${1:-0}
if [ $OPTION -eq 0 ]; then
    while true; do
        busy_pids=$(nvidia-smi -i 0,1,2,3,4,5,6,7 --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')
        if [ -z "$busy_pids" ]; then
            echo "GPUs 0,1,2,3,4,5,6,7 are idle. Sleep 5 more minutes."
            # sleep 5m
        else
            echo "GPUs 0,1,2,3,4,5,6,7 are busy. Sleeping 5 minutes."
            sleep 5m
        fi

        if [ -z "$busy_pids" ]; then
            echo "GPUs 0,1,2,3,4,5,6,7 are idle even after 5 minutes. Running script."
            break
        fi
    done

    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    num_total_runs=11
    n_proc_per_node=8
    rank_group="0 1 2 3 4 5 6 7"

    dataset="AIME2025"
    bsz=256
    prefix_len_list=(1024 2048 4096 8192 12288)
    draft_length_list=(4 3 2 1)
    
    echo "Running batch_size=$bsz, dataset=$dataset, prefix_len_list=${prefix_len_list[@]}"
    
    for draft_length in "${draft_length_list[@]}"; do
        IFS=' ' read -r -a draft_length_array <<< "$draft_length"
        echo "draft_length_array=${draft_length_array[@]}"

        torchrun --standalone --nproc_per_node=$n_proc_per_node -m RSpec.run_benchmark\
            --backend mtp\
            --target $model\
            --tokenizer_path $tokenizer_path\
            --model_name $model_name\
            --rank_group $rank_group\
            --dataset $dataset --force_budget\
            --dtype bfloat16\
            --batch_size $bsz --prefix_len_list "${prefix_len_list[@]}" --max_gen_len 128\
            --temperature 0.0\
            --printoutput\
            --profiling\
            --num_total_runs $num_total_runs\
            --lora_adapter $drafter --lora_rank 16 --lora_alpha 32\
            --draft_length "${draft_length_array[@]}"
    done


elif [ $OPTION -eq 1 ]; then
    while true; do
        busy_pids=$(nvidia-smi -i 0,1,2,3 --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')
        if [ -z "$busy_pids" ]; then
            echo "GPUs 0,1,2,3 are idle. Sleep 1 more minutes."
            sleep 1m
        else
            echo "GPUs 0,1,2,3 are busy. Sleeping 5 minutes."
            sleep 5m
        fi

        if [ -z "$busy_pids" ]; then
            echo "GPUs 0,1,2,3 are idle even after 5 minutes. Running script."
            break
        fi
    done

    export CUDA_VISIBLE_DEVICES="0,1,2,3"
    n_proc_per_node=4
    rank_group="0 1 2 3"

    bsz=32
    num_total_runs=11
    dataset_list=("AIME2025")
    prefix_len_list=(1024 2048 4096 8192 12288 16384 20480 24576 28672)
    for dataset in ${dataset_list[@]}; do
        echo "Running batch_size=$bsz, dataset=$dataset, prefix_len_list=$prefix_len_list"
        
        draft_length_list=(4 3)
        for draft_length in "${draft_length_list[@]}"; do
            IFS=' ' read -r -a draft_length_array <<< "$draft_length"
            echo "draft_length_array=${draft_length_array[@]}"

            torchrun --standalone --nproc_per_node=$n_proc_per_node -m RSpec.run_benchmark\
                --backend mtp\
                --target $model\
                --tokenizer_path $tokenizer_path\
                --model_name $model_name\
                --rank_group $rank_group\
                --dataset $dataset --force_budget\
                --dtype bfloat16\
                --batch_size $bsz --prefix_len_list "${prefix_len_list[@]}" --max_gen_len 128\
                --temperature 0.0\
                --printoutput\
                --profiling\
                --num_total_runs $num_total_runs\
                --lora_adapter $drafter --lora_rank 16 --lora_alpha 32\
                --draft_length "${draft_length_array[@]}"
        done
    done

elif [ $OPTION -eq 2 ]; then

    while true; do
        busy_pids=$(nvidia-smi -i 4,5,6,7 --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')
        if [ -z "$busy_pids" ]; then
            echo "GPUs 4,5,6,7 are idle. Sleep 1 more minutes."
            sleep 1m
        else
            echo "GPUs 4,5,6,7 are busy. Sleeping 5 minutes."
            sleep 5m
        fi

        if [ -z "$busy_pids" ]; then
            echo "GPUs 4,5,6,7 are idle even after 5 minutes. Running script."
            break
        fi
    done

    export CUDA_VISIBLE_DEVICES="4,5,6,7"
    n_proc_per_node=4
    rank_group="0 1 2 3"

    bsz=32
    num_total_runs=11
    dataset_list=("AIME2025")
    prefix_len_list=(1024 2048 4096 8192 12288 16384 20480 24576 28672)
    for dataset in ${dataset_list[@]}; do
        echo "Running batch_size=$bsz, dataset=$dataset, prefix_len_list=$prefix_len_list"
        
        draft_length_list=(2 1)
        for draft_length in "${draft_length_list[@]}"; do
            IFS=' ' read -r -a draft_length_array <<< "$draft_length"
            echo "draft_length_array=${draft_length_array[@]}"

            torchrun --standalone --nproc_per_node=$n_proc_per_node -m RSpec.run_benchmark\
                --backend mtp\
                --target $model\
                --tokenizer_path $tokenizer_path\
                --model_name $model_name\
                --rank_group $rank_group\
                --dataset $dataset --force_budget\
                --dtype bfloat16\
                --batch_size $bsz --prefix_len_list "${prefix_len_list[@]}" --max_gen_len 128\
                --temperature 0.0\
                --printoutput\
                --profiling\
                --num_total_runs $num_total_runs\
                --lora_adapter $drafter --lora_rank 16 --lora_alpha 32\
                --draft_length "${draft_length_array[@]}"
        done
    done


elif [ $OPTION -eq 3 ]; then
    while true; do
        busy_pids=$(nvidia-smi -i 0,1 --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')
        if [ -z "$busy_pids" ]; then
            echo "GPUs 0,1 are idle. Sleep 5 more minutes."
            sleep 5m
        fi

        if [ -z "$busy_pids" ]; then
            echo "GPUs 0,1 are idle even after 5 minutes. Running script."
            echo "================================================================"
            export CUDA_VISIBLE_DEVICES="0,1"
            n_proc_per_node=2
            rank_group="0 1"

            bsz=16
            num_total_runs=11
            dataset="AIME2025"
            prefix_len_list=(1024 2048 4096 8192 12288 16384 20480 24576 28672)

            echo "Running batch_size=$bsz, dataset=$dataset, prefix_len_list=$prefix_len_list"

            torchrun --standalone --nproc_per_node=$n_proc_per_node -m RSpec.run_benchmark\
                --backend mtp\
                --target $model\
                --tokenizer_path $tokenizer_path\
                --model_name $model_name\
                --rank_group $rank_group\
                --dataset $dataset --force_budget\
                --dtype bfloat16\
                --batch_size $bsz --prefix_len_list "${prefix_len_list[@]}" --max_gen_len 128\
                --temperature 0.0\
                --printoutput\
                --profiling\
                --num_total_runs $num_total_runs\
                --lora_adapter $drafter --lora_rank 16 --lora_alpha 32\
                --draft_length 4
            echo "================================================================"
            break
        else
            echo "GPUs 0,1 are busy. Sleeping 5 minutes."
            sleep 5m
        fi
    done

elif [ $OPTION -eq 4 ]; then
    while true; do
        busy_pids=$(nvidia-smi -i 2,3 --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')
        if [ -z "$busy_pids" ]; then
            echo "GPUs 2,3 are idle. Sleep 5 more minutes."
            sleep 5m
        fi

        if [ -z "$busy_pids" ]; then
            echo "GPUs 2,3 are idle even after 5 minutes. Running script."
            echo "================================================================"
            export CUDA_VISIBLE_DEVICES="2,3"
            n_proc_per_node=2
            rank_group="0 1"

            bsz=16
            num_total_runs=11
            dataset="AIME2025"
            prefix_len_list=(1024 2048 4096 8192 12288 16384 20480 24576 28672)

            echo "Running batch_size=$bsz, dataset=$dataset, prefix_len_list=$prefix_len_list"

            torchrun --standalone --nproc_per_node=$n_proc_per_node -m RSpec.run_benchmark\
                --backend mtp\
                --target $model\
                --tokenizer_path $tokenizer_path\
                --model_name $model_name\
                --rank_group $rank_group\
                --dataset $dataset --force_budget\
                --dtype bfloat16\
                --batch_size $bsz --prefix_len_list "${prefix_len_list[@]}" --max_gen_len 128\
                --temperature 0.0\
                --printoutput\
                --profiling\
                --num_total_runs $num_total_runs\
                --lora_adapter $drafter --lora_rank 16 --lora_alpha 32\
                --draft_length 3
            echo "================================================================"
            break
        else
            echo "GPUs 2,3 are busy. Sleeping 5 minutes."
            sleep 5m
        fi
    done

elif [ $OPTION -eq 5 ]; then
    while true; do
        busy_pids=$(nvidia-smi -i 4,5 --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')
        if [ -z "$busy_pids" ]; then
            echo "GPUs 4,5 are idle. Sleep 5 more minutes."
            sleep 5m
        fi

        if [ -z "$busy_pids" ]; then
            echo "GPUs 4,5 are idle even after 5 minutes. Running script."
            echo "================================================================"
            export CUDA_VISIBLE_DEVICES="4,5"
            n_proc_per_node=2
            rank_group="0 1"

            bsz=16
            num_total_runs=11
            dataset="AIME2025"
            prefix_len_list=(1024 2048 4096 8192 12288 16384 20480 24576 28672)

            echo "Running batch_size=$bsz, dataset=$dataset, prefix_len_list=$prefix_len_list"

            torchrun --standalone --nproc_per_node=$n_proc_per_node -m RSpec.run_benchmark\
                --backend mtp\
                --target $model\
                --tokenizer_path $tokenizer_path\
                --model_name $model_name\
                --rank_group $rank_group\
                --dataset $dataset --force_budget\
                --dtype bfloat16\
                --batch_size $bsz --prefix_len_list "${prefix_len_list[@]}" --max_gen_len 128\
                --temperature 0.0\
                --printoutput\
                --profiling\
                --num_total_runs $num_total_runs\
                --lora_adapter $drafter --lora_rank 16 --lora_alpha 32\
                --draft_length 2
            echo "================================================================"
            break
        else
            echo "GPUs 4,5 are busy. Sleeping 5 minutes."
            sleep 5m
        fi
    done

elif [ $OPTION -eq 6 ]; then
    while true; do
        busy_pids=$(nvidia-smi -i 6,7 --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')
        if [ -z "$busy_pids" ]; then
            echo "GPUs 6,7 are idle. Sleep 5 more minutes."
            sleep 5m
        fi

        if [ -z "$busy_pids" ]; then
            echo "GPUs 6,7 are idle even after 5 minutes. Running script."
            echo "================================================================"
            export CUDA_VISIBLE_DEVICES="6,7"
            n_proc_per_node=2
            rank_group="0 1"

            bsz=16
            num_total_runs=11
            dataset="AIME2025"
            prefix_len_list=(1024 2048 4096 8192 12288 16384 20480 24576 28672)

            echo "Running batch_size=$bsz, dataset=$dataset, prefix_len_list=$prefix_len_list"

            torchrun --standalone --nproc_per_node=$n_proc_per_node -m RSpec.run_benchmark\
                --backend mtp\
                --target $model\
                --tokenizer_path $tokenizer_path\
                --model_name $model_name\
                --rank_group $rank_group\
                --dataset $dataset --force_budget\
                --dtype bfloat16\
                --batch_size $bsz --prefix_len_list "${prefix_len_list[@]}" --max_gen_len 128\
                --temperature 0.0\
                --printoutput\
                --profiling\
                --num_total_runs $num_total_runs\
                --lora_adapter $drafter --lora_rank 16 --lora_alpha 32\
                --draft_length 1
            echo "================================================================"
            break
        else
            echo "GPUs 6,7 are busy. Sleeping 5 minutes."
            sleep 5m
        fi
    done

else
    echo "Invalid option: $OPTION"
    exit 1
fi