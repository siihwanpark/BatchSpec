#!/bin/bash

# =============================================================================
# Speculative Decoding Script
# =============================================================================

# Environment Setup
export ENABLE_INTRA_NODE_COMM=1
export TORCH_CUDA_ARCH_LIST=9.0
export FLASHINFER_JIT_VERBOSE=1

# Model Configuration
# model_name=Qwen/Qwen3-8B
# model=/home/mngcuser1/sihwan_workspace/Qwen3-8B/model.pth
# tokenizer_path=/home/mngcuser1/sihwan_workspace/Qwen3-8B
# drafter=/home/mngcuser1/sihwan_workspace/checkpoints/MTP_adapters/Qwen3-8B/am-qwen3-distilled-120k-rank16-lr2e-4-bsz4-SoftSCE-fused/step-30000/model.pth

model_name=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
model_path=/home/mngcuser1/sihwan_workspace/checkpoints/DeepSeek-R1-Distill-Llama-8B/model.pth
tokenizer_path=/home/mngcuser1/sihwan_workspace/checkpoints/DeepSeek-R1-Distill-Llama-8B
drafter=/home/mngcuser1/sihwan_workspace/checkpoints/MTP_adapters/DeepSeek-R1-Distill-Llama-8B/openthoughts-114k-rank16-lr2e-4-bsz8/step-30000/model.pth

if [ ! -f $model_path ] || [ ! -f $drafter ]; then
    echo "Model or drafter file not found! Please prepare the model first."
    exit 1
fi

num_total_runs=11
OPTION=${1:-0}
if [ $OPTION -eq 0 ]; then
    while true; do
        busy_pids=$(nvidia-smi -i 4 --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')
        if [ -z "$busy_pids" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: GPUs 4 are idle. Sleep 5 more minutes."
            sleep 5m
        fi

        if [ -z "$busy_pids" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: GPUs 4 are idle. Running script."
            break
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: GPUs 4 are busy (PIDs: $busy_pids). Sleep 15 minutes."
            sleep 15m
        fi
    done
    
    export CUDA_VISIBLE_DEVICES="4"
    n_proc_per_node=1
    rank_group="0"

    for dataset in CodeForces AIME2025 GPQA-Diamond; do
        prefix_len=2048
        max_gen_len=4096
        if [ $dataset == "CodeForces" ]; then
            num_questions_in_prompt=1
        else
            num_questions_in_prompt=4
        fi

        # Standard Autoregressive Decoding
        torchrun --standalone --nproc_per_node=$n_proc_per_node -m RSpec.run_e2e\
            --printoutput\
            --backend standard\
            --target $model\
            --tokenizer_path $tokenizer_path\
            --model_name $model_name\
            --rank_group $rank_group\
            --dataset $dataset --num_questions_in_prompt $num_questions_in_prompt --force_budget\
            --dtype bfloat16\
            --batch_size 16 --prefix_len $prefix_len --max_gen_len $max_gen_len\
            --temperature 0.0\
            --profiling\
            --num_total_runs $num_total_runs 2>&1 | tee -a logs/DSL-8B/bsz16_${dataset}_gen${max_gen_len}_tp1_standard_greedy.log

        torchrun --standalone --nproc_per_node=$n_proc_per_node -m RSpec.run_e2e\
            --printoutput\
            --backend standard\
            --target $model\
            --tokenizer_path $tokenizer_path\
            --model_name $model_name\
            --rank_group $rank_group\
            --dataset $dataset --num_questions_in_prompt $num_questions_in_prompt --force_budget\
            --dtype bfloat16\
            --batch_size 16 --prefix_len $prefix_len --max_gen_len $max_gen_len\
            --temperature 0.6 --top_p 0.95\
            --profiling\
            --num_total_runs $num_total_runs 2>&1 | tee -a logs/DSL-8B/bsz16_${dataset}_gen${max_gen_len}_tp1_standard_sampling.log

        # Self-Speculative Decoding with MTP
        draft_length_list=(4 "3 4" 3)
        for draft_length in "${draft_length_list[@]}"; do
            IFS=' ' read -r -a draft_length_array <<< "$draft_length"
            echo "draft_length_array=${draft_length_array[@]}"
            
            torchrun --standalone --nproc_per_node=$n_proc_per_node -m RSpec.run_e2e\
                --printoutput\
                --backend mtp\
                --target $model\
                --tokenizer_path $tokenizer_path\
                --model_name $model_name\
                --rank_group $rank_group\
                --dataset $dataset --num_questions_in_prompt $num_questions_in_prompt --force_budget\
                --dtype bfloat16\
                --batch_size 16 --prefix_len $prefix_len --max_gen_len $max_gen_len\
                --temperature 0.0\
                --profiling\
                --num_total_runs $num_total_runs\
                --lora_adapter $drafter --lora_rank 16 --lora_alpha 32 --draft_length "${draft_length_array[@]}"\
                2>&1 | tee -a logs/DSL-8B/bsz16_${dataset}_gen${max_gen_len}_tp1_mtp_k${draft_length_array[0]}_${draft_length_array[1]}_greedy.log
            
            torchrun --standalone --nproc_per_node=$n_proc_per_node -m RSpec.run_e2e\
                --printoutput\
                --backend mtp\
                --target $model\
                --tokenizer_path $tokenizer_path\
                --model_name $model_name\
                --rank_group $rank_group\
                --dataset $dataset --num_questions_in_prompt $num_questions_in_prompt --force_budget\
                --dtype bfloat16\
                --batch_size 16 --prefix_len $prefix_len --max_gen_len $max_gen_len\
                --temperature 0.6 --top_p 0.95 --top_k 20\
                --profiling\
                --num_total_runs $num_total_runs\
                --lora_adapter $drafter --lora_rank 16 --lora_alpha 32 --draft_length "${draft_length_array[@]}"\
                2>&1 | tee -a logs/DSL-8B/bsz16_${dataset}_gen${max_gen_len}_tp1_mtp_k${draft_length_array[0]}_${draft_length_array[1]}_sampling.log
        done

        echo "================================================"
    done


elif [ $OPTION -eq 1 ]; then
    while true; do
        busy_pids=$(nvidia-smi -i 5 --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')
        if [ -z "$busy_pids" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: GPUs 5 are idle. Sleep 5 more minutes."
            sleep 5m
        fi

        if [ -z "$busy_pids" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: GPUs 5 are idle. Running script."
            break
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: GPUs 5 are busy (PIDs: $busy_pids). Sleep 15 minutes."
            sleep 15m
        fi
    done

    export CUDA_VISIBLE_DEVICES="5"
    n_proc_per_node=1
    rank_group="0"

    for dataset in CodeForces AIME2025 GPQA-Diamond; do
        prefix_len=2048
        max_gen_len=4096
        if [ $dataset == "CodeForces" ]; then
            num_questions_in_prompt=1
        else
            num_questions_in_prompt=4
        fi

        # Standard Autoregressive Decoding
        torchrun --standalone --nproc_per_node=$n_proc_per_node -m RSpec.run_e2e\
            --printoutput\
            --backend standard\
            --target $model\
            --tokenizer_path $tokenizer_path\
            --model_name $model_name\
            --rank_group $rank_group\
            --dataset $dataset --num_questions_in_prompt $num_questions_in_prompt --force_budget\
            --dtype bfloat16\
            --batch_size 32 --prefix_len $prefix_len --max_gen_len $max_gen_len\
            --temperature 0.0\
            --profiling\
            --num_total_runs $num_total_runs 2>&1 | tee -a logs/DSL-8B/bsz32_${dataset}_gen${max_gen_len}_tp1_standard_greedy.log

        torchrun --standalone --nproc_per_node=$n_proc_per_node -m RSpec.run_e2e\
            --printoutput\
            --backend standard\
            --target $model\
            --tokenizer_path $tokenizer_path\
            --model_name $model_name\
            --rank_group $rank_group\
            --dataset $dataset --num_questions_in_prompt $num_questions_in_prompt --force_budget\
            --dtype bfloat16\
            --batch_size 32 --prefix_len $prefix_len --max_gen_len $max_gen_len\
            --temperature 0.6 --top_p 0.95 --top_k 20\
            --profiling\
            --num_total_runs $num_total_runs 2>&1 | tee -a logs/DSL-8B/bsz32_${dataset}_gen${max_gen_len}_tp1_standard_sampling.log

        # Self-Speculative Decoding with MTP
        draft_length_list=(4 "3 4" 3)
        for draft_length in "${draft_length_list[@]}"; do
            IFS=' ' read -r -a draft_length_array <<< "$draft_length"
            echo "draft_length_array=${draft_length_array[@]}"
            
            torchrun --standalone --nproc_per_node=$n_proc_per_node -m RSpec.run_e2e\
                --printoutput\
                --backend mtp\
                --target $model\
                --tokenizer_path $tokenizer_path\
                --model_name $model_name\
                --rank_group $rank_group\
                --dataset $dataset --num_questions_in_prompt $num_questions_in_prompt --force_budget\
                --dtype bfloat16\
                --batch_size 32 --prefix_len $prefix_len --max_gen_len $max_gen_len\
                --temperature 0.0\
                --profiling\
                --num_total_runs $num_total_runs\
                --lora_adapter $drafter --lora_rank 16 --lora_alpha 32 --draft_length "${draft_length_array[@]}"\
                2>&1 | tee -a logs/DSL-8B/bsz32_${dataset}_gen${max_gen_len}_tp1_mtp_k${draft_length_array[0]}_${draft_length_array[1]}_greedy.log
            
            torchrun --standalone --nproc_per_node=$n_proc_per_node -m RSpec.run_e2e\
                --printoutput\
                --backend mtp\
                --target $model\
                --tokenizer_path $tokenizer_path\
                --model_name $model_name\
                --rank_group $rank_group\
                --dataset $dataset --num_questions_in_prompt $num_questions_in_prompt --force_budget\
                --dtype bfloat16\
                --batch_size 32 --prefix_len $prefix_len --max_gen_len $max_gen_len\
                --temperature 0.6 --top_p 0.95 --top_k 20\
                --profiling\
                --num_total_runs $num_total_runs\
                --lora_adapter $drafter --lora_rank 16 --lora_alpha 32 --draft_length "${draft_length_array[@]}"\
                2>&1 | tee -a logs/DSL-8B/bsz32_${dataset}_gen${max_gen_len}_tp1_mtp_k${draft_length_array[0]}_${draft_length_array[1]}_sampling.log
        done

        echo "================================================"
    done

elif [ $OPTION -eq 2 ]; then
    while true; do
        busy_pids=$(nvidia-smi -i 6 --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')
        if [ -z "$busy_pids" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: GPUs 6 are idle. Sleep 5 more minutes."
            sleep 5m
        fi

        if [ -z "$busy_pids" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: GPUs 6 are idle. Running script."
            break
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: GPUs 6 are busy (PIDs: $busy_pids). Sleep 15 minutes."
            sleep 15m
        fi
    done

    export CUDA_VISIBLE_DEVICES="6"
    n_proc_per_node=1
    rank_group="0"

    for dataset in CodeForces AIME2025 GPQA-Diamond; do
        prefix_len=2048
        max_gen_len=4096
        if [ $dataset == "CodeForces" ]; then
            num_questions_in_prompt=1
        else
            num_questions_in_prompt=4
        fi

        # Standard Autoregressive Decoding
        torchrun --standalone --nproc_per_node=$n_proc_per_node -m RSpec.run_e2e\
            --printoutput\
            --backend standard\
            --target $model\
            --tokenizer_path $tokenizer_path\
            --model_name $model_name\
            --rank_group $rank_group\
            --dataset $dataset --num_questions_in_prompt $num_questions_in_prompt  --force_budget\
            --dtype bfloat16\
            --batch_size 64 --prefix_len $prefix_len --max_gen_len $max_gen_len\
            --temperature 0.0\
            --profiling\
            --num_total_runs $num_total_runs 2>&1 | tee -a logs/DSL-8B/bsz64_${dataset}_gen${max_gen_len}_tp1_standard_greedy.log

        # Self-Speculative Decoding with MTP
        draft_length_list=(4 "3 4" 3)
        for draft_length in "${draft_length_list[@]}"; do
            IFS=' ' read -r -a draft_length_array <<< "$draft_length"
            echo "draft_length_array=${draft_length_array[@]}"
            
            torchrun --standalone --nproc_per_node=$n_proc_per_node -m RSpec.run_e2e\
                --printoutput\
                --backend mtp\
                --target $model\
                --tokenizer_path $tokenizer_path\
                --model_name $model_name\
                --rank_group $rank_group\
                --dataset $dataset --num_questions_in_prompt $num_questions_in_prompt  --force_budget\
                --dtype bfloat16\
                --batch_size 64 --prefix_len $prefix_len --max_gen_len $max_gen_len\
                --temperature 0.0\
                --profiling\
                --num_total_runs $num_total_runs\
                --lora_adapter $drafter --lora_rank 16 --lora_alpha 32 --draft_length "${draft_length_array[@]}"\
                2>&1 | tee -a logs/DSL-8B/bsz64_${dataset}_gen${max_gen_len}_tp1_mtp_k${draft_length_array[0]}_${draft_length_array[1]}_greedy.log
        done

        echo "================================================"
    done

elif [ $OPTION -eq 3 ]; then
    while true; do
        busy_pids=$(nvidia-smi -i 7 --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')
        if [ -z "$busy_pids" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: GPUs 7 are idle. Sleep 5 more minutes."
            sleep 5m
        fi

        if [ -z "$busy_pids" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: GPUs 7 are idle. Running script."
            break
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: GPUs 7 are busy (PIDs: $busy_pids). Sleep 15 minutes."
            sleep 15m
        fi
    done

    export CUDA_VISIBLE_DEVICES="7"
    n_proc_per_node=1
    rank_group="0"

    for dataset in CodeForces AIME2025 GPQA-Diamond; do
        prefix_len=2048
        max_gen_len=4096
        if [ $dataset == "CodeForces" ]; then
            num_questions_in_prompt=1
        else
            num_questions_in_prompt=4
        fi

        # Standard Autoregressive Decoding
        torchrun --standalone --nproc_per_node=$n_proc_per_node -m RSpec.run_e2e\
            --printoutput\
            --backend standard\
            --target $model\
            --tokenizer_path $tokenizer_path\
            --model_name $model_name\
            --rank_group $rank_group\
            --dataset $dataset --num_questions_in_prompt $num_questions_in_prompt --force_budget\
            --dtype bfloat16\
            --batch_size 64 --prefix_len $prefix_len --max_gen_len $max_gen_len\
            --temperature 0.6 --top_p 0.95 --top_k 20\
            --profiling\
            --num_total_runs $num_total_runs 2>&1 | tee -a logs/DSL-8B/bsz64_${dataset}_gen${max_gen_len}_tp1_standard_sampling.log

        # Self-Speculative Decoding with MTP
        draft_length_list=("3 4" 3 "2 3")
        for draft_length in "${draft_length_list[@]}"; do
            IFS=' ' read -r -a draft_length_array <<< "$draft_length"
            echo "draft_length_array=${draft_length_array[@]}"
            
            torchrun --standalone --nproc_per_node=$n_proc_per_node -m RSpec.run_e2e\
                --printoutput\
                --backend mtp\
                --target $model\
                --tokenizer_path $tokenizer_path\
                --model_name $model_name\
                --rank_group $rank_group\
                --dataset $dataset --num_questions_in_prompt $num_questions_in_prompt --force_budget\
                --dtype bfloat16\
                --batch_size 64 --prefix_len $prefix_len --max_gen_len $max_gen_len\
                --temperature 0.6 --top_p 0.95 --top_k 20\
                --profiling\
                --num_total_runs $num_total_runs\
                --lora_adapter $drafter --lora_rank 16 --lora_alpha 32 --draft_length "${draft_length_array[@]}"\
                2>&1 | tee -a logs/DSL-8B/bsz64_${dataset}_gen${max_gen_len}_tp1_mtp_k${draft_length_array[0]}_${draft_length_array[1]}_sampling.log
        done

        echo "================================================"
    done

elif [ $OPTION -eq 4 ]; then
    while true; do
        busy_pids=$(nvidia-smi -i 4,5 --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')
        if [ -z "$busy_pids" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: GPUs 4,5 are idle. Sleep 5 more minutes."
            sleep 5m
        fi

        if [ -z "$busy_pids" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: GPUs 4,5 are idle. Running script."
            break
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: GPUs 4,5 are busy (PIDs: $busy_pids). Sleep 15 minutes."
            sleep 15m
        fi
    done

    export CUDA_VISIBLE_DEVICES="4,5"
    n_proc_per_node=2
    rank_group="0 1"

    for dataset in CodeForces AIME2025 GPQA-Diamond; do
        prefix_len=2048
        max_gen_len=4096

        if [ $dataset == "CodeForces" ]; then
            num_questions_in_prompt=1
        else
            num_questions_in_prompt=4
        fi

        # Standard Autoregressive Decoding
        torchrun --standalone --nproc_per_node=$n_proc_per_node -m RSpec.run_e2e\
            --printoutput\
            --backend standard\
            --target $model\
            --tokenizer_path $tokenizer_path\
            --model_name $model_name\
            --rank_group $rank_group\
            --dataset $dataset --num_questions_in_prompt $num_questions_in_prompt --force_budget\
            --dtype bfloat16\
            --batch_size 128 --prefix_len $prefix_len --max_gen_len $max_gen_len\
            --temperature 0.0\
            --profiling\
            --num_total_runs $num_total_runs 2>&1 | tee -a logs/DSL-8B/bsz128_${dataset}_gen${max_gen_len}_tp2_standard_greedy.log

        # Self-Speculative Decoding with MTP
        draft_length_list=("2 3" 3 "3 4")
        for draft_length in "${draft_length_list[@]}"; do
            IFS=' ' read -r -a draft_length_array <<< "$draft_length"
            echo "draft_length_array=${draft_length_array[@]}"
            torchrun --standalone --nproc_per_node=$n_proc_per_node -m RSpec.run_e2e\
                --printoutput\
                --backend mtp\
                --target $model\
                --tokenizer_path $tokenizer_path\
                --model_name $model_name\
                --rank_group $rank_group\
                --dataset $dataset --num_questions_in_prompt $num_questions_in_prompt --force_budget\
                --dtype bfloat16\
                --batch_size 128 --prefix_len $prefix_len --max_gen_len $max_gen_len\
                --temperature 0.0\
                --profiling\
                --num_total_runs $num_total_runs\
                --lora_adapter $drafter --lora_rank 16 --lora_alpha 32 --draft_length "${draft_length_array[@]}"\
                2>&1 | tee -a logs/DSL-8B/bsz128_${dataset}_gen${max_gen_len}_tp2_mtp_k${draft_length_array[0]}_${draft_length_array[1]}_greedy.log
        done
        echo "================================================"
    done

elif [ $OPTION -eq 5 ]; then
    while true; do
        busy_pids=$(nvidia-smi -i 6,7 --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')
        if [ -z "$busy_pids" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: GPUs 6,7 are idle. Sleep 5 more minutes."
            sleep 5m
        fi

        if [ -z "$busy_pids" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: GPUs 6,7 are idle. Running script."
            break
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: GPUs 6,7 are busy (PIDs: $busy_pids). Sleep 15 minutes."
            sleep 15m
        fi
    done

    export CUDA_VISIBLE_DEVICES="6,7"
    n_proc_per_node=2
    rank_group="0 1"

    for dataset in CodeForces AIME2025 GPQA-Diamond; do
        prefix_len=2048
        max_gen_len=4096

        if [ $dataset == "CodeForces" ]; then
            num_questions_in_prompt=1
        else
            num_questions_in_prompt=4
        fi

        torchrun --standalone --nproc_per_node=$n_proc_per_node -m RSpec.run_e2e\
            --printoutput\
            --backend standard\
            --target $model\
            --tokenizer_path $tokenizer_path\
            --model_name $model_name\
            --rank_group $rank_group\
            --dataset $dataset --num_questions_in_prompt $num_questions_in_prompt --force_budget\
            --dtype bfloat16\
            --batch_size 128 --prefix_len $prefix_len --max_gen_len $max_gen_len\
            --temperature 0.6 --top_p 0.95 --top_k 20\
            --profiling\
            --num_total_runs $num_total_runs 2>&1 | tee -a logs/DSL-8B/bsz128_${dataset}_gen${max_gen_len}_tp2_standard_sampling.log

        draft_length_list=("1 2" 2 "2 3")
        for draft_length in "${draft_length_list[@]}"; do
            IFS=' ' read -r -a draft_length_array <<< "$draft_length"
            echo "draft_length_array=${draft_length_array[@]}"
            torchrun --standalone --nproc_per_node=$n_proc_per_node -m RSpec.run_e2e\
                --printoutput\
                --backend mtp\
                --target $model\
                --tokenizer_path $tokenizer_path\
                --model_name $model_name\
                --rank_group $rank_group\
                --dataset $dataset --num_questions_in_prompt $num_questions_in_prompt --force_budget\
                --dtype bfloat16\
                --batch_size 128 --prefix_len $prefix_len --max_gen_len $max_gen_len\
                --temperature 0.6 --top_p 0.95 --top_k 20\
                --profiling\
                --num_total_runs $num_total_runs\
                --lora_adapter $drafter --lora_rank 16 --lora_alpha 32 --draft_length "${draft_length_array[@]}"\
                2>&1 | tee -a logs/DSL-8B/bsz128_${dataset}_gen${max_gen_len}_tp2_mtp_k${draft_length_array[0]}_${draft_length_array[1]}_sampling.log
        done
        echo "================================================"
    done

elif [ $OPTION -eq 6 ]; then
    # while true; do
    #     busy_pids=$(nvidia-smi -i 0,1,2,3 --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')
    #     if [ -z "$busy_pids" ]; then
    #         echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: GPUs 0,1,2,3 are idle. Sleep 5 more minutes."
    #         sleep 5m
    #     fi
    # done

    export CUDA_VISIBLE_DEVICES="0,1,2,3"
    n_proc_per_node=4
    rank_group="0 1 2 3"

    for dataset in CodeForces AIME2025 GPQA-Diamond; do
        prefix_len=2048
        max_gen_len=4096
        if [ $dataset = "CodeForces" ]; then
            num_questions_in_prompt=1
        else
            num_questions_in_prompt=4
        fi

        # Standard Autoregressive Decoding
        torchrun --standalone --nproc_per_node=$n_proc_per_node -m RSpec.run_e2e\
            --printoutput\
            --backend standard\
            --target $model\
            --tokenizer_path $tokenizer_path\
            --model_name $model_name\
            --rank_group $rank_group\
            --dataset $dataset --num_questions_in_prompt $num_questions_in_prompt --force_budget\
            --dtype bfloat16\
            --batch_size 256 --prefix_len $prefix_len --max_gen_len $max_gen_len\
            --temperature 0.0\
            --profiling\
            --num_total_runs $num_total_runs 2>&1 | tee -a logs/DSL-8B/bsz256_${dataset}_gen${max_gen_len}_tp4_standard_greedy.log

        torchrun --standalone --nproc_per_node=$n_proc_per_node -m RSpec.run_e2e\
            --printoutput\
            --backend standard\
            --target $model\
            --tokenizer_path $tokenizer_path\
            --model_name $model_name\
            --rank_group $rank_group\
            --dataset $dataset --num_questions_in_prompt $num_questions_in_prompt --force_budget\
            --dtype bfloat16\
            --batch_size 256 --prefix_len $prefix_len --max_gen_len $max_gen_len\
            --temperature 0.6 --top_p 0.95\
            --profiling\
            --num_total_runs $num_total_runs 2>&1 | tee -a logs/DSL-8B/bsz256_${dataset}_gen${max_gen_len}_tp4_standard_sampling.log


        draft_length_list=("1 2" 2 "2 3")
        for draft_length in "${draft_length_list[@]}"; do
            IFS=' ' read -r -a draft_length_array <<< "$draft_length"
            echo "draft_length_array=${draft_length_array[@]}"
            # Self-Speculative Decoding with MTP
            torchrun --standalone --nproc_per_node=$n_proc_per_node -m RSpec.run_e2e\
                --printoutput\
                --backend mtp\
                --target $model\
                --tokenizer_path $tokenizer_path\
                --model_name $model_name\
                --rank_group $rank_group\
                --dataset $dataset --num_questions_in_prompt $num_questions_in_prompt --force_budget\
                --dtype bfloat16\
                --batch_size 256 --prefix_len $prefix_len --max_gen_len $max_gen_len\
                --temperature 0.0\
                --profiling\
                --num_total_runs $num_total_runs\
                --lora_adapter $drafter --lora_rank 16 --lora_alpha 32 --draft_length "${draft_length_array[@]}"\
                2>&1 | tee -a logs/DSL-8B/bsz256_${dataset}_gen${max_gen_len}_tp4_mtp_k${draft_length_array[0]}_${draft_length_array[1]}_greedy.log
        done

        draft_length_list=(1 "1 2" 2)
        for draft_length in "${draft_length_list[@]}"; do
            IFS=' ' read -r -a draft_length_array <<< "$draft_length"
            echo "draft_length_array=${draft_length_array[@]}"
            # Self-Speculative Decoding with MTP
            torchrun --standalone --nproc_per_node=$n_proc_per_node -m RSpec.run_e2e\
                --printoutput\
                --backend mtp\
                --target $model\
                --tokenizer_path $tokenizer_path\
                --model_name $model_name\
                --rank_group $rank_group\
                --dataset $dataset --num_questions_in_prompt $num_questions_in_prompt --force_budget\
                --dtype bfloat16\
                --batch_size 256 --prefix_len $prefix_len --max_gen_len $max_gen_len\
                --temperature 0.6 --top_p 0.95\
                --profiling\
                --num_total_runs $num_total_runs\
                --lora_adapter $drafter --lora_rank 16 --lora_alpha 32 --draft_length "${draft_length_array[@]}"\
                2>&1 | tee -a logs/DSL-8B/bsz256_${dataset}_gen${max_gen_len}_tp4_mtp_k${draft_length_array[0]}_${draft_length_array[1]}_sampling.log
            
        done
    done
        
fi