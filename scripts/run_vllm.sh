# !/bin/bash

export TORCH_CUDA_ARCH_LIST=9.0

OPTION=${1:-0}
if [ $OPTION -eq 0 ]; then
    while true; do
        busy_pids=$(nvidia-smi -i 0,1 --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')

        if [ -z "$busy_pids" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: GPUs 0-1 are idle. Running script."
            export CUDA_VISIBLE_DEVICES=0,1
            python run_vllm.py \
                --model ~/sihwan_workspace/Qwen3-8B \
                --tp_size 2 \
                --input_jsonl data/seed_prompts/AIME2025_1000.jsonl \
                --max_gen_len 30720 \
                --max_model_len 32768 \
                --temperature 0.6 --top_p 0.95 --top_k 20
            break
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: GPUs 0-1 are busy (PIDs: $busy_pids). Sleep 15 minutes."
            sleep 15m
        fi
    done
elif [ $OPTION -eq 1 ]; then
    while true; do
        busy_pids=$(nvidia-smi -i 2,3 --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')

        if [ -z "$busy_pids" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: GPUs 2-3 are idle. Running script."
            export CUDA_VISIBLE_DEVICES=2,3
            python run_vllm.py \
                --model ~/sihwan_workspace/Qwen3-8B \
                --tp_size 2 \
                --input_jsonl data/seed_prompts/LiveMathBench_1000.jsonl \
                --max_gen_len 30720 \
                --max_model_len 32768 \
                --temperature 0.6 --top_p 0.95 --top_k 20
            break
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: GPUs 2-3 are busy (PIDs: $busy_pids). Sleep 15 minutes."
            sleep 15m
        fi
    done
elif [ $OPTION -eq 2 ]; then
    while true; do
        busy_pids=$(nvidia-smi -i 4,5 --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')

        if [ -z "$busy_pids" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: GPUs 4-5 are idle. Running script."
            export CUDA_VISIBLE_DEVICES=4,5
            python run_vllm.py \
                --model ~/sihwan_workspace/Qwen3-8B \
                --tp_size 2 \
                --input_jsonl data/seed_prompts/LiveCodeBench_1000.jsonl \
                --max_gen_len 30720 \
                --max_model_len 32768 \
                --temperature 0.6 --top_p 0.95 --top_k 20
            break
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: GPUs 4-5 are busy (PIDs: $busy_pids). Sleep 15 minutes."
            sleep 15m
        fi
    done
elif [ $OPTION -eq 3 ]; then
    while true; do
        busy_pids=$(nvidia-smi -i 6,7 --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')

        if [ -z "$busy_pids" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: GPUs 6-7 are idle. Running script."
            export CUDA_VISIBLE_DEVICES=6,7
            python run_vllm.py \
                --model ~/sihwan_workspace/Qwen3-8B \
                --tp_size 2 \
                --input_jsonl data/seed_prompts/CodeForces_1000.jsonl \
                --max_gen_len 30720 \
                --max_model_len 32768 \
                --temperature 0.6 --top_p 0.95 --top_k 20
            break
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: GPUs 6-7 are busy (PIDs: $busy_pids). Sleep 15 minutes."
            sleep 15m
        fi
    done
else
    echo "Invalid option"
    exit 1
fi