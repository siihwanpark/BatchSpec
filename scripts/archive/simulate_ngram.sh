#!/bin/bash

MODELS=("Qwen3-8B" "DSL-8B" "Qwen3-14B")
DATASETS=("AIME2025" "CodeForces" "MMLU-Pro")

for model in "${MODELS[@]}"; do
    if [ "$model" == "Qwen3-8B" ]; then
        model_name="Qwen/Qwen3-8B"
    elif [ "$model" == "DSL-8B" ]; then
        model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    elif [ "$model" == "Qwen3-14B" ]; then
        model_name="Qwen/Qwen3-14B"
    fi

    for dataset in "${DATASETS[@]}"; do
        python -m tests.simulate_ngram_drafter \
         --model $model_name \
         --dataset $dataset \
         --draft_length 10 \
         --max_ngram_size 3 \
         --max_samples 100
    done
done