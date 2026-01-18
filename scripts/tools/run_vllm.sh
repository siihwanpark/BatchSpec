#!/bin/bash
set -euo pipefail

# Suppress Python warnings
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning,ignore::DeprecationWarning"

# Environment Setup
export ENABLE_INTRA_NODE_COMM=1
export FLASHINFER_JIT_VERBOSE=1
export TORCH_CUDA_ARCH_LIST=$(python -c "import torch; cc=torch.cuda.get_device_capability(0); print(f'{cc[0]}.{cc[1]}')")
echo "Auto-detected CUDA compute capability: ${TORCH_CUDA_ARCH_LIST}"

for model in DSL-8B Qwen3-8B Qwen3-14B Qwen3-32B; do
    if [ "$model" = "DSL-8B" ]; then
        model_path="/home/jovyan/sihwan-volume/checkpoints/DeepSeek-R1-Distill-Llama-8B"
    elif [ "$model" = "Qwen3-8B" ]; then
        model_path="/home/jovyan/sihwan-volume/checkpoints/Qwen3-8B"
    elif [ "$model" = "Qwen3-14B" ]; then
        model_path="/home/jovyan/sihwan-volume/checkpoints/Qwen3-14B"
    elif [ "$model" = "Qwen3-32B" ]; then
        model_path="/home/jovyan/sihwan-volume/checkpoints/Qwen3-32B"
    fi

    extra_args=()
    if [ "$model" != "DSL-8B" ]; then
        extra_args=(--top_k 20)
    fi

    for dataset in AIME2025 CodeForces GPQA-Diamond MMLU-Pro SuperGPQA; do
        input_jsonl="/home/jovyan/sihwan-volume/BatchSpec/benchmark_data/seed_prompts/${dataset}_1000.jsonl"
        output_dir="/home/jovyan/sihwan-volume/BatchSpec/benchmark_data/responses/${model}"

        python -m tools.prepare_benchmark.run_vllm \
            --model "$model_path" \
            --tp_size 4 \
            --input_jsonl "$input_jsonl" \
            --output_dir "$output_dir" --outfile_suffix "_sampling" \
            --max_gen_len 30720 \
            --max_model_len 32768 \
            --system_prompt "You are a helpful assistant." \
            --temperature 0.6 --top_p 0.95 \
            "${extra_args[@]}"

        python -m tools.prepare_benchmark.run_vllm \
            --model "$model_path" \
            --tp_size 4 \
            --input_jsonl "$input_jsonl" \
            --output_dir "$output_dir" --outfile_suffix "_greedy" \
            --max_gen_len 30720 \
            --max_model_len 32768 \
            --system_prompt "You are a helpful assistant." \
            --temperature 0.0
    done
done

