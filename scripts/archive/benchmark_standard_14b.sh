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
base_ckpt_dir=/home/jovyan/sihwan-volume/checkpoints
model_name=Qwen3-14B
model_path=$base_ckpt_dir/Qwen3-14B/model.pth
tokenizer_path=$base_ckpt_dir/Qwen3-14B

OPTION=${1:-0}

wait_for_gpus_idle() {
  local gpus_csv="$1" # e.g., "4,5"
  local idle_confirm_sleep="${2:-5m}" # optional: sleep when idle before running
  local busy_sleep="${3:-15m}" # optional: sleep when busy

  while true; do
    # Query running compute app PIDs on given GPUs
    local busy_pids
    busy_pids=$(
      nvidia-smi -i "$gpus_csv" --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
        | tr -d '[:space:]' \
        | paste -sd, -
    )

    if [ -z "$busy_pids" ]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: GPUs ${gpus_csv} are idle. Sleep ${idle_confirm_sleep} more."
      sleep "$idle_confirm_sleep"

      # Re-check once to avoid racing with a new job starting right after we checked.
      busy_pids=$(
        nvidia-smi -i "$gpus_csv" --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
          | tr -d '[:space:]' \
          | paste -sd, -
      )
      if [ -z "$busy_pids" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: GPUs ${gpus_csv} are idle. Proceeding."
        break
      else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: GPUs ${gpus_csv} became busy (PIDs: ${busy_pids}). Sleep ${busy_sleep}."
        sleep "$busy_sleep"
      fi
    else
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: GPUs ${gpus_csv} are busy (PIDs: ${busy_pids}). Sleep ${busy_sleep}."
      sleep "$busy_sleep"
    fi
  done
}

if [ $OPTION == 0 ]; then
wait_for_gpus_idle "0,1,2,3" "3m" "5m"
for bsz in 256 128 64;do
	if [ $bsz == 128 ]; then
		prefix_len_list=(1024 2048 4096 8192 12288 16384 20480 24576 28672)
	elif [ $bsz == 256 ]; then
		prefix_len_list=(1024 2048 4096 6144 8192 10240 12288 14336)
	fi

	torchrun --standalone --nproc_per_node=4 -m batchspec.run\
		--backend standalone\
		--checkpoint_path $model_path\
		--tokenizer_path $tokenizer_path\
		--model_name $model_name\
		--rank_group 0 1 2 3\
		--dataset AIME2025\
		--dtype bfloat16\
		--batch_size $bsz --prefix_len_list "${prefix_len_list[@]}" --max_gen_len 128\
		--temperature 0.0 --force_budget\
		--printoutput\
		--profiling\
		--num_total_runs 6

	torchrun --standalone --nproc_per_node=4 -m batchspec.run\
		--backend standard\
		--checkpoint_path $model_path\
		--tokenizer_path $tokenizer_path\
		--model_name $model_name\
		--rank_group 0 1 2 3\
		--dataset AIME2025\
		--dtype bfloat16\
		--batch_size $bsz --prefix_len_list "${prefix_len_list[@]}" --max_gen_len 128\
		--temperature 0.6 --top_p 0.95 --top_k 20 --force_budget\
		--printoutput\
		--profiling\
		--num_total_runs 6
done

elif [ $OPTION == 1 ]; then
wait_for_gpus_idle "0,1" "10m" "5m"
export CUDA_VISIBLE_DEVICES="0,1"
nproc_per_node=2
rank_group="0 1"

bsz=32
prefix_len_list=(1024 2048 4096 8192 12288 16384 20480 24576 28672)
torchrun --standalone --nproc_per_node=$nproc_per_node -m batchspec.run\
	--backend standalone\
	--checkpoint_path $model_path\
	--tokenizer_path $tokenizer_path\
	--model_name $model_name\
	--rank_group $rank_group\
	--dataset AIME2025\
	--dtype bfloat16\
	--batch_size $bsz --prefix_len_list "${prefix_len_list[@]}" --max_gen_len 128\
	--temperature 0.0 --force_budget\
	--printoutput\
	--profiling\
	--num_total_runs 6

elif [ $OPTION == 2 ]; then
wait_for_gpus_idle "2,3" "10m" "5m"
export CUDA_VISIBLE_DEVICES="2,3"
nproc_per_node=2
rank_group="0 1"

bsz=32
prefix_len_list=(1024 2048 4096 8192 12288 16384 20480 24576 28672)
torchrun --standalone --nproc_per_node=$nproc_per_node -m batchspec.run\
	--backend standalone\
	--checkpoint_path $model_path\
	--tokenizer_path $tokenizer_path\
	--model_name $model_name\
	--rank_group $rank_group\
	--dataset AIME2025\
	--dtype bfloat16\
	--batch_size $bsz --prefix_len_list "${prefix_len_list[@]}" --max_gen_len 128\
	--temperature 0.0 --top_p 0.95 --top_k 20 --force_budget\
	--printoutput\
	--profiling\
	--num_total_runs 6

elif [ $OPTION == 3 ]; then
wait_for_gpus_idle "0" "15m" "5m"
export CUDA_VISIBLE_DEVICES="0"
nproc_per_node=1
rank_group="0"

bsz=16
prefix_len_list=(1024 2048 4096 8192 12288 16384 20480 24576 28672)
torchrun --standalone --nproc_per_node=$nproc_per_node -m batchspec.run\
	--backend standalone\
	--checkpoint_path $model_path\
	--tokenizer_path $tokenizer_path\
	--model_name $model_name\
	--rank_group $rank_group\
	--dataset AIME2025\
	--dtype bfloat16\
	--batch_size $bsz --prefix_len_list "${prefix_len_list[@]}" --max_gen_len 128\
	--temperature 0.0 --force_budget\
	--printoutput\
	--profiling\
	--num_total_runs 6

elif [ $OPTION == 4 ]; then
wait_for_gpus_idle "1" "15m" "5m"
export CUDA_VISIBLE_DEVICES="1"
nproc_per_node=1
rank_group="0"

bsz=16
prefix_len_list=(1024 2048 4096 8192 12288 16384 20480 24576 28672)
torchrun --standalone --nproc_per_node=$nproc_per_node -m batchspec.run\
	--backend standalone\
	--checkpoint_path $model_path\
	--tokenizer_path $tokenizer_path\
	--model_name $model_name\
	--rank_group $rank_group\
	--dataset AIME2025\
	--dtype bfloat16\
	--batch_size $bsz --prefix_len_list "${prefix_len_list[@]}" --max_gen_len 128\
	--temperature 0.0 --top_p 0.95 --top_k 20 --force_budget\
	--printoutput\
	--profiling\
	--num_total_runs 6

fi
