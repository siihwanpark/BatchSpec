#!/usr/bin/env bash

# =============================================================================
# Speculative Decoding Benchmark Script
# =============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils.sh"

enable_graceful_exit
register_cleanup 'echo "[cleanup] releasing resources..."'
register_cleanup 'pkill -P $$ || true'

set_torch_cuda_arch_list 0
log "Auto-detected CUDA compute capability: ${TORCH_CUDA_ARCH_LIST}"

# Model Configuration
OPTION=${1:-0}

if [ $OPTION == 0 ]; then
for model in Qwen3-8B DSL-8B;do
	if [ $model == "Qwen3-8B" ]; then
		extra_args=(--top_k 20)
	else
		extra_args=()
	fi

	read model_name model_path tokenizer_path <<< "$(get_model_path $model)"

	for bsz in 256 128;do
		if [ $bsz -eq 128 ]; then
			prefix_len_list=(1024 2048 4096 8192 12288 16384 20480 24576 28672)
		elif [ $bsz -eq 256 ]; then
			prefix_len_list=(1024 2048 4096 6144 8192 10240 12288 14336)
		fi

		torchrun --standalone --nproc_per_node=4 -m batchspec.run \
			--backend standard \
			--checkpoint_path $model_path \
			--tokenizer_path $tokenizer_path \
			--model_name $model_name \
			--rank_group 0 1 2 3 \
			--dataset AIME2025 \
			--dtype bfloat16 \
			--batch_size $bsz --prefix_len_list "${prefix_len_list[@]}" --max_gen_len 128 \
			--temperature 0.0 --force_budget \
			--printoutput \
			--profiling \
			--num_total_runs 6

		torchrun --standalone --nproc_per_node=4 -m batchspec.run \
			--backend standard \
			--checkpoint_path $model_path \
			--tokenizer_path $tokenizer_path \
			--model_name $model_name \
			--rank_group 0 1 2 3 \
			--dataset AIME2025 \
			--dtype bfloat16 \
			--batch_size $bsz --prefix_len_list "${prefix_len_list[@]}" --max_gen_len 128 \
			--temperature 0.6 --top_p 0.95 --force_budget \
			--printoutput \
			--profiling \
			--num_total_runs 6 \
			"${extra_args[@]}"
	done
done

read model_name model_path tokenizer_path <<< "$(get_model_path Qwen3-14B)"
for bsz in 256 128 64;do
	if [ $bsz -eq 64 ] || [ $bsz -eq 128 ]; then
		prefix_len_list=(1024 2048 4096 8192 12288 16384 20480 24576 28672)
	elif [ $bsz -eq 256 ]; then
		prefix_len_list=(1024 2048 4096 6144 8192 10240 12288 14336)
	fi

	torchrun --standalone --nproc_per_node=4 -m batchspec.run \
		--backend standard \
		--checkpoint_path $model_path \
		--tokenizer_path $tokenizer_path \
		--model_name $model_name \
		--rank_group 0 1 2 3 \
		--dataset AIME2025 \
		--dtype bfloat16 \
		--batch_size $bsz --prefix_len_list "${prefix_len_list[@]}" --max_gen_len 128 \
		--temperature 0.0 --force_budget \
		--printoutput \
		--profiling \
		--num_total_runs 6

	torchrun --standalone --nproc_per_node=4 -m batchspec.run \
		--backend standard \
		--checkpoint_path $model_path \
		--tokenizer_path $tokenizer_path \
		--model_name $model_name \
		--rank_group 0 1 2 3 \
		--dataset AIME2025 \
		--dtype bfloat16 \
		--batch_size $bsz --prefix_len_list "${prefix_len_list[@]}" --max_gen_len 128 \
		--temperature 0.6 --top_p 0.95 --top_k 20 --force_budget \
		--printoutput \
		--profiling \
		--num_total_runs 6
done

elif [ $OPTION == 1 ]; then
wait_for_gpus_idle "0,1" "3m" "5m"
export CUDA_VISIBLE_DEVICES="0,1"
nproc_per_node=2
rank_group="0 1"

for model in Qwen3-8B DSL-8B;do
	read model_name model_path tokenizer_path <<< "$(get_model_path $model)"

	bsz=64
	prefix_len_list=(1024 2048 4096 8192 12288 16384 20480 24576 28672)
	torchrun --standalone --nproc_per_node=$nproc_per_node -m batchspec.run \
		--backend standard \
		--checkpoint_path $model_path \
		--tokenizer_path $tokenizer_path \
		--model_name $model_name \
		--rank_group $rank_group \
		--dataset AIME2025 \
		--dtype bfloat16 \
		--batch_size $bsz --prefix_len_list "${prefix_len_list[@]}" --max_gen_len 128 \
		--temperature 0.0 --force_budget \
		--printoutput \
		--profiling \
		--num_total_runs 6

done

read model_name model_path tokenizer_path <<< "$(get_model_path Qwen3-14B)"
bsz=32
prefix_len_list=(1024 2048 4096 8192 12288 16384 20480 24576 28672)
torchrun --standalone --nproc_per_node=$nproc_per_node -m batchspec.run \
	--backend standard \
	--checkpoint_path $model_path \
	--tokenizer_path $tokenizer_path \
	--model_name $model_name \
	--rank_group $rank_group \
	--dataset AIME2025 \
	--dtype bfloat16 \
	--batch_size $bsz --prefix_len_list "${prefix_len_list[@]}" --max_gen_len 128 \
	--temperature 0.0 --force_budget \
	--printoutput \
	--profiling \
	--num_total_runs 6

elif [ $OPTION == 2 ]; then
wait_for_gpus_idle "2,3" "3m" "5m"
export CUDA_VISIBLE_DEVICES="2,3"
nproc_per_node=2
rank_group="0 1"

for model in Qwen3-8B DSL-8B;do
	if [ $model == "Qwen3-8B" ]; then
		extra_args=(--top_k 20)
	else
		extra_args=()
	fi

	read model_name model_path tokenizer_path <<< "$(get_model_path $model)"

	bsz=64
	prefix_len_list=(1024 2048 4096 8192 12288 16384 20480 24576 28672)
	torchrun --standalone --nproc_per_node=$nproc_per_node -m batchspec.run \
		--backend standard \
		--checkpoint_path $model_path \
		--tokenizer_path $tokenizer_path \
		--model_name $model_name \
		--rank_group $rank_group \
		--dataset AIME2025 \
		--dtype bfloat16 \
		--batch_size $bsz --prefix_len_list "${prefix_len_list[@]}" --max_gen_len 128 \
		--temperature 0.6 --top_p 0.95 --force_budget \
		--printoutput \
		--profiling \
		--num_total_runs 6 \
		"${extra_args[@]}"

done

read model_name model_path tokenizer_path <<< "$(get_model_path Qwen3-14B)"
bsz=32
prefix_len_list=(1024 2048 4096 8192 12288 16384 20480 24576 28672)
torchrun --standalone --nproc_per_node=$nproc_per_node -m batchspec.run \
	--backend standard \
	--checkpoint_path $model_path \
	--tokenizer_path $tokenizer_path \
	--model_name $model_name \
	--rank_group $rank_group \
	--dataset AIME2025 \
	--dtype bfloat16 \
	--batch_size $bsz --prefix_len_list "${prefix_len_list[@]}" --max_gen_len 128 \
	--temperature 0.6 --top_p 0.95 --top_k 20 --force_budget \
	--printoutput \
	--profiling \
	--num_total_runs 6

elif [ $OPTION == 3 ]; then
wait_for_gpus_idle "0" "10m" "5m"
export CUDA_VISIBLE_DEVICES="0"
nproc_per_node=1
rank_group="0"

for model in Qwen3-8B DSL-8B;do
	read model_name model_path tokenizer_path <<< "$(get_model_path $model)"

	bsz=32
	prefix_len_list=(1024 2048 4096 8192 12288 16384 20480 24576 28672)
	torchrun --standalone --nproc_per_node=$nproc_per_node -m batchspec.run \
		--backend standard \
		--checkpoint_path $model_path \
		--tokenizer_path $tokenizer_path \
		--model_name $model_name \
		--rank_group $rank_group \
		--dataset AIME2025 \
		--dtype bfloat16 \
		--batch_size $bsz --prefix_len_list "${prefix_len_list[@]}" --max_gen_len 128 \
		--temperature 0.0 --force_budget \
		--printoutput \
		--profiling \
		--num_total_runs 6
done

read model_name model_path tokenizer_path <<< "$(get_model_path Qwen3-14B)"
bsz=16
prefix_len_list=(1024 2048 4096 8192 12288 16384 20480 24576 28672)
torchrun --standalone --nproc_per_node=$nproc_per_node -m batchspec.run \
	--backend standard \
	--checkpoint_path $model_path \
	--tokenizer_path $tokenizer_path \
	--model_name $model_name \
	--rank_group $rank_group \
	--dataset AIME2025 \
	--dtype bfloat16 \
	--batch_size $bsz --prefix_len_list "${prefix_len_list[@]}" --max_gen_len 128 \
	--temperature 0.0 --force_budget \
	--printoutput \
	--profiling \
	--num_total_runs 6

elif [ $OPTION == 4 ]; then
# wait_for_gpus_idle "1" "10m" "5m"
export CUDA_VISIBLE_DEVICES="1"
nproc_per_node=1
rank_group="0"

for model in Qwen3-8B DSL-8B;do
	if [ $model == "Qwen3-8B" ]; then
		extra_args=(--top_k 20)
	else
		extra_args=()
	fi
	read model_name model_path tokenizer_path <<< "$(get_model_path $model)"

	bsz=32
	prefix_len_list=(1024 2048 4096 8192 12288 16384 20480 24576 28672)
	torchrun --standalone --nproc_per_node=$nproc_per_node -m batchspec.run \
		--backend standard \
		--checkpoint_path $model_path \
		--tokenizer_path $tokenizer_path \
		--model_name $model_name \
		--rank_group $rank_group \
		--dataset AIME2025 \
		--dtype bfloat16 \
		--batch_size $bsz --prefix_len_list "${prefix_len_list[@]}" --max_gen_len 128 \
		--temperature 0.6 --top_p 0.95 --force_budget \
		--printoutput \
		--profiling \
		--num_total_runs 6 \
		"${extra_args[@]}"
done

read model_name model_path tokenizer_path <<< "$(get_model_path Qwen3-14B)"
bsz=16
prefix_len_list=(1024 2048 4096 8192 12288 16384 20480 24576 28672)
torchrun --standalone --nproc_per_node=$nproc_per_node -m batchspec.run \
	--backend standard \
	--checkpoint_path $model_path \
	--tokenizer_path $tokenizer_path \
	--model_name $model_name \
	--rank_group $rank_group \
	--dataset AIME2025 \
	--dtype bfloat16 \
	--batch_size $bsz --prefix_len_list "${prefix_len_list[@]}" --max_gen_len 128 \
	--temperature 0.6 --top_p 0.95 --top_k 20 --force_budget \
	--printoutput \
	--profiling \
	--num_total_runs 6

elif [ $OPTION == 5 ]; then
wait_for_gpus_idle "2" "10m" "5m"
export CUDA_VISIBLE_DEVICES="2"
nproc_per_node=1
rank_group="0"

for model in Qwen3-8B DSL-8B;do
	read model_name model_path tokenizer_path <<< "$(get_model_path $model)"

	bsz=16
	prefix_len_list=(1024 2048 4096 8192 12288 16384 20480 24576 28672)
	torchrun --standalone --nproc_per_node=$nproc_per_node -m batchspec.run \
		--backend standard \
		--checkpoint_path $model_path \
		--tokenizer_path $tokenizer_path \
		--model_name $model_name \
		--rank_group $rank_group \
		--dataset AIME2025 \
		--dtype bfloat16 \
		--batch_size $bsz --prefix_len_list "${prefix_len_list[@]}" --max_gen_len 128 \
		--temperature 0.0 --force_budget \
		--printoutput \
		--profiling \
		--num_total_runs 6
done

elif [ $OPTION == 6 ]; then
wait_for_gpus_idle "3" "10m" "5m"
export CUDA_VISIBLE_DEVICES="3"
nproc_per_node=1
rank_group="0"

for model in Qwen3-8B DSL-8B;do
	if [ $model == "Qwen3-8B" ]; then
		extra_args=(--top_k 20)
	else
		extra_args=()
	fi
	read model_name model_path tokenizer_path <<< "$(get_model_path $model)"

	bsz=16
	prefix_len_list=(1024 2048 4096 8192 12288 16384 20480 24576 28672)
	torchrun --standalone --nproc_per_node=$nproc_per_node -m batchspec.run \
		--backend standard \
		--checkpoint_path $model_path \
		--tokenizer_path $tokenizer_path \
		--model_name $model_name \
		--rank_group $rank_group \
		--dataset AIME2025 \
		--dtype bfloat16 \
		--batch_size $bsz --prefix_len_list "${prefix_len_list[@]}" --max_gen_len 128 \
		--temperature 0.6 --top_p 0.95 --force_budget \
		--printoutput \
		--profiling \
		--num_total_runs 6 \
		"${extra_args[@]}"

done

fi