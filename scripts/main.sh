#!/usr/bin/env bash
set -u
set -o pipefail

# Import utils.sh
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils.sh"

# Set environment variables
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning,ignore::DeprecationWarning"
export FLASHINFER_JIT_VERBOSE=1

# Register cleanup functions
enable_graceful_exit
register_cleanup 'echo "[CLEANUP] releasing resources..."'

# Set CUDA architecture list
set_torch_cuda_arch_list 0
log "Auto-detected CUDA compute capability: ${TORCH_CUDA_ARCH_LIST}"

# Print usage
usage() {
  cat <<EOF
Usage:
  main.sh --gpus "0,1" --nproc 2 --rank_group "0 1" \\
    --model Qwen3-8B|DSL-8B|Qwen3-14B --backend standard|standalone|eagle|magicdec|mtp --draft_length 1|2|3|4 \\
    --dataset AIME2025|CodeForces|GPQA-Diamond|MMLU-Pro|SuperGPQA --mode greedy|sampling \\
    --bsz 16|32|64|128|256 --prefix_profile 8k|16k|32k|debug \\
    [--top_p 0.95] [--top_k 20] [--temperature 0.6] \\
    [--dtype bfloat16] [--max_gen_len 128] [--num_total_runs 6]
EOF
}

# Set default values
DATASET="AIME2025"
DTYPE="bfloat16"
MAX_GEN_LEN=128
NUM_TOTAL_RUNS=11

TOP_P="0.95"
TOP_K=""
TEMPERATURE=""

GPUS=""
NPROC=""
RANK_GROUP=""

MODEL=""
BACKEND=""
DRAFT_LENGTH=""

MODE=""
BSZ=""
PREFIX_PROFILE=""

# Parse arguments
while [ $# -gt 0 ]; do
  case "$1" in
    --gpus) GPUS="$2"; shift 2;;
    --nproc) NPROC="$2"; shift 2;;
    --rank_group) RANK_GROUP="$2"; shift 2;;

    --model) MODEL="$2"; shift 2;;
    --backend) BACKEND="$2"; shift 2;;
    --draft_length) DRAFT_LENGTH="$2"; shift 2;;

    --dataset) DATASET="$2"; shift 2;;
    --mode) MODE="$2"; shift 2;;
    --bsz) BSZ="$2"; shift 2;;
    --prefix_profile) PREFIX_PROFILE="$2"; shift 2;;

    --dtype) DTYPE="$2"; shift 2;;
    --max_gen_len) MAX_GEN_LEN="$2"; shift 2;;
    --num_total_runs) NUM_TOTAL_RUNS="$2"; shift 2;;

    --top_p) TOP_P="$2"; shift 2;;
    --top_k) TOP_K="$2"; shift 2;;
    --temperature) TEMPERATURE="$2"; shift 2;;

    -h|--help) usage; exit 0;;
    *) die "Unknown arg: $1";;
  esac
done

# Check required arguments
[ -n "$GPUS" ] || die "--gpus is required"
[ -n "$NPROC" ] || die "--nproc is required"
[ -n "$RANK_GROUP" ] || die "--rank_group is required"

[ -n "$MODEL" ] || die "--model is required"
[ -n "$BACKEND" ] || die "--backend is required"
[ -n "$DRAFT_LENGTH" ] || die "--draft_length is required"

[ -n "$MODE" ] || die "--mode is required"
[ -n "$BSZ" ] || die "--bsz is required"
[ -n "$PREFIX_PROFILE" ] || die "--prefix_profile is required"
[ -n "$DATASET" ] || die "--dataset is required"

# Set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES="$GPUS"
log "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"

# Get model name, model path, and tokenizer path
read -r model_name model_path tokenizer_path <<< "$(get_model_path "$MODEL")"

# Get prefix length list
prefix_list_for() {
  local profile="$1"
  case "$profile" in
    8k)
      echo "1024 2048 4096 6144"
      ;;
    12k)
      echo "1024 2048 4096 8192"
      ;;
    16k)
      echo "1024 2048 4096 8192 12288"
      ;;
    24k)
      echo "1024 2048 4096 8192 12288 16384 20480"
      ;;
    28k)
      echo "1024 2048 4096 8192 12288 16384 20480 24576"
      ;;
    32k)
      echo "1024 2048 4096 8192 12288 16384 20480 24576 28672"
      ;;
    debug)
      echo "1024 2048"
      ;;
    *)
      die "Unknown prefix_profile: $profile"
      ;;
  esac
}

PREFIX_LEN_LIST=( $(prefix_list_for "$PREFIX_PROFILE") )

# Get sampling arguments
SAMPLING_ARGS=()
if [ "$MODE" = "greedy" ]; then
  TEMPERATURE="${TEMPERATURE:-0.0}"
elif [ "$MODE" = "sampling" ]; then
  TEMPERATURE="${TEMPERATURE:-0.6}"
  SAMPLING_ARGS+=(--top_p "$TOP_P")
  if [ -n "$TOP_K" ]; then
    SAMPLING_ARGS+=(--top_k "$TOP_K")
  fi
else
  die "Unknown mode: $MODE (use greedy|sampling)"
fi

# Get backend-specific arguments
EXTRA_ARGS="$(get_backend_args "$BACKEND" "$MODEL" "$DRAFT_LENGTH")"

log "RUN: model=$MODEL backend=$BACKEND draft_length=$DRAFT_LENGTH dataset=$DATASET mode=$MODE bsz=$BSZ gpus=$GPUS nproc=$NPROC"
log "EXTRA_ARGS: $EXTRA_ARGS"

# Run benchmark
torchrun --standalone --nproc_per_node="$NPROC" -m batchspec.run \
  --backend "$BACKEND" \
  --checkpoint_path "$model_path" \
  --tokenizer_path "$tokenizer_path" \
  --model_name "$model_name" \
  --rank_group $RANK_GROUP \
  --dataset "$DATASET" \
  --dtype "$DTYPE" \
  --batch_size "$BSZ" --prefix_len_list "${PREFIX_LEN_LIST[@]}" --max_gen_len "$MAX_GEN_LEN" \
  --temperature "$TEMPERATURE" "${SAMPLING_ARGS[@]}" --force_budget \
  --profiling --printoutput \
  --num_total_runs "$NUM_TOTAL_RUNS" \
  $EXTRA_ARGS