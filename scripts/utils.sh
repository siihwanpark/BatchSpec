#!/usr/bin/env bash

# =============================================================================
# utils.sh — Shared utility functions for benchmark scripts
# =============================================================================

# ---- guard: avoid double-sourcing ----
if [ -n "${BATCHSPEC_UTILS_SH_LOADED:-}" ]; then
    log "utils.sh already loaded, skipping..."
    return 0
fi
BATCHSPEC_UTILS_SH_LOADED=1

: "${BASE_CKPT_DIR:=/home/jovyan/checkpoints}"

# =============================================================================
# Logging helpers
# =============================================================================
ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] INFO: $*"; }
warn() { echo "[$(ts)] WARN: $*" >&2; }
die() { echo "[$(ts)] ERROR: $*" >&2; return 1; }

# =============================================================================
# CUDA / environment helpers
# =============================================================================

# Detect CUDA compute capability
# Args:
#   $1: device index (default: 0)
# Returns: "<major>.<minor>" (e.g., 8.9, 9.0, 10.0)
detect_cuda_arch() {
    local device_idx="${1:-0}"
    python - <<EOF
import torch
idx = int("${device_idx}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA not available")
cc = torch.cuda.get_device_capability(idx)
print(f"{cc[0]}.{cc[1]}")
EOF
}

# Set TORCH_CUDA_ARCH_LIST if unset
# Args:
#   $1: device index for detection (default: 0)
set_torch_cuda_arch_list() {
    local device_idx="${1:-0}"
    if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
        export TORCH_CUDA_ARCH_LIST="$(detect_cuda_arch "$device_idx")"
    fi
}

# =============================================================================
# Experiment Argument helpers
# =============================================================================

# Get prefix length list
prefix_list_for() {
  local profile="$1"

  # Special-case for quick debugging
  #   prefix_list_for "debug"
  #   -> 1024 2048
  if [ "$profile" = "debug" ]; then
    echo "1024 2048"
    return 0
  fi

  # Expect format "<N>k" (e.g., 3k, 7k, 12k, 25k)
  # Convert to max prefix length in tokens (N * 1024)
  if [[ ! "$profile" =~ ^([0-9]+)k$ ]]; then
    die "Unknown prefix_profile format: $profile (expected e.g. 7k, 15k, 32k, debug)"
    return 1
  fi

  local k="${BASH_REMATCH[1]}"
  local max=$((k * 1024))  # Upper bound (exclusive)

  # Policy:
  #   - The returned prefix list always EXCLUDES `max`
  #   - Step size depends on the range of `max`
  #
  #   1) max <=  8k  : 1k step
  #   2) 9k ~ 16k    : 2k step (after 1k, 2k anchors)
  #   3) 17k ~ 32k   : anchors + 4k step
  #
  # Examples:
  #   prefix_list_for "3k"
  #     -> 1024 2048
  #
  #   prefix_list_for "7k"
  #     -> 1024 2048 3072 4096 5120 6144
  #
  #   prefix_list_for "12k"
  #     -> 1024 2048 4096 6144 8192 10240
  #
  #   prefix_list_for "13k"
  #     -> 1024 2048 4096 6144 8192 10240 12288
  #
  #   prefix_list_for "18k"
  #     -> 1024 2048 4096 8192 12288 16384
  #
  #   prefix_list_for "25k"
  #     -> 1024 2048 4096 8192 12288 16384 20480 24576
  #
  #   prefix_list_for "30k"
  #     -> 1024 2048 4096 8192 12288 16384 20480 24576 28672

  local -a out=()

  if [ "$max" -le 8192 ]; then
    # <= 8k: uniform 1k increments
    for ((v=1024; v<max; v+=1024)); do
      out+=("$v")
    done

  elif [ "$max" -le 16384 ]; then
    # 9k ~ 16k:
    #   keep fine granularity early, then switch to 2k steps
    out+=(1024 2048)
    for ((v=4096; v<max; v+=2048)); do
      out+=("$v")
    done

  else
    # 17k ~ 32k:
    #   coarse-grained prefixes for long-context regimes
    out+=(1024 2048 4096 8192 12288 16384)
    for ((v=20480; v<max; v+=4096)); do
      out+=("$v")
    done
  fi

  # Print as space-separated list (for array expansion)
  echo "${out[*]}"
}

# Get model name, model path, and tokenizer path
# Returns: <model_name> <model_path> <tokenizer_path>
get_model_path() {
    local model="$1"
    local base_ckpt_dir="${BASE_CKPT_DIR}"

    case "$model" in
        Qwen3-8B)
            echo "Qwen3-8B ${base_ckpt_dir}/Qwen3-8B/model.pth ${base_ckpt_dir}/Qwen3-8B"
            ;;
        DSL-8B)
            echo "DeepSeek-R1-Distill-Llama-8B ${base_ckpt_dir}/DeepSeek-R1-Distill-Llama-8B/model.pth ${base_ckpt_dir}/DeepSeek-R1-Distill-Llama-8B"
            ;;
        Qwen3-14B)
            echo "Qwen3-14B ${base_ckpt_dir}/Qwen3-14B/model.pth ${base_ckpt_dir}/Qwen3-14B"
            ;;
        *)
            echo "Unknown model: ${model}" >&2
            return 1
            ;;
    esac
}

# Get extra args for given backend
# Args:
#   $1: backend
#   $2: model
#   $3: draft length
# Returns: extra args string
get_backend_args() {
    local backend="$1"
    local model="$2"
    local draft_length="${3:-4}"
    local base_ckpt_dir="${BASE_CKPT_DIR}"

    if [ -z "$backend" ] || [ -z "$model" ]; then
        echo "Usage: get_backend_args <backend> <model> <draft_length>" >&2
        return 1
    fi
    if [ -z "$draft_length" ] || ! [[ "$draft_length" =~ ^[0-9]+$ ]] || [ "$draft_length" -le 0 ]; then
        echo "Invalid draft_length: '$draft_length' (must be a positive integer)" >&2
        return 1
    fi

    local key="${backend}:${model}"

    case "$key" in
        standard:Qwen3-8B|standard:Qwen3-14B|standard:DSL-8B)
            echo ""
            ;;

        standalone:Qwen3-8B|standalone:Qwen3-14B)
            echo "--drafter_name Qwen3-0.6B --drafter_checkpoint_path ${base_ckpt_dir}/Qwen3-0.6B/model.pth --draft_length ${draft_length}"
            ;;
        standalone:DSL-8B)
            echo "--drafter_name DeepSeek-R1-Distill-Llama-3.2-1B-Instruct --drafter_checkpoint_path ${base_ckpt_dir}/DeepSeek-R1-Distill-Llama-3.2-1B-Instruct/model.pth --draft_length ${draft_length}"
            ;;

        ngram:Qwen3-8B|ngram:Qwen3-14B|ngram:DSL-8B)
            echo "--max_ngram_size 4 --draft_length ${draft_length}"
            ;;

        eagle:Qwen3-8B)
            echo "--eagle_name Qwen3-8B_eagle3 --eagle_checkpoint_path ${base_ckpt_dir}/Qwen3-8B_eagle3/model.pth --draft_length ${draft_length}"
            ;;
        eagle:Qwen3-14B)
            echo "--eagle_name Qwen3-14B_eagle3 --eagle_checkpoint_path ${base_ckpt_dir}/Qwen3-14B_eagle3/model.pth --draft_length ${draft_length}"
            ;;
        eagle:DSL-8B)
            echo "--eagle_name EAGLE3-DeepSeek-R1-Distill-LLaMA-8B --eagle_checkpoint_path ${base_ckpt_dir}/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B/model.pth --draft_length ${draft_length}"
            ;;

        magicdec:Qwen3-8B|magicdec:Qwen3-14B|magicdec:DSL-8B)
            echo "--draft_length ${draft_length} --num_sink_tokens 16 --stream_budget 512"
            ;;

        mtp:Qwen3-8B)
            echo "--lora_checkpoint_path ${base_ckpt_dir}/MTP_adapters/Qwen3-8B/model.pth --lora_rank 16 --lora_alpha 32 --draft_length ${draft_length}"
            ;;
        mtp:Qwen3-14B)
            echo "--lora_checkpoint_path ${base_ckpt_dir}/MTP_adapters/Qwen3-14B/model.pth --lora_rank 16 --lora_alpha 32 --draft_length ${draft_length}"
            ;;
        mtp:DSL-8B)
            echo "--lora_checkpoint_path ${base_ckpt_dir}/MTP_adapters/DeepSeek-R1-Distill-Llama-8B/model.pth --lora_rank 16 --lora_alpha 32 --draft_length ${draft_length}"
            ;;

        *)
        echo "Unknown backend/model combo: $key" >&2
        return 1
        ;;
    esac
}

# Check if a checkpoint path exists
# Args:
#   $1: path
# Returns: 0 if file exists, 1 otherwise
ensure_file_exists() {
    local path="$1"
    if [ ! -f "$path" ]; then
        echo "File not found: $path" >&2
        return 1
    fi
}

# =============================================================================
# GPU waiting helpers
# =============================================================================

# Wait until given GPUs are idle according to nvidia-smi compute apps
# Args:
#   $1: GPU list in CSV format (e.g., "0,1" or "4,5")
#   $2: idle confirm sleep (default: 3m)
#   $3: busy sleep (default: 5m)

wait_for_gpus_idle() {
    local gpus_csv="$1"
    local idle_confirm_sleep="${2:-3m}"
    local busy_sleep="${3:-5m}"

    while true; do
        local busy_pids
        busy_pids=$(
            nvidia-smi -i "$gpus_csv" --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
            | tr -d '[:space:]' \
            | paste -sd, -
        )

        if [ -z "$busy_pids" ]; then
            log "GPUs ${gpus_csv} are idle. Sleep ${idle_confirm_sleep} more."
            sleep "$idle_confirm_sleep"

            busy_pids=$(
            nvidia-smi -i "$gpus_csv" --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
                | tr -d '[:space:]' \
                | paste -sd, -
            )

            if [ -z "$busy_pids" ]; then
                log "GPUs ${gpus_csv} are idle. Proceeding."
                break
            else
                log "GPUs ${gpus_csv} became busy (PIDs: ${busy_pids}). Sleep ${busy_sleep}."
                sleep "$busy_sleep"
            fi
        else
            log "GPUs ${gpus_csv} are busy (PIDs: ${busy_pids}). Sleep ${busy_sleep}."
            sleep "$busy_sleep"
        fi
    done
}

# =============================================================================
# Graceful exit helpers
# =============================================================================

# Internal cleanup handlers
__CLEANUP_HANDLERS=()

# Register a cleanup command
register_cleanup() {
  __CLEANUP_HANDLERS+=("$*")
}

# Run all registered cleanup handlers and return the exit code
_run_cleanup() {
    local rc=$?
    for cmd in "${__CLEANUP_HANDLERS[@]}"; do
        eval "$cmd" || true
    done
    return $rc
}

# Enable graceful exit traps (SIGINT and SIGTERM)
enable_graceful_exit() {
    trap '_on_signal SIGINT' SIGINT
    trap '_on_signal SIGTERM' SIGTERM
    trap '_run_cleanup' EXIT
}

# Handle signals
_on_signal() {
    local sig="$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: Caught ${sig}, killing process group..." >&2

    # Kill the whole process group to avoid orphan processes.
    kill -- -$$ 2>/dev/null || true
}

# Normalize a TSV file (replace spaces with tabs)
normalize_tsv() {
    local src="$1" out="$2"
    [ -f "$src" ] || { echo "normalize_tsv: file not found: $src" >&2; return 1; }

    local dir base out
    dir="$(dirname "$src")"
    base="$(basename "$src")"

    python - <<'PY' "$src" "$out" 1>&2
import re, sys

src, out = sys.argv[1], sys.argv[2]

lines = []
with open(src, "r", encoding="utf-8") as f:
    for line in f:
        raw = line.rstrip("\n")
        s = raw.strip()
        if not s or s.startswith("#"):
            lines.append(raw)
            continue
        cols = re.split(r"\s+", s)
        lines.append("\t".join(cols))

with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print(f"[normalize_tsv] wrote normalized tsv: {out}", file=sys.stderr)
PY

    echo "$out"
}

# =============================================================================
# Small conveniences
# =============================================================================

# Safe read helper for get_model_path style outputs
# Usage: read -r model_name model_path tokenizer_path <<< "$(get_model_path "$model")"
# (Provided as a reminder; bash built-in read is sufficient.)

log "utils.sh has been successfully loaded"
true