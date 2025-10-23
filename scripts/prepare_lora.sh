#!/bin/bash
set -e  # Exit on any error

# =============================================================================
# Configuration
# =============================================================================

# Base checkpoint directory (all models will be saved under this directory)
BASE_CHECKPOINT_DIR="/home/mngcuser1/sihwan_workspace/checkpoints/MTP_adapters"

# =============================================================================
# LoRA Adapter Checkpoint Directories
# =============================================================================
# Add or remove lora_ckpt_dirs as needed. The script will process them sequentially.
# You can comment out lora_ckpt_dirs you don't want to process.

LORA_CKPT_DIRS=(
    # "Qwen3-8B/am-qwen3-distilled-120k-rank16-lr2e-4-bsz4-SoftSCE-fused/step-30000"
    # "Qwen3-8B/am-qwen3-distilled-120k-rank128-lr2e-4-bsz8-SoftSCE/step-45000"
    # "Qwen3-8B/am-qwen3-distilled-120k-rank128-lr2e-4-bsz8-SoftSCE/step-50000"
    "DeepSeek-R1-Distill-Llama-8B/openthoughts-114k-rank16-lr2e-4-bsz8/step-30000"
)

# =============================================================================
# Functions
# =============================================================================

# Extract model name from lora_ckpt_dir
get_model_name() {
    local lora_ckpt_dir="$1"
    echo "${lora_ckpt_dir%%/*}"
}

# Log function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# =============================================================================
# Main Processing
# =============================================================================

log "Starting LoRA adapter preparation process..."
log "Base checkpoint directory: $BASE_CHECKPOINT_DIR"
log "Number of LoRA adapters to process: ${#LORA_CKPT_DIRS[@]}"

# Create base checkpoint directory if it doesn't exist
mkdir -p "$BASE_CHECKPOINT_DIR"

# Process each repository
for i in "${!LORA_CKPT_DIRS[@]}"; do
    model_name=$(get_model_name "${LORA_CKPT_DIRS[$i]}")
    lora_ckpt_dir="${BASE_CHECKPOINT_DIR}/${LORA_CKPT_DIRS[$i]}"
    
    log "----------------------------------------"
    log "Processing LoRA adapter $((i+1))/${#LORA_CKPT_DIRS[@]}"
    log "Base model name: $model_name"
    log "LoRA adapter checkpoint directory: $lora_ckpt_dir"
    log "Output directory: $lora_ckpt_dir"
    log "----------------------------------------"
    
    # Check if model already exists
    if [ -f "$lora_ckpt_dir/model.pth" ]; then
        log "LoRA adapter already exists at $lora_ckpt_dir/model.pth, skipping..."
        continue
    fi
    
    # Run the conversion script
    log "Starting LoRA conversion..."
    python -m tools.convert_weights \
        --type lora \
        --lora_path "$lora_ckpt_dir/model.safetensors" \
        --model_name "$model_name"
    
    if [ $? -eq 0 ]; then
        log "Successfully processed: $lora_ckpt_dir"
    else
        log "Failed to process: $lora_ckpt_dir"
        exit 1
    fi
    
    log "LoRA adapter saved at: $lora_ckpt_dir/model.pth"
done

log "=========================================="
log "All LoRA adapters processed successfully!"
log "LoRA adapters saved in: $BASE_CHECKPOINT_DIR"
log "==========================================" 