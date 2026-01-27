#!/bin/bash
set -e  # Exit on any error

# =============================================================================
# Configuration
# =============================================================================

# Base checkpoint directory (all models will be saved under this directory)
BASE_CHECKPOINT_DIR="/workspace/checkpoints"

# HuggingFace token (replace with your token)
HF_TOKEN=""

# =============================================================================
# Model Repository IDs
# =============================================================================
# Add or remove repo_ids as needed. The script will process them sequentially.
# You can comment out models you don't want to download.

REPO_IDS=(
    # "meta-llama/Meta-Llama-3.1-8B"
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    # "Qwen/Qwen3-0.6B"
    # "Qwen/Qwen3-1.7B"
    # "Qwen/Qwen3-8B"
    # "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-8B"
    # "Qwen/Qwen3-14B"
    # "Qwen/Qwen3-32B"
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    # "yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B"
    # "AngelSlim/Qwen3-8B_eagle3"
    # "AngelSlim/Qwen3-14B_eagle3"
    # "AngelSlim/Qwen3-32B_eagle3"
)

# =============================================================================
# Functions
# =============================================================================

# Extract model name from repo_id
get_model_name() {
    local repo_id="$1"
    # Extract the part after the last '/'
    echo "${repo_id##*/}"
}

# Create output directory path
get_output_dir() {
    local repo_id="$1"
    local model_name=$(get_model_name "$repo_id")
    echo "${BASE_CHECKPOINT_DIR}/${model_name}"
}

# Log function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# =============================================================================
# Main Processing
# =============================================================================

log "Starting model preparation process..."
log "Base checkpoint directory: $BASE_CHECKPOINT_DIR"
log "Number of models to process: ${#REPO_IDS[@]}"

# Create base checkpoint directory if it doesn't exist
mkdir -p "$BASE_CHECKPOINT_DIR"

# Process each repository
for i in "${!REPO_IDS[@]}"; do
    repo_id="${REPO_IDS[$i]}"
    output_dir=$(get_output_dir "$repo_id")
    model_name=$(get_model_name "$repo_id")
    
    log "----------------------------------------"
    log "Processing model $((i+1))/${#REPO_IDS[@]}: $model_name"
    log "Repository: $repo_id"
    log "Output directory: $output_dir"
    log "----------------------------------------"
    
    # Check if model already exists
    if [ -f "$output_dir/model.pth" ]; then
        log "Model already exists at $output_dir/model.pth, skipping..."
        continue
    fi
    
    # Run the conversion script
    log "Starting download and conversion..."
    
    # Build command with optional hf_token
    CMD="python -m tools.prepare_model.convert_weights --cleanup --type base --repo_id \"$repo_id\" --out_dir \"$output_dir\""
    [ -n "$HF_TOKEN" ] && CMD="$CMD --hf_token \"$HF_TOKEN\""
    [ -n "$model_name" ] && CMD="$CMD --model_name \"$model_name\""
    
    eval $CMD
    
    if [ $? -eq 0 ]; then
        log "Successfully processed: $model_name"
    else
        log "Failed to process: $model_name"
        exit 1
    fi
    
    log "Model saved at: $output_dir/model.pth"
done

log "=========================================="
log "All models processed successfully!"
log "Models saved in: $BASE_CHECKPOINT_DIR"
log "==========================================" 