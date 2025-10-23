#!/usr/bin/env python3
"""Common utilities for model conversion."""

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as load_safetensors_file
from requests.exceptions import HTTPError


def hf_download(out_dir: str, repo_id: str, hf_token: Optional[str] = None) -> None:
    """Download model from HuggingFace Hub."""
    print(f"\n{'='*60}")
    print(f"Downloading {repo_id} to {out_dir}")
    print(f"{'='*60}\n")
    
    if os.path.exists(out_dir):
        print(f"Directory {out_dir} already exists. Skipping download.")
        return
    
    os.makedirs(out_dir, exist_ok=True)
    
    try:
        snapshot_download(
            repo_id,
            local_dir=out_dir,
            local_dir_use_symlinks=False,
            token=hf_token
        )
        print(f"Download completed.")
    except HTTPError as e:
        if e.response.status_code == 401:
            raise RuntimeError(
                "Authentication failed. You need to pass a valid --hf_token "
                "to download private checkpoints."
            ) from e
        raise


def cleanup_original_files(checkpoint_dir: Path) -> None:
    """Remove original .bin and .safetensors files after conversion."""
    print("\nCleaning up original model files...")
    
    removed_count = 0
    
    # Remove .bin files
    for bin_file in checkpoint_dir.glob("*.bin"):
        print(f"  Removing {bin_file.name}")
        bin_file.unlink()
        removed_count += 1
    
    # Remove .safetensors files
    for safetensors_file in checkpoint_dir.glob("*.safetensors"):
        print(f"  Removing {safetensors_file.name}")
        safetensors_file.unlink()
        removed_count += 1
    
    # Remove index files
    for index_file in checkpoint_dir.glob("*.index.json"):
        if "model" in index_file.name:
            print(f"  Removing {index_file.name}")
            index_file.unlink()
            removed_count += 1
    
    print(f"Cleanup completed. Removed {removed_count} file(s).")


def find_model_files(checkpoint_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Find model weight files in a directory.
    
    Returns:
        Tuple of (index_file, weight_file). One will be None.
    """
    # Try index files first (for sharded models)
    possible_indices = [
        checkpoint_dir / 'model.safetensors.index.json',
        checkpoint_dir / 'pytorch_model.bin.index.json'
    ]
    
    for index_path in possible_indices:
        if index_path.is_file():
            print(f"Found index file: {index_path.name}")
            return index_path, None
    
    # Try single weight files
    possible_weights = [
        checkpoint_dir / 'model.safetensors',
        checkpoint_dir / 'pytorch_model.bin'
    ]
    
    for weight_path in possible_weights:
        if weight_path.is_file():
            print(f"Found weight file: {weight_path.name}")
            return None, weight_path
    
    raise FileNotFoundError(
        f"No model files found in {checkpoint_dir}. "
        f"Expected one of: {[p.name for p in possible_indices + possible_weights]}"
    )


def load_weights(
    checkpoint_dir: Path,
    index_file: Optional[Path] = None,
    weight_file: Optional[Path] = None,
    device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Load model weights from either sharded or single file.
    
    Returns:
        Dictionary of merged weights
    """
    print(f"\nLoading weights...")
    merged_result = {}
    
    if index_file is not None:
        # Load from multiple sharded files
        with open(index_file) as f:
            bin_index = json.load(f)
        
        bin_files = {checkpoint_dir / f for f in bin_index["weight_map"].values()}
        print(f"Found {len(bin_files)} shard(s) to load")
        
        for file_path in sorted(bin_files):
            print(f"  Loading {file_path.name}...")
            if "safetensors" in str(file_path):
                state_dict = load_safetensors_file(str(file_path), device=device)
            else:
                state_dict = torch.load(str(file_path), map_location=device, mmap=True, weights_only=True)
            merged_result.update(state_dict)
    
    elif weight_file is not None:
        # Load from single file
        print(f"  Loading {weight_file.name}...")
        if "safetensors" in str(weight_file):
            merged_result = load_safetensors_file(str(weight_file), device=device)
        else:
            merged_result = torch.load(str(weight_file), map_location=device, mmap=True, weights_only=True)
    
    else:
        raise ValueError("Either index_file or weight_file must be provided")
    
    print(f"Loaded {len(merged_result)} tensor(s)")
    return merged_result

