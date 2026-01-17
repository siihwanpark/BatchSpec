#!/usr/bin/env python3
"""
Unified model conversion script for base models and LoRA adapters.

This module handles:
- Base model: Download from HuggingFace + convert to custom format
- LoRA adapter: Convert and fuse LoRA weights to custom format
"""

import re
import shutil
from pathlib import Path
from typing import Dict, Optional

import torch
from safetensors.torch import load_file as load_safetensors_file

from batchspec.models import get_config
from .utils import hf_download, cleanup_original_files, find_model_files, load_weights


# ============================================================================
# Base Model Converter
# ============================================================================

def get_weight_mapping(model_name: str) -> Dict[str, Optional[str]]:
    """Get HF -> custom weight mapping based on model architecture."""
    weight_map = {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
        "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
        "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
        "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
        'model.layers.{}.self_attn.rotary_emb.inv_freq': None,
        'model.layers.{}.mlp.gate_proj.weight': 'layers.{}.feed_forward.w1.weight',
        "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
        "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
        "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
        "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "output.weight",
    }
    
    # Model-specific mappings
    model_lower = model_name.lower()
    if "qwen2" in model_lower:
        weight_map.update({
            "model.layers.{}.self_attn.q_proj.bias": "layers.{}.attention.wq.bias",
            "model.layers.{}.self_attn.k_proj.bias": "layers.{}.attention.wk.bias",
            "model.layers.{}.self_attn.v_proj.bias": "layers.{}.attention.wv.bias",
        })
    
    if "qwen3" in model_lower:
        weight_map.update({
            "model.layers.{}.self_attn.q_norm.weight": "layers.{}.attention.q_norm.weight",
            "model.layers.{}.self_attn.k_norm.weight": "layers.{}.attention.k_norm.weight",
        })

    if "eagle" in model_lower:
        weight_map.update({
            "d2t": "draft_to_target",
            "t2d": "target_to_draft",
            "midlayer.self_attn.q_proj.weight": "attn.wq.weight",
            "midlayer.self_attn.k_proj.weight": "attn.wk.weight",
            "midlayer.self_attn.v_proj.weight": "attn.wv.weight",
            "midlayer.self_attn.o_proj.weight": "attn.wo.weight",
            "midlayer.mlp.gate_proj.weight": "feed_forward.w1.weight",
            "midlayer.mlp.up_proj.weight": "feed_forward.w3.weight",
            "midlayer.mlp.down_proj.weight": "feed_forward.w2.weight",
            "midlayer.hidden_norm.weight": "hidden_norm.weight",
            "midlayer.input_layernorm.weight": "input_norm.weight",
            "midlayer.post_attention_layernorm.weight": "post_attn_norm.weight",
            "norm.weight": "norm.weight",
            "fc.weight": "fc.weight",
            "lm_head.weight": "output.weight",
        })
    
    return weight_map


@torch.inference_mode()
def convert_base_model(
    checkpoint_dir: Path,
    model_name: Optional[str] = None,
    cleanup: bool = False
) -> None:
    """
    Convert HuggingFace checkpoint to custom model.pth format.
    
    Args:
        checkpoint_dir: Directory containing the HF model
        model_name: Model name for config lookup (defaults to dir name)
        cleanup: Whether to remove original files after conversion
    """
    if model_name is None:
        model_name = checkpoint_dir.name

    # Load model config
    config = get_config(model_name)
    print(f"\nModel: {model_name}")
    print(f"Config: dim={config.dim}, n_layer={config.n_layer}, n_head={config.n_head}, vocab_size={config.vocab_size}")

    # Find and load weights
    index_file, weight_file = find_model_files(checkpoint_dir)
    merged_weights = load_weights(checkpoint_dir, index_file, weight_file)
    
    # Get weight mapping
    weight_map = get_weight_mapping(model_name)
    
    # Permutation function for RoPE
    def permute(w, n_head, eagle_attn_weight=False, qk_norm=False):
        if len(w.shape) == 2:
            # Weight matrix
            dim = config.dim
            if eagle_attn_weight:
                # Eagle attn weights (q,k,v) has shape [dim, 2 * dim] since they process the concatenated inputs (hidden_states + input_embeds)
                return (
                    w.view(n_head, 2, config.head_dim // 2, 2 * dim)
                    .transpose(1, 2)
                    .reshape(config.head_dim * n_head, 2 * dim)
                )
            else:
                return (
                    w.view(n_head, 2, config.head_dim // 2, dim)
                    .transpose(1, 2)
                    .reshape(config.head_dim * n_head, dim)
                )
        else:
            if qk_norm:
                # QK norm (Qwen3 only)
                return w.view(2, config.head_dim // 2).transpose(0, 1).reshape(config.head_dim)
            else:
                # Bias vector
                return w.view(n_head, 2, config.head_dim // 2).transpose(1, 2).reshape(config.head_dim * n_head)

    # Map weights to custom format
    print("\nMapping weights to custom format...")
    final_result = {}
    
    for key, value in merged_weights.items():
        if "layers" in key:
            # Layer-specific weight
            abstract_key = re.sub(r'(\d+)', '{}', key)
            layer_num = re.search(r'\d+', key).group(0)
            new_key = weight_map.get(abstract_key)
            if new_key is None:
                continue
            new_key = new_key.format(layer_num)
        else:
            # Global weight
            new_key = weight_map.get(key)
            if new_key is None:
                continue

        final_result[new_key] = value

    # Fuse Q, K, V weights
    print("Fusing Q/K/V weights...")
    is_eagle = "eagle" in model_name.lower()
    
    for key in tuple(final_result.keys()):
        if "wq" in key:
            q = final_result[key]
            k = final_result[key.replace("wq", "wk")]
            v = final_result[key.replace("wq", "wv")]
            
            q = permute(q, config.n_head, eagle_attn_weight=is_eagle)
            k = permute(k, config.n_local_heads, eagle_attn_weight=is_eagle)
            
            final_result[key.replace("wq", "wqkv")] = torch.cat([q, k, v])
            del final_result[key]
            del final_result[key.replace("wq", "wk")]
            del final_result[key.replace("wq", "wv")]
        
        if "q_norm" in key:
            # Qwen3 QK norm
            q_norm = final_result[key]
            k_norm = final_result[key.replace("q_norm", "k_norm")]
            final_result[key] = permute(q_norm, config.n_head, qk_norm=True)
            final_result[key.replace("q_norm", "k_norm")] = permute(k_norm, config.n_local_heads, qk_norm=True)

    # Save converted model
    output_path = checkpoint_dir / "model.pth"
    print(f"\nSaving converted model to {output_path}...")
    torch.save(final_result, output_path)
    
    # Copy tokenizer for Llama-3
    if 'llama-3' in model_name.lower():
        original_dir = checkpoint_dir / "original"
        tokenizer_model = original_dir / "tokenizer.model"
        if tokenizer_model.exists():
            tokenizer_dest = checkpoint_dir / "tokenizer.model"
            print(f"Copying tokenizer.model...")
            shutil.copy(tokenizer_model, tokenizer_dest)
    
    # Cleanup
    if cleanup:
        cleanup_original_files(checkpoint_dir)
    
    print(f"\n{'='*60}")
    print(f"Base model conversion completed!")
    print(f"Saved to: {output_path}")
    print(f"{'='*60}\n")


# ============================================================================
# LoRA Converter
# ============================================================================

def permute_lora_rows_for_rope(B: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    """Apply RoPE row permutation to LoRA-B matrix."""
    out, r = B.shape
    assert out == n_heads * head_dim, f"Shape mismatch: {out} vs {n_heads}*{head_dim}"
    return (
        B.view(n_heads, 2, head_dim // 2, r)
        .transpose(1, 2)
        .reshape(out, r)
    )


@torch.inference_mode()
def convert_lora(
    lora_path: Path,
    out_path: Path,
    model_name: str,
    n_head: Optional[int] = None,
    n_kv_head: Optional[int] = None,
    head_dim: Optional[int] = None,
    permute_qk: bool = True,
    cleanup: bool = False
) -> None:
    """
    Convert LoRA adapter from HF format to custom fused format.
    
    Args:
        lora_path: Path to LoRA model.safetensors
        out_path: Output path for converted model.pth
        model_name: Base model name for config lookup
        n_head: Number of attention heads (inferred if None)
        n_kv_head: Number of KV heads (inferred if None)
        head_dim: Head dimension (inferred if None)
        permute_qk: Apply RoPE row permutation
        cleanup: Remove original files after conversion
    """
    print(f"\n{'='*60}")
    print(f"Converting LoRA adapter: {lora_path.name}")
    print(f"{'='*60}\n")
    
    # Load LoRA weights
    print(f"Loading LoRA weights...")
    src_weights = load_safetensors_file(str(lora_path), device="cpu")
    print(f"Loaded {len(src_weights)} tensor(s)")
    
    # Infer model config
    cfg = get_config(model_name)
    n_head = n_head or cfg.n_head
    n_kv_head = n_kv_head or cfg.n_local_heads
    head_dim = head_dim or cfg.head_dim
    
    print(f"\nModel config: {model_name}")
    print(f"  n_head={n_head}, n_kv_head={n_kv_head}, head_dim={head_dim}")
    
    # Parse weights using simple regex patterns
    print("Parsing LoRA weights...")
    lora_weights = {}  # Stores intermediate parsed weights
    
    for key, tensor in src_weights.items():
        # Sampler weights - rename sampler. -> sampler_head.
        if key.startswith("sampler."):
            new_key = key.replace("sampler.", "sampler_head.", 1)
            lora_weights[new_key] = tensor
            continue
        elif key.startswith("sampler_head."):
            lora_weights[key] = tensor
            continue
        
        # Parse LoRA keys: model.model.layers.{L}.{module}.{proj}.lora_{A|B}.weight
        # Also supports fused: model.model.layers.{L}.{module}.{fused_proj}.lora_{A|B}.weight
        match = re.match(
            r"^model\.model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)\.lora_([AB])\.weight$",
            key
        )
        if not match:
            continue
        
        layer_id, module, proj, ab = match.groups()
        # Store with structured key for easy lookup
        lora_weights[f"{layer_id}.{module}.{proj}.{ab}"] = tensor
    
    print(f"Parsed {len(lora_weights)} LoRA tensor(s)")
    
    # Get unique layer IDs
    layer_ids = sorted(set(
        k.split('.')[0] for k in lora_weights.keys() 
        if not k.startswith("sampler") and '.' in k
    ))
    print(f"Processing {len(layer_ids)} layer(s)...")
    
    # Build output weights
    final_result = {}
    
    for lid in layer_ids:
        # Helper to get tensor
        def get_tensor(module, proj, ab):
            key = f"{lid}.{module}.{proj}.{ab}"
            return lora_weights.get(key)
        
        # Attention: Q/K/V -> WQKV fusion
        # Check if already fused
        qkv_A = get_tensor("self_attn", "qkv_proj", "A")
        qkv_B = get_tensor("self_attn", "qkv_proj", "B")
        
        if qkv_A is not None and qkv_B is not None:
            # Already fused
            B_qkv = qkv_B.clone()
            if permute_qk:
                out_q = n_head * head_dim
                out_k = n_kv_head * head_dim
                B_qkv[0:out_q, :] = permute_lora_rows_for_rope(B_qkv[0:out_q, :], n_head, head_dim)
                B_qkv[out_q:out_q+out_k, :] = permute_lora_rows_for_rope(B_qkv[out_q:out_q+out_k, :], n_kv_head, head_dim)
            
            final_result[f"layers.{lid}.attention.wqkv.lora_A.weight"] = qkv_A
            final_result[f"layers.{lid}.attention.wqkv.lora_B.weight"] = B_qkv
        else:
            # Non-fused - need to fuse Q, K, V
            q_A, q_B = get_tensor("self_attn", "q_proj", "A"), get_tensor("self_attn", "q_proj", "B")
            k_A, k_B = get_tensor("self_attn", "k_proj", "A"), get_tensor("self_attn", "k_proj", "B")
            v_A, v_B = get_tensor("self_attn", "v_proj", "A"), get_tensor("self_attn", "v_proj", "B")
            
            if not (q_A is not None and q_B is not None and 
                    k_A is not None and k_B is not None and 
                    v_A is not None and v_B is not None):
                raise ValueError(f"Layer {lid}: Missing Q/K/V LoRA weights")
            
            # Fuse: concatenate A, build block-diagonal B
            A_qkv = torch.cat([q_A, k_A, v_A], dim=0)
            
            # Permute B matrices for RoPE if needed
            if permute_qk:
                q_B = permute_lora_rows_for_rope(q_B, n_head, head_dim)
                k_B = permute_lora_rows_for_rope(k_B, n_kv_head, head_dim)
            
            out_q, rq = q_B.shape
            out_k, rk = k_B.shape
            out_v, rv = v_B.shape
            r_tot = rq + rk + rv
            
            B_qkv = torch.zeros((out_q + out_k + out_v, r_tot), dtype=q_B.dtype)
            B_qkv[0:out_q, 0:rq] = q_B
            B_qkv[out_q:out_q+out_k, rq:rq+rk] = k_B
            B_qkv[out_q+out_k:, rq+rk:] = v_B
            
            final_result[f"layers.{lid}.attention.wqkv.lora_A.weight"] = A_qkv
            final_result[f"layers.{lid}.attention.wqkv.lora_B.weight"] = B_qkv
        
        # Attention: O (passthrough)
        o_A, o_B = get_tensor("self_attn", "o_proj", "A"), get_tensor("self_attn", "o_proj", "B")
        if o_A is not None and o_B is not None:
            final_result[f"layers.{lid}.attention.wo.lora_A.weight"] = o_A
            final_result[f"layers.{lid}.attention.wo.lora_B.weight"] = o_B
        
        # MLP: gate/up -> W13 fusion
        # Check if already fused
        w13_A = get_tensor("mlp", "gate_up_proj", "A")
        w13_B = get_tensor("mlp", "gate_up_proj", "B")
        
        if w13_A is not None and w13_B is not None:
            # Already fused
            final_result[f"layers.{lid}.feed_forward.w13.lora_A.weight"] = w13_A
            final_result[f"layers.{lid}.feed_forward.w13.lora_B.weight"] = w13_B
        else:
            # Non-fused - need to fuse gate and up
            gate_A, gate_B = get_tensor("mlp", "gate_proj", "A"), get_tensor("mlp", "gate_proj", "B")
            up_A, up_B = get_tensor("mlp", "up_proj", "A"), get_tensor("mlp", "up_proj", "B")
            
            if not (gate_A is not None and gate_B is not None and 
                    up_A is not None and up_B is not None):
                raise ValueError(f"Layer {lid}: Missing gate/up LoRA weights")
            
            # Fuse: concatenate A, build block-diagonal B
            A_13 = torch.cat([gate_A, up_A], dim=0)
            
            out_g, rg = gate_B.shape
            out_u, ru = up_B.shape
            r_tot = rg + ru
            
            B_13 = torch.zeros((out_g + out_u, r_tot), dtype=gate_B.dtype)
            B_13[0:out_g, 0:rg] = gate_B
            B_13[out_g:, rg:] = up_B
            
            final_result[f"layers.{lid}.feed_forward.w13.lora_A.weight"] = A_13
            final_result[f"layers.{lid}.feed_forward.w13.lora_B.weight"] = B_13
        
        # MLP: down (passthrough)
        down_A, down_B = get_tensor("mlp", "down_proj", "A"), get_tensor("mlp", "down_proj", "B")
        if down_A is not None and down_B is not None:
            final_result[f"layers.{lid}.feed_forward.w2.lora_A.weight"] = down_A
            final_result[f"layers.{lid}.feed_forward.w2.lora_B.weight"] = down_B
    
    # Sampler weights (passthrough)
    for key, tensor in lora_weights.items():
        if key.startswith("sampler_head."):
            final_result[key] = tensor
    
    # Save converted LoRA
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving converted LoRA to {out_path}...")
    torch.save(final_result, out_path)
    
    # Cleanup
    if cleanup:
        cleanup_original_files(lora_path.parent)
    
    print(f"\n{'='*60}")
    print(f"LoRA conversion completed!")
    print(f"Saved to: {out_path}")
    print(f"Total tensors: {len(final_result)}")
    print(f"{'='*60}\n")


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert models to custom format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Convert base model
            python -m tools.convert --type base --checkpoint_dir ./checkpoints/Qwen3-8B
            
            # Convert base model with download
            python -m tools.convert --type base --repo_id Qwen/Qwen3-8B --out_dir ./checkpoints/Qwen3-8B
            
            # Convert LoRA adapter
            python -m tools.convert --type lora --lora_path ./lora/model.safetensors --model_name Qwen3-8B
        """
    )
    
    parser.add_argument('--type', choices=['base', 'lora'], required=True,
                        help='Conversion type')
    
    # Base model args
    parser.add_argument('--repo_id', type=str,
                        help='[base] HuggingFace repository ID for download')
    parser.add_argument('--checkpoint_dir', type=Path,
                        help='[base] Path to checkpoint directory')
    parser.add_argument('--out_dir', type=Path,
                        help='[base] Output directory (for download)')
    parser.add_argument('--hf_token', type=str,
                        help='[base] HuggingFace API token')
    
    # LoRA args
    parser.add_argument('--lora_path', type=Path,
                        help='[lora] Path to LoRA model.safetensors')
    parser.add_argument('--model_name', type=str,
                        help='Model name for config lookup')
    parser.add_argument('--no_permute_qk', action='store_true',
                        help='[lora] Disable RoPE row permutation')
    
    # Common args
    parser.add_argument('--cleanup', action='store_true',
                        help='Cleanup original files after conversion')
    
    args = parser.parse_args()
    
    if args.type == 'base':
        # Determine checkpoint directory
        if args.repo_id and args.out_dir:
            # Download first
            hf_download(str(args.out_dir), args.repo_id, args.hf_token)
            checkpoint_dir = args.out_dir
        elif args.checkpoint_dir:
            checkpoint_dir = args.checkpoint_dir
        else:
            parser.error("--type base requires either (--repo_id + --out_dir) or --checkpoint_dir")
        
        # Convert
        convert_base_model(
            checkpoint_dir=checkpoint_dir,
            model_name=args.model_name,
            cleanup=args.cleanup
        )
    
    elif args.type == 'lora':
        if not args.lora_path:
            parser.error("--type lora requires --lora_path")
        if not args.model_name:
            parser.error("--type lora requires --model_name")
        
        out_path = args.lora_path.parent / "model.pth"
        
        convert_lora(
            lora_path=args.lora_path,
            out_path=out_path,
            model_name=args.model_name,
            permute_qk=not args.no_permute_qk,
            cleanup=args.cleanup
        )


if __name__ == '__main__':
    main()

