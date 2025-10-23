#!/usr/bin/env python3
"""
Compare two PyTorch checkpoint files to verify they are identical.
"""

import torch
from pathlib import Path


def compare_checkpoints(file1: Path, file2: Path):
    """
    Compare two PyTorch checkpoint files.
    
    Args:
        file1: Path to first checkpoint
        file2: Path to second checkpoint
    """
    print(f"Loading checkpoint 1: {file1}")
    checkpoint1 = torch.load(file1, map_location='cpu')
    
    print(f"Loading checkpoint 2: {file2}")
    checkpoint2 = torch.load(file2, map_location='cpu')
    
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    # Compare keys
    keys1 = set(checkpoint1.keys())
    keys2 = set(checkpoint2.keys())
    
    print(f"\nNumber of keys in checkpoint 1: {len(keys1)}")
    print(f"Number of keys in checkpoint 2: {len(keys2)}")
    
    if keys1 != keys2:
        print("\n❌ KEY MISMATCH DETECTED!")
        
        only_in_1 = keys1 - keys2
        only_in_2 = keys2 - keys1
        
        if only_in_1:
            print(f"\nKeys only in checkpoint 1 ({len(only_in_1)}):")
            for key in sorted(only_in_1):
                print(f"  - {key}")
        
        if only_in_2:
            print(f"\nKeys only in checkpoint 2 ({len(only_in_2)}):")
            for key in sorted(only_in_2):
                print(f"  - {key}")
        
        return False
    
    print("✓ Keys match perfectly")
    
    # Compare values for each key
    print(f"\nComparing {len(keys1)} tensors...")
    all_match = True
    mismatches = []
    
    for i, key in enumerate(sorted(keys1), 1):
        val1 = checkpoint1[key]
        val2 = checkpoint2[key]
        
        # Check if both are tensors
        is_tensor1 = isinstance(val1, torch.Tensor)
        is_tensor2 = isinstance(val2, torch.Tensor)
        
        if is_tensor1 != is_tensor2:
            mismatches.append({
                'key': key,
                'error': f'Type mismatch: {type(val1).__name__} vs {type(val2).__name__}'
            })
            all_match = False
            continue
        
        if not is_tensor1:
            # Non-tensor values (e.g., scalars, strings)
            if val1 != val2:
                mismatches.append({
                    'key': key,
                    'error': f'Value mismatch: {val1} vs {val2}'
                })
                all_match = False
            continue
        
        # Compare tensor properties
        if val1.shape != val2.shape:
            mismatches.append({
                'key': key,
                'error': f'Shape mismatch: {val1.shape} vs {val2.shape}'
            })
            all_match = False
            continue
        
        if val1.dtype != val2.dtype:
            mismatches.append({
                'key': key,
                'error': f'Dtype mismatch: {val1.dtype} vs {val2.dtype}'
            })
            all_match = False
            continue
        
        # Compare tensor values
        if not torch.equal(val1, val2):
            # Get statistics about the difference
            diff = (val1 - val2).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            
            mismatches.append({
                'key': key,
                'error': f'Values differ: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}'
            })
            all_match = False
            continue
        
        # Progress indicator
        if i % 100 == 0:
            print(f"  Checked {i}/{len(keys1)} keys...")
    
    print(f"  Checked all {len(keys1)} keys")
    
    # Report results
    print("\n" + "="*80)
    if all_match:
        print("✅ RESULT: Checkpoints are IDENTICAL")
        print("="*80)
        return True
    else:
        print(f"❌ RESULT: Checkpoints are DIFFERENT")
        print(f"\nFound {len(mismatches)} mismatches:")
        print("="*80)
        
        for mismatch in mismatches[:20]:  # Show first 20 mismatches
            print(f"\nKey: {mismatch['key']}")
            print(f"  {mismatch['error']}")
        
        if len(mismatches) > 20:
            print(f"\n... and {len(mismatches) - 20} more mismatches")
        
        return False


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare two PyTorch checkpoint files')
    parser.add_argument('file1', type=Path, help='First checkpoint file')
    parser.add_argument('file2', type=Path, help='Second checkpoint file')
    
    args = parser.parse_args()
    
    if not args.file1.exists():
        print(f"Error: {args.file1} does not exist")
        exit(1)
    
    if not args.file2.exists():
        print(f"Error: {args.file2} does not exist")
        exit(1)
    
    are_identical = compare_checkpoints(args.file1, args.file2)
    exit(0 if are_identical else 1)

