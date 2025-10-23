# Backends Module

The `backends` module provides high-performance inference engines for large language models (LLMs) with advanced optimization techniques including paged attention, speculative decoding, and tensor parallelism.

## Overview

This module implements two distinct inference engines that share a common infrastructure:

1. **StandardEngine**: Standard autoregressive generation with chunked prefill and decode phases
2. **MTPEngine**: Multi-Token Prediction engine with self-speculative decoding and LoRA support

Both engines leverage:
- **Paged KV Caching**: Memory-efficient attention with dynamic page allocation
- **FlashInfer**: Optimized attention kernels for GPU acceleration
- **Tensor Parallelism**: Distributed computation across multiple GPUs

## Architecture

### Directory Structure

```
backends/
├── __init__.py             # Package exports (StandardEngine, MTPEngine)
├── standard_engine.py      # Standard autoregressive engine
├── mtp_engine.py           # Multi-token prediction engine
├── base/                   # Base classes and mixins
│   ├── engine.py           # BaseEngine abstract class
│   ├── kv_cache.py         # KV cache management mixin
│   └── attn_wrapper.py     # FlashInfer wrapper management
└── utils/                  # Utility functions
    ├── paging.py           # Page manager for KV cache
    ├── sampling.py         # Token sampling strategies
    └── tensor_parallel.py  # LoRA-aware tensor parallelism
```

### Class Hierarchy

```
BaseEngine (abstract)
├── KVCacheMixin          # Manages KV cache insert/delete/clear
├── AttentionWrapperMixin # Manages FlashInfer attention wrappers
├── StandardEngine        # Implements standard autoregressive decoding
└── MTPEngine             # Implements multi-token prediction with speculative decoding
```

## Core Components

### 1. Base Engine (`base/engine.py`)

The `BaseEngine` abstract class provides common functionality for all engines:

**Key Features:**
- Model loading and initialization
- Cache setup and management
- Sampling parameter configuration
- Compilation support via `torch.compile`
- Special token handling (EOS, suppress tokens)

**Configuration Methods:**
```python
engine.setup_caches(max_batch_size, max_seq_length, page_size, prefill_chunk_size)
engine.setup_sampling_params(temperature, top_k, top_p, force_budget)
engine.setup_special_tokens(suppress_token, replace_tokens)
engine.compile()  # Enable torch.compile optimization
```

### 2. KV Cache Management (`base/kv_cache.py`)

The `KVCacheMixin` provides efficient KV cache operations:

**Core Operations:**
- `insert_kv(dec_lens)`: Add KV entries for new tokens
- `delete_kv(del_lens)`: Remove KV entries
- `clear_kv()`: Reset all caches

**Key Features:**
- Automatic page allocation/deallocation
- Per-batch-element cache length tracking
- Efficient tensor operations avoiding Python loops

### 3. Attention Wrapper Management (`base/attn_wrapper.py`)

The `AttentionWrapperMixin` integrates FlashInfer attention kernels:

**Core Operations:**
- `_create_attention_buffer(size_mb)`: Allocate workspace buffers
- `_create_attention_wrapper(buffer, qo_length, use_custom_mask)`: Create FlashInfer wrappers
- `_register_attention_wrappers(wrappers)`: Register as custom PyTorch ops

**Features:**
- Custom attention masks for non-causal patterns
- Paged KV cache integration
- Torch.compile compatibility via custom ops

### 4. Page Manager (`utils/paging.py`)

The `PageManager` implements deterministic, vectorized page allocation:

**Algorithm:**
- Deterministic page IDs: `page_id = request_idx * max_pages_per_request + page_offset`
- CSR-style page table with `(indptr, indices)` representation
- O(N) vectorized operations for batch updates

**API:**
```python
# Vectorized operations (preferred)
page_manager.allocate_counts(add_counts, kv_page_indices, kv_page_indptr)
page_manager.free_counts(remove_counts, kv_page_indices, kv_page_indptr)

# Legacy index-based operations
page_manager.allocate(requested_indices, kv_page_indices, kv_page_indptr)
page_manager.free(requested_indices, kv_page_indices, kv_page_indptr)
```

**Invariants:**
- Each request maintains ≥1 page at all times
- Pages are append-only (allocate at tail, free from tail)
- Deterministic page ordering for regularity

### 5. Sampling Utilities (`utils/sampling.py`)

Token sampling with multiple strategies:

**Functions:**
- `sample(logits, top_p, top_k, temperature)`: Sample tokens from logits
- `get_sampling_probs(logits, top_p, top_k, temperature)`: Get probability distribution from logits

**Strategies:**
- **Greedy**: `temperature=0.0` → argmax
- **Top-k**: Keep only k highest probability tokens
- **Top-p (nucleus)**: Keep tokens with cumulative probability ≤ p
- **Temperature scaling**: Control randomness

### 6. Tensor Parallelism (`utils/tensor_parallel.py`)

LoRA-aware tensor parallelism for distributed inference:

**Key Features:**
- Automatic detection of LoRA layers (GatedLoRALinear)
- Column-parallel and row-parallel sharding
- Proper handling of fused QKV projections
- NCCL-based process group management

**Sharding Rules:**
- **Attention**:
  - `wqkv`: Column-parallel (Q/K/V splits)
  - `wo`: Row-parallel
- **Feed-Forward**:
  - `w13`: Column-parallel (fused Gate|Up split)
  - `w2`: Row-parallel
- **LoRA Adapters**:
  - Base layer: Same as regular Linear
  - `lora_A`: Replicate (col-parallel) or split (row-parallel)
  - `lora_B/BT`: Split (col-parallel) or replicate (row-parallel)

**API:**
```python
init_dist(draft_ranks=None)  # Initialize process groups
apply_tp(model, rank_group, group)  # Apply TP to transformer
apply_tp_eagle(model, rank_group, group)  # Apply TP to EAGLE model
```

## StandardEngine

### Description

`StandardEngine` implements standard autoregressive generation with optimized chunked prefill and decode phases.

### Architecture

**Forward Passes:**
1. **Prefill**: Process input sequence in chunks, return first token prediction
2. **Decode**: Generate one token at a time using cached KV states

**Key Features:**
- Chunked prefill for memory efficiency (configurable `prefill_chunk_size`)
- Efficient padding handling (skip all-padding chunks)
- Budget forcing for controlled generation
- Profiler integration for performance tracking

### Usage Example

```python
from batchspec.backends import StandardEngine
from transformers import AutoTokenizer

# Initialize
tokenizer = AutoTokenizer.from_pretrained("model_name")
engine = StandardEngine(tokenizer, dtype=torch.bfloat16, device="cuda:0")

# Load model
engine.load_model(
    model_name="llama-3-8b",
    checkpoint_path=Path("model.pt"),
    use_tp=False
)

# Setup caches
engine.setup_caches(
    max_batch_size=8,
    max_seq_length=2048,
    page_size=16,
    prefill_chunk_size=128
)

# Configure sampling
engine.setup_sampling_params(temperature=0.8, top_p=0.95, top_k=50)

# Optional: Enable compilation
engine.compile()

# Generate
input_ids = torch.tensor([[1, 2, 3, 4]])  # [batch_size, seq_len]
query_lens = torch.tensor([4])  # Actual lengths (excluding padding)

output, num_generated, num_total, model_steps = engine.generate_batch(
    input_ids=input_ids,
    query_lens=query_lens
)
```

### Generation Process

1. **Prefill Phase**:
   - Input sequence divided into chunks of `prefill_chunk_size`
   - Each chunk processed with causal attention
   - KV states cached for future use
   - Last token logits extracted for sampling

2. **Decode Loop**:
   - Generate one token at a time
   - Use cached KV states (only new token needs attention)
   - Continue until EOS or max length reached

3. **Budget Forcing** (optional):
   - Replace suppress tokens (e.g., `</think>`) with alternatives (e.g., `Wait, Alternatively`)
   - Force generation to continue until budget exhausted

## MTPEngine

### Description

`MTPEngine` implements Multi-Token Prediction with self-speculative decoding, enabling faster generation by predicting and verifying multiple tokens simultaneously.

### Architecture

**Forward Passes:**
1. **Prefill**: Process input sequence, return first token
2. **Draft**: Generate draft tokens using mask tokens and LoRA adapters
3. **Draft and Verify**: Jointly verify drafts and generate new predictions

**Key Components:**
- **Mask Tokens**: Special `<mask>` tokens for multi-token prediction
- **Gate Mask**: Binary mask distinguishing real tokens from mask tokens
- **LoRA Adapters**: Lightweight adapters for efficient draft generation
- **Sampler Head**: Separate head for generating draft tokens
- **Speculative Sampling**: Accept/reject draft tokens based on target model

### Technical Details

#### Attention Patterns

**Standard Draft** (first iteration after prefill):
```
Input: [x_0, <m1>, <m2>, ..., <mk>]
Mask:  [  0,    1,    1, ...,   1 ]  (0=token, 1=mask)
Attention: Causal (standard transformer attention)
Position IDs: [0, 1, 2, ..., k]
```

**Draft and Verify** (subsequent iterations):
```
Input: [x_0, <m1>, ..., <mk>, x_1, <m1>, ..., <mk>, ..., x_k, <m1>, ..., <mk>]
Mask:  [  0,    1, ...,   1,   0,    1, ...,   1, ...,   0,    1, ...,   1 ]
```

Attention mask (k=2 example):
```
[[1,0,0,0,0,0],    # x_0 attends to: x_0
 [1,1,0,0,0,0],    # m_1 attends to: x_0, m_1
 [1,1,1,0,0,0],    # m_2 attends to: x_0, m_1, m_2
 [1,0,0,1,0,0],    # x_1 attends to: x_0, x_1
 [1,0,0,1,1,0],    # m_1 attends to: x_0, x_1, m_1
 [1,0,0,1,1,1]]    # m_2 attends to: x_0, x_1, m_1, m_2
```

Position IDs: `[0, 1, 2, 1, 2, 3]` (derived from attention pattern)

#### Speculative Decoding Process

1. **Draft Generation**:
   - Model predicts logits for all positions
   - Sampler head generates draft tokens from mask positions
   - Draft tokens: `[x_1, x_2, ..., x_k]`

2. **Verification**:
   - Target model predicts: `[x'_1, x'_2, ..., x'_{k+1}]`
   - Compare draft vs target predictions
   - Accept longest prefix where `x_i == x'_i`

3. **Acceptance Strategies**:
   - **Greedy**: Direct token comparison
   - **Sampling**: Speculative sampling algorithm (FlashInfer)

4. **KV Cache Collation**:
   - Keep only KV entries for accepted tokens
   - Discard entries for mask tokens and rejected tokens
   - Rearrange cache to maintain contiguity

#### Budget Forcing

When enabled, the engine replaces suppress tokens (e.g., `</think>`) with alternatives (e.g., `Wait, Alternatively`) to force continued generation:

```python
# Suppress token found in draft: truncate acceptance
if draft_tokens[b, i] == suppress_token_id:
    accept_nums[b] = i
    
# Replace suppress token in bonus with random alternative
if bonus_token == suppress_token_id:
    bonus_token = random.choice(replace_token_ids)
```

### Usage Example

```python
from batchspec.backends import MTPEngine
from batchspec.models import LoRAConfig

# Initialize
tokenizer = AutoTokenizer.from_pretrained("model_name")
engine = MTPEngine(
    tokenizer=tokenizer,
    dtype=torch.bfloat16,
    device="cuda:0",
    draft_length=4  # Generate 4 draft tokens
)

# Load model with LoRA
lora_config = LoRAConfig(
    lora_rank=16,
    lora_alpha=32,
    target_modules=["attention.wqkv", "attention.wo", "feed_forward.w13", "feed_forward.w2"]
)

engine.load_model(
    model_name="llama-3-8b",
    checkpoint_path=Path("base_model.pt"),
    lora_checkpoint_path=Path("lora_adapter.pt"),
    lora_config=lora_config,
    use_tp=False
)

# Setup caches (needs extra space for draft+verify)
engine.setup_caches(
    max_batch_size=8,
    max_seq_length=2048,
    page_size=16,
    prefill_chunk_size=128
)

# Configure sampling
engine.setup_sampling_params(temperature=0.0, top_p=0.95, top_k=0)  # Greedy

# Generate
output, num_generated, num_total, model_steps = engine.generate_batch(
    input_ids=input_ids,
    query_lens=query_lens
)

# Speedup = num_generated / model_steps (typically 2-3x)
```

### Performance Characteristics

**Speedup Factors:**
- Draft length k=4: ~2-3x speedup over standard decoding
- Higher acceptance rate → greater speedup
- Longer sequences benefit more

**Memory Requirements:**
- Additional space for mask tokens: `(k+1)²` positions per step
- LoRA adapters: Minimal overhead (~1-5% of base model)
- Sampler head: Small MLP (hidden_dim → vocab_size)

**Tradeoffs:**
- More aggressive draft length → lower acceptance rate
- Sampling mode → lower acceptance than greedy
- Longer sequences → better efficiency

## Configuration Reference

### Cache Configuration

```python
setup_caches(
    max_batch_size: int = 1,          # Maximum number of sequences in batch
    max_seq_length: int = 2048,       # Maximum sequence length
    page_size: int = 16,              # Tokens per page (affects memory granularity)
    prefill_chunk_size: int = 128     # Chunk size for prefill (must divide seq_length)
)
```

**Tuning Guidelines:**
- `page_size`: Smaller = less waste, larger = fewer allocations (16 is typical)
- `prefill_chunk_size`: Larger = fewer forward passes, more memory (128-512 typical)
- `max_seq_length`: Set to actual maximum needed (extra space wastes memory)

### Sampling Configuration

```python
setup_sampling_params(
    temperature: float = 0.0,    # 0.0 = greedy, >0 = sampling
    top_k: int = 0,              # Keep top-k tokens (0 = disabled)
    top_p: float = 0.95,         # Nucleus sampling threshold (1.0 = disabled)
    force_budget: bool = False   # Force generation to continue (replace suppress tokens)
)
```

**Sampling Strategies:**
- **Greedy**: `temperature=0.0` → Most likely token always
- **Top-k**: Keep only k most probable tokens
- **Top-p**: Keep tokens with cumulative probability ≤ p
- **Temperature**: Scale logits before softmax (higher = more random)

### Compilation

```python
engine.compile()  # Enable torch.compile
```

**Effects:**
- JIT compilation of forward passes
- Optimized kernel fusion
- Reduced Python overhead
- First run slower (compilation time), subsequent runs faster

**When to Use:**
- Production deployments
- Repeated inference with same sequence lengths
- GPU-bound workloads

**When to Skip:**
- Development/debugging
- Varying sequence lengths (frequent recompilation)
- CPU inference

## Tensor Parallelism

### Setup

```python
from batchspec.backends.utils import init_dist, apply_tp

# Initialize distributed training
global_rank, global_group = init_dist()

# Define rank group for this model
rank_group = [0, 1, 2, 3]  # 4-way TP

# Load model on meta device
with torch.device('meta'):
    model = get_model("llama-3-8b", "standard")

# Load checkpoint
checkpoint = torch.load("model.pt")
model.load_state_dict(checkpoint, assign=True)

# Apply tensor parallelism
apply_tp(model, rank_group, global_group)

# Move to device
model = model.to(device=f"cuda:{global_rank}", dtype=torch.bfloat16)
```

### Requirements

**Model Requirements:**
- Number of attention heads divisible by world size
- Number of KV heads divisible by world size
- Supports both regular Linear and LoRA layers

**Hardware Requirements:**
- Multiple GPUs with high-bandwidth interconnect (NVLink preferred)
- NCCL-compatible CUDA setup
- Sufficient GPU memory per rank

### Performance Considerations

**Communication Overhead:**
- All-reduce after each layer (FFN w2, attention wo)
- Critical for performance → requires fast interconnect

**Memory Savings:**
- Model parameters divided by world size
- KV cache size reduced by world size
- Useful for models that don't fit on single GPU


## Integration with Other Components

### Model Integration

The engines work with models from `batchspec.models` that implement:

```python
model(x, gate_mask, position_ids, kv_append_indptr, kv_page_indices, 
      kv_page_indptr, kv_page_lastlen, attn_type)
```

Required components:
- `model.layers`: List of transformer layers
- `layer.attention.kv_cache`: KV cache storage
- `model.config`: Configuration object with `n_head`, `n_local_heads`, `head_dim`, `dim`

For MTPEngine:
- `model.sampler_forward(tokens, hidden_states)`: Draft token generation
- LoRA support via `GatedLoRALinear` layers

### Profiler Integration

```python
from batchspec.profiler import get_active_profiler

profiler = get_active_profiler()  # Returns NullProfiler if profiling disabled

# Profiler automatically tracks:
# - Tokens generated per step
# - Sequence lengths
# - Model forward pass timing
# - Acceptance rates (MTPEngine)
```

### Tokenizer Requirements

Standard HuggingFace tokenizers work out of the box. For MTPEngine:
- Must support adding special tokens (`<mask>`)
- Must have `eos_token_id` defined
- Encode/decode should handle special tokens correctly

