# Models Module

The `models` module provides a modular implementation of transformer architectures optimized for speculative decoding. It includes three distinct model types, each designed for specific inference strategies, along with shared components for efficient computation.

## Overview

This module implements transformer models with specialized attention mechanisms and adaptive components:

1. **StandardTransformer**: Standard autoregressive transformer with paged KV caching
2. **MTPTransformer**: Multi-Token Prediction transformer with gated LoRA adaptation
3. **EAGLETransformer**: EAGLE speculative decoding with separate target and draft models

All models share a common infrastructure built on:
- **Paged KV Caching**: Memory-efficient attention with dynamic page allocation
- **FlashInfer Integration**: Optimized attention kernels for GPU acceleration
- **Modular Architecture**: Reusable components with minimal code duplication
- **Configuration Management**: Centralized model configurations with registry system

## Architecture

### Directory Structure

```
models/
├── __init__.py              # Package exports and factory function
├── base_model.py            # Base classes for transformers
├── standard_model.py        # Standard transformer implementation
├── mtp_model.py            # Multi-token prediction transformer
├── eagle_model.py          # EAGLE speculative decoding transformer
├── configs/                # Configuration management
│   ├── model_config.py     # ModelArgs and LoRAConfig classes
│   └── registry.py         # Model configuration registry
└── modules/                # Shared neural network components
    ├── attention/          # Attention implementations
    │   ├── base.py         # Base attention classes
    │   ├── standard.py     # Standard attention
    │   ├── mtp.py          # MTP attention with gating
    │   └── eagle.py        # EAGLE attention
    ├── feedforward.py      # Feed-forward networks
    ├── lora.py             # Gated LoRA implementation
    ├── kv_cache.py         # KV cache implementations
    ├── normalization.py    # RMSNorm layer
    ├── rope.py             # RoPE mixin for positional encoding
    └── sampler_head.py     # Sampler head for MTP
```

### Class Hierarchy

```
BaseTransformer (abstract)
├── StandardTransformer       # Standard autoregressive generation
├── MTPTransformer            # Multi-token prediction with LoRA
└── EAGLETransformer          # EAGLE speculative decoding

BaseTransformerBlock
├── BaseTransformerBlock      # Standard transformer block
└── GatedLoRATransformerBlock # Block with gated LoRA

BaseAttention (abstract)
├── StandardAttention         # Standard paged attention
├── MTPAttention              # MTP attention with gating
└── EAGLEAttention            # EAGLE attention for drafting
```

## Configuration System

### ModelArgs

The `ModelArgs` dataclass defines architecture parameters for transformer models:

```python
@dataclass
class ModelArgs:
    # Core architecture
    block_size: int = 2048           # Maximum sequence length
    vocab_size: int = 32000          # Vocabulary size
    n_layer: int = 32                # Number of transformer layers
    n_head: int = 32                 # Number of attention heads
    dim: int = 4096                  # Model dimension
    intermediate_size: Optional[int] = None  # FFN intermediate size
    n_local_heads: int = -1          # Number of KV heads (for GQA)
    head_dim: int = -1               # Dimension per head
    
    # RoPE configuration
    rope_base: float = 10000.0       # Base for RoPE frequencies
    scaling_factor: float = 1.0      # Scaling factor for extended context
    
    # Model-specific options
    norm_eps: float = 1e-5           # Layer normalization epsilon
    qkv_bias: bool = False           # Bias in attention projections
    draft_vocab_size: int = 32000    # Draft vocabulary size (EAGLE)
```

**Key Features:**
- Automatic computation of derived values (head_dim, intermediate_size)
- Support for Grouped Query Attention (GQA) via `n_local_heads`
- RoPE scaling for extended context lengths
- Model-specific parameters (draft_vocab_size for EAGLE)

### LoRAConfig

The `LoRAConfig` dataclass specifies parameters for LoRA adaptation:

```python
@dataclass
class LoRAConfig:
    rank: int = 16                   # LoRA rank
    alpha: float = 32.0              # LoRA scaling factor
    lora_bias: bool = False          # Include bias in LoRA
    use_rslora: bool = False         # Use rank-stabilized LoRA
    lora_scaling: Optional[float] = None  # Computed scaling factor
```

**Scaling Methods:**
- **Standard LoRA**: `scaling = alpha / rank`
- **RS-LoRA**: `scaling = alpha / sqrt(rank)` (more stable for higher ranks)

### Configuration Registry

Pre-defined configurations for popular models:

```python
from batchspec.models import get_config

# Get configuration by name
config = get_config("llama-3-8b")
config = get_config("Qwen2.5-7b")
config = get_config("DeepSeek-R1-Distill-Llama-8B")

# Register custom configuration
from batchspec.models import register_config
register_config("custom-model", {
    "n_layer": 24,
    "n_head": 16,
    "dim": 2048,
    "vocab_size": 50000
})
```

**Supported Models:**
- Llama 2 series (7B, 13B, 70B)
- Llama 3 series (8B, 70B)
- Llama 3.1 series (8B, 70B)
- Qwen 2.5 series (7B, 14B, 32B)
- Qwen 3 series (0.6B, 1.7B, 8B)
- DeepSeek-R1 series
- Extended context models (262K tokens)
- EAGLE drafter configurations

## Core Components

### 1. Base Classes

#### BaseTransformer (`base_model.py`)

Abstract base class providing common transformer functionality:

**Key Features:**
- Token embedding layer
- Transformer layer management
- Output normalization and projection
- Distributed training support
- Token embedding resizing
- Factory method for configuration-based creation

**Key Methods:**
```python
class BaseTransformer(nn.Module, ABC):
    def __init__(self, config: ModelArgs)
    
    @abstractmethod
    def _get_attention_class(self) -> Type[BaseAttention]
    
    @abstractmethod
    def setup_caches(self, *args, **kwargs)
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> Tensor
    
    def resize_token_embeddings(self, new_num_tokens: int)
    
    @classmethod
    def from_name(cls, name: str, **kwargs)
```

#### BaseTransformerBlock (`base_model.py`)

Standard transformer block with pre-norm architecture:

**Structure:**
```
Input
  ↓
[Layer Norm] → [Attention] → [Residual Add]
  ↓
[Layer Norm] → [Feed Forward] → [Residual Add]
  ↓
Output
```

**Variants:**
- `BaseTransformerBlock`: Standard implementation
- `GatedLoRATransformerBlock`: With gated LoRA support

### 2. Attention Mechanisms

#### BaseAttention (`modules/attention/base.py`)

Abstract base class for attention implementations:

**Common Interface:**
```python
class BaseAttention(nn.Module, ABC):
    def __init__(self, config: ModelArgs)
    
    @abstractmethod
    def forward(
        self,
        x: Tensor,
        input_pos: Tensor,
        kv_append_indptr: Tensor,
        kv_page_indices: Tensor,
        kv_page_indptr: Tensor,
        kv_page_lastlen: Tensor,
        attn_type: str
    ) -> Tensor
```

**Key Features:**
- QKV projection with optional bias
- Grouped Query Attention (GQA) support
- Paged KV cache integration
- Output projection with residual connection
- Distributed inference support (all-reduce)

#### AttentionMixin (`modules/attention/base.py`)

Mixin providing shared attention functionality:

**Capabilities:**
- RoPE application (rotary position embeddings)
- KV cache updates
- QKV tensor preparation
- Output projection with all-reduce

#### StandardAttention (`modules/attention/standard.py`)

Standard paged attention for autoregressive generation:

**Attention Types:**
- `prefill`: Initial sequence processing
- `decode`: Single-token generation
- `verify`: Verification in speculative decoding

**Forward Pass:**
1. Apply QKV projection
2. Apply RoPE to queries and keys
3. Update KV cache
4. Compute attention (via FlashInfer)
5. Project output

#### MTPAttention (`modules/attention/mtp.py`)

Multi-token prediction attention with gated LoRA:

**Key Differences:**
- Gated LoRA in QKV projection
- Support for custom attention masks (non-causal)
- Position IDs instead of offsets
- Three attention modes: `prefill`, `draft`, `draft_and_verify`

**Gated LoRA:**
```python
class GatedLoRAAttention(BaseAttention):
    def __init__(self, config: ModelArgs, lora_config: LoRAConfig)
    
    def forward(
        self,
        x: Tensor,
        gate_mask: Optional[Tensor],   # Controls LoRA application
        position_ids: Tensor,          # Explicit position IDs
        ...
    )
```

#### EAGLEAttention (`modules/attention/eagle.py`)

EAGLE attention for draft generation:

**Architecture:**
- Double-width input (concatenated embeddings and hidden states)
- Separate projection for each input component
- Mode-specific attention patterns (chain vs standard)
- Support for multiple EAGLE decoding stages

**Chain Mode vs Standard Mode:**
- **Chain Mode**: Sequential draft generation (decode_1, decode_2)
- **Standard Mode**: Tree-based speculation (init_speculate, sub_speculate)

### 3. Feed-Forward Networks

#### FeedForward (`modules/feedforward.py`)

Standard SwiGLU feed-forward network:

**Architecture:**
```
x → [Linear w13] → [SwiGLU] → [Linear w2] → output
```

**Key Features:**
- Fused gate and up projections (w13) for efficiency
- SwiGLU activation: `SiLU(gate) * up`
- Distributed all-reduce support
- Legacy checkpoint compatibility

#### GatedLoRAFeedForward (`modules/feedforward.py`)

Feed-forward network with gated LoRA:

**Key Differences:**
- Gated LoRA in both w13 and w2
- Double LoRA rank for w13 (fused projections)
- Gate mask controls LoRA activation

### 4. KV Cache Implementations

#### StandardKVCache (`modules/kv_cache.py`)

Paged key-value cache for efficient attention:

**Cache Structure:**
```
Shape: [num_pages, 2 (K/V), page_size, n_heads, head_dim]
```

**Key Features:**
- Paged memory allocation
- Efficient cache updates via custom ops
- Support for variable sequence lengths
- Integration with FlashInfer

**Update Process:**
1. Compute batch indices and positions
2. Append new KV pairs to cache pages
3. Update page pointers and lengths

#### StreamingKVCache (`modules/kv_cache.py`)

Streaming KV cache with sliding window:

**Additional Features:**
- Fixed token budget (streaming_budget)
- Sink token preservation (initial tokens always kept)
- Automatic eviction of old tokens
- RoPE recalculation for shifted keys

**Use Cases:**
- Long-sequence generation with limited memory
- Streaming inference
- Extended context with fixed budget

### 5. LoRA Components

#### GatedLoRALinear (`modules/lora.py`)

Linear layer with gated LoRA adaptation:

**Architecture:**
```
x → [Base Linear] → base_output
x → [LoRA A] → [Gate Mask] → [LoRA B] → lora_output
output = base_output + lora_output * scaling
```

**Key Features:**
- Selective LoRA application via gate mask
- Efficient computation using transposed B matrix
- State dict hooks for checkpoint compatibility
- Profiler integration for performance analysis

**Gating Mechanism:**
```python
# Gate mask: 0 = base only, 1 = base + LoRA
z = lora_A(x)           # Project to low-rank space
z = z * gate_mask       # Apply gating
output = base + z @ lora_BT * scaling
```

### 6. Auxiliary Modules

#### RMSNorm (`modules/normalization.py`)

Root Mean Square Layer Normalization:

**Formula:**
```
RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
```

**Advantages:**
- Simpler than LayerNorm (no centering)
- Faster computation
- Lower memory footprint

#### RoPEMixin (`modules/rope.py`)

Mixin for Rotary Position Embeddings setup:

**Key Methods:**
```python
class RoPEMixin:
    def _setup_rope_kernels(self, use_position_ids: bool)
```

**Configuration:**
- `use_position_ids=False`: Use offsets (StandardTransformer)
- `use_position_ids=True`: Use explicit position IDs (MTPTransformer)

#### SamplerHead (`modules/sampler_head.py`)

Sampler head for multi-token prediction:

**Architecture:**
```
Input (2*dim) → [Linear + Norm + SiLU] → (dim)
              → [Linear + Norm + SiLU] → (dim)
              → [Residual Add] → Output
```

**Usage:**
```python
# Concatenate token embeddings and hidden states
sampler_input = torch.cat([token_embeds, hidden_states], dim=-1)
# Generate features for draft token prediction
features = sampler_head(sampler_input)
# Project to vocabulary
draft_logits = lm_head(features)
```

## Model Implementations

### StandardTransformer

Standard transformer for autoregressive generation.

#### Features

- Standard causal attention
- Paged KV caching
- Prefill and decode modes
- RoPE with position offsets

#### Usage

```python
from batchspec.models import StandardTransformer

# Create model from configuration
model = StandardTransformer.from_name("llama-3-8b")

# Setup caches
model.setup_caches(num_pages=1000, page_size=16)

# Forward pass
logits = model(
    idx=input_ids,              # [batch, seq_len]
    input_pos=position_offsets, # [batch]
    kv_append_indptr=kv_append_indptr,
    kv_page_indices=kv_page_indices,
    kv_page_indptr=kv_page_indptr,
    kv_page_lastlen=kv_page_lastlen,
    attn_type="prefill"         # or "decode"
)
```

#### Architecture

**Attention Pattern:**
- **Prefill**: Process input sequence in chunks, cache all KV states
- **Decode**: Generate one token at a time using cached KV states

**Cache Management:**
- RoPE uses position offsets (relative to cache length)
- Standard paged KV cache
- Efficient single-token decoding

### MTPTransformer

Multi-Token Prediction transformer with self-speculative decoding.

#### Features

- Gated LoRA adaptation
- Custom attention masks (non-causal)
- Sampler head for draft generation
- Three attention modes: prefill, draft, draft_and_verify

#### Usage

```python
from batchspec.models import MTPTransformer, LoRAConfig

# Create model with LoRA configuration
lora_config = LoRAConfig(rank=16, alpha=32)
model = MTPTransformer.from_name("llama-3-8b", lora_config)

# Setup caches (needs extra space for draft+verify)
model.setup_caches(num_pages=1000, page_size=16)

# Forward pass with gated LoRA
logits, hidden_states = model(
    idx=input_ids,
    gate_mask=gate_mask,        # 0=token, 1=mask
    position_ids=position_ids,  # Explicit positions
    kv_append_indptr=kv_append_indptr,
    kv_page_indices=kv_page_indices,
    kv_page_indptr=kv_page_indptr,
    kv_page_lastlen=kv_page_lastlen,
    attn_type="draft_and_verify"
)

# Generate draft tokens using sampler
draft_tokens = model.sampler_forward(
    idx=prev_tokens,            # [batch, 1]
    hidden_states=hidden_states # [batch, draft_len, dim]
)
```

#### Architecture

**Attention Modes:**

1. **Prefill Mode**: Standard causal attention for input processing
   ```
   Input: [x_0, x_1, ..., x_n]
   Cache: All KV states
   Output: Logits for next token
   ```

2. **Draft Mode**: Generate initial drafts after prefill
   ```
   Input: [x_0, <m1>, <m2>, ..., <mk>]  (k = draft_length)
   Mask:  [  0,    1,    1, ...,   1 ]
   Attention: Causal
   Output: Draft token predictions
   ```

3. **Draft and Verify Mode**: Joint draft generation and verification
   ```
   Input: Interleaved tokens and masks
          [x_0, <m1>, ..., <mk>, x_1, <m1>, ..., <mk>, ...]
   Mask:  [  0,    1, ...,   1,   0,    1, ...,   1, ...]
   Attention: Non-causal (see MTP attention pattern)
   Output: Verified predictions + new drafts
   ```

**Gated LoRA:**
- Base model processes all tokens
- LoRA adapters process only mask tokens (gate_mask=1)
- Enables efficient fine-tuning for draft generation

**Sampler Head:**
- Combines token embeddings and hidden states
- Generates draft token predictions
- Independent of main model forward pass

### EAGLETransformer

EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency) transformer combining target and draft models.

#### Features

- Separate target and EAGLE modules
- Hidden state extraction from target model
- Support for chain and standard modes
- Vocabulary mapping between draft and target

#### Usage

```python
from batchspec.models import EAGLETransformer

# Create model with target and drafter configurations
model = EAGLETransformer.from_name(
    target_name="llama-3-8b",
    drafter_name="EAGLE3-DeepSeek-R1-Distill-LLaMA-8B",
    use_chain_mode=True  # or False for standard mode
)

# Setup caches
model.setup_caches(num_pages=1000, page_size=16)

# Target model forward (with hidden state collection)
logits, hidden_states = model(
    idx=input_ids,
    input_pos=position_info,
    kv_append_indptr=kv_append_indptr,
    kv_page_indices=kv_page_indices,
    kv_page_indptr=kv_page_indptr,
    kv_page_lastlen=kv_page_lastlen,
    return_hidden_states=True,
    attn_type="prefill"
)

# EAGLE module forward (draft generation)
eagle_output = model.eagle_forward(
    hidden_states=torch.cat(hidden_states, dim=-1),  # Concatenate layers
    idx=input_ids,
    input_pos=position_info,
    kv_append_indptr=kv_append_indptr,
    kv_page_indices=kv_page_indices,
    kv_page_indptr=kv_page_indptr,
    kv_page_lastlen=kv_page_lastlen,
    attn_type="prefill"
)
```

#### Architecture

**Target Model:**
- Standard transformer architecture
- Hidden state extraction from layers [2, n_layer//2, n_layer-3]
- Standard or verification attention

**EAGLE Module:**
```
Hidden States (3 layers) → [FC Projection] → (dim)
                                              ↓
Input Embeddings → [Norm] ───────────┐        ↓
                                     ├─→ [Concat] → [EAGLE Attention]
Hidden States → [Norm] ──────────────┘                ↓
                                              [Feed Forward]
                                                      ↓
                                              [Output Projection]
```

**Chain Mode:**
- Sequential draft generation
- Two-stage decoding (decode_1, decode_2)
- Position offsets for RoPE

**Standard Mode:**
- Tree-based speculation
- Parallel draft generation
- Position IDs for RoPE

**Vocabulary Mapping:**
- Draft model may have different vocabulary
- Buffers for token ID conversion: `draft_to_target`, `target_to_draft`
- Automatic conversion during speculation

## Factory Function

The `get_model` factory function provides a unified interface for model creation:

```python
from batchspec.models import get_model

# Standard model
standard_model = get_model(
    model_name="llama-3-8b",
    model_type="standard"
)

# MTP model with LoRA
from batchspec.models import LoRAConfig
lora_config = LoRAConfig(rank=16, alpha=32)
mtp_model = get_model(
    model_name="llama-3-8b",
    model_type="mtp",
    lora_config=lora_config
)

# EAGLE model
eagle_model = get_model(
    model_name="llama-3-8b",
    model_type="eagle",
    drafter_name="EAGLE3-DeepSeek-R1-Distill-LLaMA-8B",
    use_chain_mode=True
)
```

**Parameters:**
- `model_name`: Configuration name for the model
- `model_type`: Type of model ("standard", "mtp", "eagle")
- `lora_config`: LoRA configuration (MTP only)
- `drafter_name`: EAGLE drafter configuration name (EAGLE only)
- `use_chain_mode`: Whether to use chain mode (EAGLE only)

## Integration with Backends

The models module integrates seamlessly with the backends module:

**StandardEngine + StandardTransformer:**
```python
from batchspec.backends import StandardEngine
from batchspec.models import get_model

# Backend creates and manages the model
engine = StandardEngine(tokenizer, dtype=torch.bfloat16)
engine.load_model(model_name="llama-3-8b", checkpoint_path=path)
```

**MTPEngine + MTPTransformer:**
```python
from batchspec.backends import MTPEngine
from batchspec.models import LoRAConfig

# Backend handles LoRA configuration
lora_config = LoRAConfig(rank=16, alpha=32)
engine = MTPEngine(tokenizer, draft_length=4)
engine.load_model(
    model_name="llama-3-8b",
    checkpoint_path=base_path,
    lora_checkpoint_path=lora_path,
    lora_config=lora_config
)
```

**Custom Operations:**
- Models register custom PyTorch operators for attention
- Backends manage attention wrappers and buffers
- Attention kernels registered at setup time

## Checkpoint Loading

### State Dict Hooks

Models include hooks for flexible checkpoint loading:

**FeedForward Hook:**
```python
# Automatically handles w1/w3 → w13 conversion
# Legacy: separate w1.weight and w3.weight
# Modern: fused w13.weight
```

**GatedLoRALinear Hook:**
```python
# Handles multiple checkpoint formats:
# 1. Non-LoRA: weight/bias → base_layer.weight/bias
# 2. LoRA: lora_B.weight → lora_BT (transposed)
```

### Loading Procedure

```python
# Load on meta device (no memory allocation)
with torch.device('meta'):
    model = get_model("llama-3-8b", "standard")

# Load checkpoint with assign=True for meta tensors
checkpoint = torch.load("model.pt", mmap=True, weights_only=True)
model.load_state_dict(checkpoint, assign=True, strict=True)

# Move to device
model = model.to(device="cuda", dtype=torch.bfloat16)
```

### Multi-Checkpoint Loading (MTP)

```python
# 1. Load base model weights
base_checkpoint = torch.load("base_model.pt")
model.load_state_dict(base_checkpoint, assign=True, strict=False)

# 2. Load LoRA adapter weights
lora_checkpoint = torch.load("lora_adapter.pt")
model.load_state_dict(lora_checkpoint, assign=True, strict=False)

# 3. Resize token embeddings if needed
model.resize_token_embeddings(new_vocab_size)
```

## Distributed Inference Support

### Tensor Parallelism

Models support tensor parallelism via the backends utilities:

```python
from batchspec.backends.utils import apply_tp

# Apply TP to model
apply_tp(model, rank_group=[0, 1, 2, 3], group=process_group)
```

**Effects on Model:**
- Attention heads split across ranks
- FFN intermediate dimension split across ranks
- KV cache size reduced per rank
- Vocabulary partitioned (output projection)

**LoRA Support:**
- Automatically handles LoRA layers
- Base and adapter layers sharded appropriately
- No code changes needed in model

### All-Reduce Operations

Models include all-reduce hooks for distributed inference:

**Feed-Forward:**
```python
# After w2 projection
if self.process_group is not None:
    dist.all_reduce(output, group=self.process_group)
```

**Attention:**
```python
# After output projection
if self.process_group is not None:
    dist.all_reduce(output, group=self.process_group)
```

**Output Projection:**
```python
# Gather logits from all ranks
if self.process_group is not None:
    logits = self._maybe_all_gather_logits(logits)
```

## Custom Operators

Models use custom PyTorch operators for performance:

### RoPE Operator

```python
torch.ops.mylib.rope(q, k, indptr, offsets_or_positions)
```

**Purpose:**
- Apply rotary position embeddings efficiently
- Integrated with FlashInfer

### KV Cache Update Operator

```python
torch.ops.mylib.update_kv(
    k, v, kv_append_indptr, kv_cache,
    kv_page_indices, kv_page_indptr, kv_page_lastlen, page_size
)
```

**Purpose:**
- Update paged KV cache with new entries
- Integrated with FlashInfer's append functions

### Attention Operators

```python
# Registered by backend engines
torch.ops.mylib.attn_prefill(q, kv_cache)
torch.ops.mylib.attn_decode(q, kv_cache)
torch.ops.mylib.attn_draft(q, kv_cache)
torch.ops.mylib.attn_draft_and_verify(q, kv_cache)
```

**Purpose:**
- Efficient attention computation using FlashInfer
- Different patterns for different decoding stages

## Configuration Examples

### Creating Custom Models

```python
from batchspec.models import ModelArgs, StandardTransformer

# Define custom configuration
config = ModelArgs(
    block_size=4096,
    vocab_size=50000,
    n_layer=24,
    n_head=16,
    n_local_heads=4,  # GQA
    dim=2048,
    intermediate_size=8192,
    rope_base=10000.0,
    norm_eps=1e-6
)

# Create model
model = StandardTransformer(config)
```

### Extended Context Configuration

```python
# Llama 3.1 with extended context
config = ModelArgs(
    block_size=131072,  # 128K context
    n_layer=32,
    n_head=32,
    n_local_heads=8,
    dim=4096,
    rope_base=500000.0,
    scaling_factor=8,
    high_freq_factor=4,
    low_freq_factor=1,
    original_max_position_embeddings=8192
)
```

### LoRA Configuration Variations

```python
# Standard LoRA
lora_config = LoRAConfig(rank=16, alpha=32)

# RS-LoRA (more stable for higher ranks)
lora_config = LoRAConfig(rank=64, alpha=128, use_rslora=True)

# Custom scaling
lora_config = LoRAConfig(rank=16, alpha=32, lora_scaling=1.0)
```

## Model Specifications

### Standard Transformer

**Input Signature:**
```python
def forward(
    idx: Tensor,                # [batch, seq_len] token IDs
    input_pos: Tensor,          # [batch] position offsets
    kv_append_indptr: Tensor,   # [batch+1] KV append indices
    kv_page_indices: Tensor,    # [num_pages] page IDs
    kv_page_indptr: Tensor,     # [batch+1] page pointers
    kv_page_lastlen: Tensor,    # [batch] last page lengths
    attn_type: str              # "prefill" or "decode"
) -> Tensor                     # [batch, seq_len, vocab_size]
```

### MTP Transformer

**Input Signature:**
```python
def forward(
    idx: Tensor,                # [batch, seq_len] token IDs
    gate_mask: Tensor,          # [batch, seq_len, 1] LoRA gate
    position_ids: Tensor,       # [batch*seq_len] positions
    kv_append_indptr: Tensor,   # [batch+1] KV append indices
    kv_page_indices: Tensor,    # [num_pages] page IDs
    kv_page_indptr: Tensor,     # [batch+1] page pointers
    kv_page_lastlen: Tensor,    # [batch] last page lengths
    attn_type: str              # "prefill", "draft", "draft_and_verify"
) -> Tuple[Tensor, Tensor]     # logits, hidden_states

def sampler_forward(
    idx: Tensor,                # [batch, 1] previous token
    hidden_states: Tensor       # [batch, draft_len, dim]
) -> Tensor                     # [batch, 1] next token
```

### EAGLE Transformer

**Target Forward:**
```python
def forward(
    idx: Tensor,
    input_pos: Tensor,
    kv_append_indptr: Tensor,
    kv_page_indices: Tensor,
    kv_page_indptr: Tensor,
    kv_page_lastlen: Tensor,
    return_hidden_states: bool = False,
    attn_type: str = "prefill"
) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]
```

**EAGLE Forward:**
```python
def eagle_forward(
    hidden_states: Tensor,      # [batch, seq_len, 3*dim] from target
    idx: Tensor,                # [batch, seq_len] token IDs
    input_pos: Tensor,          # Position info (mode-dependent)
    kv_append_indptr: Tensor,
    kv_page_indices: Tensor,
    kv_page_indptr: Tensor,
    kv_page_lastlen: Tensor,
    attn_type: str
) -> Union[Tensor, Tuple[Tensor, Tensor]]  # Chain mode returns hidden states
```

## Summary

The `models` module provides a comprehensive, modular implementation of transformer architectures optimized for speculative decoding:

- **Three Model Types**: Standard, MTP, and EAGLE transformers for different inference strategies
- **Shared Infrastructure**: Reusable components minimize code duplication
- **Configuration System**: Centralized registry with pre-defined configurations for popular models
- **Advanced Features**: Gated LoRA, paged KV caching, grouped query attention, streaming cache
- **Integration Ready**: Designed to work seamlessly with the backends module
- **Extensible Design**: Easy to add new attention mechanisms and model variants

Choose StandardTransformer for general-purpose inference, MTPTransformer for self-speculative decoding with LoRA, or EAGLETransformer for EAGLE-style speculative decoding with separate draft models.

