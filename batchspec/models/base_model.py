"""Base transformer classes."""

from abc import ABC, abstractmethod
from typing import Optional, Type, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor

from .configs import ModelArgs, LoRAConfig
from .modules import BaseAttention, RMSNorm, FeedForward, GatedLoRAFeedForward


class BaseTransformerBlock(nn.Module):
    """Basic transformer block with attention and feed-forward layers.
    
    Args:
        config: Model configuration
        attention_module: Instantiated attention module
        feed_forward_module: Instantiated feed-forward module
    """
    
    def __init__(
        self,
        config: ModelArgs,
        attention_module: BaseAttention,
        feed_forward_module: Optional[nn.Module] = None
    ):
        super().__init__()
        
        self.attention = attention_module
        self.feed_forward = feed_forward_module or FeedForward(config)
        
        # normalization
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
    
    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """Forward pass through transformer block.
        
        Applies pre-norm residual connections for both attention
        and feed-forward sublayers.
        
        Args:
            x: Input tensor
            *args, **kwargs: Arguments passed to attention layer
            
        Returns:
            Output tensor with same shape as input
        """
        # Attention with residual
        h = x + self.attention(self.attention_norm(x), *args, **kwargs)
        
        # Feed-forward with residual
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class GatedLoRATransformerBlock(BaseTransformerBlock):
    """Transformer block with gated LoRA support.
    
    Args:
        config: Model configuration
        attention_module: Gated LoRA-enabled attention module
        lora_config: LoRA configuration for feed-forward
    """
    
    def __init__(
        self,
        config: ModelArgs,
        attention_module: BaseAttention,
        lora_config: LoRAConfig
    ):
        # Use gated LoRA feed-forward
        feed_forward = GatedLoRAFeedForward(config, lora_config)
        super().__init__(config, attention_module, feed_forward)
        
    def forward(
        self, 
        x: Tensor,
        gate_mask: Optional[Tensor] = None,
        *args, 
        **kwargs
    ) -> Tensor:
        """Forward pass with gated LoRA.
        
        Args:
            x: Input tensor
            gate_mask: Optional mask for gated LoRA
            *args, **kwargs: Additional arguments for attention
            
        Returns:
            Output tensor
        """
        # Attention with residual and gating
        h = x + self.attention(self.attention_norm(x), gate_mask, *args, **kwargs)
        
        # Feed-forward with residual and gating
        out = h + self.feed_forward(self.ffn_norm(h), gate_mask)
        return out


class BaseTransformer(nn.Module, ABC):
    """Base transformer model class.
    
    Provides common functionality for transformer models including
    embedding layers, transformer blocks, and output projection.
    
    Args:
        config: Model configuration
    """
    
    def __init__(self, config: ModelArgs):
        super().__init__()
        
        self.config = config
        
        # Token embeddings
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        
        # Transformer layers (built in subclasses)
        self.layers = self._build_layers()
        
        # Output normalization and projection
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Distributed training support
        self.world_size = None
        self.rank = None
        self.process_group = None
        
        # Vocabulary range for tensor parallel
        self.vocab_start = 0
        self.vocab_end = config.vocab_size

    def _build_layers(self) -> nn.ModuleList:
        """Build transformer layers.
        
        Returns:
            ModuleList of transformer blocks
        """
        attention_class = self._get_attention_class()
        block_class = self._get_block_class()
        
        layers = []
        for _ in range(self.config.n_layer):
            attention = attention_class(self.config)
            layers.append(block_class(self.config, attention))
            
        return nn.ModuleList(layers)

    @abstractmethod
    def _get_attention_class(self) -> Type[BaseAttention]:
        """Get the attention class to use for layers.
        
        Must be implemented by subclasses.
        
        Returns:
            Attention class (not instance)
        """
        raise NotImplementedError()
    
    def _get_block_class(self) -> Type[BaseTransformerBlock]:
        """Get the transformer block class to use.
        
        Can be overridden by subclasses.
        
        Returns:
            Transformer block class (not instance)
        """
        return BaseTransformerBlock
    
    @abstractmethod
    def setup_caches(self, *args, **kwargs):
        """Setup KV caches and attention kernels.
        
        Must be implemented by subclasses.
        """
        raise NotImplementedError()
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:
        """Forward pass through the transformer.
        
        Must be implemented by subclasses.
        """
        raise NotImplementedError()
    
    def _maybe_all_gather_logits(self, logits: Tensor) -> Tensor:
        """All-gather logits across all ranks for tensor parallel.
        
        Args:
            logits: Local logits tensor
            
        Returns:
            All-gathered logits from all ranks
        """
        if self.process_group is None:
            return logits

        gathered_logits = [
            torch.empty_like(logits) 
            for _ in range(self.world_size)
        ]
        dist.all_gather(gathered_logits, logits, group=self.process_group)
        return torch.cat(gathered_logits, dim=-1)
    
    def resize_token_embeddings(
        self, 
        new_num_tokens: Optional[int] = None
    ) -> nn.Embedding:
        """Resize token embeddings for vocabulary changes.
        
        Args:
            new_num_tokens: New vocabulary size
            
        Returns:
            Updated embedding layer
        """
        old_embeddings = self.tok_embeddings
        
        if new_num_tokens is None:
            return old_embeddings
            
        if new_num_tokens <= 0:
            raise ValueError(f"new_num_tokens must be positive, got {new_num_tokens}")

        old_vocab = old_embeddings.num_embeddings
        emb_dim = old_embeddings.embedding_dim
        
        if new_num_tokens == old_vocab:
            return old_embeddings

        device = old_embeddings.weight.device
        dtype = old_embeddings.weight.dtype
        
        # Create new embeddings
        new_embeddings = nn.Embedding(
            new_num_tokens, emb_dim, 
            device=device, dtype=dtype
        )

        with torch.no_grad():
            # Copy existing embeddings
            n_copy = min(old_vocab, new_num_tokens)
            new_embeddings.weight[:n_copy].copy_(old_embeddings.weight[:n_copy])

            # Initialize new embeddings with mean of existing
            if new_num_tokens > old_vocab and old_vocab > 0:
                mean_vec = old_embeddings.weight[:old_vocab].mean(dim=0, keepdim=True)
                new_embeddings.weight[old_vocab:new_num_tokens].copy_(mean_vec)

            # Zero out padding token if it exists
            pad_id = getattr(self.config, "pad_token_id", None)
            if pad_id is not None and 0 <= pad_id < new_num_tokens:
                new_embeddings.weight[pad_id].zero_()

        # Preserve gradient requirements
        new_embeddings.requires_grad_(old_embeddings.weight.requires_grad)

        self.tok_embeddings = new_embeddings
        return self.tok_embeddings

    @classmethod
    def from_name(cls, name: str, **kwargs):
        """Create model from configuration name.
        
        Args:
            name: Configuration name
            **kwargs: Additional arguments passed to constructor
            
        Returns:
            Transformer model instance
        """
        from ..configs import get_config
        config = get_config(name)
        return cls(config, **kwargs)
