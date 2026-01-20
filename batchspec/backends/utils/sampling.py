"""Sampling utilities for token generation.

"""

import torch
from torch import Tensor
from flashinfer.sampling import (
    top_k_renorm_probs,
    top_p_renorm_probs,
    top_k_sampling_from_probs,
    top_p_sampling_from_probs,
    top_k_top_p_sampling_from_logits,
    sampling_from_logits,
    softmax,
)


def get_sampling_probs(
    logits: Tensor,
    top_p: float,
    top_k: int,
    temperature: float
) -> Tensor:
    """Get sampling probabilities from logits.
    
    Applies temperature scaling, top-k, and top-p filtering.
    
    Args:
        logits: Logits tensor of shape (batch, vocab) or (batch, seq, vocab)
        top_p: Top-p (nucleus) sampling parameter
        top_k: Top-k sampling parameter
        temperature: Temperature for scaling
        
    Returns:
        Probability tensor with same shape as logits
    """
    # Handle different input shapes
    if len(logits.shape) == 2:
        bsz, voc_size = logits.shape
        seqlen = 1
    elif len(logits.shape) == 3:
        bsz, seqlen, voc_size = logits.shape
        logits = logits.reshape(-1, voc_size)
    else:
        raise ValueError(f"Given `logits` has an invalid shape: {logits.shape}")

    # Apply temperature and convert to probabilities
    probs = softmax(logits, temperature=temperature)
    
    # Apply top-p filtering
    if top_p > 0 and top_p < 1.0:
        probs = top_p_renorm_probs(probs, top_p)
    
    # Apply top-k filtering
    if top_k > 0:
        probs = top_k_renorm_probs(probs, top_k)
    
    return probs.reshape(bsz, seqlen, voc_size)


def sample(
    logits: Tensor,
    top_p: float,
    top_k: int,
    temperature: float
) -> Tensor:
    """Sample tokens from logits with top-k, top-p, and temperature.
    
    Args:
        logits: Logits tensor of shape (batch, vocab) or (batch, seq, vocab)
        top_p: Top-p (nucleus) sampling parameter
        top_k: Top-k sampling parameter
        temperature: Sampling temperature (0.0 for greedy)
        
    Returns:
        Sampled token indices of shape (batch, seq)
    """
    # Handle different input shapes
    if len(logits.shape) == 2:
        bsz, voc_size = logits.shape
        seqlen = 1
    elif len(logits.shape) == 3:
        bsz, seqlen, voc_size = logits.shape
        logits = logits.reshape(-1, voc_size)
    else:
        raise ValueError(f"Given `logits` has an invalid shape: {logits.shape}")

    # Greedy sampling (temperature = 0)
    if temperature == 0:
        return torch.argmax(logits, dim=-1).reshape(bsz, seqlen).long()
    
    # Non-greedy sampling with various strategies
    if top_k > 0:
        if top_p > 0:
            # Simultaneously apply top-k and top-p
            samples = top_k_top_p_sampling_from_logits(
                logits=logits / temperature,
                top_k=top_k,
                top_p=top_p
            )
        else:
            # Apply top-k only
            probs = softmax(logits, temperature=temperature)
            samples = top_k_sampling_from_probs(probs=probs, top_k=top_k)
    else:
        if top_p > 0:
            # Apply top-p only
            probs = softmax(logits, temperature=temperature)
            samples = top_p_sampling_from_probs(probs=probs, top_p=top_p)
        else:
            # No top-k or top-p (full distribution sampling)
            samples = sampling_from_logits(logits=logits / temperature)
    
    samples = samples.reshape(bsz, seqlen).long()
    return samples


def fix_invalid_draft_tokens(
    draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
):
    """
    Fix invalid draft tokens for speculative sampling.

    Why this is needed:
    - In speculative sampling, the drafter proposes tokens x ~ q(x),and correctness requires
      `q(x) > 0` for every sampled token.
    - With approximate top-k/top-p sampling (`top_k_top_p_sampling_from_logits` in FlashInfer),
      it can rarely happen that a token outside the support (`q(x) == 0` after renormalization)
      is sampled.
    - Such tokens break acceptance ratio computation and can cause incorrect behavior or NaNs.

    What this function does:
    - If the sampled draft token has `q(x) == 0`, replace it with `argmax(q)`, which is guaranteed
      to lie inside the support of the drafter distribution.
    - Otherwise, keep the original sampled token unchanged.

    This acts as a lightweight safety guard that only triggers on rare sampler failures
    and minimally perturbs the algorithm.

    Args:
        draft_tokens: Sampled draft tokens. Shape: [bsz, seqlen]
        draft_probs: Drafter probability distribution. Shape: [bsz, seqlen, vocab]

    Returns:
        Fixed draft tokens. Shape: [bsz, seqlen]
    """

    # Probability of the sampled draft token p(x)
    sampled_prob = torch.gather(
        draft_probs, dim=-1, index=draft_tokens[..., None]
    )[..., 0]  # [bsz, seqlen]

    # Detect invalid samples: sampled token not in the support (p(x) == 0)
    zero_mask = sampled_prob == 0

    # Fast path: no invalid tokens
    if not zero_mask.any():
        return draft_tokens

    # Fallback token: highest-probability token under p
    argmax_tokens = draft_probs.argmax(dim=-1)  # [bsz, seqlen]

    # Replace only invalid positions
    fixed_tokens = torch.where(zero_mask, argmax_tokens, draft_tokens)

    return fixed_tokens