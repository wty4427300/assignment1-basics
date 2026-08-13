from __future__ import annotations
import torch
from collections.abc import Iterable

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the average cross-entropy loss.
    
    Args:
        logits: (N, C) where C is the number of classes.
        targets: (N,) containing the index of the correct class.
        
    Returns:
        The average cross-entropy loss across examples.
    """
    # 1. Compute Log-Sum-Exp for numerical stability
    # m = max(z)
    # LSE = m + log(sum(exp(z - m)))
    m, _ = torch.max(logits, dim=-1, keepdim=True)
    lse = m + torch.log(torch.sum(torch.exp(logits - m), dim=-1, keepdim=True))
    
    # 2. Extract logits for the correct classes
    # Use advanced indexing to get logits[i, targets[i]]
    # targets shape is (N,), we need to match the batch dimension
    batch_indices = torch.arange(logits.size(0), device=logits.device)
    correct_logits = logits[batch_indices, targets].unsqueeze(-1)
    
    # 3. Loss = LSE - correct_logits
    loss = lse - correct_logits
    
    # 4. Return the mean loss across the batch
    return loss.mean()

def clip_gradients(parameters: torch.nn.Parameter | Iterable[torch.nn.Parameter], max_norm: float):
    """
    Clips the gradients of an iterable of parameters based on their total L2 norm.
    The gradients are modified in-place.
    
    Args:
        parameters: An iterable of Tensors or a single Tensor that will have gradients normalized.
        max_norm: Max L2 norm of the gradients.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    
    # 1. Collect all non-None gradients and compute their square sum in float32
    total_norm_sq = 0.0
    grads = []
    for p in parameters:
        if p.grad is not None:
            grad = p.grad.detach()
            grads.append(grad)
            # Use float32 for summation to avoid precision issues
            total_norm_sq += grad.to(torch.float32).pow(2).sum().item()
            
    total_norm = total_norm_sq ** 0.5
    
    # 2. Compute clipping coefficient
    # If total_norm > max_norm, clip_coeff = max_norm / total_norm
    if total_norm > max_norm:
        clip_coeff = max_norm / (total_norm + 1e-6)
        # 3. Apply the scaling factor to each gradient in-place
        for g in grads:
            g.mul_(clip_coeff)

import math

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    # 1. Linear warmup
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    
    # 2. After cycle, return min_lr
    if it > cosine_cycle_iters:
        return min_learning_rate
    
    # 3. Cosine decay
    decay_ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)

def sample_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Samples from the top-p (nucleus) distribution.
    
    Args:
        logits: (batch_size, vocab_size) tensor of logits.
        p: Float between 0 and 1.
        
    Returns:
        (batch_size,) tensor of sampled token indices.
    """
    # 1. Convert logits to probabilities
    # We use the softmax from model.py to ensure numerical stability
    from .model import softmax
    probs = softmax(logits, dim=-1)
    
    # 2. Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    
    # 3. Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 4. Remove tokens with cumulative probability above p
    # Shift cumulative_probs to the right by 1 to ensure we keep the first token that exceeds p
    # Example: if sorted_probs = [0.4, 0.3, 0.2, 0.1] and p = 0.5
    # cumulative_probs = [0.4, 0.7, 0.9, 1.0]
    # mask = [False, True, True, True] (if we used cumulative_probs > p)
    # But we want to keep [0.4, 0.3] because 0.4 < 0.5 and 0.4 + 0.3 > 0.5
    # So we want mask = [False, False, True, True]
    sorted_indices_to_remove = cumulative_probs > p
    # Shift mask to the right to keep the first token that crossed the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    
    # 5. Set probabilities of removed tokens to 0
    sorted_probs[sorted_indices_to_remove] = 0.0
    
    # 6. Re-normalize
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    
    # 7. Sample from the modified distribution
    # Multinoulli sampling
    next_token_idx_in_sorted = torch.multinomial(sorted_probs, num_samples=1)
    
    # 8. Map back to original indices
    next_token_idx = torch.gather(sorted_indices, -1, next_token_idx_in_sorted)
    
    return next_token_idx.squeeze(-1)
