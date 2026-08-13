from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    from cs336_basics.model import Linear
    module = Linear(d_in, d_out)
    module.weight.data = weights
    return module(in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    from cs336_basics.model import Embedding
    module = Embedding(vocab_size, d_model)
    module.weight.data = weights
    return module(token_ids)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    from cs336_basics.model import SwiGLU
    module = SwiGLU(d_model, d_ff)
    module.w1.weight.data = w1_weight
    module.w2.weight.data = w2_weight
    module.w3.weight.data = w3_weight
    return module(in_features)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    from cs336_basics.model import MultiheadSelfAttention
    batch_size = Q.shape[0]
    num_heads = 1
    d_model = Q.shape[-1]
    module = MultiheadSelfAttention(d_model, num_heads)
    return module._scaled_dot_product_attention(Q, K, V, mask)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    from cs336_basics.model import MultiheadSelfAttention
    module = MultiheadSelfAttention(d_model, num_heads)
    with torch.no_grad():
        module.q_proj.weight.copy_(q_proj_weight)
        module.k_proj.weight.copy_(k_proj_weight)
        module.v_proj.weight.copy_(v_proj_weight)
        module.output_proj.weight.copy_(o_proj_weight)
    return module(in_features)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    from cs336_basics.model import MultiheadSelfAttention, RoPE
    module = MultiheadSelfAttention(d_model, num_heads)
    with torch.no_grad():
        module.q_proj.weight.copy_(q_proj_weight)
        module.k_proj.weight.copy_(k_proj_weight)
        module.v_proj.weight.copy_(v_proj_weight)
        module.output_proj.weight.copy_(o_proj_weight)
    
    rope = RoPE(d_model // num_heads, max_seq_len, theta)
    return module(in_features, rope=rope, token_positions=token_positions)


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    from cs336_basics.model import RoPE
    module = RoPE(d_k, max_seq_len, theta)
    return module(in_query_or_key, token_positions)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    from cs336_basics.model import TransformerBlock, RoPE
    module = TransformerBlock(d_model, num_heads, d_ff)
    
    with torch.no_grad():
        module.ln1.weight.copy_(weights["ln1.weight"])
        module.attn.q_proj.weight.copy_(weights["attn.q_proj.weight"])
        module.attn.k_proj.weight.copy_(weights["attn.k_proj.weight"])
        module.attn.v_proj.weight.copy_(weights["attn.v_proj.weight"])
        module.attn.output_proj.weight.copy_(weights["attn.output_proj.weight"])
        module.ln2.weight.copy_(weights["ln2.weight"])
        module.ffn.w1.weight.copy_(weights["ffn.w1.weight"])
        module.ffn.w2.weight.copy_(weights["ffn.w2.weight"])
        module.ffn.w3.weight.copy_(weights["ffn.w3.weight"])

    batch_size, seq_len, _ = in_features.shape
    token_positions = torch.arange(seq_len, device=in_features.device).repeat(batch_size, 1)
    rope = RoPE(d_model // num_heads, max_seq_len, theta)
    
    return module(in_features, rope=rope, token_positions=token_positions)


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    from cs336_basics.model import TransformerLM
    module = TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)
    
    with torch.no_grad():
        module.token_embeddings.weight.copy_(weights["token_embeddings.weight"])
        for i in range(num_layers):
            block = module.blocks[i]
            block.ln1.weight.copy_(weights[f"layers.{i}.ln1.weight"])
            block.attn.q_proj.weight.copy_(weights[f"layers.{i}.attn.q_proj.weight"])
            block.attn.k_proj.weight.copy_(weights[f"layers.{i}.attn.k_proj.weight"])
            block.attn.v_proj.weight.copy_(weights[f"layers.{i}.attn.v_proj.weight"])
            block.attn.output_proj.weight.copy_(weights[f"layers.{i}.attn.output_proj.weight"])
            block.ln2.weight.copy_(weights[f"layers.{i}.ln2.weight"])
            block.ffn.w1.weight.copy_(weights[f"layers.{i}.ffn.w1.weight"])
            block.ffn.w2.weight.copy_(weights[f"layers.{i}.ffn.w2.weight"])
            block.ffn.w3.weight.copy_(weights[f"layers.{i}.ffn.w3.weight"])
        
        module.ln_final.weight.copy_(weights["ln_final.weight"])
        module.lm_head.weight.copy_(weights["lm_head.weight"])
        
    return module(in_indices)


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    from cs336_basics.model import RMSNorm
    module = RMSNorm(d_model, eps)
    module.weight.data = weights
    return module(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    from cs336_basics.model import silu
    return silu(in_features)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    max_idx = len(dataset) - context_length - 1
    ix = torch.randint(0, max_idx, (batch_size,))
    x = torch.stack([torch.from_numpy(dataset[i : i + context_length].astype("int64")) for i in ix])
    y = torch.stack([torch.from_numpy(dataset[i + 1 : i + context_length + 1].astype("int64")) for i in ix])
    return x.to(device), y.to(device)


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    from cs336_basics.model import softmax
    return softmax(in_features, dim)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    from cs336_basics.nn_utils import cross_entropy
    return cross_entropy(inputs, targets)


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    from cs336_basics.nn_utils import clip_gradients
    clip_gradients(parameters, max_l2_norm)


def get_adamw_cls() -> Any:
    from cs336_basics.optimizer import AdamW
    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    from cs336_basics.nn_utils import get_lr_cosine_schedule
    return get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    from cs336_basics.tokenizer import Tokenizer
    return Tokenizer(vocab, merges, special_tokens)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    from cs336_basics.bpe_training import train_bpe
    return train_bpe(str(input_path), vocab_size, special_tokens)
