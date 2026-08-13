import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight.t())

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(variance + self.eps)
        x_normed = (x_f32 * inv_rms).to(orig_dtype)
        return self.weight * x_normed

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = silu(self.w1(x))
        signal = self.w3(x)
        return self.w2(gate * signal)

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_max = x_max.masked_fill(torch.isinf(x_max), 0.0)
    exp_x = torch.exp(x - x_max)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp.masked_fill(sum_exp == 0, 1.0)

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # Interleaved (RoFormer / GPT-NeoX) style rotation: pairs adjacent dims (2i, 2i+1).
    # Produces [-x1, x0, -x3, x2, ...], consistent with the interleaved cos/sin table below.
    x_interleaved = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x_interleaved.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).reshape(x.shape)

class RoPE(nn.Module):
    def __init__(self, d_k: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.theta = theta

        indices = torch.arange(0, d_k, 2).float()
        freqs = 1.0 / (theta ** (indices / d_k))
        t = torch.arange(max_seq_len).float()
        freqs_m = torch.outer(t, freqs)
        # Interleaved (RoFormer) convention: each frequency is written twice consecutively,
        # giving [f0, f0, f1, f1, ..., f_{d/2-1}, f_{d/2-1}]. This must stay consistent with
        # the interleaved rotate_half above (which pairs adjacent dims 2i, 2i+1).
        emb = torch.stack((freqs_m, freqs_m), dim=-1).reshape(max_seq_len, d_k)

        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (..., seq, d)
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        
        if x.ndim > cos.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)
            
        return (x * cos) + (rotate_half(x) * sin)

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        orig_dtype = q.dtype
        q, k, v = q.float(), k.float(), v.float()
        
        d_k = q.shape[-1]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
        
        if mask is not None:
            # Broadcast the mask to prefix the leading (batch, heads) dims of
            # attn_scores by prepending singleton dims at the front.
            while mask.ndim < attn_scores.ndim:
                mask = mask.unsqueeze(0)
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
            
        attn_weights = softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        return out.to(orig_dtype)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        rope: RoPE | None = None,
        token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Causal (lower-triangular) self-attention by default: position i may only
        # attend to positions <= i. The reference applies this mask even when the
        # caller does not pass one explicitly.
        if mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # TRANSPOSE HEAD DIM AGAIN: (batch, seq, num_heads, head_dim)
        # Try: reshape(..., num_heads, head_dim).transpose(1, 2)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        if rope is not None and token_positions is not None:
            q = rope(q, token_positions)
            k = rope(k, token_positions)
            
        out = self._scaled_dot_product_attention(q, k, v, mask)
        
        # Merge back: (batch, num_heads, seq, head_dim) -> (batch, seq, num_heads, head_dim) -> (batch, seq, d_model)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.output_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiheadSelfAttention(d_model, num_heads)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        rope: RoPE | None = None,
        token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        # 1. Attention sub-layer with Pre-norm and residual
        x = x + self.attn(self.ln1(x), mask=mask, rope=rope, token_positions=token_positions)
        
        # 2. FFN sub-layer with Pre-norm and residual
        x = x + self.ffn(self.ln2(x))
        
        return x

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()
        self.context_length = context_length
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
        
        # Share a single RoPE instance across all layers
        self.rope = RoPE(d_model // num_heads, context_length, rope_theta)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        # in_indices: (batch, seq_len)
        batch_size, seq_len = in_indices.shape
        
        # 1. Generate token_positions for RoPE
        # Shape: (batch, seq_len)
        token_positions = torch.arange(seq_len, device=in_indices.device).repeat(batch_size, 1)
        
        # 2. Embedding lookup
        x = self.token_embeddings(in_indices)
        
        # 3. Apply Transformer Blocks
        for block in self.blocks:
            x = block(x, rope=self.rope, token_positions=token_positions)
            
        # 4. Final normalization
        x = self.ln_final(x)
        
        # 5. Output projection to logits
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        in_indices: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 0.0,
    ) -> torch.Tensor:
        """
        Generate new tokens starting from a prompt.
        
        Args:
            in_indices: (batch, seq_len) prompt indices.
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature (higher = more random).
            top_p: Nucleus sampling threshold (0.0 = greedy).
            
        Returns:
            (batch, seq_len + max_new_tokens) tensor containing prompt + generated tokens.
        """
        from .nn_utils import sample_top_p
        
        for _ in range(max_new_tokens):
            # 1. Crop input if it exceeds context_length
            # The model can only handle up to context_length tokens
            curr_indices = in_indices if in_indices.size(1) <= self.context_length else in_indices[:, -self.context_length:]
            
            # 2. Forward pass to get logits for the LAST token
            # Output shape: (batch, seq_len, vocab_size)
            logits = self.forward(curr_indices)
            
            # Get logits of the last time step: (batch, vocab_size)
            logits = logits[:, -1, :]
            
            # 3. Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # 4. Sample next token
            if top_p > 0.0:
                next_token = sample_top_p(logits, top_p)
            else:
                # Greedy sampling
                next_token = torch.argmax(logits, dim=-1)
            
            # 5. Append to the sequence: (batch, seq_len + 1)
            in_indices = torch.cat((in_indices, next_token.unsqueeze(-1)), dim=1)
            
        return in_indices
