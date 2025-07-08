import torch
import math
from torch import nn
from einops import einsum, rearrange
from cs336_basics.utils import softmax, initial_torch_linear_weight
from cs336_basics.position_embed import RoPE
from cs336_basics.linear import Linear

def scaled_dot_product_attention(Q, K, V: torch.Tensor, mask: torch.Tensor = None):
    d_k = K.shape[-1]
    attn_scores = einsum(Q, K, "... q d_k, ... k d_k -> ... q k")
    attn_scores /= math.sqrt(d_k)
    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask == False, float("-inf"))
    attn_weights = softmax(attn_scores, dim=-1)
    return einsum(attn_weights, V, "... q k, ... k d_v -> ... q d_v")

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device: torch.device | None = None, dtype: torch.dtype | None = None):

        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        seq_len = x.shape[-2]

        Q = rearrange(self.q_proj(x), "... seq_len (h d_k) -> ... h seq_len d_k", h = self.num_heads)
        K = rearrange(self.k_proj(x), "... seq_len (h d_k) -> ... h seq_len d_k", h = self.num_heads)
        V = rearrange(self.v_proj(x), "... seq_len (h d_v) -> ... h seq_len d_v", h = self.num_heads)

        mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
        
        attn = scaled_dot_product_attention(Q, K, V, mask)
        attn = rearrange(attn, "... h seq_len d_v -> ... seq_len (h d_v)", h = self.num_heads)

        return self.output_proj(attn)

class MultiheadSelfAttentionWithRoPE(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float, max_seq_len: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

        self.RoPE_layer = RoPE(theta, d_model // num_heads, max_seq_len, device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        seq_len = x.shape[-2]
        
        Q = rearrange(self.q_proj(x), "... seq_len (h d_k) -> ... h seq_len d_k", h = self.num_heads)
        Q = self.RoPE_layer(Q, token_positions)

        K = rearrange(self.k_proj(x), "... seq_len (h d_k) -> ... h seq_len d_k", h = self.num_heads)
        K = self.RoPE_layer(K, token_positions)

        V = rearrange(self.v_proj(x), "... seq_len (h d_v) -> ... h seq_len d_v", h = self.num_heads)

        mask = torch.tril(torch.ones(seq_len, seq_len)).bool()

        attn = scaled_dot_product_attention(Q, K, V, mask)
        attn = rearrange(attn, "... h seq_len d_v -> ... seq_len (h d_v)", h = self.num_heads)
        
        return self.output_proj(attn)