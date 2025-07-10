import torch
import math
from torch import nn
from einops import einsum, rearrange
from cs336_basics.utils import softmax
from cs336_basics.position_embed import RoPE
from cs336_basics.linear import Linear

def scaled_dot_product_attention(Q, K, V: torch.Tensor, mask: torch.Tensor = None):
    """
        输入变量的 shape:
            [1] Q: [batch, num_heads, seq_len, d_model // num_heads]
            [2] K: [batch, num_heads, seq_len, d_model // num_heads]
            [3] V: [batch, num_heads, seq_len, d_model // num_heads]
            [3] mask: [seq_len, seq_len]
        需要的总矩阵乘法 FLOPs 为 4 * d_model * context_length * context_length
    """
    d_k = K.shape[-1]
    # 计算 attn 分数, 需要的矩阵乘法 FLOPs 为 2 * (batch * num_heads) * seq_len * d_model // num_heads * seq_len
    # 等于 2 * d_model * context_length * context_length
    attn_scores = einsum(Q, K, "... q d_k, ... k d_k -> ... q k") # shape 为 [batch, num_heads, seq_len, seq_len]
    attn_scores /= math.sqrt(d_k)
    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask == False, float("-inf"))
    attn_weights = softmax(attn_scores, dim=-1)
    
    # 需要的 FLOPs: 2 * batch * num_heads * seq_len * seq_len * d_model // num_heads
    # 等于 2 * d_model * context_length * context_length 
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

        mask = torch.tril(torch.ones(seq_len, seq_len)).bool().to(x.device)
        
        attn = scaled_dot_product_attention(Q, K, V, mask)
        attn = rearrange(attn, "... h seq_len d_v -> ... seq_len (h d_v)", h = self.num_heads)

        return self.output_proj(attn)

class MultiheadSelfAttentionWithRoPE(nn.Module):
    """
        总的参数大小 4 * d_model * d_model
        总的矩阵乘法 flops 构成：

    """
    def __init__(self, d_model: int, num_heads: int, theta: float, max_seq_len: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.q_proj = Linear(d_model, d_model, device, dtype) # 参数大小：d_model * d_model
        self.k_proj = Linear(d_model, d_model, device, dtype) # 参数大小：d_model * d_model
        self.v_proj = Linear(d_model, d_model, device, dtype) # 参数大小：d_model * d_model
        self.output_proj = Linear(d_model, d_model, device, dtype) # 参数大小：d_model * d_model

        self.RoPE_layer = RoPE(theta, d_model // num_heads, max_seq_len, device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        """
            输入的 x 的形状为 [batch, seq_len, d_model]
            总矩阵乘法 FLOPs 的构成:
                [1] Q's proj: 2 * context_length * d_model * d_model
                [2] K's proj: 2 * context_length * d_model * d_model
                [3] V's proj: 2 * context_length * d_model * d_model
                [4] scaled_dot_product_attention: 4 * d_model * context_length * context_length
                [5] O's proj: 2 * context_length * d_model * d_model
                total: 8 * context_length * d_model * d_model + 4 * d_model * context_length * context_length
        """
        seq_len = x.shape[-2]
        
        # flops: 2 * (batch * seq_len)  * d_model * d_model = 2 * context_length * d_model * d_model
        Q = rearrange(self.q_proj(x), "... seq_len (h d_k) -> ... h seq_len d_k", h = self.num_heads) # shape: [batch, seq_len, d_model] -> [batch, num_heads, seq_len, d_model // num_heads]
        # flops: 4 * context_length * d_model 可以忽略不计
        Q = self.RoPE_layer(Q, token_positions)

        # flops: 2 * context_length * d_model * d_model
        K = rearrange(self.k_proj(x), "... seq_len (h d_k) -> ... h seq_len d_k", h = self.num_heads)
        # flops: 4 * context_length * d_model 可以忽略不计
        K = self.RoPE_layer(K, token_positions)

        # flops: 2 * context_length * d_model * d_model
        V = rearrange(self.v_proj(x), "... seq_len (h d_v) -> ... h seq_len d_v", h = self.num_heads)

        mask = torch.tril(torch.ones(seq_len, seq_len)).bool().to(x.device) # shape 为 [seq_len, seq_len]

        # flops: 4 * d_model * context_length * context_length
        attn = scaled_dot_product_attention(Q, K, V, mask)
        attn = rearrange(attn, "... h seq_len d_v -> ... seq_len (h d_v)", h = self.num_heads) # shape: [batch, num_heads, seq_len, d_model // num_heads] -> [batch, seq_len, d_model]
        
        # flops: 2 * context_length * d_model * d_model
        return self.output_proj(attn)