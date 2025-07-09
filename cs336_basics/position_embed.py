import torch

from torch import nn
from einops import einsum, rearrange

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        freqs = torch.outer(torch.arange(max_seq_len), 1.0 / (theta ** (torch.arange(0, d_k, 2) / d_k)))
        self.register_buffer('sin_table', torch.sin(freqs), persistent=False)
        self.register_buffer('cos_table', torch.cos(freqs), persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
            输入的 x 的 shape 为 [batch, num_heads, seq_len, d_model // num_heads]
        """
        cos_theta = self.cos_table[token_positions]
        sin_theta = self.sin_table[token_positions]

        # 二维向量旋转：https://github.com/berwin/Blog/issues/57
        rot_matrix = torch.stack([
            cos_theta, -sin_theta,
            sin_theta,  cos_theta
        ], dim=-1)
        
        rot_matrix = rearrange(rot_matrix, "... (row col) -> ... row col", row = 2, col = 2) # shape 为 [seq_len, 2, 2]
        rerange_mat = torch.stack([x[..., ::2], x[..., 1::2]], dim=-1) # shape 为 [batch, num_heads, seq_len, (d_model // num_heads) / 2, 2]
        
        # 这里的矩阵乘法是为了加入旋转位置编码, 需要计算开销是 2 * (batch * num_heads * seq_len) * ((d_model // num_heads) / 2 * 2 * 2) FLOPs
        # 等于 4 * context_length * d_model, 可以忽略不计
        rotated = einsum(rot_matrix, rerange_mat, "... dim1 dim2, ... dim2 -> ... dim1") # shape 为 [batch, num_heads, seq_len, (d_model // num_heads) / 2, 2]

        result = torch.zeros_like(x)
        result[..., ::2], result[..., 1::2] = rotated[..., 0], rotated[..., 1]
        # 恢复原来的 shape: [batch, num_heads, seq_len, d_model // num_heads]

        return result
