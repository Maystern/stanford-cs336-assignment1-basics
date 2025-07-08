import torch
import math

from torch import nn
from einops import einsum
from cs336_basics.utils import SiLU, initial_torch_linear_weight, initial_torch_embedding_weight

class Linear(nn.Module):
    def __init__(self, in_features, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.weight = nn.Parameter(initial_torch_linear_weight(out_features, in_features, device, dtype), requires_grad=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.weight = nn.Parameter(initial_torch_embedding_weight(num_embeddings, embedding_dim, device, dtype), requires_grad=True)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class SwiGLU(nn.Module):
    """
        如果采用 Linear 层，通用的定式是 d_model -> 4 * d_model -> d_model 需要的 参数大小为 8 * d_model * d_model
        如果采用 SwiGLU 层，参数大小: 3 * d_model * d_ff, 其中 d_ff = 8/3 * d_model 约等于 2.67 * d_model, 可以保证总的参数大小为 8 * d_model * d_model

        矩阵乘法总 flops: 4 * d_model * d_model * d_ff + 2 * d_ff * d_ff * d_model
    """
    def __init__(self, d_model, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype) # 参数大小 d_model * d_ff
        self.w2 = Linear(d_ff, d_model, device, dtype) # 参数大小 d_model * d_ff
        self.w3 = Linear(d_model, d_ff, device, dtype) # 参数大小 d_model * d_ff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # flops: 2 * d_model * d_model * d_ff
        p = SiLU(self.w1(x))
        # flops: 2 * d_model * d_model * d_ff
        q = self.w3(x)
        # flops: 2 * d_ff * d_Ff * d_model
        return self.w2(p * q)