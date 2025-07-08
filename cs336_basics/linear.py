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
    def __init__(self, d_model, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = SiLU(self.w1(x))
        q = self.w3(x)
        return self.w2(p * q)