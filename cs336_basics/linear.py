import torch
import math

from torch import nn
from einops import einsum
from cs336_basics.utils import SiLU

def initial_torch_linear_weight(in_features, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
    weight = torch.empty((in_features, out_features), device=device, dtype=dtype)
    std = math.sqrt(2.0 / (in_features + out_features))
    nn.init.trunc_normal_(weight, 0, std, -3.0 * std, 3.0 * std)
    return weight

def initial_torch_embedding_weight(num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
    weight = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
    weight = nn.init.trunc_normal_(weight, 0, 1, -3.0, 3.0)
    return weight

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
        self.w1_weight = nn.Parameter(initial_torch_linear_weight(d_ff, d_model, device, dtype), requires_grad=True)
        self.w2_weight = nn.Parameter(initial_torch_linear_weight(d_model, d_ff, device, dtype), requires_grad=True)
        self.w3_weight = nn.Parameter(initial_torch_linear_weight(d_ff, d_model, device, dtype), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = SiLU(einsum(self.w1_weight, x, "d_ff d_model, ... d_model -> ... d_ff"))
        q = einsum(self.w3_weight, x, "d_ff d_model, ... d_model -> ... d_ff")
        return einsum(self.w2_weight, p * q, "d_model d_ff, ... d_ff -> ... d_model")