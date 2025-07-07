import torch
import math

from torch import nn
from einops import einsum

class Linear(nn.Module):
    def __init__(self, in_features, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        weight = torch.empty((out_features, in_features), device=device, dtype=dtype)
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(weight, 0, std, -3.0 * std, 3.0 * std)
        self.weight = nn.Parameter(weight, requires_grad=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        weight = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        weight = nn.init.trunc_normal_(weight, 0, 1, -3.0, 3.0)
        self.weight = nn.Parameter(weight, requires_grad=True)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
