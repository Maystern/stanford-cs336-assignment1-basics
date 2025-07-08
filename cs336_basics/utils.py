import torch
import math

from torch import nn

def SiLU(x: torch.Tensor):
    return x * torch.sigmoid(x)

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_value, _ = x.max(dim=dim, keepdim=True)
    y = x - max_value
    exp_y = torch.exp(y)
    return exp_y / torch.sum(exp_y, dim=dim, keepdim=True)

def initial_torch_linear_weight(in_features, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
    weight = torch.empty((in_features, out_features), device=device, dtype=dtype)
    std = math.sqrt(2.0 / (in_features + out_features))
    nn.init.trunc_normal_(weight, 0, std, -3.0 * std, 3.0 * std)
    return weight

def initial_torch_embedding_weight(num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
    weight = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
    weight = nn.init.trunc_normal_(weight, 0, 1, -3.0, 3.0)
    return weight