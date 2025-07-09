import torch
import math

from torch import nn
from einops import einsum, rearrange
from typing import Iterable

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

def get_module_memory_bytes(module: torch.nn.Module) -> int:
    total_bytes = 0
    for param in module.parameters():
        total_bytes += param.numel() * param.element_size()
    return total_bytes

def cross_entropy(input, target: torch.Tensor) -> torch.Tensor:
    max_value, _ = input.max(dim=-1, keepdim=True)
    div = max_value + torch.log(torch.sum(torch.exp(input - max_value), dim=-1))
    r = -(input[torch.arange(input.shape[0]), target] - div)
    return torch.mean(r)

def cosine_annealing_lr_schedule(t: int, alpha_max, alpha_min: float, Tw, Tc: int):
    if t < Tw:
        return alpha_max * t / Tw
    elif t <= Tc:
        return alpha_min + 0.5 * (1 + math.cos(math.pi * (t - Tw) / (Tc - Tw))) * (alpha_max - alpha_min)
    else: 
        return alpha_min

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6):
    l2_norm_sum = None
    for p in parameters:
        if p.grad is None:
            continue
        l2_norm = torch.sum(p.grad ** 2)
        if l2_norm_sum is None:
            l2_norm_sum = l2_norm
        else:
            l2_norm_sum += l2_norm

    if l2_norm_sum is None: return

    l2_norm_sum = torch.sqrt(l2_norm_sum)

    if l2_norm_sum < max_l2_norm:
        return
    
    scale_factor = max_l2_norm / (l2_norm_sum + eps)
    for p in parameters:
        if p.grad is None:
            continue
        p.grad.mul_(scale_factor)
