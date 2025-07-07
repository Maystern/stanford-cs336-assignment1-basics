import torch

def SiLU(x: torch.Tensor):
    return x * torch.sigmoid(x)

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_value, _ = x.max(dim=dim, keepdim=True)
    y = x - max_value
    exp_y = torch.exp(y)
    return exp_y / torch.sum(exp_y, dim=dim, keepdim=True)