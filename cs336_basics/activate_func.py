import torch

def SiLU(x: torch.Tensor):
    return x * torch.sigmoid(x)