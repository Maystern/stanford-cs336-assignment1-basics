import torch
from torch import nn
from einops import einsum, rearrange

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.gain_param = nn.Parameter(torch.ones((d_model), device=device, dtype=dtype), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = rearrange(torch.sqrt(torch.mean(x ** 2, dim=-1) + self.eps), "... -> ... 1")
        g = rearrange(self.gain_param, "d_model -> 1 1 d_model")
        result = (x / rms) * g
        return result.to(in_dtype)
    
if __name__ == "__main__":
    rmsNorm = RMSNorm(512, 1e-5, torch.device("cpu"), torch.float16)
    x = torch.rand((32, 100, 512), dtype=torch.float16)
    y = rmsNorm(x)
    print(y.shape)