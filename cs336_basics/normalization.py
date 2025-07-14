import torch
from torch import nn
from einops import rearrange

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.ones((d_model), device=device, dtype=dtype), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            输入 x 的 shape 为 [batch, seq_len, d_model]
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = rearrange(torch.sqrt(torch.mean(x ** 2, dim=-1) + self.eps), "... -> ... 1") # 形状为 [batch, seq_len, d_model]
        result = (x / rms) * self.weight # [batch, seq_len, d_model] * [d_model] 需要进行广播，这里的乘法是按元素乘法，并不是矩阵乘法，计算量为 batch * seq_len * d_model (可以忽略不计)
        return result.to(in_dtype)