import torch
import math

from typing import Optional, Tuple
from collections.abc import Callable

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr = 1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss

class AdamW(torch.optim.Optimizer):

    def __init__(self, params, lr: float, weight_decay: float, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        defaults = {"lr": lr, "weight_decay": weight_decay, "beta": betas, "eps": eps}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr, weight_decay, beta, eps = group["lr"], group["weight_decay"], group["beta"], group["eps"]
            beta1, beta2 = beta
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                g = p.grad.data
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p.grad))
                v = state.get("v", torch.zeros_like(p.grad))
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * (g ** 2)
                alpha_t = lr * math.sqrt(1 - math.pow(beta2, t)) / (1 - math.pow(beta1, t))
                p.data -= alpha_t * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p.data
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss        

if __name__ == "__main__":
    
    print(f"============= lr = {1e1} =============")
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1e1)
    for t in range(10):
        opt.zero_grad() # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean() # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward() # Run backward pass, which computes gradients.
        opt.step() # Run optimizer step.
    
    print(f"============= lr = {1e2} =============")
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1e2)
    for t in range(10):
        opt.zero_grad() # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean() # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward() # Run backward pass, which computes gradients.
        opt.step() # Run optimizer step.
    
    print(f"============= lr = {1e3} =============")
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1e3)
    for t in range(10):
        opt.zero_grad() # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean() # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward() # Run backward pass, which computes gradients.
        opt.step() # Run optimizer step.

    """
        ============= lr = 10.0 =============
        21.938270568847656
        14.04049301147461
        10.350052833557129
        8.097809791564941
        6.5592265129089355
        5.438349723815918
        4.586527347564697
        3.919318199157715
        3.384639263153076
        2.948397159576416
        ============= lr = 100.0 =============
        23.860456466674805
        23.860454559326172
        4.093806266784668
        0.09797400236129761
        1.4919488778682497e-16
        1.6628697082843524e-18
        5.599467324767595e-20
        3.3356409591402463e-21
        2.861528688857099e-22
        3.1794761105894243e-23
        ============= lr = 1000.0 =============
        20.490581512451172
        7397.10009765625
        1277595.5
        142118848.0
        11511626752.0
        726515515392.0
        37296933961728.0
        1604673467318272.0
        5.914484358198067e+16
        1.8992066381836452e+18
    """