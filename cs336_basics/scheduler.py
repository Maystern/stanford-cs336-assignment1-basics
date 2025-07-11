import math

class CosineAnnealingLRScheduler:
    def __init__(self, alpha_max, alpha_min: float, Tw, Tc: int):
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.Tw = Tw
        self.Tc = Tc
    def __call__(self, t: int) -> float:
        if t < self.Tw:
            return self.alpha_max * t / self.Tw
        elif t <= self.Tc:
            return self.alpha_min + 0.5 * (1 + math.cos(math.pi * (t - self.Tw) / (self.Tc - self.Tw))) * (self.alpha_max - self.alpha_min)
        else: 
            return self.alpha_min

class CostantLRScheduler:
    def __init__(self, lr: float):
        self.lr = lr
    def __call__(self, t: int) -> float:
        return self.lr