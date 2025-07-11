import torch
import os
import typing

from torch import nn

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    checkpoint = {
        "model_weights": model.state_dict(),
        "optim_states": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpoint, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    weights = torch.load(src)
    assert ("model_weights" in weights) and ("optim_states" in weights) and ("iteration" in weights)
    model.load_state_dict(weights["model_weights"])
    optimizer.load_state_dict(weights["optim_states"])
    return weights["iteration"]