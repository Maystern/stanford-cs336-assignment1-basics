import torch
import random
import numpy.typing as npt

def data_loading(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
    sampled_idxes = random.sample(range(len(dataset) - context_length), batch_size)
    batch = torch.stack([torch.tensor(dataset[i: i + context_length], dtype=torch.int32) for i in sampled_idxes], dim = 0)
    targets = torch.stack([torch.tensor(dataset[i + 1: i + context_length + 1], dtype=torch.int32) for i in sampled_idxes], dim = 0)
    return batch.to(device), targets.to(device)