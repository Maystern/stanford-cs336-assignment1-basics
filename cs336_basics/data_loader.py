import os
import torch
import random
import numpy as np
import numpy.typing as npt
import multiprocessing as mp

from typing import List
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.pretokenization import find_chunk_boundaries

def init_pool(lock):
    global tqdm_lock
    tqdm_lock = lock

def tokenization(args) -> List[int]:
    chunk: str
    tokenizer: Tokenizer
    chunk, tokenizer = args
    return tokenizer.encode(chunk)

def dataset_loading(tokenizer: Tokenizer, dataset_path: str, special_tokens: List[str], cache_path: str) -> npt.NDArray:
    dataset_name = dataset_path.split("/")[-1].split(".")[0]
    data_file_name = cache_path +  dataset_name + "_encoded_tokens.dat"
    try:
        mmap_array = np.memmap(
            data_file_name,
            dtype=np.int32,
            mode='r',
        )
        print(f"""loading [{dataset_name + "_encoded_tokens.dat"}] from cache successfully!""")
        return mmap_array
    except Exception as e:
        print(f"""在 cache 中没有 [{dataset_name + "_encoded_tokens.dat"}] 文件, 需要重新映射""")
        pass

    num_processes = os.cpu_count()
    print(f"tokenize the dataset with {num_processes} cpus.")
    params = []
    task_id = 0
    special_tokens_utf8 = [special_token.encode("utf-8") for special_token in special_tokens]
    with open(dataset_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens_utf8)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            params.append((chunk, tokenizer))
            task_id += 1
    with mp.Manager() as manager:
        tqdm_lock = manager.Lock()
        with mp.Pool(len(params), initializer=init_pool, initargs=(tqdm_lock,)) as pool:
            results = pool.map(tokenization, params)

    encode_list = []
    for result in results:
        encode_list.extend(result)

    encode_mmap_array = np.memmap(
        data_file_name, 
        dtype=np.int32, 
        mode='w+', 
        shape=(len(encode_list),)
    )
    encode_mmap_array[:] = encode_list
    del encode_list

    return encode_mmap_array

def data_loading(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
    sampled_idxes = random.sample(range(len(dataset) - context_length), batch_size)
    batch = torch.stack([torch.tensor(dataset[i: i + context_length], dtype=torch.int32) for i in sampled_idxes], dim = 0)
    targets = torch.stack([torch.tensor(dataset[i + 1: i + context_length + 1], dtype=torch.int32) for i in sampled_idxes], dim = 0)
    return batch.to(device), targets.to(device)