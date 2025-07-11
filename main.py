import os
import yaml
import torch
import pickle
import multiprocessing as mp
import numpy as np
import wandb

from tqdm import tqdm
from typing import List
from einops import rearrange
from cs336_basics.transformer import Transformer
from cs336_basics.utils import get_module_memory_bytes, cross_entropy, gradient_lr_norm_sum_calc
from cs336_basics.optimizer import AdamW
from cs336_basics.train_bpe import train_bpe
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.pretokenization import find_chunk_boundaries
from cs336_basics.data_loader import data_loading
from cs336_basics.checkpoint import save_checkpoint
from cs336_basics.common import determined_optimizer, get_train_config, get_model_config, construct_model, construct_tokenizer

def init_pool(lock):
    global tqdm_lock
    tqdm_lock = lock

def tokenization(args) -> List[int]:
    chunk: str
    tokenizer: Tokenizer
    chunk, tokenizer = args
    return tokenizer.encode(chunk)

if __name__ == "__main__":

    wandb.init(
        project='cs336_assignment1_jiacheng',
        entity='jiacheng-luo',
        # name="cosine_annealing_lr_schedule",
        name="costant_lr_schedule",
    )

    model_config_path = "./config/gpt2_tiny.yaml"
    # model_config_path = "./config/gpt2_xl.yaml"
    # model_config_path = "./config/gpt2_small.yaml"
    model_config = get_model_config(model_config_path)
    try:
        model = construct_model(model_config)
    except Exception as exc:
        print(f"模型加载失败 {exc}")

    print(f"实际占用内存: {get_module_memory_bytes(model) / (1024 * 1024):.4f} MB")

    train_config_path = "./config/train.yaml"
    train_config = get_train_config(train_config_path)
    
    optim = determined_optimizer(
        model,
        train_config,
    )
    
    special_tokens = train_config["special_tokens"]
    special_tokens_utf8 = [special_token.encode("utf-8") for special_token in special_tokens]
    dataset_path = train_config["dataset_path"]

    bpe_tokenizer = construct_tokenizer(train_config, model_config)
    
    num_processes = os.cpu_count()
    print(f"tokenize the dataset with {num_processes} cpus.")
    params = []
    task_id = 0
    with open(dataset_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens_utf8)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            params.append((chunk, bpe_tokenizer))
            task_id += 1
    with mp.Manager() as manager:
        tqdm_lock = manager.Lock()
        with mp.Pool(len(params), initializer=init_pool, initargs=(tqdm_lock,)) as pool:
            results = pool.map(tokenization, params)

    encode_list = []
    for result in results:
        encode_list.extend(result)
    
    encode_mmap_array = np.memmap(
        "encoded_tokens.dat", 
        dtype=np.int32, 
        mode='w+', 
        shape=(len(encode_list),)
    )
    encode_mmap_array[:] = encode_list
    
    del encode_list
    
    model_saved_path = train_config["model_saved_path"] + model_config["name"]
    for t in tqdm(range(train_config["num_epoch"])):
        X, targets = data_loading(
            encode_mmap_array,
            train_config["batch_size"],
            context_length=model_config["context_length"],
            device=model_config["device"]  # 确保是 "mps" 设备
        )
        wandb.log({
            "Gradient": gradient_lr_norm_sum_calc(model.parameters())
        }, step=t)
        optim.zero_grad() 
        out = model(X) # [batch, seq_len, vocab_size]
        out = rearrange(out, "batch seq_len vocab_size -> (batch seq_len) vocab_size")
        targets = rearrange(targets, "batch seq_len -> (batch seq_len)")
        loss = cross_entropy(out, targets)
        wandb.log({
            "Loss": loss.detach().float()
        }, step=t)
        loss.backward()
        optim.step()
    
    save_checkpoint(model, optim, t, f"{model_saved_path}_checkpoint_{t}.pt")