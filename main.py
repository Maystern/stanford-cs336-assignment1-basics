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
from cs336_basics.utils import get_module_memory_bytes, cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_basics.train_bpe import train_bpe, split_by_special_tokens
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.pretokenization import find_chunk_boundaries, find_all_regex
from cs336_basics.data_loader import data_loading
from cs336_basics.scheduler import CostantLRScheduler, CosineAnnealingLRScheduler

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
        name="cosine_annealing_lr_schedule",
    )

    model_config_path = "./config/gpt2_tiny.yaml"
    # model_config_path = "./config/gpt2_xl.yaml"
    # model_config_path = "./config/gpt2_small.yaml"
    with open(model_config_path, "r") as file:
        try:
            model_config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(f"模型配置解析错误 {exc}")

    try:
        model = Transformer(
            vocab_size=model_config["vocab_size"],
            context_length=model_config["context_length"],
            d_model=model_config["d_model"],
            num_layers=model_config["num_layers"],
            num_heads=model_config["num_heads"],
            d_ff=model_config["d_ff"],
            rope_theta=model_config["rope_theta"],
            device=torch.device(model_config["device"]),
            dtype=torch.float32
        )
    except Exception as exc:
        print(f"模型加载失败 {exc}")

    print(f"实际占用内存: {get_module_memory_bytes(model) / (1024 * 1024):.4f} MB")

    train_config_path = "./config/train.yaml"
    with open(train_config_path, "r") as file:
        try:
            train_config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(f"模型配置解析错误 {exc}")
    
    lr_schedule = CostantLRScheduler(lr=float(train_config["lr"]))
    
    try:
        lr_schedule_config = train_config["lr_schedule"]
        if lr_schedule_config["type"] == "costant":
            lr_schedule = CostantLRScheduler(float(lr_schedule_config["params"]["lr"]))
        elif lr_schedule_config["type"] == "cosine_annealing":
            params = lr_schedule_config["params"]
            lr_schedule = CosineAnnealingLRScheduler(
                alpha_max=float(params["alpha_max"]),
                alpha_min=float(params["alpha_min"]),
                Tw=int(params["Tw"]),
                Tc=int(params["Tc"])
            )
    except Exception as e:
        print(f"""学习率配置读取失败, 采用恒定学习率, lr = {float(train_config["lr"])}""")

    optim = AdamW(
        params=model.parameters(),
        lr = float(train_config["lr"]),
        weight_decay=float(train_config["weight_decay"]),
        betas=train_config["betas"],
        eps=float(train_config["eps"]),
        lr_schedule=lr_schedule,
        disabled_wandb_log=False
    )

    dataset_path = train_config["dataset_path"]
    store_path_cache_base = train_config["dataset_vocab_merges_cache_dir"]
    dataset_name = store_path_cache_base + dataset_path.split("/")[-1].split(".")[0]
    vocab_path = dataset_name + "_vocab.pkl"
    merges_path = dataset_name + "_merges.pkl"
    special_tokens = train_config["special_tokens"]
    special_tokens_utf8 = [special_token.encode("utf-8") for special_token in special_tokens]

    vocab_size = model_config["vocab_size"]

    vocab, merges = None, None
    try:
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        print("loading [vocab] from cache successfully!")
    except Exception as exc:
        pass
    try:
        with open(merges_path + "", 'rb') as f:
            merges = pickle.load(f)
        print("loading [merges] from cache successfully!")
    except Exception as exc:
        pass

    if vocab is None or merges is None or len(vocab) != vocab_size:
        print(f"train bpe tokenizer...")
        vocab, merges = train_bpe(dataset_path, vocab_size, special_tokens)
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        os.makedirs(os.path.dirname(merges_path), exist_ok=True)
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab, f)
        with open(merges_path, 'wb') as f:
            pickle.dump(merges, f)
    
    bpe_tokenizer = Tokenizer(vocab, merges, special_tokens)

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

    for t in tqdm(range(train_config["num_epoch"])):
        X, targets = data_loading(
            encode_mmap_array,
            train_config["batch_size"],
            context_length=model_config["context_length"],
            device=model_config["device"]  # 确保是 "mps" 设备
        )
        optim.zero_grad() 
        out = model(X) # [batch, seq_len, vocab_size]
        out = rearrange(out, "batch seq_len vocab_size -> (batch seq_len) vocab_size")
        targets = rearrange(targets, "batch seq_len -> (batch seq_len)")
        loss = cross_entropy(out, targets)
        wandb.log({
            "Test Loss": loss.detach().float()
        }, step=t)
        loss.backward()
        optim.step()