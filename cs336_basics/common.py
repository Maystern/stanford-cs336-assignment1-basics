import yaml
import sys
import torch
import os
import pickle

from torch import nn
from typing import Dict
from cs336_basics.optimizer import AdamW
from cs336_basics.scheduler import CostantLRScheduler, CosineAnnealingLRScheduler
from cs336_basics.transformer import Transformer, TransformerInfoCalc
from cs336_basics.train_bpe import train_bpe
from cs336_basics.tokenizer import Tokenizer

def determined_lr_schedule(train_config: Dict):
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
    return lr_schedule

def determined_optimizer(model: nn.Module, train_config: Dict):
    return AdamW(
        params=model.parameters(),
        lr = float(train_config["lr"]),
        weight_decay=float(train_config["weight_decay"]),
        betas=train_config["betas"],
        eps=float(train_config["eps"]),
        lr_schedule=determined_lr_schedule(train_config),
        gradient_clipping=train_config["gradient_clipping"],
        disabled_wandb_log=False
    )

def get_train_config(train_config_path: str):
    with open(train_config_path, "r") as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(f"训练配置解析错误 {exc}")
            sys.exit(1)

def get_model_config(model_config_path: str):
    with open(model_config_path, "r") as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(f"模型配置解析错误 {exc}")
            sys.exit(1)

def construct_model(model_config: Dict):
    device = model_config["device"]
    model = Transformer(
            vocab_size=model_config["vocab_size"],
            context_length=model_config["context_length"],
            d_model=model_config["d_model"],
            num_layers=model_config["num_layers"],
            num_heads=model_config["num_heads"],
            d_ff=model_config["d_ff"],
            rope_theta=model_config["rope_theta"],
            device=torch.device(device),
            dtype=torch.float32
        )
    if device == "cpu":
        model = torch.compile(model)
    elif device == "mps":
        model = torch.compile(model, backend="aot_eager")
    return model

def construct_tokenizer(train_config: Dict, model_config: Dict) -> Tokenizer:

    dataset_path = train_config["train_dataset_path"]
    store_path_cache_base = train_config["cache_dir"]
    dataset_name = store_path_cache_base + dataset_path.split("/")[-1].split(".")[0]
    
    
    special_tokens = train_config["special_tokens"]

    vocab_size = model_config["vocab_size"]

    vocab_path = dataset_name + "_vocab.pkl"
    merges_path = dataset_name + "_merges.pkl"


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
    
    return Tokenizer(vocab, merges, special_tokens)