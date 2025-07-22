import os
import wandb
import argparse
import torch
from tqdm import tqdm
from einops import rearrange
from cs336_basics.utils import get_module_memory_bytes, cross_entropy, gradient_lr_norm_sum_calc, get_module_param_count

from cs336_basics.data_loader import data_loading, dataset_loading
from cs336_basics.checkpoint import save_checkpoint, load_checkpoint
from cs336_basics.common import determined_optimizer, get_train_config, get_model_config, construct_model, construct_tokenizer


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', '-m', help='输入模型配置文件路径')
    parser.add_argument('--train', '-t', help='输出文输入训练配置文件路径件路径')

    args = parser.parse_args()

    model_config = get_model_config(args.model)
    train_config = get_train_config(args.train)

    load_local_training_rec = True

    if "run_id" in train_config["wandb"] and train_config["wandb"]["run_id"] != "":
        run = wandb.init(
            project=train_config["wandb"]["project"],
            id=train_config["wandb"]["run_id"],
            resume="must"
        )
        print(f"[run_id] load run_id from train_config, run_id = {run.id}")
    else:
        run = wandb.init(
            project=train_config["wandb"]["project"],
            entity=train_config["wandb"]["entity"],
            name=model_config["name"] + "_" + train_config["name"],
            config={
                "model_config": model_config,
                "train_config": train_config
            },
            resume="allow"
        )
        print(f"[run_id] new run_id = {run.id}")
        load_local_training_rec = False

    try:
        model = construct_model(model_config)
    except Exception as exc:
        print(f"模型加载失败 {exc}")

    print(f"实际可训练参数: {get_module_param_count(model) / (1000 * 1000):.4f} M")
    print(f"实际占用内存: {get_module_memory_bytes(model) / (1024 * 1024):.4f} MB")

    optim = determined_optimizer(
        model,
        train_config,
    )
    
    special_tokens = train_config["special_tokens"]
    bpe_tokenizer = construct_tokenizer(train_config, model_config)
    train_dataset = dataset_loading(bpe_tokenizer, train_config["train_dataset_path"], special_tokens, train_config["cache_dir"])
    test_dataset = dataset_loading(bpe_tokenizer, train_config["test_dataset_path"], special_tokens, train_config["cache_dir"])

    saved_t = 0
    saved_t_rec_path = None

    if load_local_training_rec:
        for t in range(train_config["model_saved_interval"], train_config["num_epoch"] + 1, train_config["model_saved_interval"]):
            model_saved_path = train_config["model_saved_path"] + train_config["name"] + "/" + model_config["name"] + f"""_checkpoint_{t}.pt"""
            if os.path.exists(model_saved_path):
                saved_t = t
                saved_t_rec_path = model_saved_path
            else:
                break
        
        if saved_t_rec_path is not None:
            print(f"resume model and optimizer from {saved_t_rec_path}")
            load_checkpoint(saved_t_rec_path, model, optim)
    
    for t in tqdm(range(saved_t, train_config["num_epoch"]), desc="Training Model"):
        optim.zero_grad()

        # 训练
        train_batch_input, train_batch_target = data_loading(
            dataset = train_dataset,
            batch_size = int(train_config["batch_size"]["train"]),
            context_length=int(model_config["context_length"]),
            device=str(model_config["device"])
        )
        model.train()
        train_batch_out = model(train_batch_input)
        train_batch_out = rearrange(train_batch_out, "batch seq_len vocab_size -> (batch seq_len) vocab_size")
        train_batch_target = rearrange(train_batch_target, "batch seq_len -> (batch seq_len)")
        
        # 计算损失，并计算梯度
        loss = cross_entropy(train_batch_out, train_batch_target)
        loss.backward()
        wandb.log({
            "Gradient": torch.sqrt(gradient_lr_norm_sum_calc(model.parameters()))
        }, step=t)
        
        # 优化
        optim.step()
        
        wandb.log({
            "Train Loss": loss.detach().float()
        }, step=t)

        # 验证
        valid_batch_input, valid_batch_target = data_loading(
            dataset=test_dataset,
            batch_size=int(train_config["batch_size"]["eval"]),
            context_length=int(model_config["context_length"]),
            device=str(model_config["device"])
        )
        model.eval()
        with torch.no_grad():
            valid_batch_out = model(valid_batch_input)
            valid_batch_out = rearrange(valid_batch_out, "batch seq_len vocab_size -> (batch seq_len) vocab_size")
            valid_batch_target = rearrange(valid_batch_target, "batch seq_len -> (batch seq_len)")
            loss = cross_entropy(valid_batch_out, valid_batch_target)
            wandb.log({
                "Validation Loss": loss.detach().float()
            }, step=t)
        
        if (t + 1) % int(train_config["model_saved_interval"]) == 0:
            model_saved_dir = train_config["model_saved_path"] + train_config["name"] + "/"
            model_saved_path = model_saved_dir + model_config["name"] + f"""_checkpoint_{t + 1}.pt"""
            os.makedirs(model_saved_dir, exist_ok=True)
            save_checkpoint(model, optim, t, model_saved_path)
