import wandb
import argparse
import torch

from tqdm import tqdm
from einops import rearrange
from cs336_basics.utils import get_module_memory_bytes, cross_entropy, gradient_lr_norm_sum_calc, get_module_param_count

from cs336_basics.data_loader import data_loading, dataset_loading
from cs336_basics.checkpoint import save_checkpoint
from cs336_basics.common import determined_optimizer, get_train_config, get_model_config, construct_model, construct_tokenizer


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='处理命令行参数示例')
    parser.add_argument('--model', '-m', help='输入模型配置文件路径')
    parser.add_argument('--train', '-t', help='输出文输入训练配置文件路径件路径')

    args = parser.parse_args()

    model_config = get_model_config(args.model)
    train_config = get_train_config(args.train)

    wandb.init(
        project='cs336_assignment1_jiacheng',
        entity='jiacheng-luo',
        name=model_config["name"] + "_" + train_config["name"]
    )
    
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
    
    for t in tqdm(range(train_config["num_epoch"]), desc="Training Model"):
        optim.zero_grad()

        # 训练
        train_batch_input, train_batch_target = data_loading(
            train_dataset,
            train_config["batch_size"],
            context_length=model_config["context_length"],
            device=model_config["device"]  # 确保是 "mps" 设备
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
            test_dataset,
            train_config["batch_size"],
            context_length=model_config["context_length"],
            device=model_config["device"]  # 确保是 "mps" 设备
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
    model_saved_path = train_config["model_saved_path"] + model_config["name"] + f"""_checkpoint_{t}_{train_config["name"]}.pt"""
    save_checkpoint(model, optim, t, model_saved_path)
    print(f"""[done] saved model [{model_config["name"]}_checkpoint_{t}_{train_config["name"]}.pt] in {train_config["model_saved_path"]} !""")