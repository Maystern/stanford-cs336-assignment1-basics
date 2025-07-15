import torch

from cs336_basics.checkpoint import load_checkpoint
from cs336_basics.transformer import decoding
from cs336_basics.common import determined_optimizer, get_train_config, get_model_config, construct_model, construct_tokenizer

if __name__ == "__main__":
    model_config_path = "./config/gpt2_tiny.yaml"
    train_config_path = "./config/train.yaml"
    train_config = get_train_config(train_config_path)
    model_config = get_model_config(model_config_path)
    
    try:
        model = construct_model(model_config)
    except Exception as exc:
        print(f"模型加载失败 {exc}")

    optim = determined_optimizer(model, train_config)

    load_checkpoint("/Users/bytedance/code/stanford-cs336-assignment1-basics/models/gpt2_tiny_checkpoint_9_more_total_tokens.pt", model, optim)
    
    text = "A bug in red boots crawled into my pencil case. When I opened it, there was a tiny letter: 'Help, the erasers are"

    bpe_tokenizer = construct_tokenizer(train_config, model_config)
    encoded_list = bpe_tokenizer.encode(text)
    
    special_tokens = train_config["special_tokens"]
    encoded_special_tokens = []
    for special_token in special_tokens:
        encoded_special_token = bpe_tokenizer.encode(special_token)
        encoded_special_tokens.extend(encoded_special_token)

    print(f"encoded_list = {encoded_list}")
    print(f"encoded_special_tokens = {encoded_special_tokens}")

    decoded_result = decoding(model, encoded_list, 100, encoded_special_tokens, 0.8, 0.9, torch.device(model_config["device"]))
    print(decoded_result)
    print(f"生产的 token 总数为 {len(decoded_result)} 总文本 token 为 {len(encoded_list) + len(decoded_result)}")

    result = bpe_tokenizer.decode(decoded_result)
    
    print(text + result)