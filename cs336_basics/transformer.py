import torch

from torch import nn
from typing import List
from einops import rearrange
from cs336_basics.normalization import RMSNorm
from cs336_basics.attention import MultiheadSelfAttentionWithRoPE
from cs336_basics.linear import SwiGLU, Embedding, Linear
from cs336_basics.utils import get_module_memory_bytes, softmax

class TransformerBlock(nn.Module):
    """
        总参数量: 4 * d_model * d_model + 3 * d_model * d_ff + 2 * d_model

        总计算量: 
            [1] attn: 8 * context_length * d_model * d_model + 4 * d_model * context_length * context_length
            [2] ffn: 6 * context_length * d_model * d_ff
            total: 8 * context_length * d_model * d_model + 4 * d_model * context_length * context_length + 6 * context_length * d_model * d_ff
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        
        # flops: batch * seq_len * d_model (非矩阵乘法，可以忽略不计)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype) # 参数大小 d_model

        # flops = 8 * context_length * d_model * d_model + 4 * d_model * context_length * context_length
        self.attn = MultiheadSelfAttentionWithRoPE(d_model, num_heads, theta, max_seq_len, device, dtype) # 参数大小 4 * d_model * d_model

        # flops: batch * seq_len * d_model (非矩阵乘法，可以忽略不计)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype) # 参数大小 d_model
        
        # flops = 6 * context_length * d_model * d_ff
        self.ffn = SwiGLU(d_model, d_ff, device, dtype) # 参数大小 3 * d_model * d_ff
    
    def forward(self, x: torch.Tensor):
        """
            输入 x 的 shape 为 [batch, seq_len, d_model]
        """
        x = x + self.attn(self.ln1(x), torch.arange(x.shape[-2]).to(x.device))
        x = x + self.ffn(self.ln2(x))
        return x

class Transformer(nn.Module):
    """
        总参数量 2 * self.vocab_size * self.d_model + 4 * self.num_layers * self.d_model * self.d_model + 3 * self.num_layers * self.d_model * self.d_ff + (2 * self.num_layers + 1) * self.d_model

        总的 forward 矩阵乘法 FLOPs:
            [1] layers: num_layers * (
                8 * context_length * d_model * d_model + 4 * d_model * context_length * context_length +
                6 * context_length * d_model * d_ff
            )
            [2] lm_head: 2 * context_length * d_model * vocab_size
        total: num_layers * (8 * context_length * d_model * d_model + 4 * d_model * context_length * context_length + 6 * context_length * d_model * d_ff) + 2 * context_length * d_model * vocab_size
    """
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype) # 参数大小 vocab_size * d_model
        self.layers = nn.Sequential( # 参数大小 num_layers * (4 * d_model * d_model + 3 * d_model * d_ff + 2 * d_model)
            *[TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device, dtype) for _ in range(num_layers)]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype) # 参数大小 d_model
        self.lm_head = Linear(d_model, vocab_size, device, dtype) # 参数大小 d_model * vocab_size
    
    def forward(self, x: torch.Tensor):
        """
            输入 x 的 shape: [batch, seq_len], 总的 token 数 batch * seq_len = context_length
        """
        x = self.token_embeddings(x) # flops: 0 不涉及 matrix multiplication, [batch, seq_len] -> [batch, seq_len, d_model]
        x = self.layers(x) # flops: num_layers * (8 * context_length * d_model * d_model + 4 * d_model * context_length * context_length + 6 * context_length * d_model * d_ff)
        x = self.ln_final(x) # flops: 0 
        x = self.lm_head(x) # flops: 2 * context_length * d_model * vocab_size
        return x
    
    def param_count(self) -> int:
        return 2 * self.vocab_size * self.d_model + 4 * self.num_layers * self.d_model * self.d_model + 3 * self.num_layers * self.d_model * self.d_ff + (2 * self.num_layers + 1) * self.d_model
    
    def mul_flops(self) -> int:
        attn_flops = self.num_layers * 8 * self.context_length * self.d_model * self.d_model + 4 * self.d_model * self.context_length * self.context_length
        ffn_flops = self.num_layers * (6 * self.context_length * self.d_model * self.d_ff)
        lm_head_flops = 2 * self.context_length * self.d_model * self.vocab_size

        attn_flops /= (1000 * 1000 * 1000)
        ffn_flops /= (1000 * 1000 * 1000)
        lm_head_flops /= (1000 * 1000 * 1000)
        print(f"attn_flops: {attn_flops:.4f} GFLOPs\t ffn_flops: {ffn_flops:.4f} GFLOPs\t lm_head_flops: {lm_head_flops:.4f} GFLOPs")
        return attn_flops + ffn_flops + lm_head_flops

class TransformerInfoCalc(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.context_length = context_length
    
    def forward(self, x: torch.Tensor):
        raise RuntimeError("TransformerInfoCalc 类只做参数计算, 不能调用 forward 方法")
    
    def param_count(self) -> int:
        embedding_params = 1 * self.vocab_size * self.d_model
        non_embedding_params = 4 * self.num_layers * self.d_model * self.d_model + 3 * self.num_layers * self.d_model * self.d_ff + (2 * self.num_layers + 1) * self.d_model + self.vocab_size * self.d_model
        return embedding_params + non_embedding_params
    
    def mul_flops(self) -> int:
        attn_flops = self.num_layers * 8 * self.context_length * self.d_model * self.d_model + 4 * self.d_model * self.context_length * self.context_length
        ffn_flops = self.num_layers * (6 * self.context_length * self.d_model * self.d_ff)
        lm_head_flops = 2 * self.context_length * self.d_model * self.vocab_size

        attn_flops /= (1000 * 1000 * 1000)
        ffn_flops /= (1000 * 1000 * 1000)
        lm_head_flops /= (1000 * 1000 * 1000)
        total_flops = attn_flops + ffn_flops + lm_head_flops
        print(f"attn_flops: {attn_flops:.4f} GFLOPs ({attn_flops / total_flops * 100.0:.2f}%)\t ffn_flops: {ffn_flops:.4f} GFLOPs({ffn_flops / total_flops * 100.0:.2f}%)\t lm_head_flops: {lm_head_flops:.4f} GFLOPs ({lm_head_flops / total_flops * 100.0:.2f}%)")
        return total_flops


def decoding(model: Transformer, x: List[int], max_token_num: int, encoded_special_token_list: List[int], temperature: float, p: float, device: torch.device = torch.device("cpu")) -> List[int]:
    """
        x: [seq_len] -> [token_num]
    """
    model = model.to(device)
    results = []
    encoded_special_token_set = set(encoded_special_token_list)
    while len(results) < max_token_num and len(x) + len(results) <= model.context_length:
        input = rearrange(torch.tensor(x + results, device=device), "seq_len -> 1 seq_len")
        probs = rearrange(model(input)[:,-1,:], "1 seq_len -> seq_len")
        probs = softmax(probs / temperature, dim=-1)
        sorted_indices = torch.argsort(probs, descending=True)
        sorted_probs = probs[sorted_indices]
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff = torch.searchsorted(cumulative_probs, p, side="right") + 1
        sample_indices = sorted_indices[: cutoff]
        sample_probs = sorted_probs[: cutoff]
        sample_probs = sample_probs / torch.sum(sample_probs)
        sampled_index = torch.multinomial(sample_probs, num_samples=1).item()
        next_token = sample_indices[sampled_index].item()
        results.append(next_token)
        if next_token in encoded_special_token_set:
            break
    return results

if __name__ == "__main__":
    """
        (a) trainable parameters: 2_127_057_600, if each parameter is torch.float32, GPT-2 XL model needs 7.92 GB memory
        (b) matrix multiplies FLOPs occur on:
            [1] attn: num_layers * (8 * context_length * d_model * d_model + 4 * d_model * context_length * context_length)
            [2] ffn: num_layers * (6 * context_length * d_model * d_ff)
            [3] lm_head: 2 * context_length * d_model * vocab_size
            In GPT-2 XL model, total multiplies FLOPs is 4197.9249 GFLOPs, and the breakdown is
                [1] attn_flops: 1013.3438 GFLOPs    [2] ffn_flops: 3019.8989 GFLOPs    [3] lm_head_flops: 164.6821 GFLOPs
        (c) ffn (or SwiGLU)
        (d) ========== GPT-2 small ==========
            模型参数个数: 162148608
            手动计算内存 618.5479 MB
            attn_flops: 61.2033 GFLOPs (23.89%)      ffn_flops: 115.9641 GFLOPs(45.26%)      lm_head_flops: 79.0474 GFLOPs (30.85%)
            模型推理总乘法FLOPs: 256.2148 GFLOPs
            ========== GPT-2 medium ==========
            模型参数个数: 404917248
            手动计算内存 1544.6367 MB
            attn_flops: 210.4534 GFLOPs (28.91%)     ffn_flops: 412.2162 GFLOPs(56.62%)      lm_head_flops: 105.3966 GFLOPs (14.48%)
            模型推理总乘法FLOPs: 728.0662 GFLOPs
            ========== GPT-2 large ==========
            模型参数个数: 836494080
            手动计算内存 3190.9717 MB
            attn_flops: 488.5525 GFLOPs (30.79%)     ffn_flops: 966.2733 GFLOPs(60.90%)      lm_head_flops: 131.7457 GFLOPs (8.30%)
            模型推理总乘法FLOPs: 1586.5715 GFLOPs

            As the model size increases,
               [1] the proportion of attn and ffn's FLOPs increases
               [2] the proportion of lm_head_flops's FLOPs decreases
        (e) 
            ========== GPT-2 XL ==========
            模型参数个数: 2127057600
            手动计算内存 8114.0808 MB
            实际占用内存: 0.0000 MB
            attn_flops: 1013.3438 GFLOPs (24.14%)    ffn_flops: 3019.8989 GFLOPs(71.94%)     lm_head_flops: 164.6821 GFLOPs (3.92%)
            模型推理总乘法FLOPs: 4197.9249 GFLOPs
            ========== GPT-2 XL (context_length = 16,384) ==========
            模型参数个数: 2127057600
            手动计算内存 8114.0808 MB
            实际占用内存: 0.0000 MB
            attn_flops: 17824.1143 GFLOPs (25.92%)   ffn_flops: 48318.3821 GFLOPs(70.25%)    lm_head_flops: 2634.9142 GFLOPs (3.83%)
            模型推理总乘法FLOPs: 68777.4106 GFLOPs

            total FLOPs change: 4197.9249 GFLOPs -> 68777.4106 GFLOPs
            contribution of FLOPs of the model components change:
                [1] attn_flops: 24.14% -> 25.92%
                [2] ffn_flops: 71.94% -> 70.25%
                [3] lm_head_flops: -> 3.83%
    """
    # GPT-2 XL
    print(f"========== GPT-2 XL ==========")
    vocab_size  = 50_257
    context_length = 1_024
    num_layers =  48 #
    d_model = 1_600 #
    num_heads = 25 #
    d_ff = 6_400 #
    rope_theta = 100_000
    transformer = TransformerInfoCalc(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, torch.device("cpu"), torch.float32)
    print(f"模型参数个数: {transformer.param_count()}") # 2_127_057_600 个参数
    print(f"手动计算内存 {transformer.param_count() * 4 / (1024 * 1024):.4f} MB")
    print(f"实际占用内存: {get_module_memory_bytes(transformer) / (1024 * 1024):.4f} MB")
    print(f"模型推理总乘法FLOPs: {transformer.mul_flops():.4f} GFLOPs")

    # GPT-2 small
    print(f"========== GPT-2 small ==========")
    num_layers = 12
    d_model = 768
    num_heads = 12
    d_ff = 8 * d_model // 3
    transformer = TransformerInfoCalc(vocab_size, context_length,d_model, num_layers, num_heads, d_ff, rope_theta, torch.device("cpu"), torch.float32)
    print(f"模型参数个数: {transformer.param_count()}") # 162_148_608 个参数
    print(f"手动计算内存 {transformer.param_count() * 4 / (1024 * 1024):.4f} MB")
    print(f"模型推理总乘法FLOPs: {transformer.mul_flops():.4f} GFLOPs")

    # GPT-2 medium
    print(f"========== GPT-2 medium ==========")
    num_layers = 24
    d_model = 1_024
    num_heads = 16
    d_ff = 8 * d_model // 3
    transformer = TransformerInfoCalc(vocab_size, context_length,d_model, num_layers, num_heads, d_ff, rope_theta, torch.device("cpu"), torch.float32)
    print(f"模型参数个数: {transformer.param_count()}") # 404_917_248 个参数
    print(f"手动计算内存 {transformer.param_count() * 4 / (1024 * 1024):.4f} MB")
    print(f"模型推理总乘法FLOPs: {transformer.mul_flops():.4f} GFLOPs")

    # GPT-2 large
    print(f"========== GPT-2 large ==========")
    num_layers = 36
    d_model = 1_280
    num_heads = 20
    d_ff = 8 * d_model // 3
    transformer = TransformerInfoCalc(vocab_size, context_length,d_model, num_layers, num_heads, d_ff, rope_theta, torch.device("cpu"), torch.float32)
    print(f"模型参数个数: {transformer.param_count()}") # 836_494_080 个参数
    print(f"手动计算内存 {transformer.param_count() * 4 / (1024 * 1024):.4f} MB")
    print(f"模型推理总乘法FLOPs: {transformer.mul_flops():.4f} GFLOPs")

    # GPT-2 XL
    print(f"========== GPT-2 XL (context_length = 16,384) ==========")
    vocab_size  = 50_257
    context_length = 16_384
    num_layers =  48 #
    d_model = 1_600 #
    num_heads = 25 #
    d_ff = 6_400 #
    rope_theta = 100_000
    transformer = TransformerInfoCalc(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, torch.device("cpu"), torch.float32)
    print(f"模型参数个数: {transformer.param_count()}") # 2_127_057_600 个参数
    print(f"手动计算内存 {transformer.param_count() * 4 / (1024 * 1024):.4f} MB")
    print(f"实际占用内存: {get_module_memory_bytes(transformer) / (1024 * 1024):.4f} MB")
    print(f"模型推理总乘法FLOPs: {transformer.mul_flops():.4f} GFLOPs")