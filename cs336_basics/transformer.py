import torch

from torch import nn

from cs336_basics.normalization import RMSNorm
from cs336_basics.attention import MultiheadSelfAttentionWithRoPE
from cs336_basics.linear import SwiGLU, Embedding, Linear
from cs336_basics.utils import get_module_memory_bytes

class TransformerBlock(nn.Module):
    """
        总参数量: 4 * d_model * d_model + 3 * d_model * d_ff + 2 * d_model

        总计算量: 6 * d_model * d_model * d_model + 4 * context_length * context_length * d_model + 2 * context_length * d_model * d_model
                + 4 * d_model * d_model * d_ff + 2 * d_ff * d_ff * d_model
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype) # 参数大小 d_model

        # flops = 6 * d_model * d_model * d_model + 4 * context_length * context_length * d_model + 2 * context_length * d_model * d_model
        self.attn = MultiheadSelfAttentionWithRoPE(d_model, num_heads, theta, max_seq_len, device, dtype) # 参数大小 4 * d_model * d_model

        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype) # 参数大小 d_model
        
        # flops = 4 * d_model * d_model * d_ff + 2 * d_ff * d_ff * d_model
        self.ffn = SwiGLU(d_model, d_ff, device, dtype) # 参数大小 3 * d_model * d_ff
    
    def forward(self, x: torch.Tensor):
        x += self.attn(self.ln1(x), torch.arange(x.shape[-2]))
        x += self.ffn(self.ln2(x))
        return x

class Transformer(nn.Module):
    """
        总参数量 2 * self.vocab_size * self.d_model + 4 * self.num_layers * self.d_model * self.d_model + 3 * self.num_layers * self.d_model * self.d_ff + (2 * self.num_layers + 1) * self.d_model
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
        x = self.token_embeddings(x) # flops: 0
        x = self.layers(x) # flops: num_layers * (6 * d_model * d_model * d_model + 4 * context_length * context_length * d_model + 2 * context_length * d_model * d_model + 4 * d_model * d_model * d_ff + 2 * d_ff * d_ff * d_model)
        x = self.ln_final(x) 
        x = self.lm_head(x) # flops: 2 * d_model * d_model * vocab_size
        return x
    
    def param_count(self) -> int:
        return 2 * self.vocab_size * self.d_model + 4 * self.num_layers * self.d_model * self.d_model + 3 * self.num_layers * self.d_model * self.d_ff + (2 * self.num_layers + 1) * self.d_model
    
    def mul_flops(self) -> int:
        attn_flops = self.num_layers * (6 * self.d_model * self.d_model * self.d_model + 4 * self.context_length * self.context_length * self.d_model + 2 * self.context_length * self.d_model * self.d_model)
        ffn_flops = self.num_layers * (4 * self.d_model * self.d_model * self.d_ff + 2 * self.d_ff * self.d_ff * self.d_model)
        lm_head_flops = 2 * self.d_model * self.d_model * self.vocab_size
        print(f"attn_flops: {attn_flops}\t ffn_flops: {ffn_flops}\t lm_head_flops:{lm_head_flops}")
        return attn_flops + ffn_flops + lm_head_flops

class TransformerInfoCalc(nn.Module):
    """
        总参数量 2 * self.vocab_size * self.d_model + 4 * self.num_layers * self.d_model * self.d_model + 3 * self.num_layers * self.d_model * self.d_ff + (2 * self.num_layers + 1) * self.d_model
    """
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.context_length = context_length
        """
            self.token_embeddings = Embedding(vocab_size, d_model, device, dtype) # 参数大小 vocab_size * d_model
            self.layers = nn.Sequential( # 参数大小 num_layers * (4 * d_model * d_model + 3 * d_model * d_ff + 2 * d_model)
                *[TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device, dtype) for _ in range(num_layers)]
            )
            self.ln_final = RMSNorm(d_model, device=device, dtype=dtype) # 参数大小 d_model
            self.lm_head = Linear(d_model, vocab_size, device, dtype) # 参数大小 d_model * vocab_size
        """
    
    def forward(self, x: torch.Tensor):
        """
            x = self.token_embeddings(x) # flops: 0
            x = self.layers(x) # flops: num_layers * (6 * d_model * d_model * d_model + 4 * context_length * context_length * d_model + 2 * context_length * d_model * d_model + 4 * d_model * d_model * d_ff + 2 * d_ff * d_ff * d_model)
            x = self.ln_final(x) 
            x = self.lm_head(x) # flops: 2 * d_model * d_model * vocab_size
        """
        return x
    
    def param_count(self) -> int:
        return 2 * self.vocab_size * self.d_model + 4 * self.num_layers * self.d_model * self.d_model + 3 * self.num_layers * self.d_model * self.d_ff + (2 * self.num_layers + 1) * self.d_model
    
    def mul_flops(self) -> int:
        attn_flops = self.num_layers * (6 * self.d_model * self.d_model * self.d_model + 4 * self.context_length * self.context_length * self.d_model + 2 * self.context_length * self.d_model * self.d_model)
        ffn_flops = self.num_layers * (4 * self.d_model * self.d_model * self.d_ff + 2 * self.d_ff * self.d_ff * self.d_model)
        lm_head_flops = 2 * self.d_model * self.d_model * self.vocab_size

        attn_flops /= (1000 * 1000 * 1000)
        ffn_flops /= (1000 * 1000 * 1000)
        lm_head_flops /= (1000 * 1000 * 1000)
        print(f"attn_flops: {attn_flops:.4f} GFLOPs\t ffn_flops: {ffn_flops:.4f} GFLOPs\t lm_head_flops: {lm_head_flops:.4f} GFLOPs")
        return attn_flops + ffn_flops + lm_head_flops

if __name__ == "__main__":
    """
        (a) trainable parameters: 2_127_057_600, if each parameter is torch.float32, GPT-2 XL model needs 7.92 GB memory
        (b) matrix multiplies FLOPs occur on:
            [1] attn: num_layers * (6 * d_model * d_model * d_model + 4 * context_length * context_length * d_model + 2 * context_length * d_model * d_model)
            [2] ffn: num_layers * (4 * d_model * d_model * d_ff + 2 * d_ff * d_ff * d_model)
            [3] lm_head: 2 * d_model * d_model * vocab_size
        
    """
    vocab_size  = 50_257
    context_length = 1_024
    num_layers =  48
    d_model = 1_600
    num_heads = 25
    d_ff = 6_400
    rope_theta = 100_000
    transformer = TransformerInfoCalc(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, torch.device("cpu"), torch.float32)
    print(f"模型参数个数: {transformer.param_count()}") # 2_127_057_600 个参数
    print(f"手动计算内存 {transformer.param_count() * 4 / (1024 * 1024):.4f} MB")
    print(f"实际占用内存: {get_module_memory_bytes(transformer) / (1024 * 1024):.4f} MB")
    print("=================================")
    print(f"模型推理总乘法FLOPs: {transformer.mul_flops():.4f} GFLOPs")
