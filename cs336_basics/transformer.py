import torch

from torch import nn
from cs336_basics.normalization import RMSNorm
from cs336_basics.attention import MultiheadSelfAttentionWithRoPE
from cs336_basics.linear import SwiGLU, Embedding, Linear
from cs336_basics.utils import softmax

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiheadSelfAttentionWithRoPE(d_model, num_heads, theta, max_seq_len, device, dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)
    
    def forward(self, x: torch.Tensor):
        x += self.attn(self.ln1(x), torch.arange(x.shape[-2]))
        x += self.ffn(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.layers = nn.Sequential(
            *[TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device, dtype) for _ in range(num_layers)]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)
    
    def forward(self, x: torch.Tensor):
        x = self.token_embeddings(x)
        x = self.layers(x) 
        x = self.ln_final(x) 
        x = self.lm_head(x)
        return x