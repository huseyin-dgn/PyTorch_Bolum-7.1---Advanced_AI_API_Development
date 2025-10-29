import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import math


class DropPath(nn.Module):
    def __init__(self, drop_prob = 0.2):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self,x):
        if self.drop_prob == 0.0 or not self.training:
            return x
    
        keep_prob = 1 - self.drop_prob
        shape = (x.size(0) , ) + (1,) * (x.ndim - 1)
        random_tensors = keep_prob + torch.rand(shape , dtype=x.dtype , device=x.device)
        random_tensors.floor_()
        return x.div(keep_prob) * random_tensors
    

class TokenEmbed(nn.Module):
    def __init__(self, vocab_size , embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size , embed_dim)
    
    def forward(self,x):
        return self.embedding(x)
    

class PositionelEncod(nn.Module):
    def __init__(self, embed_dim , max_len = 5000):
        super().__init__()

        pe = torch.zeros(max_len , embed_dim)
        position = torch.arange( 0 ,max_len , dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0 , embed_dim , 2).float() * (-math.log(10000.0)/embed_dim))
        pe[: , 0::2] = torch.sin(position * div_term)
        pe[: , 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe' , pe)
    
    def forward(self,x):
        seq_len = x.size(1)
        return x + self.pe[: , :seq_len , :]
    

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=16, dp=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dp)

    def split_heads(self, x):
        B, T, C = x.size()
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, T, head_dim]

    def combine_heads(self, x):
        B, H, T, D = x.size()
        return x.transpose(1, 2).contiguous().view(B, T, H * D)

    def forward(self, query, key, value, mask=None):
        Q = self.split_heads(self.q_proj(query))
        K = self.split_heads(self.k_proj(key))
        V = self.split_heads(self.v_proj(value))

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, T_q, T_k]

        if mask is not None:
            # Mask broadcast: [B, T] -> [B, 1, 1, T] -> broadcastable
            if mask.dim() == 2:
                mask = mask[:, None, None, :]
            elif mask.dim() == 3:
                mask = mask[:, None, :, :]  # [B,1,T_q,T_k] (cross attention)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        return self.out_proj(self.combine_heads(out))

class FeedForward(nn.Module):
    def __init__(self, embed_dim, expansion=8, dp=0.1, use_swiglu=False):
        super().__init__()
        if use_swiglu:
            # SwiGLU activation
            self.net = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * expansion * 2),
                nn.SiLU(),
                nn.Dropout(dp),
                nn.Linear(embed_dim * expansion, embed_dim),
                nn.Dropout(dp)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * expansion),
                nn.GELU(),
                nn.Dropout(dp),
                nn.Linear(embed_dim * expansion, embed_dim),
                nn.Dropout(dp)
            )
    def forward(self, x):
        return self.net(x)
    
# ENCODER
class TransformerEncoderBlockLLM(nn.Module):
    def __init__(self, embed_dim=1024, num_heads=16, dp=0.1, drop_path=0.1, expansion=8, use_swiglu=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dp)
        self.ffn = FeedForward(embed_dim, expansion, dp, use_swiglu)

        self.drop_path = DropPath(drop_path)
        self.gamma_1 = nn.Parameter(torch.ones(embed_dim) * 1e-2)
        self.gamma_2 = nn.Parameter(torch.ones(embed_dim) * 1e-2)

    def forward(self, x, mask=None):
        # Self-Attention
        attn_out = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + self.drop_path(self.gamma_1 * attn_out)
        # FeedForward
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.drop_path(self.gamma_2 * ffn_out)
        return x
    
class TransformersEncoderLLM(nn.Module):
    def __init__(self, vocab_size , embed_dim = 1024 , num_layers = 12 , dp = 0.1 ,num_heads=16 ,  expansion = 8 , max_len= 5000 , drop_path = 0.1 , use_swiglu =False):
        super().__init__()

        self.tok_emb = TokenEmbed(vocab_size,embed_dim)
        self.pos_enc = PositionelEncod(embed_dim , max_len)
        self.layers = nn.ModuleList(
            [TransformerEncoderBlockLLM(embed_dim , num_heads , dp , drop_path , expansion , use_swiglu) for _ in range(num_layers)]
            )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self,src_tokens , src_mask =None):
        x = self.tok_emb(src_tokens)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x,mask = src_mask)
        x = self.norm(x)
        return x
    
# DECODER
class TransformerDecoderBlockLLM(nn.Module):
    def __init__(self, embed_dim=1024, num_heads=16, dp=0.1, drop_path=0.1, expansion=8, use_swiglu=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dp)
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads, dp)
        self.ffn = FeedForward(embed_dim, expansion, dp, use_swiglu)

        self.drop_path = DropPath(drop_path)
        self.gamma_1 = nn.Parameter(torch.ones(embed_dim) * 1e-2)
        self.gamma_2 = nn.Parameter(torch.ones(embed_dim) * 1e-2)
        self.gamma_3 = nn.Parameter(torch.ones(embed_dim) * 1e-2)

    def forward(self, x, enc_out=None, self_mask=None, enc_mask=None):
        # Masked Self-Attention
        if self_mask is not None:
            # [B, T] -> [B, 1, T, T] (triangular mask veya pad mask)
            if self_mask.dim() == 2:
                # Causal mask: üst üçgen mask için manuel ekleme gerekebilir
                causal_mask = torch.tril(torch.ones((x.size(1), x.size(1)), device=x.device)).bool()
                self_mask = self_mask[:, None, :] & causal_mask[None, :, :]
        attn_out = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask=self_mask)
        x = x + self.drop_path(self.gamma_1 * attn_out)

        # Cross-Attention
        if enc_out is not None:
            if enc_mask is not None and enc_mask.dim() == 2:
                enc_mask = enc_mask[:, None, None, :]  # [B,1,1,T_enc]
            cross_out = self.cross_attn(self.norm2(x), self.norm2(enc_out), self.norm2(enc_out), mask=enc_mask)
            x = x + self.drop_path(self.gamma_2 * cross_out)

        # FeedForward
        ffn_out = self.ffn(self.norm3(x))
        x = x + self.drop_path(self.gamma_3 * ffn_out)

        return x

class TransformerDecoderLLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=1024, num_layers=12, num_heads=16, dp=0.1, drop_path=0.1, expansion=8, max_len=5000, use_swiglu=False):
        super().__init__()
        self.embedding = TokenEmbed(vocab_size, embed_dim)
        self.pos_encoding = PositionelEncod(embed_dim, max_len)
        self.layers = nn.ModuleList([
            TransformerDecoderBlockLLM(embed_dim, num_heads, dp, drop_path, expansion, use_swiglu) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, enc_out=None, self_mask=None, enc_mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, enc_out, self_mask, enc_mask)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
    

class Seq2SeqLLM(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, src_tokens, tgt_tokens, src_mask=None, tgt_mask=None):
        enc_out = self.encoder(src_tokens, src_mask)
        logits = self.decoder(tgt_tokens, enc_out, self_mask=tgt_mask, enc_mask=src_mask)
        return logits