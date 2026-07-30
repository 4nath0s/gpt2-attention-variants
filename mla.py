import torch
import torch.nn as nn
from mask import build_mask

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, d_latent, qkv_bias=False, window_size=None):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.d_latent = d_latent
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_dkv = nn.Linear(d_in, d_latent, bias=qkv_bias)   
        self.W_uk = nn.Linear(d_latent, d_out, bias=qkv_bias)   
        self.W_uv = nn.Linear(d_latent, d_out, bias=qkv_bias)   
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.window_size = window_size
        self.register_buffer("mask", build_mask(context_length, window_size), persistent=False)
        self.cache_ckv = None
    
    def forward(self, x, use_cache=False, kv=None):
        b, num_tokens, d_in = x.shape
        queries = self.W_query(x)
        c_kv = self.W_dkv(x)                                    

        if use_cache:
            if self.cache_ckv is None:
                self.cache_ckv = c_kv
            else:
                self.cache_ckv = torch.cat([self.cache_ckv, c_kv], dim=1)
            c_kv = self.cache_ckv
            
            if self.window_size is not None and self.cache_ckv.shape[1] > self.window_size:
                self.cache_ckv = self.cache_ckv[:, -self.window_size:, :]

        T_k = c_kv.shape[1]
        keys = self.W_uk(c_kv)     
        values = self.W_uv(c_kv)    

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, T_k, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        past_len = T_k - num_tokens
        mask_bool = self.mask.bool()[past_len:T_k, :T_k]

        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2)

        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec, kv

    
    def reset_cache(self):
        self.cache_ckv = None