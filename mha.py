import torch
import torch.nn as nn
from mask import build_mask
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False, window_size=None, share_kv=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.window_size = window_size
        self.register_buffer("mask", build_mask(context_length, window_size), persistent=False)
        self.cache_K = None
        self.cache_V = None
        self.share_kv = share_kv
        if not share_kv:
            self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        
    def forward(self, x, use_cache=False, kv=None):
        b, num_tokens, d_in = x.shape
        queries = self.W_query(x)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.transpose(1, 2)

        if self.share_kv:
            assert kv is not None
            keys, values = kv
        else:
            keys = self.W_key(x)
            values = self.W_value(x)
            keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
            values = values.view(b, num_tokens, self.num_heads, self.head_dim)
            keys = keys.transpose(1, 2)
            values = values.transpose(1, 2)

            if use_cache:
                if self.cache_K is None:
                    self.cache_K = keys
                    self.cache_V = values
                else:
                    self.cache_K = torch.cat([self.cache_K, keys], dim=2)
                    self.cache_V = torch.cat([self.cache_V, values], dim=2)
            
                keys = self.cache_K
                values = self.cache_V
                
                if self.window_size is not None and self.cache_K.shape[2] > self.window_size:
                    self.cache_K = self.cache_K[:, :, -self.window_size:, :]
                    self.cache_V = self.cache_V[:, :, -self.window_size:, :]

            kv = (keys, values)

        attn_scores = queries @ keys.transpose(2, 3)
        T_k = keys.shape[2]
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
        self.cache_K = None
        self.cache_V = None