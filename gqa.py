import torch
import torch.nn as nn
from mask import build_mask

# Grouped-query attention: num_heads query heads but only num_groups K/V heads.
# Each K/V head is shared by group_size = num_heads // num_groups query heads.
# num_groups = 1 -> MQA, num_groups = num_heads -> MHA.
# KV cache per token and per layer: 2 * num_groups * head_dim values.
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, num_groups, qkv_bias=False, window_size=None, share_kv=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"
        assert (num_heads % num_groups == 0), "num_groups must divide num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.num_groups =  num_groups
        self.group_size = num_heads // num_groups
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.window_size = window_size
        # persistent=False: the mask is not saved in the checkpoint, it is rebuilt from the config
        self.register_buffer("mask", build_mask(context_length, window_size), persistent=False)
        self.cache_K = None
        self.cache_V = None
        # share_kv=True (cross-layer attention): no K/V projections in this layer,
        # it reuses the K/V computed by the previous layer
        self.share_kv = share_kv
        if not share_kv:
            # K and V only have num_groups heads
            self.W_key = nn.Linear(d_in, self.head_dim * self.num_groups, bias=qkv_bias)
            self.W_value = nn.Linear(d_in, self.head_dim * self.num_groups, bias=qkv_bias)
        
        
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
            # (b, T, num_groups * head_dim) -> (b, num_groups, T, head_dim)
            keys   = keys.view(b, num_tokens, self.num_groups, self.head_dim)
            values = values.view(b, num_tokens, self.num_groups, self.head_dim)
            keys = keys.transpose(1, 2)
            values = values.transpose(1, 2)

            # Generation: append the new K/V to the cache and attend over the whole cache
            if use_cache:
                if self.cache_K is None:
                    self.cache_K = keys
                    self.cache_V = values
                else:
                    self.cache_K = torch.cat([self.cache_K, keys], dim=2)
                    self.cache_V = torch.cat([self.cache_V, values], dim=2)
            
                keys = self.cache_K
                values = self.cache_V
                
                # Sliding window: keep only the last window_size tokens (same logic as mha.py)
                if self.window_size is not None and self.cache_K.shape[2] > self.window_size:
                    self.cache_K = self.cache_K[:, :, -self.window_size:, :]
                    self.cache_V = self.cache_V[:, :, -self.window_size:, :]

            kv = (keys, values)

        # The cache stays compact (num_groups heads), K/V are only duplicated here
        # so that each query head gets the K/V head of its group
        keys   = keys.repeat_interleave(self.group_size, dim=1)     # (b, 4, T, 64) → (b, 12, T, 64)
        values = values.repeat_interleave(self.group_size, dim=1)
        
        attn_scores = queries @ keys.transpose(2, 3)
        T_k = keys.shape[2]
        # rows of the mask for the current queries (indices relative to the cache, see mha.py)
        past_len = T_k - num_tokens
        mask_bool = self.mask.bool()[past_len:T_k, :T_k]

        attn_scores.masked_fill_(mask_bool, -torch.inf)
        # scaled dot-product attention: scores / sqrt(head_dim)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2)

        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec, kv
    
    def reset_cache(self):
        self.cache_K = None
        self.cache_V = None
