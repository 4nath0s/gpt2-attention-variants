# GPT-2 architecture: pre-LayerNorm transformer blocks, learned absolute position embeddings.
# Only the attention module changes between variants (cfg.attention).
import torch
import torch.nn as nn
from mha import MultiHeadAttention
from gqa import GroupedQueryAttention
from mla import MultiHeadLatentAttention

# same as nn.LayerNorm, written by hand (scale = gamma, shift = beta)
class LayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(d_model))
        self.shift = nn.Parameter(torch.zeros(d_model))
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


# tanh approximation of GELU, as in the original GPT-2
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.const = torch.sqrt(torch.tensor(2.0 / torch.pi))
        
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(self.const * (x + 0.044715 * torch.pow(x, 3))))
    
    
# MLP: d_model -> 4 * d_model -> d_model
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(cfg.d_model, 4 * cfg.d_model), GELU(), nn.Linear(4 * cfg.d_model, cfg.d_model))
        
    def forward(self, x):
        return self.layers(x)
    
    
class TransformerBlock(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        # Cross-layer attention: with cla_factor = k, only layers 0, k, 2k, ... compute K/V,
        # the k - 1 next layers reuse them. Only for mha and gqa.
        # getattr: older checkpoints were saved with a config without these fields
        cla_factor = getattr(cfg, "cla_factor", 1)
        share_kv = (cla_factor > 1) and (layer_idx % cla_factor != 0)
        if cfg.attention == "mha":
            self.att = MultiHeadAttention(
                d_in=cfg.d_model,
                d_out=cfg.d_model,
                context_length=cfg.context_length,
                num_heads=cfg.n_heads,
                dropout=cfg.dropout,
                qkv_bias=cfg.qkv_bias,
                window_size = getattr(cfg, "window_size", None),
                share_kv=share_kv)
        elif cfg.attention == "gqa":
            self.att = GroupedQueryAttention(d_in=cfg.d_model,
                d_out=cfg.d_model,
                context_length=cfg.context_length,
                num_heads=cfg.n_heads,
                dropout=cfg.dropout,
                num_groups=cfg.n_groups,
                qkv_bias=cfg.qkv_bias,
                window_size = getattr(cfg, "window_size", None),
                share_kv=share_kv)
        elif cfg.attention == "mla":
            self.att = MultiHeadLatentAttention(
                d_in=cfg.d_model,
                d_out=cfg.d_model,
                context_length=cfg.context_length,
                num_heads=cfg.n_heads,
                dropout=cfg.dropout,
                d_latent=cfg.d_latent,
                qkv_bias=cfg.qkv_bias,
                window_size = getattr(cfg, "window_size", None))
        else:
            raise ValueError(f"Unknown attention variant : {cfg.attention!r}")
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg.d_model)
        self.norm2 = LayerNorm(cfg.d_model)
        self.drop_shortcut = nn.Dropout(cfg.dropout)
        
    # pre-norm residual block: x + Attn(LN(x)), then x + FF(LN(x))
    def forward(self, x, use_cache=False, kv=None):
        shortcut = x
        x = self.norm1(x)
        x, kv = self.att(x, use_cache=use_cache, kv=kv)
        x = self.drop_shortcut(x)
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x, kv
    
    
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.d_model)
        self.drop_emb = nn.Dropout(cfg.dropout)

        self.trf_blocks = nn.ModuleList([TransformerBlock(cfg, layer_idx=i) for i in range(cfg.n_layers)])

        self.final_norm = LayerNorm(cfg.d_model)
        # no weight tying with tok_emb (unlike the original GPT-2)
        self.out_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # number of tokens already in the KV cache, to get the right positions
        self.current_pos = 0
        
    def forward(self, in_idx, use_cache=False):
        batch_size, seq_len = in_idx.shape

        if use_cache:
            # with the cache only the new tokens are given, so positions start after the cached ones
            start = self.current_pos
            self.current_pos += seq_len
        else:
            start = 0
        pos = torch.arange(start, start + seq_len, device=in_idx.device)

        x = self.tok_emb(in_idx) + self.pos_emb(pos)
        x = self.drop_emb(x)
        # kv carries the K/V from one block to the next (used by cross-layer attention)
        kv = None
        for blk in self.trf_blocks:
            x, kv = blk(x, use_cache=use_cache, kv=kv)
        x = self.final_norm(x)
        return self.out_head(x)
    
    
    # call before every new generation
    def reset_kv_cache(self):
        for blk in self.trf_blocks:
            blk.att.reset_cache()
        self.current_pos = 0