import torch
import torch.nn as nn
from mha import MultiHeadAttention
from gqa import GroupedQueryAttention
from mla import MultiHeadLatentAttention

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


class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.const = torch.sqrt(torch.tensor(2.0 / torch.pi))
        
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(self.const * (x + 0.044715 * torch.pow(x, 3))))
    
    
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(cfg.d_model, 4 * cfg.d_model), GELU(), nn.Linear(4 * cfg.d_model, cfg.d_model))
        
    def forward(self, x):
        return self.layers(x)
    
    
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.attention == "mha":
            self.att = MultiHeadAttention(
                d_in=cfg.d_model,
                d_out=cfg.d_model,
                context_length=cfg.context_length,
                num_heads=cfg.n_heads,
                dropout=cfg.dropout,
                qkv_bias=cfg.qkv_bias,
                window_size = getattr(cfg, "window_size", None))
        elif cfg.attention == "gqa":
            self.att = GroupedQueryAttention(d_in=cfg.d_model,
                d_out=cfg.d_model,
                context_length=cfg.context_length,
                num_heads=cfg.n_heads,
                dropout=cfg.dropout,
                num_groups=cfg.n_groups,
                qkv_bias=cfg.qkv_bias,
                window_size = getattr(cfg, "window_size", None))
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
        
    def forward(self, x, use_cache=False):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, use_cache=use_cache)
        x = self.drop_shortcut(x)
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x
    
    
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.d_model)
        self.drop_emb = nn.Dropout(cfg.dropout)

        self.trf_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])

        self.final_norm = LayerNorm(cfg.d_model)
        self.out_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.current_pos = 0
        
    def forward(self, in_idx, use_cache=False):
        batch_size, seq_len = in_idx.shape

        if use_cache:
            start = self.current_pos
            self.current_pos += seq_len
        else:
            start = 0
        pos = torch.arange(start, start + seq_len, device=in_idx.device)

        x = self.tok_emb(in_idx) + self.pos_emb(pos)
        x = self.drop_emb(x)
        for blk in self.trf_blocks:
            x = blk(x, use_cache=use_cache)
        x = self.final_norm(x)
        return self.out_head(x)
    
    
    def reset_kv_cache(self):
        for blk in self.trf_blocks:
            blk.att.reset_cache()
        self.current_pos = 0