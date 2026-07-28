from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    context_length: int = 1024
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    dropout: float = 0.1
    n_groups: int = 4 # 1(MQA), 2, 3, 4, 6 or 12(MHA)
    attention: str = "gqa" # "mha" | "gqa" | "mla"
    qkv_bias:bool = False
    seed_number:int = 123
    d_latent: int = 256
    window_size: int | None = 256 #1024 or None = classical causal mask 

