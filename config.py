from dataclasses import dataclass


# Model and run configuration, read by training.py.
# To train another variant, change attention / n_groups / window_size / cla_factor here.
# Default sizes = GPT-2 small (768 dims, 12 layers, 12 heads, 1024 context).
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
    # MLA only: size of the latent vector c_kv, which is the only thing cached
    d_latent: int = 256
    window_size: int | None = None #1024 or None = classical causal mask 
    cla_factor: int = 1 # 1 = Cross layer attention disabled

