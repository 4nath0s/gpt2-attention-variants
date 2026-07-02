from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    context_length: int = 1024
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    dropout: float = 0.1
    attention: str = "mha" 

