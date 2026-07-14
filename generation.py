import time
from pathlib import Path

import torch
import tiktoken

from config import GPTConfig
from model import GPTModel
from training import text_to_token_ids, token_ids_to_text, generate_text_simple


def generate_text_cached(model, idx, max_new_tokens, context_size):
    assert idx.shape[1] + max_new_tokens <= context_size, "depassement du contexte maximal"
    model.reset_kv_cache()
    with torch.no_grad():
        logits = model(idx, use_cache=True)  
        for _ in range(max_new_tokens):
            idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            idx = torch.cat([idx, idx_next], dim=1)
            logits = model(idx_next, use_cache=True)  
    return idx

def load_model(script_dir, device, variant):
    ckpt_path = script_dir / "models" / f"model_and_optimizer_{variant}.pth"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = GPTModel(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, cfg




def main():
    script_dir = Path(__file__).parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")

    model, cfg = load_model(script_dir, device, "mha")
 

    prompt = "Once upon a time"
    encoded = text_to_token_ids(prompt, tokenizer).to(device)
    N = 200

    t0 = time.time()
    out = generate_text_cached(model, encoded, N, cfg.context_length)
    t = time.time() - t0

    print(f"Time : {t}")
    print(token_ids_to_text(out, tokenizer))


if __name__ == "__main__":
    main()