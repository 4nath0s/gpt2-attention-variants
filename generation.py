import time
from pathlib import Path

import torch
import tiktoken

from config import GPTConfig
from model import GPTModel
from training import text_to_token_ids, token_ids_to_text, generate_text_simple


def generate_text_cached(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None):
    assert idx.shape[1] + max_new_tokens <= context_size, "depassement du contexte maximal"
    model.reset_kv_cache()
    with torch.no_grad():
        logits = model(idx, use_cache=True)  
        for _ in range(max_new_tokens):
            logits = logits[:, -1, :]
            if top_k is not None:
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1:]
                logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)
            if temperature > 0.0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)
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

    t0 = time.time()
    model, cfg = load_model(script_dir, device, "mha")
    tm = time.time() - t0

 
    prompt = "Once upon a time"
    t0 = time.time()
    encoded = text_to_token_ids(prompt, tokenizer).to(device)
    te = time.time() - t0
    N = 200

    t0 = time.time()
    out = generate_text_cached(model, encoded, N, cfg.context_length, temperature=1, top_k=35)
    t = time.time() - t0

    print(f"Time loading model : {tm}")
    print(f"Time encoding : {te}")
    print(f"Time : {t}")
    print(token_ids_to_text(out, tokenizer))


if __name__ == "__main__":
    main()