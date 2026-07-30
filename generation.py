import time
from pathlib import Path
import matplotlib.pyplot as plt

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
    incompat = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    assert not incompat.missing_keys, incompat.missing_keys
    assert all(k.endswith(".mask") for k in incompat.unexpected_keys), incompat.unexpected_keys
    model.to(device)
    model.eval()
    return model, cfg


def comparison(script_dir, device, attentions, text, tokenizer, max_new_tokens=200):
    results = {}

    for variant in attentions:
        try:
            model, cfg = load_model(script_dir, device, variant)
        except FileNotFoundError:
            print(f"{variant} not found")
            continue

        encoded = text_to_token_ids(text, tokenizer).to(device)

        t0 = time.time()
        out = generate_text_cached(model, encoded, max_new_tokens, cfg.context_length, temperature=1, top_k=35)
        gen_time = time.time() - t0
        if cfg.attention != "mla":
            cache_bytes = sum(blk.att.cache_K.numel() * blk.att.cache_K.element_size() + blk.att.cache_V.numel() * blk.att.cache_V.element_size() for blk in model.trf_blocks if blk.att.cache_K is not None)
        else :
            cache_bytes = sum(blk.att.cache_ckv.numel() * blk.att.cache_ckv.element_size()for blk in model.trf_blocks)
        seq_len = out.shape[1]

        results[variant] = {
            "params": sum(p.numel() for p in model.parameters()),
            "gen_time_s": gen_time,
            "tok_per_s": max_new_tokens / gen_time,
            "cache_MB": cache_bytes / 1e6,
            "cache_KB_per_token": cache_bytes / seq_len / 1e3,
            "sample": token_ids_to_text(out, tokenizer),
        }
        
        del model     
        torch.cuda.empty_cache()

    return results

def plot_generation(results, save_dir=None):
    metrics = {
        "gen_time_s": ("Generation time (s)", 1, "{:.2f}"),
        "tok_per_s": ("Flow rate (tokens/s)", 1, "{:.1f}"),
        "cache_MB": ("Cache size KV (MB)", 1, "{:.1f}"),
        "cache_KB_per_token": ("KV cache per token (KB)", 1, "{:.2f}"),
    }
    variants = list(results.keys())

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

    for key, (title, scale, fmt) in metrics.items():
        values = [results[v][key] / scale for v in variants]
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(variants, values, color="#170089")
        ax.bar_label(bars, labels=[fmt.format(v) for v in values], padding=3)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        fig.tight_layout()
        if save_dir is not None:
            fig.savefig(save_dir / f"{key}.png", dpi=150)

    plt.show()

def main():
    script_dir = Path(__file__).parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    tokenizer = tiktoken.get_encoding("gpt2")

    attentions = ["mha", "mqa", "gqa2", "gqa3", "gqa3", "gqa4", "gqa6", "mla"]
    prompt = "Once upon a time"
    results = comparison(script_dir, device, attentions, prompt, tokenizer, max_new_tokens=1000)
    plot_generation(results)


if __name__ == "__main__":
    main()