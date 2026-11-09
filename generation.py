# Loads the trained checkpoints and compares them: validation perplexity,
# generation speed with KV cache and KV cache size. Plots are saved next to this script.
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import math
import torch.nn.functional as F

import torch
import tiktoken

from config import GPTConfig
from model import GPTModel
from training import text_to_token_ids, token_ids_to_text, generate_text_simple


# Generation with KV cache: the prompt goes through the model once, then only the
# new token is given at each step. Top-k + temperature sampling (greedy if temperature = 0).
def generate_text_cached(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None):
    # positions are learned up to context_length, so no generation past it (even with a sliding window)
    assert idx.shape[1] + max_new_tokens <= context_size, "prompt + new tokens exceed the context length"
    model.reset_kv_cache()
    with torch.no_grad():
        logits = model(idx, use_cache=True)  
        for _ in range(max_new_tokens):
            logits = logits[:, -1, :]
            # keep the top_k logits, the others are set to -inf
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
    # strict=False: older checkpoints still contain the mask buffer, now rebuilt from the config
    incompat = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    assert not incompat.missing_keys, incompat.missing_keys
    assert all(k.endswith(".mask") for k in incompat.unexpected_keys), incompat.unexpected_keys
    model.to(device)
    model.eval()
    return model, cfg


@torch.no_grad()
# Perplexity = exp(mean cross-entropy) on val.bin, non-overlapping blocks of context_length tokens.
# max_tokens limits the number of evaluated tokens (faster).
def perplexity(model, cfg, device, val_path, batch_size=4, max_tokens=None):
    data = np.memmap(val_path, dtype=np.uint16, mode="r")
    T = cfg.context_length

    n_blocks = (len(data) - 1) // T
    if max_tokens is not None:
        n_blocks = min(n_blocks, max_tokens // T)
    n_batches = n_blocks // batch_size
    assert n_batches > 0, "val.bin is too small for this context_length"

    total_nll, total_tokens = 0.0, 0
    for b in range(n_batches):
        s = b * batch_size * T
        x = np.stack([data[s + i*T     : s + i*T + T]     for i in range(batch_size)])
        y = np.stack([data[s + i*T + 1 : s + i*T + T + 1] for i in range(batch_size)])
        x = torch.from_numpy(x.astype(np.int64)).to(device)
        y = torch.from_numpy(y.astype(np.int64)).to(device)

        logits = model(x)
        total_nll += F.cross_entropy(
            logits.float().flatten(0, 1), y.flatten(), reduction="sum"
        ).item()
        total_tokens += y.numel()

    return math.exp(total_nll / total_tokens)


def comparison(script_dir, device, attentions, text, tokenizer, max_new_tokens=200):
    results = {}

    for variant in attentions:
        try:
            model, cfg = load_model(script_dir, device, variant)
        except FileNotFoundError:
            print(f"{variant} not found")
            continue

        encoded = text_to_token_ids(text, tokenizer).to(device)
        
        ppl = perplexity(model, cfg, device, script_dir / "val.bin")

        t0 = time.time()
        out = generate_text_cached(model, encoded, max_new_tokens, cfg.context_length, temperature=1, top_k=35)
        gen_time = time.time() - t0
        # KV cache size after generation, summed over the layers (fp32, 4 bytes per value).
        # With cross-layer attention, the layers that reuse K/V have cache_K = None.
        if cfg.attention != "mla":
            cache_bytes = sum(blk.att.cache_K.numel() * blk.att.cache_K.element_size() + blk.att.cache_V.numel() * blk.att.cache_V.element_size() for blk in model.trf_blocks if blk.att.cache_K is not None)
        else :
            cache_bytes = sum(blk.att.cache_ckv.numel() * blk.att.cache_ckv.element_size()for blk in model.trf_blocks)
        seq_len = out.shape[1]

        # tok_per_s includes the prompt pass. cache_KB_per_token divides by the full sequence length,
        # so for sliding window models it is an average (the cache is capped at window_size tokens).
        results[variant] = {
            "params": sum(p.numel() for p in model.parameters()),
            "gen_time_s": gen_time,
            "tok_per_s": max_new_tokens / gen_time,
            "cache_MB": cache_bytes / 1e6,
            "cache_KB_per_token": cache_bytes / seq_len / 1e3,
            "sample": token_ids_to_text(out, tokenizer),
            "ppl": ppl,
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
        "ppl": ("Perplexity (val)", 1, "{:.2f}"),
    }
    variants = list(results.keys())

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

    for key, (title, scale, fmt) in metrics.items():
        values = [results[v][key] / scale for v in variants]
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(variants, values, color="#170089")
        ax.bar_label(bars, labels=[fmt.format(v) for v in values], padding=3)
        ax.set_title(title)

        # zoom on the y axis, the perplexities are very close
        if key == "ppl":
            lo, hi = min(values), max(values)
            margin = 0.15 * (hi - lo) + 0.02
            ax.set_ylim(lo - margin, hi + margin)

        ax.tick_params(axis="x", rotation=45)
        plt.setp(ax.get_xticklabels(), ha="right")
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        fig.tight_layout()
        if save_dir is not None:
            fig.savefig(save_dir / f"{key}.png", dpi=150)

    # combined plot: KV cache (left axis) and perplexity (right axis)
    x = np.arange(len(variants))
    width = 0.38
    caches = [results[v]["cache_MB"] for v in variants]
    ppls = [results[v]["ppl"] for v in variants]

    fig, ax_cache = plt.subplots(figsize=(9, 4.5))
    ax_ppl = ax_cache.twinx()

    b1 = ax_cache.bar(x - width / 2, caches, width, color="#170089", label="Cache KV")
    b2 = ax_ppl.bar(x + width / 2, ppls, width, color="#e07b39", label="Perplexity")

    ax_cache.bar_label(b1, labels=[f"{v:.1f}" for v in caches], padding=3, fontsize=8)
    ax_ppl.bar_label(b2, labels=[f"{v:.2f}" for v in ppls], padding=3, fontsize=8)

    ax_cache.set_ylabel("Cache size KV (MB)", color="#170089")
    ax_ppl.set_ylabel("Perplexity (val)", color="#e07b39")
    ax_cache.tick_params(axis="y", colors="#170089")
    ax_ppl.tick_params(axis="y", colors="#e07b39")

    ax_cache.set_xticks(x)
    ax_cache.set_xticklabels(variants, rotation=45, ha="right")
    ax_cache.set_ylim(0, max(caches) * 1.18)
    ax_ppl.set_ylim(0, max(ppls) * 1.18)

    ax_cache.grid(axis="y", alpha=0.3)
    ax_cache.set_axisbelow(True)
    ax_cache.set_title("KV cache and perplexity")
    ax_cache.legend(handles=[b1, b2], frameon=False, loc="upper right")

    fig.tight_layout()
    if save_dir is not None:
        fig.savefig(save_dir / "cache_vs_ppl.png", dpi=150)

    plt.show()

def main():
    script_dir = Path(__file__).parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    tokenizer = tiktoken.get_encoding("gpt2")

    # checkpoint suffixes to compare: models/model_and_optimizer_<suffix>.pth
    attentions = ["mha", "mqa", "gqa2", "gqa3", "gqa4", "gqa6", "mla", "mha_w256", "gqa1_w256", "gqa4_w256", "mla_w256"]
    prompt = "Once upon a time"
    results = comparison(script_dir, device, attentions, prompt, tokenizer, max_new_tokens=1000)
    plot_generation(results, save_dir=script_dir)


if __name__ == "__main__":
    main()