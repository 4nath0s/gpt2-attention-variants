import time
from pathlib import Path

import torch
import tiktoken
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from torch.utils.data import DataLoader

from config import GPTConfig
from model import GPTModel
from dataset import GPTDatasetBin


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def generate_and_print_sample(model, tokenizer, device, start_context,
                              context_size):
    model.eval()
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size,
        )
    print(token_ids_to_text(token_ids, tokenizer).replace("\n", " "))
    model.train()


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                        enabled=(device.type == "cuda")):
        logits = model(input_batch)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), target_batch.flatten()
        )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device,
                                      num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device,
                                    num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train_model_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs, eval_freq, eval_iter, start_context,
                       tokenizer, context_size):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    t0 = time.time()

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                tok_per_sec = tokens_seen / (time.time() - t0)
                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f} "
                    f"[{tok_per_sec:,.0f} tok/s]"
                )

                if global_step % (eval_freq * 4) == 0:
                    generate_and_print_sample(model, tokenizer, device,
                                              start_context, context_size)

        generate_and_print_sample(model, tokenizer, device,
                                  start_context, context_size)

    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses,
                save_path=None):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.",
             label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Curves saved in : {save_path}")
    plt.show()


def main():
    torch.manual_seed(123)

    script_dir = Path(__file__).parent

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")

    cfg = GPTConfig()

    model = GPTModel(cfg)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=4e-4, weight_decay=0.1)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device} | Parametres: {n_params:,}")

    train_loader = DataLoader(
        GPTDatasetBin(script_dir / "train.bin", cfg.context_length),
        batch_size=4, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        GPTDatasetBin(script_dir / "val.bin", cfg.context_length),
        batch_size=4, shuffle=False, drop_last=False,
    )

    print(f"Train batches: {len(train_loader)}, "
          f"Val batches: {len(val_loader)}")
    assert len(val_loader) > 0, (
        "val_loader vide : val.bin manquant ou plus court que "
        "context_length. Lancer preprocess.py d'abord."
    )

    
    num_epochs = 1
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=250, eval_iter=10,
        start_context="Once upon a time",
        tokenizer=tokenizer,
        context_size=cfg.context_length,
    )

    ckpt_path = script_dir / "models" / f"model_and_optimizer_{cfg.attention}.pth"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "cfg": cfg,
    }, ckpt_path)
    print(f"Checkpoint sauvegarde dans {ckpt_path}")

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses,
                save_path=script_dir / "curves" / f"loss_curves_{cfg.attention}.png")


if __name__ == "__main__":
    main()