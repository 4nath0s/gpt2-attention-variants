# Downloads TinyStories (Hugging Face), tokenizes it with the GPT-2 tokenizer
# and writes train.bin / val.bin next to this script. Run once before training.
# Format: raw uint16 token ids, with <|endoftext|> after each story.
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset

script_dir = Path(__file__).parent
tokenizer = tiktoken.get_encoding("gpt2")
eot = tokenizer._special_tokens["<|endoftext|>"]

# int = only keep the first N training tokens (quick tests), None = whole dataset
MAX_TOKENS = None


def write_bin(split, out_path, max_tokens=None):
    ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    total = 0
    with open(out_path, "wb") as f:
        for doc in ds:
            ids = tokenizer.encode_ordinary(doc["text"]) + [eot]
            # uint16 is enough since vocab_size = 50257 < 65536
            np.array(ids, dtype=np.uint16).tofile(f)
            total += len(ids)
            if max_tokens and total >= max_tokens:
                break
    print(f"{out_path.name}: {total:,} tokens")


write_bin("validation", script_dir / "val.bin")
write_bin("train", script_dir / "train.bin", MAX_TOKENS)