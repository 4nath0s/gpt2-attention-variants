from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset

script_dir = Path(__file__).parent
tokenizer = tiktoken.get_encoding("gpt2")
eot = tokenizer._special_tokens["<|endoftext|>"]

MAX_TOKENS = None


def write_bin(split, out_path, max_tokens=None):
    ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    total = 0
    with open(out_path, "wb") as f:
        for doc in ds:
            ids = tokenizer.encode_ordinary(doc["text"]) + [eot]
            np.array(ids, dtype=np.uint16).tofile(f)
            total += len(ids)
            if max_tokens and total >= max_tokens:
                break
    print(f"{out_path.name}: {total:,} tokens")


write_bin("validation", script_dir / "val.bin")
write_bin("train", script_dir / "train.bin", MAX_TOKENS)