import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, tokenizer=None, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    
import numpy as np

class GPTDatasetBin(Dataset):
    def __init__(self, bin_path, max_length):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.max_length = max_length

    def __len__(self):
        return (len(self.data) - 1) // self.max_length

    def __getitem__(self, idx):
        i = idx * self.max_length
        x = torch.from_numpy(self.data[i:i + self.max_length].astype(np.int64))
        y = torch.from_numpy(self.data[i + 1:i + self.max_length + 1].astype(np.int64))
        return x, y
