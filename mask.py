import torch

def build_mask(context_length, window_size=None):
    ones = torch.ones(context_length, context_length, dtype=torch.bool)
    future_mask = torch.triu(ones, diagonal=1)   
    if window_size is None:
        return future_mask  #classical causal mask   
    old_mask = torch.tril(ones, diagonal=-window_size)  
    return torch.logical_or(future_mask, old_mask)