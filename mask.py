import torch

# Boolean attention mask, True = masked (set to -inf before the softmax).
# Row i = query position, column j = key position.
def build_mask(context_length, window_size=None):
    ones = torch.ones(context_length, context_length, dtype=torch.bool)
    # j > i: future tokens
    future_mask = torch.triu(ones, diagonal=1)   
    if window_size is None:
        return future_mask  #classical causal mask   
    # j <= i - window_size: tokens too far in the past (sliding window).
    # Each query then sees at most window_size keys, itself included.
    old_mask = torch.tril(ones, diagonal=-window_size)  
    return torch.logical_or(future_mask, old_mask)