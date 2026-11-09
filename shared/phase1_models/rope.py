# TODO 3.1: rotary positional embeddings from scratch (split-half convention).
import torch


def build_rope_cache(max_seq_len, d_head, base=10000.0):
    assert d_head % 2 == 0, "RoPE requires an even head dimension"
    inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))   # (d_head/2,)
    positions = torch.arange(max_seq_len).float()
    angles = torch.outer(positions, inv_freq)                                  # (T, d_head/2)
    angles = torch.cat([angles, angles], dim=-1)                               # (T, d_head)
    return angles.cos(), angles.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin
