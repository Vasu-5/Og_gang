import torch
import math

def apply_rope(x, seq_len):
    # x shape: [batch_size, num_heads, seq_len, head_dim]
    head_dim = x.size(-1)
    half_dim = head_dim // 2

    # Position indices [seq_len]
    position = torch.arange(seq_len, dtype=torch.float32, device=x.device)

    # Frequencies [half_dim]
    dim_indices = torch.arange(half_dim, dtype=torch.float32, device=x.device)
    inv_freq = 1.0 / (10000 ** (dim_indices / half_dim))

    # [seq_len, half_dim]
    sinusoid = torch.einsum("i,j->ij", position, inv_freq)
    sin = torch.sin(sinusoid)[None, None, :, :]  # shape: [1, 1, seq_len, half_dim]
    cos = torch.cos(sinusoid)[None, None, :, :]

    # Split last dim
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]

    x_rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_rotated
