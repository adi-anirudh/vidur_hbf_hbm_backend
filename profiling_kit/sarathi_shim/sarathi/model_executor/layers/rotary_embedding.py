import torch
import torch.nn as nn


class _RotaryEmbedding(nn.Module):
    """Standard neox-style RoPE applied to q and k. Shapes: q [T, n_q*hd],
    k [T, n_kv*hd]. Cost-faithful for profiling (real elementwise rotation)."""
    def __init__(self, head_dim, rotary_dim, max_position, base):
        super().__init__()
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
        # Size tables generously: the MLP profiler sweeps num_tokens (positions)
        # independently of the model's max_position_embeddings, so clamp up.
        max_position = max(int(max_position), 1 << 17)
        t = torch.arange(max_position).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def _rotate(self, x, positions):
        # x: [T, n_heads*head_dim] -> [T, n_heads, head_dim]
        T = x.shape[0]
        x = x.view(T, -1, self.head_dim)
        rd = self.rotary_dim
        xr, xpass = x[..., :rd], x[..., rd:]
        cos = self.cos[positions].unsqueeze(1)  # [T,1,rd/2]
        sin = self.sin[positions].unsqueeze(1)
        x1, x2 = xr[..., : rd // 2], xr[..., rd // 2:]
        rot = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        out = torch.cat([rot, xpass], dim=-1).view(T, -1)
        return out

    def forward(self, positions, q, k):
        return self._rotate(q, positions), self._rotate(k, positions)


def get_rope(head_dim, rotary_dim, max_position, base, is_neox_style=True,
             rope_scaling=None):
    return _RotaryEmbedding(head_dim, rotary_dim, max_position, int(base)).cuda().half()
