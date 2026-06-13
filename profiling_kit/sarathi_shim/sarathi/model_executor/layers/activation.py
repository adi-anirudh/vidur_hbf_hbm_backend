import torch
import torch.nn as nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    """Gated SiLU: input is [..., 2*d] -> silu(a) * b where a,b = split."""
    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return F.silu(a) * b
