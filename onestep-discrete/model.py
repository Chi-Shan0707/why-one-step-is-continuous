"""One-step generator: z -> L x V logits (an MLP with 3 linear layers, width 256)."""
import torch.nn as nn

from data import L, V


class MLP(nn.Module):
    def __init__(self, d_in, L=L, V=V, width=256):
        super().__init__()
        self.L, self.V = L, V
        self.net = nn.Sequential(
            nn.Linear(d_in, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, L * V))

    def forward(self, z):
        return self.net(z).view(-1, self.L, self.V)
