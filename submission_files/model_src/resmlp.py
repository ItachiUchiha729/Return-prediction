"""
ResidualMLP architecture used for training and inference.
Defined here (not in the notebook) so model_predictor.py can import it.
"""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d, d),
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_blocks: int, dropout: float):
        super().__init__()
        self.proj   = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.blocks = nn.Sequential(*[ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.head   = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.proj(x))).squeeze(-1)
