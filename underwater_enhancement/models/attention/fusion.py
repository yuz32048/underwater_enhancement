from __future__ import annotations

import torch
from torch import nn


class AttentionFusion(nn.Module):
    def __init__(self, num_branches: int = 4, channels: int = 3):
        super().__init__()
        self.num_branches = num_branches
        self.attn = nn.Sequential(
            nn.Conv2d(num_branches * channels + channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_branches, 1),
        )
        self.last_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, branch_outputs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        stacked = torch.stack(branch_outputs, dim=1)
        logits = self.attn(torch.cat([x] + branch_outputs, dim=1))
        weights = torch.softmax(logits, dim=1)
        self.last_weights = weights.detach()
        fused = (stacked * weights.unsqueeze(2)).sum(dim=1)
        return fused.clamp(0, 1), weights


class AverageFusion(nn.Module):
    def __init__(self, num_branches: int = 4):
        super().__init__()
        self.num_branches = num_branches
        self.last_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, branch_outputs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        weights = x.new_full((x.shape[0], len(branch_outputs), x.shape[2], x.shape[3]), 1.0 / len(branch_outputs))
        self.last_weights = weights.detach()
        return torch.stack(branch_outputs, dim=1).mean(dim=1).clamp(0, 1), weights


class ConcatFusion(nn.Module):
    def __init__(self, num_branches: int = 4):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(num_branches * 3, 32, 1), nn.ReLU(inplace=True), nn.Conv2d(32, 3, 1), nn.Sigmoid())
        self.last_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, branch_outputs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        weights = x.new_full((x.shape[0], len(branch_outputs), x.shape[2], x.shape[3]), 1.0 / len(branch_outputs))
        self.last_weights = weights.detach()
        return self.conv(torch.cat(branch_outputs, dim=1)), weights

