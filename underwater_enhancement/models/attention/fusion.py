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


class TemperatureAttentionFusion(nn.Module):
    def __init__(self, num_branches: int = 4, channels: int = 3, temperature: float = 2.0):
        super().__init__()
        self.num_branches = num_branches
        self.temperature = temperature
        self.attn = nn.Sequential(
            nn.Conv2d(num_branches * channels + channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_branches, 1),
        )
        self.last_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, branch_outputs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        stacked = torch.stack(branch_outputs, dim=1)
        logits = self.attn(torch.cat([x] + branch_outputs, dim=1))
        weights = torch.softmax(logits / self.temperature, dim=1)
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


class ResidualConcatFusion(nn.Module):
    def __init__(self, num_branches: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_branches * 3, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1),
            nn.Tanh(),
        )
        self.last_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, branch_outputs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        weights = x.new_full((x.shape[0], len(branch_outputs), x.shape[2], x.shape[3]), 1.0 / len(branch_outputs))
        self.last_weights = weights.detach()
        residuals = [out - x for out in branch_outputs]
        residual = self.conv(torch.cat(residuals, dim=1))
        return (x + residual).clamp(0, 1), weights


class GatedResidualFusion(nn.Module):
    def __init__(self, num_branches: int = 4, channels: int = 3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(num_branches * channels + channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_branches, 1),
            nn.Sigmoid(),
        )
        self.last_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, branch_outputs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        gates = self.gate(torch.cat([x] + branch_outputs, dim=1))
        weights = gates / (gates.sum(dim=1, keepdim=True) + 1e-6)
        self.last_weights = weights.detach()
        residuals = torch.stack([out - x for out in branch_outputs], dim=1)
        residual = (residuals * gates.unsqueeze(2)).sum(dim=1)
        return (x + residual).clamp(0, 1), weights


class ConcatSEFusion(nn.Module):
    def __init__(self, num_branches: int = 4, channels: int = 3, reduction: int = 4):
        super().__init__()
        in_channels = num_branches * channels
        hidden = max(in_channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, 1),
            nn.Sigmoid(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1),
            nn.Sigmoid(),
        )
        self.last_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, branch_outputs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        weights = x.new_full((x.shape[0], len(branch_outputs), x.shape[2], x.shape[3]), 1.0 / len(branch_outputs))
        self.last_weights = weights.detach()
        features = torch.cat(branch_outputs, dim=1)
        features = features * self.se(features)
        return self.conv(features), weights

