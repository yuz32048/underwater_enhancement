from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def simple_white_balance(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = x.mean(dim=(2, 3), keepdim=True)
    gray = mean.mean(dim=1, keepdim=True)
    return (x * gray / (mean + eps)).clamp(0, 1)


class VGGFeatureBlock(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            from torchvision import models

            weights = models.VGG16_Weights.IMAGENET1K_FEATURES
            vgg = models.vgg16(weights=weights).features[:9]
            out_ch = 128
        except Exception:
            vgg = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            )
            out_ch = 128
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.reduce = nn.Conv2d(out_ch, 16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.vgg(x)
        feat = nn.functional.interpolate(feat, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.reduce(feat)


class BlueCastBranch(nn.Module):
    def __init__(self, channels: int = 32):
        super().__init__()
        self.vgg = VGGFeatureBlock()
        self.cnn = nn.Sequential(ConvBlock(19, channels), nn.Conv2d(channels, 3, 3, padding=1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wb = simple_white_balance(x)
        return self.cnn(torch.cat([wb, self.vgg(wb)], dim=1))


class GreenCastBranch(nn.Module):
    def __init__(self, channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(ConvBlock(3, channels), ConvBlock(channels, channels), nn.Conv2d(channels, 3, 3, padding=1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compensated = x.clone()
        compensated[:, 0:1] = (x[:, 0:1] + 0.15 * x[:, 1:2]).clamp(0, 1)
        compensated[:, 2:3] = (x[:, 2:3] + 0.08 * x[:, 1:2]).clamp(0, 1)
        return self.net(compensated)


class LowLightBranch(nn.Module):
    def __init__(self, channels: int = 32, gamma: float = 0.7):
        super().__init__()
        self.gamma = gamma
        self.net = nn.Sequential(ConvBlock(3, channels), ConvBlock(channels, channels), nn.Conv2d(channels, 3, 3, padding=1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(torch.pow(x.clamp(1e-6, 1), self.gamma))


class BlurBranch(nn.Module):
    def __init__(self, channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(ConvBlock(3, channels), ConvBlock(channels, channels), nn.Conv2d(channels, 3, 3, padding=1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def branch_from_name(name: str) -> nn.Module:
    key = name.lower()
    if key in {"blue", "blue_cast"}:
        return BlueCastBranch()
    if key in {"green", "green_cast"}:
        return GreenCastBranch()
    if key in {"lowlight", "low_light"}:
        return LowLightBranch()
    if key == "blur":
        return BlurBranch()
    raise ValueError(f"Unknown branch: {name}")

