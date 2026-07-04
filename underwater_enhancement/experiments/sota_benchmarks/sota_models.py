from __future__ import annotations

import torch
from torch import nn

from models.cyclegan import CycleGAN
from models.waternet import WaterNet


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: bool = True, activation: str = "relu", stride: int = 1):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_ch, out_ch, 4 if stride == 2 else 3, stride=stride, padding=1)]
        if norm:
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        if activation == "leaky":
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        else:
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: bool = False):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetGenerator(nn.Module):
    """Pix2Pix/UGAN-style U-Net generator."""

    def __init__(self, base: int = 32):
        super().__init__()
        self.d1 = ConvBlock(3, base, norm=False, activation="leaky", stride=2)
        self.d2 = ConvBlock(base, base * 2, activation="leaky", stride=2)
        self.d3 = ConvBlock(base * 2, base * 4, activation="leaky", stride=2)
        self.d4 = ConvBlock(base * 4, base * 8, activation="leaky", stride=2)
        self.bottleneck = ConvBlock(base * 8, base * 8, activation="relu", stride=2)
        self.u4 = UpBlock(base * 8, base * 8, dropout=True)
        self.u3 = UpBlock(base * 16, base * 4, dropout=True)
        self.u2 = UpBlock(base * 8, base * 2)
        self.u1 = UpBlock(base * 4, base)
        self.out = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base * 2, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        b = self.bottleneck(d4)
        u4 = self.u4(b)
        u3 = self.u3(torch.cat([u4, d4], dim=1))
        u2 = self.u2(torch.cat([u3, d3], dim=1))
        u1 = self.u1(torch.cat([u2, d2], dim=1))
        return self.out(torch.cat([u1, d1], dim=1))


class FunieGenerator(nn.Module):
    """Compact FUnIE-GAN-style encoder-decoder with skip fusion."""

    def __init__(self, base: int = 32):
        super().__init__()
        self.e1 = ConvBlock(3, base, norm=False, activation="leaky", stride=2)
        self.e2 = ConvBlock(base, base * 2, activation="leaky", stride=2)
        self.e3 = ConvBlock(base * 2, base * 4, activation="leaky", stride=2)
        self.e4 = ConvBlock(base * 4, base * 8, activation="leaky", stride=2)
        self.d3 = UpBlock(base * 8, base * 4)
        self.d2 = UpBlock(base * 8, base * 2)
        self.d1 = UpBlock(base * 4, base)
        self.out = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base * 2, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        d3 = self.d3(e4)
        d2 = self.d2(torch.cat([d3, e3], dim=1))
        d1 = self.d1(torch.cat([d2, e2], dim=1))
        return self.out(torch.cat([d1, e1], dim=1))


class ConditionalPatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 6, base: int = 64):
        super().__init__()

        def block(in_ch: int, out_ch: int, norm: bool = True, stride: int = 2) -> list[nn.Module]:
            layers: list[nn.Module] = [nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_ch, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *block(in_channels, base, norm=False),
            *block(base, base * 2),
            *block(base * 2, base * 4),
            *block(base * 4, base * 8, stride=1),
            nn.Conv2d(base * 8, 1, 4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, y], dim=1))


class UWCNN(nn.Module):
    """UWCNN-style residual CNN for direct paired reconstruction."""

    def __init__(self, channels: int = 64, blocks: int = 7):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(3, channels, 3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(blocks):
            layers.extend([
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.ReLU(inplace=True),
            ])
        layers.append(nn.Conv2d(channels, 3, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x) + x)


def build_model(name: str) -> nn.Module:
    key = name.lower().replace("_", "-")
    if key == "cyclegan":
        return CycleGAN(fusion="average", use_multibranch=False)
    if key == "ugan":
        return nn.ModuleDict({"G": UNetGenerator(), "D": ConditionalPatchDiscriminator()})
    if key == "funie-gan":
        return nn.ModuleDict({"G": FunieGenerator(), "D": ConditionalPatchDiscriminator()})
    if key == "uwcnn":
        return UWCNN()
    if key == "waternet":
        return WaterNet()
    raise ValueError(f"Unsupported model: {name}")


def generator_for(model: nn.Module, name: str) -> nn.Module:
    key = name.lower().replace("_", "-")
    if key == "cyclegan":
        return model.G_AB  # type: ignore[attr-defined]
    if key in {"ugan", "funie-gan"}:
        return model["G"]  # type: ignore[index]
    return model
