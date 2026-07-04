from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from models.attention import (
    AttentionFusion,
    AverageFusion,
    ConcatFusion,
    ConcatSEFusion,
    GatedResidualFusion,
    ResidualConcatFusion,
    TemperatureAttentionFusion,
)
from models.branches import BlurBranch, BlueCastBranch, GreenCastBranch, LowLightBranch


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ResNetGenerator(nn.Module):
    def __init__(self, in_channels: int = 3, base: int = 64, num_blocks: int = 4):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, base, 7, padding=3),
            nn.InstanceNorm2d(base, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(base * 2, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(base * 4, affine=True),
            nn.ReLU(inplace=True),
        ]
        layers += [ResidualBlock(base * 4) for _ in range(num_blocks)]
        layers += [
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base * 4, base * 2, 3, padding=1),
            nn.InstanceNorm2d(base * 2, affine=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base * 2, base, 3, padding=1),
            nn.InstanceNorm2d(base, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, 3, 7, padding=3),
            nn.Sigmoid(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiBranchGenerator(nn.Module):
    def __init__(
        self,
        fusion: str = "attention",
        enabled_branches: list[str] | None = None,
        freeze_branches: bool = False,
        use_multibranch: bool = True,
    ):
        super().__init__()
        self.use_multibranch = use_multibranch
        if not use_multibranch:
            self.single = ResNetGenerator(base=32, num_blocks=3)
            self.last_attention: torch.Tensor | None = None
            return

        branch_map = {
            "blue": BlueCastBranch(),
            "green": GreenCastBranch(),
            "lowlight": LowLightBranch(),
            "low_light": LowLightBranch(),
            "blur": BlurBranch(),
        }
        enabled_branches = enabled_branches or ["blue", "green", "lowlight", "blur"]
        self.branch_names = [b for b in enabled_branches if b in branch_map]
        self.branches = nn.ModuleDict({name: branch_map[name] for name in self.branch_names})
        if freeze_branches:
            for p in self.branches.parameters():
                p.requires_grad = False
        if fusion == "concat":
            self.fusion = ConcatFusion(len(self.branch_names))
        elif fusion == "residual_concat":
            self.fusion = ResidualConcatFusion(len(self.branch_names))
        elif fusion == "gated_residual":
            self.fusion = GatedResidualFusion(len(self.branch_names))
        elif fusion == "concat_se":
            self.fusion = ConcatSEFusion(len(self.branch_names))
        elif fusion == "temperature_attention":
            self.fusion = TemperatureAttentionFusion(len(self.branch_names))
        elif fusion == "average":
            self.fusion = AverageFusion(len(self.branch_names))
        else:
            self.fusion = AttentionFusion(len(self.branch_names))
        self.refine = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True), ResidualBlock(32), nn.Conv2d(32, 3, 3, padding=1), nn.Sigmoid())
        self.last_attention: torch.Tensor | None = None

    def load_branch_weights(self, weight_dir: str | Path, strict: bool = False) -> None:
        if not self.use_multibranch:
            return
        names = {
            "blue": "blue_branch.pth",
            "green": "green_branch.pth",
            "lowlight": "lowlight_branch.pth",
            "low_light": "lowlight_branch.pth",
            "blur": "blur_branch.pth",
        }
        weight_dir = Path(weight_dir)
        for name, module in self.branches.items():
            path = weight_dir / names[name]
            if path.exists():
                state = torch.load(path, map_location="cpu")
                module.load_state_dict(state.get("model", state), strict=strict)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        if not self.use_multibranch:
            out = self.single(x)
            self.last_attention = None
            return (out, None) if return_attention else out
        outputs = [branch(x) for branch in self.branches.values()]
        fused, weights = self.fusion(x, outputs)
        self.last_attention = weights.detach()
        out = self.refine(fused).clamp(0, 1)
        return (out, weights) if return_attention else out

