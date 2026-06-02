import torch
from torch import nn

from models.branches import BlueCastBranch, BlurBranch, GreenCastBranch, LowLightBranch


class AttentionFusion(nn.Module):
    def __init__(self, num_branches: int = 4):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(num_branches * 3 + 3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_branches, 1),
        )

    def forward(self, x: torch.Tensor, branch_outputs: list[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(branch_outputs, dim=1)
        attn_in = torch.cat([x] + branch_outputs, dim=1)
        weights = torch.softmax(self.attn(attn_in), dim=1).unsqueeze(2)
        return (stacked * weights).sum(dim=1).clamp(0, 1)


class MultiBranchGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.blue_branch = BlueCastBranch()
        self.green_branch = GreenCastBranch()
        self.low_light_branch = LowLightBranch()
        self.blur_branch = BlurBranch()
        self.fusion = AttentionFusion(4)
        self.refine = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [
            self.blue_branch(x),
            self.green_branch(x),
            self.low_light_branch(x),
            self.blur_branch(x),
        ]
        fused = self.fusion(x, outputs)
        return self.refine(fused)
