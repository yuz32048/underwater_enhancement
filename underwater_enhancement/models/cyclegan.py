from torch import nn

from models.discriminator import PatchDiscriminator
from models.generator import MultiBranchGenerator, ResNetGenerator


class CycleGAN(nn.Module):
    def __init__(self, fusion: str = "attention", enabled_branches: list[str] | None = None, freeze_branches: bool = False, use_multibranch: bool = True):
        super().__init__()
        self.G_AB = MultiBranchGenerator(fusion=fusion, enabled_branches=enabled_branches, freeze_branches=freeze_branches, use_multibranch=use_multibranch)
        self.G_BA = ResNetGenerator(base=32, num_blocks=3)
        self.D_A = PatchDiscriminator()
        self.D_B = PatchDiscriminator()
