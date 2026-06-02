from torch import nn

from models.discriminator import PatchDiscriminator
from models.generator import MultiBranchGenerator


class CycleGAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.G_AB = MultiBranchGenerator()
        self.G_BA = MultiBranchGenerator()
        self.D_A = PatchDiscriminator()
        self.D_B = PatchDiscriminator()
