from torch import nn


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 3, base: int = 64):
        super().__init__()

        def block(in_ch, out_ch, norm=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_ch, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *block(in_channels, base, norm=False),
            *block(base, base * 2),
            *block(base * 2, base * 4),
            nn.Conv2d(base * 4, base * 8, 4, stride=1, padding=1),
            nn.InstanceNorm2d(base * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 8, 1, 4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.net(x)

