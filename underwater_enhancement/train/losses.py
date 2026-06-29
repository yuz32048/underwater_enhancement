import torch
from torch import nn


class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size
        self.avg = nn.AvgPool2d(window_size, stride=1, padding=window_size // 2)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        mu_x = self.avg(x)
        mu_y = self.avg(y)
        sigma_x = self.avg(x * x) - mu_x * mu_x
        sigma_y = self.avg(y * y) - mu_y * mu_y
        sigma_xy = self.avg(x * y) - mu_x * mu_y
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
            (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2) + 1e-8
        )
        return 1.0 - ssim.mean()


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()

        from torchvision import models

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES)
        self.features = vgg.features[:16].eval()

        for param in self.features.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss()

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x/y 默认是 [-1, 1]，先转成 [0, 1]
        x = (x + 1.0) / 2.0
        y = (y + 1.0) / 2.0

        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        x_feat = self.features(x)
        y_feat = self.features(y)

        return self.criterion(x_feat, y_feat)


def gan_loss(pred: torch.Tensor, target_is_real: bool, criterion: nn.Module) -> torch.Tensor:
    target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
    return criterion(pred, target)
