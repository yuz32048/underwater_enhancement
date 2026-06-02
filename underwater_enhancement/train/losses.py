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
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2) + 1e-8)
        return 1.0 - ssim.mean()


def gan_loss(pred: torch.Tensor, target_is_real: bool, criterion: nn.Module) -> torch.Tensor:
    target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
    return criterion(pred, target)
