from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn


def gray_world_white_balance(image_rgb: np.ndarray) -> np.ndarray:
    image = image_rgb.astype(np.float32)
    means = image.reshape(-1, 3).mean(axis=0)
    gray = float(means.mean())
    scale = gray / (means + 1e-6)
    return np.clip(image * scale.reshape(1, 1, 3), 0, 255).astype(np.uint8)


def lab_clahe(image_rgb: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    enhanced_l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([enhanced_l, a, b]), cv2.COLOR_LAB2RGB)


def gamma_correction(image_rgb: np.ndarray, gamma: float = 0.7) -> np.ndarray:
    image = image_rgb.astype(np.float32) / 255.0
    corrected = np.power(np.clip(image, 0.0, 1.0), gamma)
    return (corrected * 255.0).round().astype(np.uint8)


class WaterNet(nn.Module):
    """WaterNet-style confidence fusion over WB, CLAHE, and gamma corrected inputs."""

    def __init__(self, channels: int = 16):
        super().__init__()
        self.confidence = nn.Sequential(
            nn.Conv2d(9, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, wb: torch.Tensor, ce: torch.Tensor, gc: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([wb, ce, gc], dim=1)
        weights = self.confidence(inputs)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)
        stacked = torch.stack([wb, ce, gc], dim=1)
        return (stacked * weights.unsqueeze(2)).sum(dim=1).clamp(0.0, 1.0)
