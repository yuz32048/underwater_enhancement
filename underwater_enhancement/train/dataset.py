from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset

from ..utils.image_io import image_to_tensor, list_images, pil_loader


class UnpairedImageDataset(Dataset):
    def __init__(self, degraded_dir: str | Path, clean_dir: str | Path, image_size: int = 256):
        self.degraded = [p for p in list_images(degraded_dir) if "depth" not in p.parts and "comparisons" not in p.parts]
        self.clean = list_images(clean_dir)
        if not self.degraded:
            raise FileNotFoundError(f"No degraded images found in {degraded_dir}")
        if not self.clean:
            raise FileNotFoundError(f"No clean images found in {clean_dir}")
        self.image_size = image_size

    def __len__(self) -> int:
        return max(len(self.degraded), len(self.clean))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        a = pil_loader(self.degraded[idx % len(self.degraded)])
        b = pil_loader(self.clean[idx % len(self.clean)])
        a = a.resize((self.image_size, self.image_size))
        b = b.resize((self.image_size, self.image_size))
        return image_to_tensor(__import__("numpy").array(a)), image_to_tensor(__import__("numpy").array(b))
