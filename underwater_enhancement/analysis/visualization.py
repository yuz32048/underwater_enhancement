from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from ..utils.image_io import save_image_rgb


def save_rgb_histogram(image_rgb: np.ndarray, out_path: str | Path, bins: int = 256) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    colors = {"R": "red", "G": "green", "B": "blue"}
    plt.figure(figsize=(8, 4))
    for idx, name in enumerate(["R", "G", "B"]):
        hist = cv2.calcHist([image_rgb], [idx], None, [bins], [0, 256])
        plt.plot(hist, color=colors[name], label=name)
    plt.xlim([0, 255])
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def save_brightness_histogram(image_rgb: np.ndarray, out_path: str | Path, bins: int = 256) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2]
    plt.figure(figsize=(8, 4))
    plt.hist(v.ravel(), bins=bins, range=(0, 255), color="gray")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def save_edge_visualization(edges: np.ndarray, out_path: str | Path) -> None:
    edge_rgb = np.stack([edges] * 3, axis=-1)
    save_image_rgb(out_path, edge_rgb)
