from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np
import torch
from PIL import Image

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(root: str | Path) -> List[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS])


def read_image_rgb(path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_image_rgb(path: str | Path, image_rgb: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(image_rgb, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


def resize_rgb(image_rgb: np.ndarray, size: int | tuple[int, int]) -> np.ndarray:
    if isinstance(size, int):
        size = (size, size)
    return cv2.resize(image_rgb, size, interpolation=cv2.INTER_AREA)


def image_to_tensor(image_rgb: np.ndarray) -> torch.Tensor:
    arr = image_rgb.astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    t = tensor.detach().cpu().clamp(0, 1)
    if t.ndim == 4:
        t = t[0]
    arr = t.permute(1, 2, 0).numpy() * 255.0
    return arr.round().astype(np.uint8)


def load_config(path: str | Path) -> dict:
    import yaml

    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_comparison(path: str | Path, images: Iterable[np.ndarray], labels: Iterable[str] | None = None) -> None:
    imgs = [np.clip(im, 0, 255).astype(np.uint8) for im in images]
    canvas = np.concatenate(imgs, axis=1)
    canvas = np.ascontiguousarray(canvas)
    if labels:
        import cv2

        h = canvas.shape[0]
        for i, label in enumerate(labels):
            x = i * imgs[0].shape[1] + 8
            cv2.putText(canvas, str(label), (x, min(28, h - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    save_image_rgb(path, canvas)


def pil_loader(path: str | Path, size: int | None = None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize((size, size), Image.BICUBIC)
    return img
