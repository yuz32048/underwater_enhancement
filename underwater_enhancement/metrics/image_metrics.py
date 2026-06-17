from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(target, pred, data_range=255))


def ssim(pred: np.ndarray, target: np.ndarray) -> float:
    return float(structural_similarity(target, pred, channel_axis=2, data_range=255))


def uciqe(image_rgb: np.ndarray) -> float:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l = lab[:, :, 0] / 255.0
    a = lab[:, :, 1] - 128.0
    b = lab[:, :, 2] - 128.0
    chroma = np.sqrt(a * a + b * b)
    sigma_c = float(np.std(chroma))
    con_l = float(np.percentile(l, 99) - np.percentile(l, 1))
    sat = chroma / np.sqrt(chroma * chroma + l * l + 1e-8)
    return float(0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * np.mean(sat))


def _eme(gray: np.ndarray, block: int = 8) -> float:
    gray = gray.astype(np.float32) + 1.0
    vals = []
    h, w = gray.shape
    for y in range(0, h - block + 1, block):
        for x in range(0, w - block + 1, block):
            patch = gray[y:y + block, x:x + block]
            vals.append(np.log(patch.max() / patch.min()))
    return float(2.0 * np.mean(vals)) if vals else 0.0


def uiqm(image_rgb: np.ndarray) -> float:
    img = image_rgb.astype(np.float32)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    uicm = -0.0268 * np.hypot(np.mean(rg), np.mean(yb)) + 0.1586 * np.sqrt(np.var(rg) + np.var(yb))
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    uism = _eme(gray)
    uiconm = float(np.std(gray) / 255.0)
    return float(0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm)


def calculate_metrics(pred: np.ndarray, target: np.ndarray | None = None) -> dict[str, float | str]:
    row: dict[str, float | str] = {"UIQM": uiqm(pred), "UCIQE": uciqe(pred)}
    if target is None:
        row["PSNR"] = ""
        row["SSIM"] = ""
    else:
        if target.shape[:2] != pred.shape[:2]:
            target = cv2.resize(target, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_AREA)
        row["PSNR"] = psnr(pred, target)
        row["SSIM"] = ssim(pred, target)
    return row

