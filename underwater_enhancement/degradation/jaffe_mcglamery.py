from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np


@dataclass
class DegradationParams:
    degradation_type: str
    beta_r: float
    beta_g: float
    beta_b: float
    background_r: float
    background_g: float
    background_b: float
    depth_min: float
    depth_max: float
    blur_kernel: int = 0


def generate_pseudo_depth(shape: Tuple[int, int], depth_range: Tuple[float, float], rng: np.random.Generator) -> np.ndarray:
    h, w = shape
    y = np.linspace(depth_range[0], depth_range[1], h, dtype=np.float32)[:, None]
    noise = rng.normal(0, 0.04, (h, w)).astype(np.float32)
    depth = np.repeat(y, w, axis=1) + noise
    depth = cv2.GaussianBlur(depth, (21, 21), 0)
    return np.clip(depth, depth_range[0], depth_range[1])


def _sample_params(degradation_type: str, cfg: dict, rng: np.random.Generator) -> DegradationParams:
    beta_lo, beta_hi = cfg.get("beta_range", [0.35, 1.8])
    bg_lo, bg_hi = cfg.get("background_light_range", [0.55, 1.0])
    kernels = cfg.get("blur_kernel_choices", [3, 5, 7])
    beta = rng.uniform(beta_lo, beta_hi, 3)
    background = rng.uniform(bg_lo, bg_hi, 3)

    if degradation_type == "blue_shift":
        beta = np.array([1.4, 0.85, 0.45]) * rng.uniform(0.8, 1.25)
        background = np.array([0.25, 0.55, 0.95]) * rng.uniform(0.85, 1.1)
    elif degradation_type == "green_shift":
        beta = np.array([1.25, 0.55, 0.85]) * rng.uniform(0.8, 1.25)
        background = np.array([0.25, 0.9, 0.45]) * rng.uniform(0.85, 1.1)
    elif degradation_type == "low_light":
        beta = beta * rng.uniform(0.9, 1.5)
        background = background * rng.uniform(0.25, 0.55)
    elif degradation_type == "blur":
        beta = beta * rng.uniform(0.55, 1.0)

    k = int(rng.choice(kernels)) if degradation_type == "blur" else 0
    if k % 2 == 0:
        k += 1
    return DegradationParams(
        degradation_type=degradation_type,
        beta_r=float(beta[0]),
        beta_g=float(beta[1]),
        beta_b=float(beta[2]),
        background_r=float(np.clip(background[0], 0, 1)),
        background_g=float(np.clip(background[1], 0, 1)),
        background_b=float(np.clip(background[2], 0, 1)),
        depth_min=float(cfg.get("depth_range", [0.2, 1.0])[0]),
        depth_max=float(cfg.get("depth_range", [0.2, 1.0])[1]),
        blur_kernel=k,
    )


def apply_jaffe_mcglamery(image_rgb: np.ndarray, degradation_type: str, cfg: dict, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, Dict]:
    img = image_rgb.astype(np.float32) / 255.0
    params = _sample_params(degradation_type, cfg, rng)
    depth = generate_pseudo_depth(img.shape[:2], (params.depth_min, params.depth_max), rng)
    beta = np.array([params.beta_r, params.beta_g, params.beta_b], dtype=np.float32).reshape(1, 1, 3)
    background = np.array([params.background_r, params.background_g, params.background_b], dtype=np.float32).reshape(1, 1, 3)
    transmission = np.exp(-beta * depth[:, :, None])
    degraded = img * transmission + background * (1.0 - transmission)
    if degradation_type == "low_light":
        degraded = np.power(np.clip(degraded, 0, 1), rng.uniform(1.35, 2.4)) * rng.uniform(0.65, 0.9)
    if degradation_type == "blur":
        degraded = cv2.GaussianBlur(degraded, (params.blur_kernel, params.blur_kernel), 0)
    degraded_u8 = (np.clip(degraded, 0, 1) * 255).round().astype(np.uint8)
    depth_u8 = ((depth - depth.min()) / max(depth.max() - depth.min(), 1e-6) * 255).round().astype(np.uint8)
    return degraded_u8, depth_u8, params.__dict__
