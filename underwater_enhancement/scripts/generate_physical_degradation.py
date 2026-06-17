from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.image_io import list_images, read_image_rgb, save_image_rgb


def pseudo_depth(shape: tuple[int, int], depth_min: float, depth_max: float) -> np.ndarray:
    h, w = shape
    y = np.linspace(depth_min, depth_max, h, dtype=np.float32)[:, None]
    depth = np.repeat(y, w, axis=1)
    return cv2.GaussianBlur(depth, (21, 21), 0)


def jaffe_mcglamery(image_rgb: np.ndarray, beta: tuple[float, float, float], background: tuple[float, float, float], depth: np.ndarray) -> np.ndarray:
    j = image_rgb.astype(np.float32) / 255.0
    beta_arr = np.array(beta, dtype=np.float32).reshape(1, 1, 3)
    bg = np.array(background, dtype=np.float32).reshape(1, 1, 3)
    t = np.exp(-beta_arr * depth[:, :, None])
    return np.clip(j * t + bg * (1.0 - t), 0, 1)


def degrade(image_rgb: np.ndarray, kind: str, args: argparse.Namespace) -> np.ndarray:
    depth = pseudo_depth(image_rgb.shape[:2], args.depth_min, args.depth_max)
    if kind == "blue_cast":
        out = jaffe_mcglamery(image_rgb, (args.blue_beta_r, args.blue_beta_g, args.blue_beta_b), (0.22, 0.50, args.blue_background_b), depth)
    elif kind == "green_cast":
        out = jaffe_mcglamery(image_rgb, (args.green_beta_r, args.green_beta_g, args.green_beta_b), (0.22, args.green_background_g, 0.36), depth)
    elif kind == "low_light":
        out = jaffe_mcglamery(image_rgb, (args.low_beta, args.low_beta, args.low_beta), (0.05, 0.07, 0.08), depth)
        out = np.power(out, args.low_gamma) * args.low_scale
    elif kind == "blur":
        out = image_rgb.astype(np.float32) / 255.0
        k = args.blur_kernel if args.blur_kernel % 2 == 1 else args.blur_kernel + 1
        out = cv2.GaussianBlur(out, (k, k), args.blur_sigma)
    else:
        raise ValueError(kind)
    return (np.clip(out, 0, 1) * 255).round().astype(np.uint8)


def run(args: argparse.Namespace) -> pd.DataFrame:
    reference_dir = Path(args.input_dir)
    output_root = Path(args.output_dir)
    mapping_path = Path(args.mapping_csv)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    kinds = ["blue_cast", "green_cast", "low_light", "blur"]
    for kind in kinds:
        (output_root / kind).mkdir(parents=True, exist_ok=True)

    rows = []
    suffix = {"blue_cast": "blue", "green_cast": "green", "low_light": "lowlight", "blur": "blur"}
    for ref in tqdm(list_images(reference_dir), desc="Generating physical degradation"):
        image = read_image_rgb(ref)
        for kind in kinds:
            out = degrade(image, kind, args)
            degraded_name = f"{ref.stem}_{suffix[kind]}.png"
            out_path = output_root / kind / degraded_name
            save_image_rgb(out_path, out)
            rows.append({
                "degraded_image": out_path.as_posix(),
                "degradation_type": kind,
                "reference_image": ref.name,
                "source_reference_path": ref.as_posix(),
            })
    df = pd.DataFrame(rows)
    df.to_csv(mapping_path, index=False)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Jaffe-McGlamery physical degradations from UIEB reference-890.")
    parser.add_argument("--input-dir", default="data/raw_underwater/UIEB/reference-890")
    parser.add_argument("--output-dir", default="data/processed/physical_degradation")
    parser.add_argument("--mapping-csv", default="results/physical_degradation_mapping.csv")
    parser.add_argument("--depth-min", type=float, default=0.25)
    parser.add_argument("--depth-max", type=float, default=1.1)
    parser.add_argument("--blue-beta-r", type=float, default=1.45)
    parser.add_argument("--blue-beta-g", type=float, default=0.85)
    parser.add_argument("--blue-beta-b", type=float, default=0.38)
    parser.add_argument("--blue-background-b", type=float, default=0.95)
    parser.add_argument("--green-beta-r", type=float, default=1.25)
    parser.add_argument("--green-beta-g", type=float, default=0.48)
    parser.add_argument("--green-beta-b", type=float, default=0.92)
    parser.add_argument("--green-background-g", type=float, default=0.92)
    parser.add_argument("--low-beta", type=float, default=1.15)
    parser.add_argument("--low-gamma", type=float, default=1.8)
    parser.add_argument("--low-scale", type=float, default=0.72)
    parser.add_argument("--blur-kernel", type=int, default=7)
    parser.add_argument("--blur-sigma", type=float, default=1.8)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

