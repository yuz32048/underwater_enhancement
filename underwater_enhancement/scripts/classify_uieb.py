from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.image_io import list_images, read_image_rgb


def extract_degradation_features(image_rgb: np.ndarray, canny1: int = 80, canny2: int = 180) -> dict[str, float]:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, canny1, canny2)
    a = lab[:, :, 1] - 128.0
    b = lab[:, :, 2] - 128.0
    return {
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "color_variance": float(np.var(a) + np.var(b)),
        "mean_v": float(hsv[:, :, 2].mean()),
        "laplacian_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "edge_density": float(np.count_nonzero(edges) / gray.size),
    }


def classify_image(features: dict[str, float], args: argparse.Namespace) -> dict[str, bool]:
    return {
        "blue_cast": features["mean_b"] <= args.blue_b_threshold,
        "green_cast": features["mean_a"] <= args.green_a_threshold and features["mean_b"] >= args.green_b_threshold,
        "low_light": features["mean_v"] <= args.low_light_v_threshold,
        "blur": features["laplacian_var"] <= args.blur_laplacian_threshold or features["edge_density"] <= args.blur_edge_threshold,
    }


def run(args: argparse.Namespace) -> pd.DataFrame:
    input_dir = Path(args.input_dir)
    output_root = Path(args.output_dir)
    csv_path = Path(args.csv_path)
    classes = ["blue_cast", "green_cast", "low_light", "blur"]
    for cls in classes:
        (output_root / cls).mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in tqdm(list_images(input_dir), desc="Classifying UIEB raw-890"):
        image = read_image_rgb(path)
        features = extract_degradation_features(image, args.canny1, args.canny2)
        labels = classify_image(features, args)
        for cls, active in labels.items():
            if active:
                shutil.copy2(path, output_root / cls / path.name)
        rows.append({"image_name": path.name, **{k: int(v) for k, v in labels.items()}, **features})

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    return df


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数，配置水下图像退化分类的各项阈值与路径。
    
    Args:
        无（参数通过命令行传入）
    
    Returns:
        argparse.Namespace: 包含所有解析后的命令行参数的对象，包括输入/输出目录、
            CSV路径以及各类退化检测阈值（蓝色通道b阈值、绿色通道a/b阈值、
            低光V阈值、模糊Laplacian阈值、模糊边缘阈值、Canny边缘检测参数等）
    """
    parser = argparse.ArgumentParser(description="Classify UIEB raw-890 into degradation labels.")
    parser.add_argument("--input-dir", default="data/raw_underwater/UIEB/raw-890")
    parser.add_argument("--output-dir", default="data/processed/UIEB_classified")
    parser.add_argument("--csv-path", default="results/classification_result.csv")
    parser.add_argument("--blue-b-threshold", type=float, default=-4.0)
    parser.add_argument("--green-a-threshold", type=float, default=-2.0)
    parser.add_argument("--green-b-threshold", type=float, default=2.0)
    parser.add_argument("--low-light-v-threshold", type=float, default=85.0)
    parser.add_argument("--blur-laplacian-threshold", type=float, default=80.0)
    parser.add_argument("--blur-edge-threshold", type=float, default=0.025)
    parser.add_argument("--canny1", type=int, default=80)
    parser.add_argument("--canny2", type=int, default=180)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

