import argparse
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..analysis.visualization import save_brightness_histogram, save_edge_visualization, save_rgb_histogram
from ..utils.image_io import list_images, load_config, read_image_rgb


def extract_features(image_rgb: np.ndarray, cfg: dict) -> Dict[str, float]:
    analysis_cfg = cfg.get("analysis", {})
    low_thr = int(analysis_cfg.get("low_light_pixel_threshold", 50))
    c1 = int(analysis_cfg.get("canny_threshold1", 80))
    c2 = int(analysis_cfg.get("canny_threshold2", 180))

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    edges = cv2.Canny(gray, c1, c2)

    r, g, b = [image_rgb[:, :, i].astype(np.float32) for i in range(3)]
    l, a, lab_b = [lab[:, :, i].astype(np.float32) for i in range(3)]
    v = hsv[:, :, 2].astype(np.float32)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edge_count = int(np.count_nonzero(edges))
    total = gray.size
    brightness_mean = float(v.mean())
    rgb_means = np.array([r.mean(), g.mean(), b.mean()])
    color_shift = float(np.std(rgb_means) + abs(float(a.mean()) - 128.0) + abs(float(lab_b.mean()) - 128.0))

    return {
        "height": int(image_rgb.shape[0]),
        "width": int(image_rgb.shape[1]),
        "red_mean": float(r.mean()),
        "green_mean": float(g.mean()),
        "blue_mean": float(b.mean()),
        "red_std": float(r.std()),
        "green_std": float(g.std()),
        "blue_std": float(b.std()),
        "lab_l_mean": float(l.mean()),
        "lab_l_std": float(l.std()),
        "lab_a_mean": float(a.mean()),
        "lab_a_std": float(a.std()),
        "lab_b_mean": float(lab_b.mean()),
        "lab_b_std": float(lab_b.std()),
        "hsv_v_mean": brightness_mean,
        "hsv_v_std": float(v.std()),
        "low_light_ratio": float((v < low_thr).sum() / total),
        "canny_edge_count": edge_count,
        "edge_density": float(edge_count / total),
        "laplacian_var": lap_var,
        "sharpness": lap_var,
        "brightness": brightness_mean,
        "color_shift": color_shift,
        "blue_red_diff": float(b.mean() - r.mean()),
        "green_red_diff": float(g.mean() - r.mean()),
    }


def analyze_folder(cfg: dict, root: str | Path = ".") -> pd.DataFrame:
    root = Path(root)
    in_dir = root / cfg["paths"]["input_dir"]
    out_dir = root / cfg["paths"]["analysis_dir"]
    hist_dir = out_dir / "rgb_histograms"
    bright_dir = out_dir / "brightness_histograms"
    edge_dir = out_dir / "edges"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    bins = int(cfg.get("analysis", {}).get("histogram_bins", 256))
    images = list_images(in_dir)
    for path in tqdm(images, desc="Analyzing images"):
        image = read_image_rgb(path)
        feats = extract_features(image, cfg)
        rel = path.relative_to(in_dir).as_posix()
        feats["image_path"] = rel
        rows.append(feats)
        stem = path.stem
        save_rgb_histogram(image, hist_dir / f"{stem}_rgb_hist.png", bins)
        save_brightness_histogram(image, bright_dir / f"{stem}_brightness_hist.png", bins)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, int(cfg["analysis"].get("canny_threshold1", 80)), int(cfg["analysis"].get("canny_threshold2", 180)))
        save_edge_visualization(edges, edge_dir / f"{stem}_edges.png")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "image_features.csv", index=False)
    if not df.empty:
        df.to_excel(out_dir / "image_features.xlsx", index=False)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config.yaml")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    analyze_folder(load_config(config_path), config_path.parent)
