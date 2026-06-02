import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from analysis.feature_extraction import analyze_folder
from utils.image_io import load_config

CLASSES = ["color_distortion_blue", "color_distortion_green", "low_light", "blurry", "normal"]


def classify_row(row: pd.Series, thresholds: dict) -> Tuple[List[str], List[str]]:
    labels: List[str] = []
    reasons: List[str] = []

    if row["blue_mean"] >= thresholds.get("blue_mean_threshold", 115) and row["blue_red_diff"] >= thresholds.get("blue_red_diff_threshold", 18):
        labels.append("color_distortion_blue")
        reasons.append(f"blue_mean={row['blue_mean']:.2f}, blue_red_diff={row['blue_red_diff']:.2f}")
    if row["green_mean"] >= thresholds.get("green_mean_threshold", 105) and row["green_red_diff"] >= thresholds.get("green_red_diff_threshold", 15):
        labels.append("color_distortion_green")
        reasons.append(f"green_mean={row['green_mean']:.2f}, green_red_diff={row['green_red_diff']:.2f}")
    if row["brightness"] < thresholds.get("brightness_threshold", 75) or row["low_light_ratio"] > thresholds.get("low_light_ratio_threshold", 0.35):
        labels.append("low_light")
        reasons.append(f"brightness={row['brightness']:.2f}, low_light_ratio={row['low_light_ratio']:.3f}")
    if row["laplacian_var"] < thresholds.get("laplacian_var_threshold", 80) or row["edge_density"] < thresholds.get("blurry_edge_density_threshold", 0.025):
        labels.append("blurry")
        reasons.append(f"laplacian_var={row['laplacian_var']:.2f}, edge_density={row['edge_density']:.4f}")
    if not labels:
        labels = ["normal"]
        reasons = ["all configured thresholds within normal range"]
    return labels, reasons


def classify_folder(cfg: dict, root: str | Path = ".") -> pd.DataFrame:
    root = Path(root)
    in_dir = root / cfg["paths"]["input_dir"]
    analysis_dir = root / cfg["paths"]["analysis_dir"]
    feature_csv = analysis_dir / "image_features.csv"
    if feature_csv.exists():
        df = pd.read_csv(feature_csv)
    else:
        df = analyze_folder(cfg, root)

    out_dir = root / cfg["paths"]["classified_dir"]
    for name in CLASSES:
        (out_dir / name).mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    thresholds = cfg.get("classification", {})
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Classifying images"):
        labels, reasons = classify_row(row, thresholds)
        src = in_dir / row["image_path"]
        for label in labels:
            dst = out_dir / label / Path(row["image_path"]).name
            if src.exists():
                shutil.copy2(src, dst)
        rows.append({"image_path": row["image_path"], "labels": ";".join(labels), "reasons": " | ".join(reasons)})

    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "classification_results.csv", index=False)
    if not result.empty:
        result.to_excel(out_dir / "classification_results.xlsx", index=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config.yaml")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    classify_folder(load_config(config_path), config_path.parent)
