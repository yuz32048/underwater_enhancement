from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.classify_uieb import classify_image, extract_degradation_features
from utils.image_io import list_images, read_image_rgb


def _parse_values(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _score(row: dict[str, float], target: float) -> float:
    counts = [row["blue_cast"], row["green_cast"], row["low_light"], row["blur"]]
    imbalance = max(counts) - min(counts)
    target_error = sum(abs(count - target) for count in counts) / len(counts)
    return imbalance + 0.5 * target_error


def run(args: argparse.Namespace) -> None:
    raw_dir = Path(args.raw_dir)
    images = list_images(raw_dir)
    if not images:
        raise FileNotFoundError(f"No images found: {raw_dir}")

    feature_rows = []
    for path in tqdm(images, desc="Extracting train features"):
        feature_rows.append({
            "path": path,
            "features": extract_degradation_features(read_image_rgb(path), args.canny1, args.canny2),
        })

    blue_values = _parse_values(args.blue_b_values)
    green_a_values = _parse_values(args.green_a_values)
    green_b_values = _parse_values(args.green_b_values)
    low_values = _parse_values(args.low_light_v_values)
    blur_lap_values = _parse_values(args.blur_laplacian_values)
    blur_edge_values = _parse_values(args.blur_edge_values)
    target = args.target_count if args.target_count > 0 else len(images) * args.target_ratio

    rows = []
    grid = itertools.product(blue_values, green_a_values, green_b_values, low_values, blur_lap_values, blur_edge_values)
    for blue_b, green_a, green_b, low_v, blur_lap, blur_edge in tqdm(list(grid), desc="Sweeping thresholds"):
        threshold_args = argparse.Namespace(
            blue_b_threshold=blue_b,
            green_a_threshold=green_a,
            green_b_threshold=green_b,
            low_light_v_threshold=low_v,
            blur_laplacian_threshold=blur_lap,
            blur_edge_threshold=blur_edge,
        )
        counts = {"blue_cast": 0, "green_cast": 0, "low_light": 0, "blur": 0}
        for item in feature_rows:
            labels = classify_image(item["features"], threshold_args)
            for name, active in labels.items():
                counts[name] += int(active)
        row = {
            "blue_b_threshold": blue_b,
            "green_a_threshold": green_a,
            "green_b_threshold": green_b,
            "low_light_v_threshold": low_v,
            "blur_laplacian_threshold": blur_lap,
            "blur_edge_threshold": blur_edge,
            **counts,
        }
        row["min_count"] = min(counts.values())
        row["max_count"] = max(counts.values())
        row["imbalance"] = row["max_count"] - row["min_count"]
        row["score"] = _score(row, target)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["score", "imbalance"], ascending=True)
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[sweep] wrote {out}")
    print(df.head(args.topk).to_string(index=False))


def parse_args() -> argparse.Namespace:
    default_workdir = Path(__file__).resolve().parent / "workdir"
    parser = argparse.ArgumentParser(description="Sweep UIEB train classification thresholds without copying files.")
    parser.add_argument("--raw-dir", default=str("/root/autodl-tmp/experiments/three_stage_uieb/workdir/v3/splits/train/raw"))
    parser.add_argument("--output-csv", default=str(default_workdir / "logs/classification_threshold_sweep.csv"))
    parser.add_argument("--target-ratio", type=float, default=0.35)
    parser.add_argument("--target-count", type=float, default=0)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--canny1", type=int, default=80)
    parser.add_argument("--canny2", type=int, default=180)
    parser.add_argument("--blue-b-values", default="-10,-8,-6,-4")
    parser.add_argument("--green-a-values", default="-2,-1,0,1")
    parser.add_argument("--green-b-values", default="0,1,2")
    parser.add_argument("--low-light-v-values", default="110,120,130,140,150")
    parser.add_argument("--blur-laplacian-values", default="80,100,120")
    parser.add_argument("--blur-edge-values", default="0.025,0.03,0.035")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
