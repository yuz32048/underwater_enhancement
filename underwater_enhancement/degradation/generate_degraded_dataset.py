import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from degradation.jaffe_mcglamery import apply_jaffe_mcglamery
from utils.archive_extraction import prepare_archives
from utils.image_io import list_images, load_config, read_image_rgb, resize_rgb, save_comparison, save_image_rgb


def generate_dataset(cfg: dict, root: str | Path = ".") -> pd.DataFrame:
    root = Path(root)
    in_dir = root / cfg["paths"]["input_dir"]
    out_root = root / cfg["paths"]["degraded_dir"]
    prepare_archives(in_dir, cfg)
    deg_cfg = cfg.get("degradation", {})
    types = deg_cfg.get("types", ["blue_shift", "green_shift", "low_light", "blur"])
    num_per_image = int(deg_cfg.get("num_per_image", 2))
    image_size = int(deg_cfg.get("image_size", 256))
    rng = __import__("numpy").random.default_rng(int(cfg.get("seed", 42)))

    rows: List[Dict] = []
    images = list_images(in_dir)
    for src in tqdm(images, desc="Generating degraded images"):
        image = resize_rgb(read_image_rgb(src), image_size)
        for deg_type in types:
            for idx in range(num_per_image):
                degraded, depth, params = apply_jaffe_mcglamery(image, deg_type, deg_cfg, rng)
                rel_stem = f"{src.stem}_{idx:03d}"
                out_path = out_root / deg_type / f"{rel_stem}.png"
                depth_path = out_root / deg_type / "depth" / f"{rel_stem}_depth.png"
                cmp_path = out_root / deg_type / "comparisons" / f"{rel_stem}_comparison.png"
                save_image_rgb(out_path, degraded)
                save_image_rgb(depth_path, __import__("numpy").stack([depth] * 3, axis=-1))
                save_comparison(cmp_path, [image, degraded], ["before", "after"])
                row = {"source_image": src.relative_to(in_dir).as_posix(), "output_image": out_path.relative_to(out_root).as_posix(), "depth_image": depth_path.relative_to(out_root).as_posix()}
                row.update(params)
                rows.append(row)

    df = pd.DataFrame(rows)
    out_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_root / "degradation_params.csv", index=False)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config.yaml")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    generate_dataset(load_config(config_path), config_path.parent)
