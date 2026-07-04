from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import build_uieb_pairs
from datasets.image_datasets import save_average_metrics
from metrics import calculate_metrics
from models.cyclegan import CycleGAN
from utils.image_io import image_to_tensor, pil_loader, read_image_rgb, save_comparison, save_image_rgb, tensor_to_image


def _device(name: str) -> torch.device:
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else ("cpu" if name == "auto" else name))


def _load_model(checkpoint: str | Path, device: torch.device) -> CycleGAN:
    ckpt = torch.load(checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})
    enabled_raw = ckpt_args.get("enabled_branches", "")
    enabled_branches = [item.strip() for item in enabled_raw.split(",") if item.strip()] if enabled_raw else None
    model = CycleGAN(
        fusion=ckpt_args.get("fusion", "attention"),
        enabled_branches=enabled_branches,
        use_multibranch=not ckpt_args.get("single_generator", False),
    ).to(device)
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model.eval()
    return model


def run(args: argparse.Namespace) -> None:
    device = _device(args.device)
    model = _load_model(args.checkpoint, device)
    test_raw = Path(args.test_raw_dir) if args.test_raw_dir else Path(args.workdir) / "splits/test/raw"
    test_reference = Path(args.test_reference_dir) if args.test_reference_dir else Path(args.workdir) / "splits/test/reference"
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    comparison_dir = output_dir / "comparisons"
    metrics_csv = output_dir / "metrics.csv"
    average_csv = output_dir / "average_metrics.csv"
    attention_csv = output_dir / "attention.csv"
    pairs = build_uieb_pairs(test_raw, test_reference)
    if not pairs:
        raise FileNotFoundError(f"No test pairs found in {test_raw} and {test_reference}")
    if getattr(args, "test_max_images", 0) > 0:
        pairs = pairs[: args.test_max_images]

    rows = []
    attn_rows = []
    with torch.no_grad():
        for raw_path, ref_path in tqdm(pairs, desc="Testing three-stage checkpoint"):
            raw = np.array(pil_loader(raw_path, args.image_size))
            tensor = image_to_tensor(raw).unsqueeze(0).to(device)
            enhanced = model.G_AB(tensor)
            enhanced_img = tensor_to_image(enhanced)
            rel = raw_path.relative_to(test_raw).with_suffix(".png")
            out_path = image_dir / rel
            save_image_rgb(out_path, enhanced_img)
            target = np.array(pil_loader(ref_path, args.image_size))
            save_comparison(comparison_dir / rel, [raw, enhanced_img, target], ["raw", "enhanced", "reference"])
            metric_row = calculate_metrics(enhanced_img, target)
            rows.append({
                "dataset": "UIEB_test_split",
                "image_name": raw_path.name,
                "input_path": raw_path.as_posix(),
                "reference_path": ref_path.as_posix(),
                "enhanced_path": out_path.as_posix(),
                **metric_row,
            })
            attn = getattr(model.G_AB, "last_attention", None)
            if attn is not None:
                vals = attn.mean(dim=(0, 2, 3)).detach().cpu().tolist()
                names = list(getattr(model.G_AB, "branch_names", []))
                attn_rows.append({
                    "dataset": "UIEB_test_split",
                    "image_name": raw_path.name,
                    **{f"attention_{names[i]}": vals[i] for i in range(min(len(names), len(vals)))},
                })

    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(metrics_csv, index=False)
    save_average_metrics(metrics_csv, average_csv)
    if attn_rows:
        pd.DataFrame(attn_rows).to_csv(attention_csv, index=False)
    print(f"[test] wrote {metrics_csv}")
    print(f"[test] wrote {average_csv}")


def parse_args() -> argparse.Namespace:
    default_workdir = Path(__file__).resolve().parent / "workdir"
    parser = argparse.ArgumentParser(description="Test the three-stage UIEB experiment on the held-out test split.")
    parser.add_argument("--workdir", default=str(default_workdir))
    parser.add_argument("--checkpoint", default=str(default_workdir / "checkpoints/stage3/stage3_best.pth"))
    parser.add_argument("--test-raw-dir", default="")
    parser.add_argument("--test-reference-dir", default="")
    parser.add_argument("--output-dir", default=str(default_workdir / "test_results"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--test-max-images", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
