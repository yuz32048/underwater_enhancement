from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from datasets import TestImageDataset
from datasets.image_datasets import save_average_metrics
from metrics import calculate_metrics
from models.waternet import WaterNet, gamma_correction, gray_world_white_balance, lab_clahe
from utils.image_io import (
    image_to_tensor,
    pil_loader,
    read_image_rgb,
    resize_rgb,
    save_comparison,
    save_image_rgb,
    tensor_to_image,
)


def _device(name: str) -> torch.device:
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else ("cpu" if name == "auto" else name))


def _state_dict_from_checkpoint(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "net", "waternet"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break
    if not isinstance(checkpoint, dict):
        raise TypeError("Unsupported WaterNet checkpoint format.")

    state = {}
    for key, value in checkpoint.items():
        if not torch.is_tensor(value):
            continue
        clean_key = key.removeprefix("module.").removeprefix("model.")
        state[clean_key] = value
    return state


def _load_model(args: argparse.Namespace, device: torch.device) -> WaterNet:
    model = WaterNet(channels=args.channels).to(device)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"WaterNet checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as exc:
        print(f"Safe checkpoint loading failed ({exc}); falling back to full checkpoint loading.")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = _state_dict_from_checkpoint(checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=args.strict)
    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")
    model.eval()
    return model


def _waternet_inputs(image_rgb: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    wb = gray_world_white_balance(image_rgb)
    ce = lab_clahe(image_rgb)
    gc = gamma_correction(image_rgb)
    return (
        image_to_tensor(wb).unsqueeze(0).to(device),
        image_to_tensor(ce).unsqueeze(0).to(device),
        image_to_tensor(gc).unsqueeze(0).to(device),
    )


def build_test_sets(args: argparse.Namespace) -> list[TestImageDataset]:
    uieb = Path(args.uieb_root)
    euvp = Path(args.euvp_root)
    sets = [
        TestImageDataset("UIEB_raw_890", uieb / "raw-890", uieb / "reference-890"),
        TestImageDataset("UIEB_challenging_60", uieb / "challenging-60", None),
    ]
    for name in ["underwater_dark", "underwater_imagenet", "underwater_scenes"]:
        base = euvp / "EUVP_Paired" / name
        inp = base / "trainA" if (base / "trainA").exists() else base
        ref = base / "trainB" if (base / "trainB").exists() else None
        sets.append(TestImageDataset(f"EUVP_{name}", inp, ref))
    sets.extend([
        TestImageDataset("EUVP_Unpaired", euvp / "Unpaired" / "trainA", None),
        TestImageDataset("EUVP_test_samples", euvp / "test_samples" / "Inp", euvp / "test_samples" / "GTr"),
        TestImageDataset("EUVP_eval_data", euvp / "eval_data", None),
    ])
    return sets


def run(args: argparse.Namespace) -> None:
    device = _device(args.device)
    model = _load_model(args, device)
    out_root = Path(args.output_dir)
    cmp_root = Path(args.comparison_dir)
    rows = []

    with torch.no_grad():
        for dataset in build_test_sets(args):
            if not dataset.images:
                continue
            for path in tqdm(dataset.images, desc=f"Testing {dataset.name}"):
                pil = pil_loader(path, args.image_size)
                image_rgb = np.array(pil)
                wb, ce, gc = _waternet_inputs(image_rgb, device)
                enhanced = model(wb, ce, gc)
                enhanced_img = tensor_to_image(enhanced)

                rel_name = path.relative_to(dataset.input_dir).with_suffix(".png")
                out_path = out_root / dataset.name / rel_name
                save_image_rgb(out_path, enhanced_img)

                ref_path = dataset.reference_for(path)
                target = read_image_rgb(ref_path) if ref_path else None
                comparison = [image_rgb, enhanced_img]
                labels = ["input", "waternet"]
                if target is not None:
                    comparison.append(resize_rgb(target, (enhanced_img.shape[1], enhanced_img.shape[0])))
                    labels.append("reference")
                save_comparison(cmp_root / dataset.name / rel_name, comparison, labels)

                metric_row = calculate_metrics(enhanced_img, target)
                rows.append({
                    "dataset": dataset.name,
                    "image_name": path.name,
                    "input_path": path.as_posix(),
                    "reference_path": ref_path.as_posix() if ref_path else "",
                    "enhanced_path": out_path.as_posix(),
                    **metric_row,
                })

    Path(args.metrics_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.metrics_csv, index=False)
    save_average_metrics(args.metrics_csv, args.average_csv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test WaterNet on UIEB and EUVP and compute PSNR/SSIM/UIQM/UCIQE.")
    parser.add_argument("--checkpoint", default="checkpoints/waternet/waternet.pth")
    parser.add_argument("--uieb-root", default="data/raw_underwater/UIEB")
    parser.add_argument("--euvp-root", default="data/raw_underwater/EUVP")
    parser.add_argument("--output-dir", default="outputs/waternet_test_results")
    parser.add_argument("--comparison-dir", default="outputs/waternet_visual_comparisons")
    parser.add_argument("--metrics-csv", default="results/waternet_evaluation_metrics.csv")
    parser.add_argument("--average-csv", default="results/waternet_average_metrics.csv")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
