from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from datasets import TestImageDataset
from metrics import calculate_metrics
from models.cyclegan import CycleGAN
from utils.image_io import image_to_tensor, pil_loader, read_image_rgb, tensor_to_image


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def epoch_num(path: Path) -> int:
    match = re.search(r"epoch_(\d+)\.pth", path.name)
    return int(match.group(1)) if match else -1


def load_model(checkpoint_path: Path, device: torch.device) -> CycleGAN:
    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_args = ckpt.get("args", {})

    model = CycleGAN(
        fusion=ckpt_args.get("fusion", "attention"),
        enabled_branches=None,
        freeze_branches=False,
        use_multibranch=not ckpt_args.get("plain_cyclegan", False),
    ).to(device)

    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def evaluate_one_checkpoint(
    checkpoint_path: Path,
    dataset: TestImageDataset,
    device: torch.device,
    image_size: int,
    max_images: int | None = None,
) -> dict:
    model = load_model(checkpoint_path, device)

    rows = []
    images = dataset.images[:max_images] if max_images else dataset.images

    with torch.no_grad():
        for img_path in tqdm(images, desc=f"Eval {checkpoint_path.name}", leave=False):
            pil = pil_loader(img_path, image_size)
            tensor = image_to_tensor(__import__("numpy").array(pil)).unsqueeze(0).to(device)

            enhanced = model.G_AB(tensor)
            enhanced_img = tensor_to_image(enhanced)

            ref_path = dataset.reference_for(img_path)
            if ref_path is None:
                continue

            target = read_image_rgb(ref_path)
            metric_row = calculate_metrics(enhanced_img, target)

            rows.append(metric_row)

    if not rows:
        return {
            "checkpoint": checkpoint_path.name,
            "epoch": epoch_num(checkpoint_path),
            "PSNR": None,
            "SSIM": None,
            "UIQM": None,
            "UCIQE": None,
            "num_images": 0,
        }

    df = pd.DataFrame(rows)

    return {
        "checkpoint": checkpoint_path.name,
        "epoch": epoch_num(checkpoint_path),
        "PSNR": df["PSNR"].mean(skipna=True),
        "SSIM": df["SSIM"].mean(skipna=True),
        "UIQM": df["UIQM"].mean(skipna=True),
        "UCIQE": df["UCIQE"].mean(skipna=True),
        "num_images": len(df),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate multiple CycleGAN checkpoints on UIEB raw-890 validation/test set."
    )

    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/generator",
        help="Directory containing epoch_*.pth checkpoints.",
    )
    parser.add_argument(
        "--uieb-root",
        default="data/raw_underwater/UIEB",
        help="UIEB root directory.",
    )
    parser.add_argument(
        "--output-csv",
        default="results/checkpoint_evaluation.csv",
        help="CSV file to save checkpoint evaluation results.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--device",
        default="auto",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Evaluate every N epochs.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Maximum number of images to evaluate. 0 means all images.",
    )

    args = parser.parse_args()

    device = get_device(args.device)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoints = sorted(
        checkpoint_dir.glob("epoch_*.pth"),
        key=epoch_num,
    )

    if args.interval > 1:
        checkpoints = [
            p for p in checkpoints
            if epoch_num(p) > 0 and epoch_num(p) % args.interval == 0
        ]

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

    uieb_root = Path(args.uieb_root)
    dataset = TestImageDataset(
        "UIEB_raw_890",
        uieb_root / "raw-890",
        uieb_root / "reference-890",
    )

    max_images = args.max_images if args.max_images > 0 else None

    results = []

    for ckpt_path in checkpoints:
        result = evaluate_one_checkpoint(
            checkpoint_path=ckpt_path,
            dataset=dataset,
            device=device,
            image_size=args.image_size,
            max_images=max_images,
        )
        results.append(result)

        print(
            f"Epoch {result['epoch']:>3} | "
            f"PSNR={result['PSNR']:.4f} | "
            f"SSIM={result['SSIM']:.4f} | "
            f"UIQM={result['UIQM']:.4f} | "
            f"UCIQE={result['UCIQE']:.4f}"
        )

    df = pd.DataFrame(results)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    best_psnr = df.loc[df["PSNR"].idxmax()]
    best_ssim = df.loc[df["SSIM"].idxmax()]

    print("\nBest by PSNR:")
    print(best_psnr)

    print("\nBest by SSIM:")
    print(best_ssim)

    print(f"\nSaved results to: {output_csv}")


if __name__ == "__main__":
    main()