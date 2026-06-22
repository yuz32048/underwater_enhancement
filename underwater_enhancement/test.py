from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from datasets import TestImageDataset
from datasets.image_datasets import save_average_metrics
from metrics import calculate_metrics
from models.cyclegan import CycleGAN
from utils.image_io import image_to_tensor, pil_loader, read_image_rgb, save_comparison, save_image_rgb, tensor_to_image


def _device(name: str) -> torch.device:
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else ("cpu" if name == "auto" else name))


def _load_model(args: argparse.Namespace, device: torch.device) -> CycleGAN:
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})
    all_branches = ["blue", "green", "lowlight", "blur"]
    disabled = set(ckpt_args.get("disable_branch") or [])
    enabled_branches = [b for b in all_branches if b not in disabled]

    model = CycleGAN(
        fusion=ckpt_args.get("fusion", "attention"),
        enabled_branches=enabled_branches,
        freeze_branches=False,
        use_multibranch=not ckpt_args.get("plain_cyclegan", False),
    ).to(device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _euvp_paired_set(root: Path, name: str) -> tuple[Path, Path | None]:
    candidates = list((root / "EUVP_Paired").glob(name + "*"))
    base = candidates[0] if candidates else root / "EUVP_Paired" / name
    if (base / "trainA").exists() and (base / "trainB").exists():
        return base / "trainA", base / "trainB"
    return base, None


def build_test_sets(args: argparse.Namespace) -> list[TestImageDataset]:
    uieb = Path(args.uieb_root)
    euvp = Path(args.euvp_root)
    sets = [
        TestImageDataset("UIEB_raw_890", uieb / "raw-890", uieb / "reference-890"),
        TestImageDataset("UIEB_challenging_60", uieb / "challenging-60", None),
    ]
    for name in ["underwater_dark", "underwater_imagenet", "underwater_scenes"]:
        inp, ref = _euvp_paired_set(euvp, name)
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
    attn_rows = []

    with torch.no_grad():
        for dataset in build_test_sets(args):
            if not dataset.images:
                continue
            for path in tqdm(dataset.images, desc=f"Testing {dataset.name}"):
                pil = pil_loader(path, args.image_size)
                tensor = image_to_tensor(__import__("numpy").array(pil)).unsqueeze(0).to(device)
                enhanced = model.G_AB(tensor)
                enhanced_img = tensor_to_image(enhanced)
                rel_name = path.relative_to(dataset.input_dir).with_suffix(".png")
                out_path = out_root / dataset.name / rel_name
                save_image_rgb(out_path, enhanced_img)
                save_comparison(cmp_root / dataset.name / rel_name, [tensor_to_image(tensor), enhanced_img], ["input", "enhanced"])
                ref_path = dataset.reference_for(path)
                target = read_image_rgb(ref_path) if ref_path else None
                metric_row = calculate_metrics(enhanced_img, target)
                rows.append({
                    "dataset": dataset.name,
                    "image_name": path.name,
                    "input_path": path.as_posix(),
                    "reference_path": ref_path.as_posix() if ref_path else "",
                    "enhanced_path": out_path.as_posix(),
                    **metric_row,
                })
                attn = getattr(model.G_AB, "last_attention", None)
                if attn is not None:
                    vals = attn.mean(dim=(0, 2, 3)).detach().cpu().tolist()
                    names = list(getattr(model.G_AB, "branch_names", []))

                    attn_data = {}
                    for i in range(min(len(names), len(vals))):
                        attn_data[f"attention_{names[i]}"] = vals[i]

                    attn_rows.append({
                        "dataset": dataset.name,
                        "image_name": path.name,
                        **attn_data,
                    })

    Path(args.metrics_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.metrics_csv, index=False)
    save_average_metrics(args.metrics_csv, args.average_csv)
    if attn_rows:
        pd.DataFrame(attn_rows).to_csv(args.attention_csv, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test G_AB on UIEB and EUVP and compute PSNR/SSIM/UIQM/UCIQE.")
    parser.add_argument("--checkpoint", default="checkpoints/best_model/generator_best.pth")
    parser.add_argument("--uieb-root", default="data/raw_underwater/UIEB")
    parser.add_argument("--euvp-root", default="data/raw_underwater/EUVP")
    parser.add_argument("--output-dir", default="outputs/test_results")
    parser.add_argument("--comparison-dir", default="outputs/visual_comparisons")
    parser.add_argument("--metrics-csv", default="results/evaluation_metrics.csv")
    parser.add_argument("--average-csv", default="results/average_metrics.csv")
    parser.add_argument("--attention-csv", default="results/attention_statistics.csv")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

