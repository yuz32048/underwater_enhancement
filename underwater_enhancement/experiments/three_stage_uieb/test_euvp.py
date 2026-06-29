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

from datasets import TestImageDataset
from datasets.image_datasets import save_average_metrics
from metrics import calculate_metrics
from models.cyclegan import CycleGAN
from utils.image_io import image_to_tensor, pil_loader, read_image_rgb, save_comparison, save_image_rgb, tensor_to_image


def _device(name: str) -> torch.device:
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else ("cpu" if name == "auto" else name))


def _load_model(checkpoint: str | Path, device: torch.device) -> CycleGAN:
    ckpt = torch.load(checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})
    model = CycleGAN(fusion=ckpt_args.get("fusion", "attention")).to(device)
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model.eval()
    return model


def _euvp_paired_set(root: Path, name: str) -> tuple[Path, Path | None]:
    candidates = list((root / "EUVP_Paired").glob(name + "*"))
    base = candidates[0] if candidates else root / "EUVP_Paired" / name
    if (base / "trainA").exists() and (base / "trainB").exists():
        return base / "trainA", base / "trainB"
    return base, None


def build_euvp_sets(euvp_root: str | Path) -> list[TestImageDataset]:
    euvp = Path(euvp_root)
    sets: list[TestImageDataset] = []
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
    model = _load_model(args.checkpoint, device)
    out_root = Path(args.output_dir) / "images"
    cmp_root = Path(args.output_dir) / "comparisons"
    metrics_csv = Path(args.output_dir) / "metrics.csv"
    average_csv = Path(args.output_dir) / "average_metrics.csv"
    attention_csv = Path(args.output_dir) / "attention.csv"
    rows = []
    attn_rows = []

    with torch.no_grad():
        for dataset in build_euvp_sets(args.euvp_root):
            if not dataset.images:
                continue
            for path in tqdm(dataset.images, desc=f"Testing {dataset.name}"):
                pil = pil_loader(path, args.image_size)
                raw = np.array(pil)
                tensor = image_to_tensor(raw).unsqueeze(0).to(device)
                enhanced = model.G_AB(tensor)
                enhanced_img = tensor_to_image(enhanced)
                rel = path.relative_to(dataset.input_dir).with_suffix(".png")
                out_path = out_root / dataset.name / rel
                save_image_rgb(out_path, enhanced_img)

                ref_path = dataset.reference_for(path)
                target = read_image_rgb(ref_path) if ref_path else None
                comparison_images = [raw, enhanced_img] if target is None else [raw, enhanced_img, target]
                comparison_labels = ["input", "enhanced"] if target is None else ["input", "enhanced", "reference"]
                save_comparison(cmp_root / dataset.name / rel, comparison_images, comparison_labels)

                rows.append({
                    "dataset": dataset.name,
                    "image_name": path.name,
                    "input_path": path.as_posix(),
                    "reference_path": ref_path.as_posix() if ref_path else "",
                    "enhanced_path": out_path.as_posix(),
                    **calculate_metrics(enhanced_img, target),
                })
                attn = getattr(model.G_AB, "last_attention", None)
                if attn is not None:
                    vals = attn.mean(dim=(0, 2, 3)).detach().cpu().tolist()
                    names = list(getattr(model.G_AB, "branch_names", []))
                    attn_rows.append({
                        "dataset": dataset.name,
                        "image_name": path.name,
                        **{f"attention_{names[i]}": vals[i] for i in range(min(len(names), len(vals)))},
                    })

    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(metrics_csv, index=False)
    save_average_metrics(metrics_csv, average_csv)
    if attn_rows:
        pd.DataFrame(attn_rows).to_csv(attention_csv, index=False)
    print(f"[euvp-test] wrote {metrics_csv}")
    print(f"[euvp-test] wrote {average_csv}")


def parse_args() -> argparse.Namespace:
    default_workdir = Path(__file__).resolve().parent / "workdir"
    parser = argparse.ArgumentParser(description="External EUVP test for the three-stage UIEB experiment.")
    parser.add_argument("--checkpoint", default=str(default_workdir / "checkpoints/stage3/stage3_best.pth"))
    parser.add_argument("--euvp-root", default=str(PROJECT_ROOT / "data/raw_underwater/EUVP"))
    parser.add_argument("--output-dir", default=str(default_workdir / "euvp_test_results"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
