from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from metrics import calculate_metrics
from models.branches import branch_from_name
from utils.image_io import (
    image_to_tensor,
    list_images,
    pil_loader,
    read_image_rgb,
    resize_rgb,
    save_comparison,
    save_image_rgb,
    tensor_to_image,
)


BRANCHES = ["blue", "green", "lowlight", "blur"]

BRANCH_DIRS = {
    "blue": "blue_cast",
    "green": "green_cast",
    "lowlight": "low_light",
    "blur": "blur",
}

CATEGORY_DIRS = {
    "blue_cast": "blue_cast",
    "green_cast": "green_cast",
    "low_light": "low_light",
    "blur": "blur",
}

WEIGHT_FILES = {
    "blue": "blue_branch_np.pth",
    "green": "green_branch_np.pth",
    "lowlight": "lowlight_branch_np.pth",
    "blur": "blur_branch_np.pth",
}


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _build_reference_index(reference_dir: str | Path) -> dict[str, Path]:
    refs = list_images(reference_dir)
    index: dict[str, Path] = {}
    for path in refs:
        index[path.name.lower()] = path
        index[path.stem.lower()] = path
    return index


def _find_reference_for_real(image_path: Path, reference_index: dict[str, Path]) -> Path | None:
    return reference_index.get(image_path.name.lower()) or reference_index.get(image_path.stem.lower())


def _load_physical_mapping(mapping_csv: str | Path) -> dict[str, Path]:
    mapping_path = Path(mapping_csv)
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Physical degradation mapping file not found: {mapping_path}. "
            "Run scripts/generate_physical_degradation.py before testing physical_generated samples."
        )

    df = pd.read_csv(mapping_path)
    required = {"degraded_image", "reference_image", "source_reference_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Mapping file {mapping_path} is missing columns: {', '.join(sorted(missing))}")

    result: dict[str, Path] = {}
    for _, row in df.iterrows():
        degraded = Path(str(row["degraded_image"]))
        ref = Path(str(row["source_reference_path"])) if str(row.get("source_reference_path", "")) else Path(str(row["reference_image"]))
        result[degraded.as_posix().lower()] = ref
        result[degraded.name.lower()] = ref
        result[degraded.stem.lower()] = ref
    return result


def _find_reference_for_physical(
    image_path: Path,
    mapping: dict[str, Path],
    reference_index: dict[str, Path],
) -> Path | None:
    ref = (
        mapping.get(image_path.as_posix().lower())
        or mapping.get(image_path.name.lower())
        or mapping.get(image_path.stem.lower())
    )
    if ref is None:
        return None
    if ref.exists():
        return ref
    return reference_index.get(ref.name.lower()) or reference_index.get(ref.stem.lower())


def _load_branch_model(branch: str, checkpoint_dir: str | Path, device: torch.device) -> torch.nn.Module:
    model = branch_from_name(branch).to(device)
    ckpt_path = Path(checkpoint_dir) / WEIGHT_FILES[branch]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Branch checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _evaluate_image(
    model: torch.nn.Module,
    image_path: Path,
    ref_path: Path | None,
    branch: str,
    source_type: str,
    args: argparse.Namespace,
    device: torch.device,
    test_category: str | None = None,
) -> dict:
    pil = pil_loader(image_path, args.image_size)
    tensor = image_to_tensor(__import__("numpy").array(pil)).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)

    output_img = tensor_to_image(output)
    input_img = tensor_to_image(tensor)

    if args.mode == "cross":
        category = test_category or "unknown"
        out_path = Path(args.cross_output_dir) / category / branch / f"{image_path.stem}.png"
        cmp_path = Path(args.cross_comparison_dir) / category / branch / f"{image_path.stem}.png"
    else:
        out_path = Path(args.output_dir) / branch / source_type / f"{image_path.stem}.png"
        cmp_path = Path(args.comparison_dir) / branch / source_type / f"{image_path.stem}.png"

    save_image_rgb(out_path, output_img)

    target_img = None
    comparison_images = [input_img, output_img]
    comparison_labels = ["input", "output"]

    if ref_path is not None and ref_path.exists():
        target_img = read_image_rgb(ref_path)
        resized_ref = resize_rgb(target_img, (output_img.shape[1], output_img.shape[0]))
        comparison_images.append(resized_ref)
        comparison_labels.append("reference")
    else:
        warnings.warn(f"Reference not found for {image_path}. Metrics will skip PSNR/SSIM.")

    save_comparison(cmp_path, comparison_images, comparison_labels)
    metric_row = calculate_metrics(output_img, target_img)

    row = {
        "branch": branch,
        "source_type": source_type,
        "image_name": image_path.name,
        "input_path": image_path.as_posix(),
        "reference_path": ref_path.as_posix() if ref_path is not None and ref_path.exists() else "",
        "output_path": out_path.as_posix(),
        **metric_row,
    }

    if test_category is not None:
        row["test_category"] = test_category

    return row


def _save_average(rows: list[dict], average_csv: str | Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    numeric_cols = ["PSNR", "SSIM", "UIQM", "UCIQE"]
    numeric = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    grouped = (
        df[["branch", "source_type"]]
        .join(numeric)
        .groupby(["branch", "source_type"], dropna=False)
        .agg(
            PSNR_mean=("PSNR", "mean"),
            SSIM_mean=("SSIM", "mean"),
            UIQM_mean=("UIQM", "mean"),
            UCIQE_mean=("UCIQE", "mean"),
            num_images=("UIQM", "count"),
        )
        .reset_index()
    )

    overall = pd.DataFrame(
        [{
            "branch": "overall",
            "source_type": "overall",
            "PSNR_mean": numeric["PSNR"].mean(),
            "SSIM_mean": numeric["SSIM"].mean(),
            "UIQM_mean": numeric["UIQM"].mean(),
            "UCIQE_mean": numeric["UCIQE"].mean(),
            "num_images": len(df),
        }]
    )

    out = pd.concat([grouped, overall], ignore_index=True)
    Path(average_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(average_csv, index=False)
    return out


def _save_cross_matrix(rows: list[dict], matrix_csv: str | Path, best_csv: str | Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    numeric_cols = ["PSNR", "SSIM", "UIQM", "UCIQE"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    matrix = (
        df.groupby(["test_category", "branch"], dropna=False)
        .agg(
            PSNR_mean=("PSNR", "mean"),
            SSIM_mean=("SSIM", "mean"),
            UIQM_mean=("UIQM", "mean"),
            UCIQE_mean=("UCIQE", "mean"),
            num_images=("UIQM", "count"),
        )
        .reset_index()
    )

    Path(matrix_csv).parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(matrix_csv, index=False)

    best_rows = []
    for category, group in matrix.groupby("test_category"):
        best_psnr_row = group.loc[group["PSNR_mean"].idxmax()]
        best_ssim_row = group.loc[group["SSIM_mean"].idxmax()]

        best_rows.append({
            "test_category": category,
            "best_branch_by_PSNR": best_psnr_row["branch"],
            "best_PSNR": best_psnr_row["PSNR_mean"],
            "best_branch_by_SSIM": best_ssim_row["branch"],
            "best_SSIM": best_ssim_row["SSIM_mean"],
        })

    best_df = pd.DataFrame(best_rows)
    Path(best_csv).parent.mkdir(parents=True, exist_ok=True)
    best_df.to_csv(best_csv, index=False)

    return matrix


def test_one_branch(
    branch: str,
    args: argparse.Namespace,
    device: torch.device,
    reference_index: dict[str, Path],
) -> list[dict]:
    model = _load_branch_model(branch, args.checkpoint_dir, device)
    branch_dir = BRANCH_DIRS[branch]
    rows: list[dict] = []

    real_dir = Path(args.classified_root) / branch_dir
    real_images = list_images(real_dir)

    for image_path in tqdm(real_images, desc=f"{branch} real_classified"):
        ref_path = _find_reference_for_real(image_path, reference_index)
        rows.append(
            _evaluate_image(
                model=model,
                image_path=image_path,
                ref_path=ref_path,
                branch=branch,
                source_type="real_classified",
                args=args,
                device=device,
            )
        )

    physical_dir = Path(args.physical_root) / branch_dir
    physical_images = list_images(physical_dir)

    if physical_images:
        physical_mapping = _load_physical_mapping(args.mapping_csv)
        for image_path in tqdm(physical_images, desc=f"{branch} physical_generated"):
            ref_path = _find_reference_for_physical(image_path, physical_mapping, reference_index)
            rows.append(
                _evaluate_image(
                    model=model,
                    image_path=image_path,
                    ref_path=ref_path,
                    branch=branch,
                    source_type="physical_generated",
                    args=args,
                    device=device,
                )
            )

    return rows


def run_normal(args: argparse.Namespace) -> pd.DataFrame:
    device = _device(args.device)
    reference_index = _build_reference_index(args.reference_dir)
    branches = BRANCHES if args.branch == "all" else [args.branch]

    rows: list[dict] = []
    for branch in branches:
        rows.extend(test_one_branch(branch, args, device, reference_index))

    metrics_path = Path(args.metrics_csv)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        rows,
        columns=[
            "branch",
            "source_type",
            "image_name",
            "input_path",
            "reference_path",
            "output_path",
            "PSNR",
            "SSIM",
            "UIQM",
            "UCIQE",
        ],
    )
    df.to_csv(metrics_path, index=False)

    avg = _save_average(rows, args.average_csv) if rows else pd.DataFrame()
    if not avg.empty:
        print(avg.to_string(index=False))

    return df


def run_cross(args: argparse.Namespace) -> pd.DataFrame:
    device = _device(args.device)
    reference_index = _build_reference_index(args.reference_dir)

    models = {
        branch: _load_branch_model(branch, args.checkpoint_dir, device)
        for branch in BRANCHES
    }

    rows: list[dict] = []

    for test_category, category_dir in CATEGORY_DIRS.items():
        image_dir = Path(args.classified_root) / category_dir
        images = list_images(image_dir)

        if args.max_images > 0:
            images = images[: args.max_images]

        for branch, model in models.items():
            for image_path in tqdm(images, desc=f"{test_category} -> {branch}"):
                ref_path = _find_reference_for_real(image_path, reference_index)
                rows.append(
                    _evaluate_image(
                        model=model,
                        image_path=image_path,
                        ref_path=ref_path,
                        branch=branch,
                        source_type="cross_real_classified",
                        args=args,
                        device=device,
                        test_category=test_category,
                    )
                )

    metrics_path = Path(args.cross_metrics_csv)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        rows,
        columns=[
            "test_category",
            "branch",
            "source_type",
            "image_name",
            "input_path",
            "reference_path",
            "output_path",
            "PSNR",
            "SSIM",
            "UIQM",
            "UCIQE",
        ],
    )
    df.to_csv(metrics_path, index=False)

    matrix = _save_cross_matrix(rows, args.cross_matrix_csv, args.cross_best_csv)

    print(matrix.to_string(index=False))
    print(f"\nCross metrics saved to: {args.cross_metrics_csv}")
    print(f"Cross matrix saved to: {args.cross_matrix_csv}")
    print(f"Best branch table saved to: {args.cross_best_csv}")

    return df


def run(args: argparse.Namespace) -> pd.DataFrame:
    if args.mode == "cross":
        return run_cross(args)
    return run_normal(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test pretrained expert branches independently or by cross-category matrix.")

    parser.add_argument("--mode", choices=["normal", "cross"], default="normal")

    parser.add_argument("--branch", choices=BRANCHES + ["all"], default="all")
    parser.add_argument("--classified-root", default="data/processed/UIEB_classified")
    parser.add_argument("--physical-root", default="data/processed/physical_degradation")
    parser.add_argument("--reference-dir", default="data/raw_underwater/UIEB/reference-890")
    parser.add_argument("--mapping-csv", default="results/physical_degradation_mapping.csv")
    parser.add_argument("--checkpoint-dir", default="checkpoints/pretrained_branches")

    parser.add_argument("--output-dir", default="outputs/branch_test_results")
    parser.add_argument("--comparison-dir", default="outputs/branch_visual_comparisons")
    parser.add_argument("--metrics-csv", default="results/branch_test_metrics.csv")
    parser.add_argument("--average-csv", default="results/branch_average_metrics.csv")

    parser.add_argument("--cross-output-dir", default="outputs/cross_branch_test")
    parser.add_argument("--cross-comparison-dir", default="outputs/cross_branch_comparisons")
    parser.add_argument("--cross-metrics-csv", default="results/cross_branch_metrics.csv")
    parser.add_argument("--cross-matrix-csv", default="results/cross_branch_matrix.csv")
    parser.add_argument("--cross-best-csv", default="results/best_branch_by_category.csv")

    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-images", type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())