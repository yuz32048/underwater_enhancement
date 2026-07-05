from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.three_stage_uieb import test_euvp, test_three_stage, train_three_stage


@dataclass(frozen=True)
class AblationSpec:
    ablation_id: str
    name: str
    description: str
    overrides: dict[str, object] = field(default_factory=dict)


ABLATIONS: list[AblationSpec] = [
    AblationSpec(
        "A0",
        "full_model",
        "Full model with concat fusion, synthetic degraded domain augmentation, branch expert pretraining, and all four degradation branches.",
    ),
    AblationSpec(
        "A1",
        "wo_synthetic_degradation",
        "Remove synthetic degraded images from the Stage 2/3 degraded domain.",
        {"synthetic_ratio": 0.0},
    ),
    AblationSpec(
        "A2",
        "wo_branch_expert_pretrain",
        "Disable branch expert pretraining before Stage 1.",
        {"no_branch_expert_pretrain": True},
    ),
    AblationSpec(
        "A3",
        "wo_multibranch_experts",
        "Replace the multi-branch degradation expert generator with a single generator.",
        {"single_generator": True},
    ),
    AblationSpec(
        "A4",
        "wo_blue_branch",
        "Remove the blue-cast branch from the degradation expert set.",
        {"enabled_branches": "green,lowlight,blur"},
    ),
    AblationSpec(
        "A5",
        "wo_green_branch",
        "Remove the green-cast branch from the degradation expert set.",
        {"enabled_branches": "blue,lowlight,blur"},
    ),
    AblationSpec(
        "A6",
        "wo_lowlight_branch",
        "Remove the low-light branch from the degradation expert set.",
        {"enabled_branches": "blue,green,blur"},
    ),
    AblationSpec(
        "A7",
        "wo_blur_branch",
        "Remove the blur branch from the degradation expert set.",
        {"enabled_branches": "blue,green,lowlight"},
    ),
]

ABLATION_BY_ID = {item.ablation_id.lower(): item for item in ABLATIONS}
ABLATION_BY_NAME = {item.name.lower(): item for item in ABLATIONS}


def _slug(spec: AblationSpec) -> str:
    return f"{spec.ablation_id}_{spec.name}"


def _resolve_specs(selection: str) -> list[AblationSpec]:
    if selection.lower() == "all":
        return ABLATIONS
    specs: list[AblationSpec] = []
    for raw in selection.split(","):
        key = raw.strip().lower()
        if not key:
            continue
        spec = ABLATION_BY_ID.get(key) or ABLATION_BY_NAME.get(key)
        if spec is None:
            valid = ", ".join(item.ablation_id for item in ABLATIONS)
            raise ValueError(f"Unknown ablation '{raw}'. Valid choices: all,{valid}")
        specs.append(spec)
    return specs


def _base_train_args(args: argparse.Namespace, spec: AblationSpec) -> argparse.Namespace:
    train_args = argparse.Namespace(**vars(args))
    train_args.workdir = str(Path(args.ablation_root) / _slug(spec))
    train_args.overwrite = args.overwrite
    train_args.enabled_branches = ""
    train_args.single_generator = False
    train_args.command = "all"
    train_args.stage1_checkpoint = ""
    train_args.stage2_checkpoint = ""
    for key, value in spec.overrides.items():
        setattr(train_args, key, value)
    return train_args


def _prepare(args: argparse.Namespace) -> None:
    train_three_stage.prepare_splits(args)
    if args.synthetic_ratio > 0 or args.branch_use_synthetic:
        train_three_stage.prepare_synthetic(args)


def _train_and_test(args: argparse.Namespace, spec: AblationSpec) -> None:
    run_args = _base_train_args(args, spec)
    workdir = Path(run_args.workdir)
    print(f"[ablation] {spec.ablation_id} {spec.name}: {spec.description}")
    _prepare(run_args)
    train_three_stage.run_train(run_args)
    checkpoint = workdir / "checkpoints/stage3/stage3_best.pth"
    test_args = argparse.Namespace(
        workdir=str(workdir),
        checkpoint=str(checkpoint),
        test_raw_dir="",
        test_reference_dir="",
        output_dir=str(workdir / "test_results"),
        image_size=run_args.image_size,
        device=run_args.device,
        test_max_images=args.test_max_images,
    )
    test_three_stage.run(test_args)
    euvp_args = argparse.Namespace(
        checkpoint=str(checkpoint),
        euvp_root=args.euvp_root,
        output_dir=str(workdir / "euvp_test_results"),
        image_size=run_args.image_size,
        device=run_args.device,
        test_max_images=args.euvp_test_max_images,
        euvp_datasets=args.euvp_datasets,
    )
    test_euvp.run(euvp_args)
    metadata = {
        "ablation_id": spec.ablation_id,
        "name": spec.name,
        "description": spec.description,
        "overrides": spec.overrides,
        "workdir": str(workdir),
    }
    workdir.mkdir(parents=True, exist_ok=True)
    with (workdir / "ablation.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def _test_euvp_only(args: argparse.Namespace, spec: AblationSpec) -> None:
    run_args = _base_train_args(args, spec)
    workdir = Path(run_args.workdir)
    checkpoint = workdir / "checkpoints/stage3/stage3_best.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint for {spec.ablation_id} {spec.name}: {checkpoint}")
    euvp_args = argparse.Namespace(
        checkpoint=str(checkpoint),
        euvp_root=args.euvp_root,
        output_dir=str(workdir / "euvp_test_results"),
        image_size=run_args.image_size,
        device=run_args.device,
        test_max_images=args.euvp_test_max_images,
        euvp_datasets=args.euvp_datasets,
    )
    print(f"[ablation-euvp] {spec.ablation_id} {spec.name}")
    test_euvp.run(euvp_args)


def summarize(args: argparse.Namespace) -> None:
    rows: list[dict[str, object]] = []
    root = Path(args.ablation_root)
    for spec in ABLATIONS:
        metrics_path = root / _slug(spec) / "test_results/average_metrics.csv"
        if not metrics_path.exists():
            continue
        df = pd.read_csv(metrics_path, index_col=0)
        if "overall" not in df.index:
            continue
        row: dict[str, object] = {
            "ablation_id": spec.ablation_id,
            "name": spec.name,
            "description": spec.description,
        }
        for key in ["PSNR", "SSIM", "UIQM", "UCIQE"]:
            if key in df.columns:
                row[key] = float(df.loc["overall", key])
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No ablation metrics found under {root}")
    out = root / "summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[ablation] wrote {out}")


def summarize_euvp(args: argparse.Namespace) -> None:
    rows: list[dict[str, object]] = []
    root = Path(args.ablation_root)
    for spec in ABLATIONS:
        metrics_path = root / _slug(spec) / "euvp_test_results/average_metrics.csv"
        if not metrics_path.exists():
            continue
        df = pd.read_csv(metrics_path, index_col=0)
        if "overall" not in df.index:
            continue
        row: dict[str, object] = {
            "ablation_id": spec.ablation_id,
            "name": spec.name,
            "description": spec.description,
        }
        for key in ["PSNR", "SSIM", "UIQM", "UCIQE"]:
            if key in df.columns and pd.notna(df.loc["overall", key]):
                row[key] = float(df.loc["overall", key])
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No EUVP ablation metrics found under {root}")
    out = root / "euvp_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[ablation] wrote {out}")


def list_ablations() -> None:
    for spec in ABLATIONS:
        overrides = ", ".join(f"{k}={v}" for k, v in spec.overrides.items()) or "default"
        print(f"{spec.ablation_id:>2}  {spec.name:<28} {overrides}")
        print(f"    {spec.description}")


def add_args(parser: argparse.ArgumentParser) -> None:
    train_three_stage.add_common_args(parser)
    parser.set_defaults(workdir="", fusion="concat")
    parser.add_argument("--ablation-root", default=str(Path(__file__).resolve().parent / "workdir"))
    parser.add_argument("--ablations", default="all", help="Comma-separated IDs/names, e.g. A0,A1,A4, or all.")
    parser.add_argument("--test-max-images", type=int, default=0)
    parser.add_argument("--euvp-root", default=str(PROJECT_ROOT / "data/raw_underwater/EUVP"))
    parser.add_argument("--euvp-test-max-images", type=int, default=0)
    parser.add_argument("--euvp-datasets", default="all", help="Comma-separated EUVP subsets: all, underwater_dark, underwater_imagenet, underwater_scenes, unpaired, test_samples, eval_data.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UIEB ablation experiments from one entrypoint.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    run = sub.add_parser("run")
    add_args(run)
    euvp = sub.add_parser("test-euvp")
    add_args(euvp)
    summary = sub.add_parser("summary")
    add_args(summary)
    euvp_summary = sub.add_parser("summary-euvp")
    add_args(euvp_summary)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "list":
        list_ablations()
        return
    if args.command == "summary":
        summarize(args)
        return
    if args.command == "summary-euvp":
        summarize_euvp(args)
        return
    specs = _resolve_specs(args.ablations)
    if args.command == "test-euvp":
        for spec in specs:
            _test_euvp_only(args, spec)
        summarize_euvp(args)
        return
    for spec in specs:
        _train_and_test(args, spec)
    summarize(args)
    summarize_euvp(args)


if __name__ == "__main__":
    main()
