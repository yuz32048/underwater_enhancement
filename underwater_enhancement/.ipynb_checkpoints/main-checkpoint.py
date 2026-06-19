import argparse
from pathlib import Path

from analysis.classify_images import classify_folder
from analysis.feature_extraction import analyze_folder
from degradation.generate_degraded_dataset import generate_dataset
from eval.test import run_test
from scripts.classify_uieb import run as run_uieb_classification
from scripts.generate_physical_degradation import run as run_physical_degradation
from test import run as run_full_test
from train_branch import BRANCHES, train_one
from train_cyclegan import train as train_multibranch_cyclegan
from train.train_cyclegan import train
from utils.archive_extraction import prepare_archives
from utils.image_io import load_config
from utils.seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parent


def _add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--uieb-root", default="data/raw_underwater/UIEB")
    parser.add_argument("--euvp-root", default="data/raw_underwater/EUVP")
    parser.add_argument("--classified-root", default="data/processed/UIEB_classified")
    parser.add_argument("--physical-root", default="data/processed/physical_degradation")
    parser.add_argument("--classification-csv", default="results/classification_result.csv")
    parser.add_argument("--mapping-csv", default="results/physical_degradation_mapping.csv")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)


def _classification_args(args: argparse.Namespace, root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        input_dir=str(root / args.uieb_root / "raw-890"),
        output_dir=str(root / args.classified_root),
        csv_path=str(root / args.classification_csv),
        blue_b_threshold=-4.0,
        green_a_threshold=-2.0,
        green_b_threshold=2.0,
        low_light_v_threshold=85.0,
        blur_laplacian_threshold=80.0,
        blur_edge_threshold=0.025,
        canny1=80,
        canny2=180,
    )


def _physical_args(args: argparse.Namespace, root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        input_dir=str(root / args.uieb_root / "reference-890"),
        output_dir=str(root / args.physical_root),
        mapping_csv=str(root / args.mapping_csv),
        depth_min=0.25,
        depth_max=1.1,
        blue_beta_r=1.45,
        blue_beta_g=0.85,
        blue_beta_b=0.38,
        blue_background_b=0.95,
        green_beta_r=1.25,
        green_beta_g=0.48,
        green_beta_b=0.92,
        green_background_g=0.92,
        low_beta=1.15,
        low_gamma=1.8,
        low_scale=0.72,
        blur_kernel=7,
        blur_sigma=1.8,
    )


def _branch_args(args: argparse.Namespace, root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        classified_root=str(root / args.classified_root),
        physical_root=str(root / args.physical_root),
        reference_dir=str(root / args.uieb_root / "reference-890"),
        mapping_csv=str(root / args.mapping_csv),
        save_dir=str(root / "checkpoints/pretrained_branches"),
        sample_dir=str(root / "outputs/train_samples/branches"),
        log_csv=str(root / "logs/branch_train_log.csv"),
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        image_size=args.image_size,
        device=args.device,
        num_workers=args.num_workers,
        lambda_l1=1.0,
        lambda_ssim=0.5,
        sample_every=200,
    )


def _cyclegan_args(args: argparse.Namespace, root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        uieb_raw_dir=str(root / args.uieb_root / "raw-890"),
        uieb_reference_dir=str(root / args.uieb_root / "reference-890"),
        physical_root=str(root / args.physical_root),
        pretrained_branch_dir=str(root / "checkpoints/pretrained_branches"),
        generator_dir=str(root / "checkpoints/generator"),
        discriminator_dir=str(root / "checkpoints/discriminator"),
        best_dir=str(root / "checkpoints/best_model"),
        sample_dir=str(root / "outputs/train_samples/cyclegan"),
        log_csv=str(root / "logs/cyclegan_train_log.csv"),
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        image_size=args.image_size,
        device=args.device,
        num_workers=args.num_workers,
        resume="",
        sample_every=200,
        lambda_adv=1.0,
        lambda_cycle=10.0,
        lambda_identity=5.0,
        lambda_ssim=1.0,
        fusion="attention",
        freeze_branches=False,
        plain_cyclegan=False,
        no_physical_degradation=False,
        no_branch_pretrain=False,
        no_attention=False,
        disable_branch=None,
        ablation_name="",
    )


def _test_args(args: argparse.Namespace, root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint=str(root / "checkpoints/best_model/generator_best.pth"),
        uieb_root=str(root / args.uieb_root),
        euvp_root=str(root / args.euvp_root),
        output_dir=str(root / "outputs/test_results"),
        comparison_dir=str(root / "outputs/visual_comparisons"),
        metrics_csv=str(root / "results/evaluation_metrics.csv"),
        average_csv=str(root / "results/average_metrics.csv"),
        attention_csv=str(root / "results/attention_statistics.csv"),
        image_size=args.image_size,
        device=args.device,
    )


def run_branch_pretraining(args: argparse.Namespace, root: Path) -> None:
    branch_args = _branch_args(args, root)
    branches = BRANCHES if args.branch == "all" else [args.branch]
    for branch in branches:
        train_one(branch, branch_args)


def run_full_pipeline(args: argparse.Namespace, root: Path) -> None:
    if not args.skip_classify:
        run_uieb_classification(_classification_args(args, root))
    if not args.skip_physical:
        run_physical_degradation(_physical_args(args, root))
    if not args.skip_branch:
        run_branch_pretraining(args, root)
    if not args.skip_cyclegan:
        train_multibranch_cyclegan(_cyclegan_args(args, root))
    if not args.skip_test:
        run_full_test(_test_args(args, root))


def main() -> None:
    parser = argparse.ArgumentParser(description="Underwater image enhancement project")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("analyze")
    sub.add_parser("classify")
    sub.add_parser("degrade")
    sub.add_parser("train")
    sub.add_parser("test")
    sub.add_parser("prepare-data")

    classify_uieb_parser = sub.add_parser("classify-uieb", help="Pipeline UIEB raw-890 degradation classification")
    _add_pipeline_args(classify_uieb_parser)

    physical_parser = sub.add_parser("physical-degrade", help="Generate Jaffe-McGlamery physical degradations from UIEB reference-890")
    _add_pipeline_args(physical_parser)

    branch_parser = sub.add_parser("pretrain-branches", help="Supervised pretraining for expert branches")
    _add_pipeline_args(branch_parser)
    branch_parser.add_argument("--branch", choices=BRANCHES + ["all"], default="all")

    cyclegan_parser = sub.add_parser("train-cyclegan", help="Train the pipeline Multi-Branch CycleGAN")
    _add_pipeline_args(cyclegan_parser)

    test_parser = sub.add_parser("test-all", help="Test trained G_AB on UIEB and EUVP")
    _add_pipeline_args(test_parser)

    full_parser = sub.add_parser("full-pipeline", help="Run classify, physical generation, branch pretraining, CycleGAN training, and test")
    _add_pipeline_args(full_parser)
    full_parser.add_argument("--branch", choices=BRANCHES + ["all"], default="all")
    full_parser.add_argument("--skip-classify", action="store_true")
    full_parser.add_argument("--skip-physical", action="store_true")
    full_parser.add_argument("--skip-branch", action="store_true")
    full_parser.add_argument("--skip-cyclegan", action="store_true")
    full_parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    root = Path(args.config).resolve().parent

    if args.command == "analyze":
        analyze_folder(cfg, root)
    elif args.command == "classify":
        classify_folder(cfg, root)
    elif args.command == "degrade":
        generate_dataset(cfg, root)
    elif args.command == "train":
        train(cfg, root)
    elif args.command == "test":
        run_test(cfg, root)
    elif args.command == "prepare-data":
        prepare_archives(root / cfg["paths"]["input_dir"], cfg)
    elif args.command == "classify-uieb":
        run_uieb_classification(_classification_args(args, root))
    elif args.command == "physical-degrade":
        run_physical_degradation(_physical_args(args, root))
    elif args.command == "pretrain-branches":
        run_branch_pretraining(args, root)
    elif args.command == "train-cyclegan":
        train_multibranch_cyclegan(_cyclegan_args(args, root))
    elif args.command == "test-all":
        run_full_test(_test_args(args, root))
    elif args.command == "full-pipeline":
        run_full_pipeline(args, root)


if __name__ == "__main__":
    main()
