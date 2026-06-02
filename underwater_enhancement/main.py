import argparse
from pathlib import Path

from analysis.classify_images import classify_folder
from analysis.feature_extraction import analyze_folder
from degradation.generate_degraded_dataset import generate_dataset
from eval.test import run_test
from train.train_cyclegan import train
from utils.image_io import load_config
from utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Underwater image enhancement project")
    parser.add_argument("--config", default="config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("analyze")
    sub.add_parser("classify")
    sub.add_parser("degrade")
    sub.add_parser("train")
    sub.add_parser("test")
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


if __name__ == "__main__":
    main()
