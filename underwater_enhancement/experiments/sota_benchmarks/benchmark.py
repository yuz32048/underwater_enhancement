from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import PairedImageDataset, build_uieb_pairs
from datasets.image_datasets import save_average_metrics
from metrics import calculate_metrics
from models.waternet import gamma_correction, gray_world_white_balance, lab_clahe
from train.losses import SSIMLoss, gan_loss
from utils.image_io import image_to_tensor, list_images, pil_loader, save_comparison, save_image_rgb, tensor_to_image
from utils.logger import CSVLogger
from utils.seed import set_seed

from experiments.sota_benchmarks.sota_models import build_model, generator_for

MODELS = ["cyclegan", "ugan", "funie-gan", "uwcnn", "waternet"]


def _device(name: str) -> torch.device:
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else ("cpu" if name == "auto" else name))


def _split_dir(args: argparse.Namespace, split: str, subdir: str) -> Path:
    return Path(args.workdir) / "splits" / split / subdir


def prepare_splits(args: argparse.Namespace) -> None:
    split_root = Path(args.workdir) / "splits"
    manifest = split_root / "split_manifest.csv"
    if manifest.exists() and not args.overwrite:
        print(f"[prepare] using existing split manifest: {manifest}")
        return
    if split_root.exists() and args.overwrite:
        shutil.rmtree(split_root)
    pairs = build_uieb_pairs(args.uieb_raw_dir, args.uieb_reference_dir)
    if not pairs:
        raise FileNotFoundError("No UIEB raw/reference pairs found.")
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    train_count = int(round(len(pairs) * args.train_ratio))
    val_count = int(round(len(pairs) * args.val_ratio))
    splits = {
        "train": pairs[:train_count],
        "validation": pairs[train_count:train_count + val_count],
        "test": pairs[train_count + val_count:],
    }
    rows: list[dict[str, str]] = []
    for split, split_pairs in splits.items():
        for raw_path, ref_path in split_pairs:
            raw_out = split_root / split / "raw" / raw_path.name
            ref_out = split_root / split / "reference" / ref_path.name
            raw_out.parent.mkdir(parents=True, exist_ok=True)
            ref_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(raw_path, raw_out)
            shutil.copy2(ref_path, ref_out)
            rows.append({
                "split": split,
                "raw_source": raw_path.as_posix(),
                "reference_source": ref_path.as_posix(),
                "raw_path": raw_out.as_posix(),
                "reference_path": ref_out.as_posix(),
            })
    manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(manifest, index=False)
    print("[prepare] split counts: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))


class UnpairedDataset(Dataset):
    def __init__(self, raw_dir: str | Path, reference_dir: str | Path, image_size: int, seed: int):
        self.raw = list_images(raw_dir)
        self.reference = list_images(reference_dir)
        self.image_size = image_size
        self.rng = random.Random(seed)
        if not self.raw or not self.reference:
            raise FileNotFoundError("CycleGAN needs non-empty raw and reference domains.")

    def __len__(self) -> int:
        return max(len(self.raw), len(self.reference))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        a_path = self.raw[idx % len(self.raw)]
        b_path = self.reference[self.rng.randrange(len(self.reference))]
        a = np.array(pil_loader(a_path, self.image_size))
        b = np.array(pil_loader(b_path, self.image_size))
        return image_to_tensor(a), image_to_tensor(b)


class WaterNetDataset(Dataset):
    def __init__(self, raw_dir: str | Path, reference_dir: str | Path, image_size: int):
        self.pairs = build_uieb_pairs(raw_dir, reference_dir)
        self.image_size = image_size
        if not self.pairs:
            raise FileNotFoundError("No paired images found for WaterNet.")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_path, ref_path = self.pairs[idx]
        raw = np.array(pil_loader(raw_path, self.image_size))
        ref = np.array(pil_loader(ref_path, self.image_size))
        return (
            image_to_tensor(raw),
            image_to_tensor(gray_world_white_balance(raw)),
            image_to_tensor(lab_clahe(raw)),
            image_to_tensor(gamma_correction(raw)),
            image_to_tensor(ref),
        )


def _paired_loader(args: argparse.Namespace, split: str, shuffle: bool) -> DataLoader:
    pairs = build_uieb_pairs(_split_dir(args, split, "raw"), _split_dir(args, split, "reference"))
    return DataLoader(PairedImageDataset(pairs, args.image_size), batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers)


def _waternet_loader(args: argparse.Namespace, split: str, shuffle: bool) -> DataLoader:
    dataset = WaterNetDataset(_split_dir(args, split, "raw"), _split_dir(args, split, "reference"), args.image_size)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers)


def _unpaired_loader(args: argparse.Namespace, split: str) -> DataLoader:
    dataset = UnpairedDataset(_split_dir(args, split, "raw"), _split_dir(args, split, "reference"), args.image_size, args.seed)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)


def _save_checkpoint(path: Path, model: nn.Module, optimizers: dict[str, torch.optim.Optimizer], epoch: int, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizers": {k: v.state_dict() for k, v in optimizers.items()},
        "epoch": epoch,
        "args": vars(args),
    }, path)


def _load_checkpoint(path: Path, model: nn.Module, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)


def _val_loss(generator: nn.Module, loader: DataLoader, device: torch.device, max_batches: int) -> float:
    generator.eval()
    l1 = nn.L1Loss()
    losses: list[float] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            if max_batches > 0 and batch_idx > max_batches:
                break
            if len(batch) == 5:
                _, wb, ce, gc, target = batch
                out = generator(wb.to(device), ce.to(device), gc.to(device))
            else:
                raw, target = batch
                out = generator(raw.to(device))
            losses.append(l1(out, target.to(device)).item())
    generator.train()
    return float(np.mean(losses)) if losses else float("inf")


def train_paired(args: argparse.Namespace, model_name: str, device: torch.device) -> Path:
    model = build_model(model_name).to(device)
    generator = generator_for(model, model_name)
    train_loader = _waternet_loader(args, "train", True) if model_name == "waternet" else _paired_loader(args, "train", True)
    val_loader = _waternet_loader(args, "validation", False) if model_name == "waternet" else _paired_loader(args, "validation", False)
    optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    l1 = nn.L1Loss()
    ssim = SSIMLoss().to(device)
    best = float("inf")
    ckpt_dir = Path(args.workdir) / model_name / "checkpoints"
    logger = CSVLogger(Path(args.workdir) / model_name / "logs" / "train.csv", ["epoch", "loss", "val_loss"])
    for epoch in range(1, args.epochs + 1):
        losses: list[float] = []
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"{model_name} epoch {epoch}/{args.epochs}"), start=1):
            if args.max_train_batches > 0 and batch_idx > args.max_train_batches:
                break
            optimizer.zero_grad(set_to_none=True)
            if model_name == "waternet":
                _, wb, ce, gc, target = batch
                out = generator(wb.to(device), ce.to(device), gc.to(device))
            else:
                raw, target = batch
                out = generator(raw.to(device))
            target = target.to(device)
            loss = args.lambda_l1 * l1(out, target) + args.lambda_ssim * ssim(out, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        val = _val_loss(generator, val_loader, device, args.val_max_batches)
        logger.log({"epoch": epoch, "loss": float(np.mean(losses)) if losses else 0.0, "val_loss": val})
        _save_checkpoint(ckpt_dir / "latest.pth", model, {"optimizer": optimizer}, epoch, args)
        if val < best:
            best = val
            _save_checkpoint(ckpt_dir / "best.pth", model, {"optimizer": optimizer}, epoch, args)
    return ckpt_dir / "best.pth"


def train_cgan(args: argparse.Namespace, model_name: str, device: torch.device) -> Path:
    model = build_model(model_name).to(device)
    generator = model["G"]
    discriminator = model["D"]
    train_loader = _paired_loader(args, "train", True)
    val_loader = _paired_loader(args, "validation", False)
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    l1 = nn.L1Loss()
    mse = nn.MSELoss()
    ssim = SSIMLoss().to(device)
    ckpt_dir = Path(args.workdir) / model_name / "checkpoints"
    logger = CSVLogger(Path(args.workdir) / model_name / "logs" / "train.csv", ["epoch", "g_loss", "d_loss", "val_loss"])
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        g_losses: list[float] = []
        d_losses: list[float] = []
        for batch_idx, (raw, target) in enumerate(tqdm(train_loader, desc=f"{model_name} epoch {epoch}/{args.epochs}"), start=1):
            if args.max_train_batches > 0 and batch_idx > args.max_train_batches:
                break
            raw = raw.to(device)
            target = target.to(device)
            fake = generator(raw)
            opt_d.zero_grad(set_to_none=True)
            d_real = gan_loss(discriminator(raw, target), True, mse)
            d_fake = gan_loss(discriminator(raw, fake.detach()), False, mse)
            d_loss = 0.5 * (d_real + d_fake)
            d_loss.backward()
            opt_d.step()
            opt_g.zero_grad(set_to_none=True)
            fake = generator(raw)
            adv = gan_loss(discriminator(raw, fake), True, mse)
            rec = args.lambda_l1 * l1(fake, target) + args.lambda_ssim * ssim(fake, target)
            g_loss = args.lambda_adv * adv + rec
            g_loss.backward()
            opt_g.step()
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
        val = _val_loss(generator, val_loader, device, args.val_max_batches)
        logger.log({"epoch": epoch, "g_loss": np.mean(g_losses), "d_loss": np.mean(d_losses), "val_loss": val})
        _save_checkpoint(ckpt_dir / "latest.pth", model, {"opt_g": opt_g, "opt_d": opt_d}, epoch, args)
        if val < best:
            best = val
            _save_checkpoint(ckpt_dir / "best.pth", model, {"opt_g": opt_g, "opt_d": opt_d}, epoch, args)
    return ckpt_dir / "best.pth"


def train_cyclegan(args: argparse.Namespace, device: torch.device) -> Path:
    model = build_model("cyclegan").to(device)
    loader = _unpaired_loader(args, "train")
    val_loader = _paired_loader(args, "validation", False)
    opt_g = torch.optim.Adam(list(model.G_AB.parameters()) + list(model.G_BA.parameters()), lr=args.lr, betas=(args.beta1, args.beta2))
    opt_d = torch.optim.Adam(list(model.D_A.parameters()) + list(model.D_B.parameters()), lr=args.lr, betas=(args.beta1, args.beta2))
    l1 = nn.L1Loss()
    mse = nn.MSELoss()
    ckpt_dir = Path(args.workdir) / "cyclegan" / "checkpoints"
    logger = CSVLogger(Path(args.workdir) / "cyclegan" / "logs" / "train.csv", ["epoch", "g_loss", "d_loss", "val_loss"])
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        g_losses: list[float] = []
        d_losses: list[float] = []
        for batch_idx, (real_a, real_b) in enumerate(tqdm(loader, desc=f"cyclegan epoch {epoch}/{args.epochs}"), start=1):
            if args.max_train_batches > 0 and batch_idx > args.max_train_batches:
                break
            real_a = real_a.to(device)
            real_b = real_b.to(device)
            fake_b = model.G_AB(real_a)
            fake_a = model.G_BA(real_b)
            opt_d.zero_grad(set_to_none=True)
            d_a = 0.5 * (gan_loss(model.D_A(real_a), True, mse) + gan_loss(model.D_A(fake_a.detach()), False, mse))
            d_b = 0.5 * (gan_loss(model.D_B(real_b), True, mse) + gan_loss(model.D_B(fake_b.detach()), False, mse))
            d_loss = d_a + d_b
            d_loss.backward()
            opt_d.step()
            opt_g.zero_grad(set_to_none=True)
            fake_b = model.G_AB(real_a)
            fake_a = model.G_BA(real_b)
            rec_a = model.G_BA(fake_b)
            rec_b = model.G_AB(fake_a)
            idt_a = model.G_BA(real_a)
            idt_b = model.G_AB(real_b)
            adv = gan_loss(model.D_B(fake_b), True, mse) + gan_loss(model.D_A(fake_a), True, mse)
            cycle = l1(rec_a, real_a) + l1(rec_b, real_b)
            identity = l1(idt_a, real_a) + l1(idt_b, real_b)
            g_loss = args.lambda_adv * adv + args.lambda_cycle * cycle + args.lambda_identity * identity
            g_loss.backward()
            opt_g.step()
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
        val = _val_loss(model.G_AB, val_loader, device, args.val_max_batches)
        logger.log({"epoch": epoch, "g_loss": np.mean(g_losses), "d_loss": np.mean(d_losses), "val_loss": val})
        _save_checkpoint(ckpt_dir / "latest.pth", model, {"opt_g": opt_g, "opt_d": opt_d}, epoch, args)
        if val < best:
            best = val
            _save_checkpoint(ckpt_dir / "best.pth", model, {"opt_g": opt_g, "opt_d": opt_d}, epoch, args)
    return ckpt_dir / "best.pth"


def train(args: argparse.Namespace) -> Path:
    set_seed(args.seed)
    prepare_splits(args)
    device = _device(args.device)
    if args.model == "cyclegan":
        return train_cyclegan(args, device)
    if args.model in {"ugan", "funie-gan"}:
        return train_cgan(args, args.model, device)
    return train_paired(args, args.model, device)


def _enhance(model: nn.Module, model_name: str, raw_img: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = image_to_tensor(raw_img).unsqueeze(0).to(device)
    with torch.no_grad():
        if model_name == "waternet":
            wb = image_to_tensor(gray_world_white_balance(raw_img)).unsqueeze(0).to(device)
            ce = image_to_tensor(lab_clahe(raw_img)).unsqueeze(0).to(device)
            gc = image_to_tensor(gamma_correction(raw_img)).unsqueeze(0).to(device)
            out = model(wb, ce, gc)
        else:
            out = generator_for(model, model_name)(tensor)
    return tensor_to_image(out)


def test(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = _device(args.device)
    model = build_model(args.model).to(device)
    checkpoint = Path(args.checkpoint) if args.checkpoint else Path(args.workdir) / args.model / "checkpoints" / "best.pth"
    _load_checkpoint(checkpoint, model, device)
    model.eval()
    pairs = build_uieb_pairs(_split_dir(args, "test", "raw"), _split_dir(args, "test", "reference"))
    if not pairs:
        raise FileNotFoundError("No held-out test pairs found. Run prepare/train first.")
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.workdir) / args.model / "test_results"
    rows: list[dict[str, str | float]] = []
    if args.test_max_images > 0:
        pairs = pairs[: args.test_max_images]
    for raw_path, ref_path in tqdm(pairs, desc=f"testing {args.model}"):
        raw = np.array(pil_loader(raw_path, args.image_size))
        ref = np.array(pil_loader(ref_path, args.image_size))
        enhanced = _enhance(model, args.model, raw, device)
        rel = raw_path.relative_to(_split_dir(args, "test", "raw")).with_suffix(".png")
        out_path = output_dir / "images" / rel
        save_image_rgb(out_path, enhanced)
        save_comparison(output_dir / "comparisons" / rel, [raw, enhanced, ref], ["raw", "enhanced", "reference"])
        rows.append({
            "dataset": "UIEB_test_split",
            "model": args.model,
            "image_name": raw_path.name,
            "input_path": raw_path.as_posix(),
            "reference_path": ref_path.as_posix(),
            "enhanced_path": out_path.as_posix(),
            **calculate_metrics(enhanced, ref),
        })
    metrics_csv = output_dir / "metrics.csv"
    average_csv = output_dir / "average_metrics.csv"
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(metrics_csv, index=False)
    save_average_metrics(metrics_csv, average_csv)
    print(f"[test] wrote {metrics_csv}")
    print(f"[test] wrote {average_csv}")


def summarize(args: argparse.Namespace) -> None:
    rows: list[dict[str, str | float]] = []
    for model_name in MODELS:
        path = Path(args.workdir) / model_name / "test_results" / "average_metrics.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0)
        if "overall" not in df.index:
            continue
        row = {"model": model_name}
        row.update({k: float(df.loc["overall", k]) for k in ["PSNR", "SSIM", "UIQM", "UCIQE"] if k in df.columns})
        rows.append(row)
    if not rows:
        raise FileNotFoundError("No model average_metrics.csv files found.")
    out = Path(args.workdir) / "summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[summary] wrote {out}")


def run_all(args: argparse.Namespace) -> None:
    prepare_splits(args)
    for model_name in MODELS:
        model_args = argparse.Namespace(**vars(args))
        model_args.model = model_name
        model_args.checkpoint = ""
        model_args.output_dir = ""
        model_args.overwrite = False
        train(model_args)
        test(model_args)
    summarize(args)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    default_workdir = Path(__file__).resolve().parent / "workdir"
    parser.add_argument("--workdir", default=str(default_workdir))
    parser.add_argument("--uieb-raw-dir", default=str(PROJECT_ROOT / "data/raw_underwater/UIEB/raw-890"))
    parser.add_argument("--uieb-reference-dir", default=str(PROJECT_ROOT / "data/raw_underwater/UIEB/reference-890"))
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--lambda-l1", type=float, default=1.0)
    parser.add_argument("--lambda-ssim", type=float, default=0.5)
    parser.add_argument("--lambda-adv", type=float, default=0.5)
    parser.add_argument("--lambda-cycle", type=float, default=10.0)
    parser.add_argument("--lambda-identity", type=float, default=7.5)
    parser.add_argument("--val-max-batches", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--test-max-images", type=int, default=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified SOTA benchmark for underwater image enhancement.")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    add_common_args(prepare)
    train_parser = sub.add_parser("train")
    add_common_args(train_parser)
    train_parser.add_argument("--model", choices=MODELS, required=True)
    test_parser = sub.add_parser("test")
    add_common_args(test_parser)
    test_parser.add_argument("--model", choices=MODELS, required=True)
    test_parser.add_argument("--checkpoint", default="")
    test_parser.add_argument("--output-dir", default="")
    summary_parser = sub.add_parser("summary")
    add_common_args(summary_parser)
    all_parser = sub.add_parser("run-all")
    add_common_args(all_parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare_splits(args)
    elif args.command == "train":
        train(args)
    elif args.command == "test":
        test(args)
    elif args.command == "summary":
        summarize(args)
    elif args.command == "run-all":
        run_all(args)
    else:
        raise ValueError(args.command)
    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    config_path = Path(args.workdir) / "last_args.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
