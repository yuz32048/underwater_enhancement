from __future__ import annotations

import argparse
import csv
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
from metrics import calculate_metrics
from models.cyclegan import CycleGAN
from scripts.generate_physical_degradation import run as generate_physical_degradation
from train.losses import SSIMLoss, gan_loss
from utils.image_io import image_to_tensor, list_images, pil_loader, read_image_rgb, save_comparison, tensor_to_image
from utils.logger import CSVLogger
from utils.seed import set_seed

BRANCHES = ["blue", "green", "lowlight", "blur"]
SYNTHETIC_DIRS = ["blue_cast", "green_cast", "low_light", "blur"]


def _device(name: str) -> torch.device:
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else ("cpu" if name == "auto" else name))


def _copy_pair(raw_path: Path, ref_path: Path, raw_out: Path, ref_out: Path) -> None:
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    ref_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw_path, raw_out)
    shutil.copy2(ref_path, ref_out)


def prepare_splits(args: argparse.Namespace) -> None:
    split_root = Path(args.workdir) / "splits"
    manifest = split_root / "split_manifest.csv"
    if manifest.exists() and not args.overwrite:
        print(f"[prepare] Using existing split manifest: {manifest}")
        return
    if split_root.exists() and args.overwrite:
        shutil.rmtree(split_root)

    pairs = build_uieb_pairs(args.uieb_raw_dir, args.uieb_reference_dir)
    if not pairs:
        raise FileNotFoundError("No UIEB raw/reference pairs found.")

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    total = len(pairs)
    train_count = int(round(total * args.train_ratio))
    val_count = int(round(total * args.val_ratio))
    train_pairs = pairs[:train_count]
    val_pairs = pairs[train_count:train_count + val_count]
    test_pairs = pairs[train_count + val_count:]

    rows: list[dict[str, str]] = []
    for split_name, split_pairs in (("train", train_pairs), ("validation", val_pairs), ("test", test_pairs)):
        for raw_path, ref_path in split_pairs:
            raw_out = split_root / split_name / "raw" / raw_path.name
            ref_out = split_root / split_name / "reference" / ref_path.name
            _copy_pair(raw_path, ref_path, raw_out, ref_out)
            rows.append({
                "split": split_name,
                "key": raw_path.stem.lower(),
                "raw_source": raw_path.as_posix(),
                "reference_source": ref_path.as_posix(),
                "raw_path": raw_out.as_posix(),
                "reference_path": ref_out.as_posix(),
            })

    manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(manifest, index=False)
    print(f"[prepare] split counts: train={len(train_pairs)}, validation={len(val_pairs)}, test={len(test_pairs)}")
    print(f"[prepare] wrote {manifest}")


def prepare_synthetic(args: argparse.Namespace) -> None:
    train_reference = Path(args.workdir) / "splits" / "train" / "reference"
    output_root = Path(args.workdir) / "synthetic_degraded" / "train"
    mapping_csv = Path(args.workdir) / "synthetic_degraded" / "train_mapping.csv"
    if mapping_csv.exists() and not args.overwrite:
        print(f"[prepare] Using existing synthetic mapping: {mapping_csv}")
        return
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    if not list_images(train_reference):
        raise FileNotFoundError(f"No train reference images found: {train_reference}")

    deg_args = argparse.Namespace(
        input_dir=str(train_reference),
        output_dir=str(output_root),
        mapping_csv=str(mapping_csv),
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        blue_beta_r=args.blue_beta_r,
        blue_beta_g=args.blue_beta_g,
        blue_beta_b=args.blue_beta_b,
        blue_background_b=args.blue_background_b,
        green_beta_r=args.green_beta_r,
        green_beta_g=args.green_beta_g,
        green_beta_b=args.green_beta_b,
        green_background_g=args.green_background_g,
        low_beta=args.low_beta,
        low_gamma=args.low_gamma,
        low_scale=args.low_scale,
        blur_kernel=args.blur_kernel,
        blur_sigma=args.blur_sigma,
    )
    generate_physical_degradation(deg_args)
    print(f"[prepare] synthetic degraded images are train-only domain augmentation: {output_root}")


class MixedUnpairedDataset(Dataset):
    def __init__(
        self,
        real_raw_dir: str | Path,
        synthetic_dirs: list[str | Path],
        clean_dir: str | Path,
        image_size: int,
        synthetic_ratio: float,
        seed: int,
    ):
        self.real_raw = list_images(real_raw_dir)
        synthetic_by_dir = [list_images(d) for d in synthetic_dirs]
        synthetic_all = [p for bucket in synthetic_by_dir for p in bucket]
        target_synth = int(round(len(self.real_raw) * synthetic_ratio))
        rng = random.Random(seed)
        rng.shuffle(synthetic_all)
        self.synthetic = synthetic_all[:target_synth] if synthetic_ratio > 0 else []
        self.domain_a = self.real_raw + self.synthetic
        self.domain_b = list_images(clean_dir)
        self.image_size = image_size
        if not self.real_raw:
            raise FileNotFoundError(f"No real raw train images found: {real_raw_dir}")
        if not self.domain_a:
            raise FileNotFoundError("No degraded domain images found.")
        if not self.domain_b:
            raise FileNotFoundError(f"No clean domain images found: {clean_dir}")

    def __len__(self) -> int:
        return max(len(self.domain_a), len(self.domain_b))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        a = np.array(pil_loader(self.domain_a[idx % len(self.domain_a)], self.image_size))
        b = np.array(pil_loader(self.domain_b[idx % len(self.domain_b)], self.image_size))
        return image_to_tensor(a), image_to_tensor(b)


class SafePerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            from torchvision import models

            weights = models.VGG16_Weights.IMAGENET1K_FEATURES
            self.features = models.vgg16(weights=weights).features[:16].eval()
            self.backend = "vgg16"
        except Exception:
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(inplace=True),
            ).eval()
            self.backend = "fallback_cnn"
        for param in self.features.parameters():
            param.requires_grad = False
        self.criterion = nn.L1Loss()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = (x.clamp(0, 1) - self.mean) / self.std
        y = (y.clamp(0, 1) - self.mean) / self.std
        return self.criterion(self.features(x), self.features(y))


def _load_model(path: str | Path, model: CycleGAN, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=False)


def _save_checkpoint(path: str | Path, model: CycleGAN, optim_g, optim_d, epoch: int, step: int, args: argparse.Namespace, stage: str, score: float) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "stage": stage,
        "epoch": epoch,
        "step": step,
        "score": score,
        "model": model.state_dict(),
        "optim_g": optim_g.state_dict() if optim_g is not None else None,
        "optim_d": optim_d.state_dict() if optim_d is not None else None,
        "args": vars(args),
    }, path)


def _paired_loader(raw_dir: Path, reference_dir: Path, image_size: int, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    pairs = build_uieb_pairs(raw_dir, reference_dir)
    dataset = PairedImageDataset(pairs, image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, drop_last=False)


def evaluate_paired(model: CycleGAN, raw_dir: Path, reference_dir: Path, args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    model.eval()
    rows = []
    pairs = build_uieb_pairs(raw_dir, reference_dir)
    if args.val_max_images > 0:
        pairs = pairs[:args.val_max_images]
    with torch.no_grad():
        for raw_path, ref_path in pairs:
            raw = np.array(pil_loader(raw_path, args.image_size))
            tensor = image_to_tensor(raw).unsqueeze(0).to(device)
            enhanced = tensor_to_image(model.G_AB(tensor))
            target = read_image_rgb(ref_path)
            rows.append(calculate_metrics(enhanced, target))
    model.train()
    if not rows:
        return {"PSNR": 0.0, "SSIM": 0.0, "UIQM": 0.0, "UCIQE": 0.0, "score": 0.0}
    df = pd.DataFrame(rows)
    psnr = float(pd.to_numeric(df["PSNR"], errors="coerce").mean())
    ssim = float(pd.to_numeric(df["SSIM"], errors="coerce").mean())
    uiqm = float(pd.to_numeric(df["UIQM"], errors="coerce").mean())
    uciqe = float(pd.to_numeric(df["UCIQE"], errors="coerce").mean())
    return {"PSNR": psnr, "SSIM": ssim, "UIQM": uiqm, "UCIQE": uciqe, "score": psnr + args.val_ssim_weight * ssim}


def train_stage1(args: argparse.Namespace) -> Path:
    device = _device(args.device)
    model = CycleGAN(fusion=args.fusion).to(device)
    train_loader = _paired_loader(
        Path(args.workdir) / "splits/train/raw",
        Path(args.workdir) / "splits/train/reference",
        args.image_size,
        args.batch_size,
        args.num_workers,
        True,
    )
    optimizer = torch.optim.Adam(model.G_AB.parameters(), lr=args.stage1_lr, betas=(0.5, 0.999))
    l1 = nn.L1Loss()
    ssim = SSIMLoss().to(device)
    perceptual = SafePerceptualLoss().to(device)
    train_log = CSVLogger(Path(args.workdir) / "logs/stage1_train.csv", ["epoch", "step", "loss", "l1", "ssim", "perceptual"])
    val_records = []
    best_score = -float("inf")
    step = 0
    sample_dir = Path(args.workdir) / "samples/stage1"
    ckpt_dir = Path(args.workdir) / "checkpoints/stage1"

    for epoch in range(1, args.stage1_epochs + 1):
        pbar = tqdm(train_loader, desc=f"Stage 1 epoch {epoch}/{args.stage1_epochs}")
        for raw, target in pbar:
            raw = raw.to(device)
            target = target.to(device)
            output = model.G_AB(raw)
            loss_l1 = l1(output, target)
            loss_ssim = ssim(output, target)
            loss_perc = perceptual(output, target)
            loss = args.lambda_l1 * loss_l1 + args.lambda_ssim * loss_ssim + args.lambda_perceptual * loss_perc
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            train_log.log({"epoch": epoch, "step": step, "loss": loss.item(), "l1": loss_l1.item(), "ssim": loss_ssim.item(), "perceptual": loss_perc.item()})
            if step % args.sample_every == 0:
                save_comparison(sample_dir / f"step_{step}.png", [tensor_to_image(raw), tensor_to_image(output), tensor_to_image(target)], ["raw", "stage1", "reference"])

        val = evaluate_paired(model, Path(args.workdir) / "splits/validation/raw", Path(args.workdir) / "splits/validation/reference", args, device)
        val_records.append({"epoch": epoch, "step": step, **val})
        pd.DataFrame(val_records).to_csv(Path(args.workdir) / "logs/stage1_validation.csv", index=False)
        _save_checkpoint(ckpt_dir / "latest.pth", model, optimizer, None, epoch, step, args, "stage1", val["score"])
        if val["score"] > best_score:
            best_score = val["score"]
            _save_checkpoint(ckpt_dir / "stage1_best.pth", model, optimizer, None, epoch, step, args, "stage1", best_score)
            print(f"[Stage 1] best score={best_score:.4f}")
    return ckpt_dir / "stage1_best.pth"


def freeze_branches(model: CycleGAN) -> None:
    if not getattr(model.G_AB, "use_multibranch", False):
        return
    for param in model.G_AB.branches.parameters():
        param.requires_grad = False


def partial_unfreeze_branch_cnns(model: CycleGAN) -> None:
    if not getattr(model.G_AB, "use_multibranch", False):
        return
    for param in model.G_AB.branches.parameters():
        param.requires_grad = False
    for branch in model.G_AB.branches.values():
        module = getattr(branch, "cnn", None) or getattr(branch, "net", None)
        if module is not None:
            for param in module.parameters():
                param.requires_grad = True


def _trainable_params(modules: list[nn.Module]) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    for module in modules:
        params.extend([p for p in module.parameters() if p.requires_grad])
    return params


def _unpaired_dataset(args: argparse.Namespace) -> MixedUnpairedDataset:
    synthetic_root = Path(args.workdir) / "synthetic_degraded/train"
    return MixedUnpairedDataset(
        real_raw_dir=Path(args.workdir) / "splits/train/raw",
        synthetic_dirs=[synthetic_root / name for name in SYNTHETIC_DIRS],
        clean_dir=Path(args.workdir) / "splits/train/reference",
        image_size=args.image_size,
        synthetic_ratio=args.synthetic_ratio,
        seed=args.seed,
    )


def train_cycle_stage(args: argparse.Namespace, stage: str, resume: Path, epochs: int, lr: float, partial_unfreeze: bool) -> Path:
    device = _device(args.device)
    model = CycleGAN(fusion=args.fusion).to(device)
    _load_model(resume, model, device)
    if partial_unfreeze:
        partial_unfreeze_branch_cnns(model)
    else:
        freeze_branches(model)

    dataset = _unpaired_dataset(args)
    print(f"[{stage}] degraded_domain: real_raw={len(dataset.real_raw)}, synthetic_used={len(dataset.synthetic)}, clean={len(dataset.domain_b)}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    optim_g = torch.optim.Adam(_trainable_params([model.G_AB, model.G_BA]), lr=lr, betas=(0.5, 0.999))
    optim_d = torch.optim.Adam(list(model.D_A.parameters()) + list(model.D_B.parameters()), lr=lr, betas=(0.5, 0.999))
    adv = nn.MSELoss()
    l1 = nn.L1Loss()
    ssim = SSIMLoss().to(device)
    log = CSVLogger(Path(args.workdir) / f"logs/{stage}_train.csv", ["epoch", "step", "loss_g", "loss_d", "loss_adv", "loss_cycle", "loss_identity", "loss_ssim"])
    ckpt_dir = Path(args.workdir) / f"checkpoints/{stage}"
    sample_dir = Path(args.workdir) / f"samples/{stage}"
    val_records = []
    best_score = -float("inf")
    step = 0

    for epoch in range(1, epochs + 1):
        pbar = tqdm(loader, desc=f"{stage} epoch {epoch}/{epochs}")
        for real_a, real_b in pbar:
            real_a = real_a.to(device)
            real_b = real_b.to(device)
            fake_b = model.G_AB(real_a)
            rec_a = model.G_BA(fake_b)
            fake_a = model.G_BA(real_b)
            rec_b = model.G_AB(fake_a)
            id_b = model.G_AB(real_b)

            loss_adv = gan_loss(model.D_B(fake_b), True, adv) + gan_loss(model.D_A(fake_a), True, adv)
            loss_cycle = l1(rec_a, real_a) + l1(rec_b, real_b)
            loss_identity = l1(id_b, real_b)
            loss_ssim = ssim(rec_a, real_a) + ssim(rec_b, real_b)
            loss_g = args.lambda_adv * loss_adv + args.lambda_cycle * loss_cycle + args.lambda_identity * loss_identity + args.lambda_ssim_cycle * loss_ssim
            optim_g.zero_grad()
            loss_g.backward()
            optim_g.step()

            loss_d_a = 0.5 * (gan_loss(model.D_A(real_a), True, adv) + gan_loss(model.D_A(fake_a.detach()), False, adv))
            loss_d_b = 0.5 * (gan_loss(model.D_B(real_b), True, adv) + gan_loss(model.D_B(fake_b.detach()), False, adv))
            loss_d = loss_d_a + loss_d_b
            optim_d.zero_grad()
            loss_d.backward()
            optim_d.step()

            step += 1
            pbar.set_postfix(g=f"{loss_g.item():.3f}", d=f"{loss_d.item():.3f}")
            log.log({"epoch": epoch, "step": step, "loss_g": loss_g.item(), "loss_d": loss_d.item(), "loss_adv": loss_adv.item(), "loss_cycle": loss_cycle.item(), "loss_identity": loss_identity.item(), "loss_ssim": loss_ssim.item()})
            if step % args.sample_every == 0:
                save_comparison(sample_dir / f"step_{step}.png", [tensor_to_image(real_a), tensor_to_image(fake_b), tensor_to_image(real_b)], ["degraded", "G_AB", "clean"])

        val = evaluate_paired(model, Path(args.workdir) / "splits/validation/raw", Path(args.workdir) / "splits/validation/reference", args, device)
        val_records.append({"epoch": epoch, "step": step, **val})
        pd.DataFrame(val_records).to_csv(Path(args.workdir) / f"logs/{stage}_validation.csv", index=False)
        _save_checkpoint(ckpt_dir / "latest.pth", model, optim_g, optim_d, epoch, step, args, stage, val["score"])
        if val["score"] > best_score:
            best_score = val["score"]
            _save_checkpoint(ckpt_dir / f"{stage}_best.pth", model, optim_g, optim_d, epoch, step, args, stage, best_score)
            print(f"[{stage}] best score={best_score:.4f}")
    return ckpt_dir / f"{stage}_best.pth"


def run_prepare(args: argparse.Namespace) -> None:
    prepare_splits(args)
    prepare_synthetic(args)


def run_train(args: argparse.Namespace) -> None:
    stage1 = Path(args.stage1_checkpoint) if args.stage1_checkpoint else train_stage1(args)
    stage2 = Path(args.stage2_checkpoint) if args.stage2_checkpoint else train_cycle_stage(args, "stage2", stage1, args.stage2_epochs, args.stage2_lr, partial_unfreeze=False)
    train_cycle_stage(args, "stage3", stage2, args.stage3_epochs, args.stage3_lr, partial_unfreeze=True)


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
    parser.add_argument("--fusion", choices=["attention", "concat", "average"], default="attention")
    parser.add_argument("--sample-every", type=int, default=200)
    parser.add_argument("--val-max-images", type=int, default=0)
    parser.add_argument("--val-ssim-weight", type=float, default=10.0)
    parser.add_argument("--stage1-epochs", type=int, default=20)
    parser.add_argument("--stage2-epochs", type=int, default=80)
    parser.add_argument("--stage3-epochs", type=int, default=20)
    parser.add_argument("--stage1-lr", type=float, default=2e-4)
    parser.add_argument("--stage2-lr", type=float, default=2e-4)
    parser.add_argument("--stage3-lr", type=float, default=5e-5)
    parser.add_argument("--stage1-checkpoint", default="")
    parser.add_argument("--stage2-checkpoint", default="")
    parser.add_argument("--synthetic-ratio", type=float, default=1.0)
    parser.add_argument("--lambda-l1", type=float, default=1.0)
    parser.add_argument("--lambda-ssim", type=float, default=0.5)
    parser.add_argument("--lambda-perceptual", type=float, default=0.1)
    parser.add_argument("--lambda-adv", type=float, default=1.0)
    parser.add_argument("--lambda-cycle", type=float, default=10.0)
    parser.add_argument("--lambda-identity", type=float, default=5.0)
    parser.add_argument("--lambda-ssim-cycle", type=float, default=1.0)
    parser.add_argument("--depth-min", type=float, default=0.25)
    parser.add_argument("--depth-max", type=float, default=1.1)
    parser.add_argument("--blue-beta-r", type=float, default=1.45)
    parser.add_argument("--blue-beta-g", type=float, default=0.85)
    parser.add_argument("--blue-beta-b", type=float, default=0.38)
    parser.add_argument("--blue-background-b", type=float, default=0.95)
    parser.add_argument("--green-beta-r", type=float, default=1.25)
    parser.add_argument("--green-beta-g", type=float, default=0.48)
    parser.add_argument("--green-beta-b", type=float, default=0.92)
    parser.add_argument("--green-background-g", type=float, default=0.92)
    parser.add_argument("--low-beta", type=float, default=1.15)
    parser.add_argument("--low-gamma", type=float, default=1.8)
    parser.add_argument("--low-scale", type=float, default=0.72)
    parser.add_argument("--blur-kernel", type=int, default=7)
    parser.add_argument("--blur-sigma", type=float, default=1.8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Three-stage UIEB experiment with leak-free splits.")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "stage1", "stage2", "stage3", "train", "all"):
        p = sub.add_parser(command)
        add_common_args(p)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.command == "prepare":
        run_prepare(args)
    elif args.command == "stage1":
        train_stage1(args)
    elif args.command == "stage2":
        checkpoint = Path(args.stage1_checkpoint) if args.stage1_checkpoint else Path(args.workdir) / "checkpoints/stage1/stage1_best.pth"
        train_cycle_stage(args, "stage2", checkpoint, args.stage2_epochs, args.stage2_lr, partial_unfreeze=False)
    elif args.command == "stage3":
        checkpoint = Path(args.stage2_checkpoint) if args.stage2_checkpoint else Path(args.workdir) / "checkpoints/stage2/stage2_best.pth"
        train_cycle_stage(args, "stage3", checkpoint, args.stage3_epochs, args.stage3_lr, partial_unfreeze=True)
    elif args.command == "train":
        run_train(args)
    elif args.command == "all":
        run_prepare(args)
        run_train(args)


if __name__ == "__main__":
    main()
