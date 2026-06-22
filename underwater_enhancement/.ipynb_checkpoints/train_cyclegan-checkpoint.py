from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.cyclegan import CycleGAN
from train.losses import SSIMLoss, gan_loss
from utils.logger import CSVLogger

import numpy as np
import pandas as pd

from datasets import CycleGANDataset, TestImageDataset
from metrics import calculate_metrics
from utils.image_io import (
    save_comparison,
    save_image_rgb,
    tensor_to_image,
    image_to_tensor,
    pil_loader,
    read_image_rgb,
)

def _device(name: str) -> torch.device:
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else ("cpu" if name == "auto" else name))


def _domain_a_dirs(args: argparse.Namespace) -> list[str]:
    dirs = [args.uieb_raw_dir]
    if not args.no_physical_degradation:
        dirs.extend([
            f"{args.physical_root}/blue_cast",
            f"{args.physical_root}/green_cast",
            f"{args.physical_root}/low_light",
            f"{args.physical_root}/blur",
        ])
    return dirs


def _physical_dirs(args: argparse.Namespace) -> list[str]:
    if args.no_physical_degradation:
        return []
    return [
        f"{args.physical_root}/blue_cast",
        f"{args.physical_root}/green_cast",
        f"{args.physical_root}/low_light",
        f"{args.physical_root}/blur",
    ]


def _enabled_branches(args: argparse.Namespace) -> list[str]:
    branches = ["blue", "green", "lowlight", "blur"]
    disabled = set(args.disable_branch or [])
    return [b for b in branches if b not in disabled]


def _save_checkpoint(path: Path, model: CycleGAN, optim_g, optim_d, epoch: int, step: int, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "step": step, "model": model.state_dict(), "optim_g": optim_g.state_dict(), "optim_d": optim_d.state_dict(), "args": vars(args)}, path)


def _load_checkpoint(path: Path, model: CycleGAN, optim_g, optim_d, device: torch.device) -> tuple[int, int]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    if "optim_g" in ckpt:
        optim_g.load_state_dict(ckpt["optim_g"])
    if "optim_d" in ckpt:
        optim_d.load_state_dict(ckpt["optim_d"])
    return int(ckpt.get("epoch", 0)) + 1, int(ckpt.get("step", 0))

def _evaluate_validation(
    model: CycleGAN,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    model.eval()

    val_input_dir = Path(args.val_input_dir) if args.val_input_dir else Path(args.uieb_raw_dir)
    val_reference_dir = Path(args.val_reference_dir) if args.val_reference_dir else Path(args.uieb_reference_dir)

    dataset = TestImageDataset(
        "UIEB_validation",
        val_input_dir,
        val_reference_dir,
    )

    rows = []
    images = dataset.images
    if args.val_max_images > 0:
        images = images[: args.val_max_images]

    with torch.no_grad():
        for img_path in tqdm(images, desc="Validation", leave=False):
            pil = pil_loader(img_path, args.image_size)
            tensor = image_to_tensor(np.array(pil)).unsqueeze(0).to(device)

            enhanced = model.G_AB(tensor)
            enhanced_img = tensor_to_image(enhanced)

            ref_path = dataset.reference_for(img_path)
            if ref_path is None:
                continue

            target = read_image_rgb(ref_path)
            metric_row = calculate_metrics(enhanced_img, target)
            rows.append(metric_row)

    model.train()

    if not rows:
        return {
            "PSNR": 0.0,
            "SSIM": 0.0,
            "UIQM": 0.0,
            "UCIQE": 0.0,
            "score": 0.0,
            "num_images": 0,
        }

    df = pd.DataFrame(rows)

    psnr = float(df["PSNR"].mean(skipna=True))
    ssim = float(df["SSIM"].mean(skipna=True))
    uiqm = float(df["UIQM"].mean(skipna=True))
    uciqe = float(df["UCIQE"].mean(skipna=True))

    score = psnr + args.val_ssim_weight * ssim

    return {
        "PSNR": psnr,
        "SSIM": ssim,
        "UIQM": uiqm,
        "UCIQE": uciqe,
        "score": score,
        "num_images": len(df),
    }

def train(args: argparse.Namespace) -> None:
    device = _device(args.device)
    for path in [args.generator_dir, args.discriminator_dir, args.best_dir, args.sample_dir, Path(args.log_csv).parent]:
        Path(path).mkdir(parents=True, exist_ok=True)
    exp_root = Path(args.ablation_name) if args.ablation_name else None
    if exp_root:
        result_dir = Path("results/ablation") / exp_root
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
        args.log_csv = str(result_dir / "cyclegan_train_log.csv")

    dataset = CycleGANDataset(
        _domain_a_dirs(args),
        args.uieb_reference_dir,
        args.image_size,
        raw_dir=args.uieb_raw_dir,
        physical_dirs=_physical_dirs(args),
        physical_sample_ratio=args.physical_sample_ratio,
    )
    print(f"raw_count={dataset.raw_count}")
    print(f"physical_total_count={dataset.physical_total_count}")
    print(f"physical_used_count={dataset.physical_used_count}")
    print(f"raw_to_physical_ratio={dataset.raw_to_physical_ratio:.4f}" if dataset.physical_used_count else "raw_to_physical_ratio=inf")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    model = CycleGAN(
        fusion=("average" if args.no_attention else args.fusion),
        enabled_branches=_enabled_branches(args),
        freeze_branches=args.freeze_branches,
        use_multibranch=not args.plain_cyclegan,
    ).to(device)
    if not args.no_branch_pretrain and not args.plain_cyclegan:
        model.G_AB.load_branch_weights(args.pretrained_branch_dir, strict=False)

    optim_g = torch.optim.Adam(list(model.G_AB.parameters()) + list(model.G_BA.parameters()), lr=args.lr, betas=(0.5, 0.999))
    optim_d = torch.optim.Adam(list(model.D_A.parameters()) + list(model.D_B.parameters()), lr=args.lr, betas=(0.5, 0.999))
    adv = nn.MSELoss()
    l1 = nn.L1Loss()
    ssim = SSIMLoss().to(device)
    log = CSVLogger(args.log_csv, ["epoch", "step", "loss_g", "loss_d", "loss_cycle", "loss_identity", "loss_ssim", "attention_blue", "attention_green", "attention_lowlight", "attention_blur"])
    start_epoch, step = 1, 0
    if args.resume:
        start_epoch, step = _load_checkpoint(Path(args.resume), model, optim_g, optim_d, device)

    best_score = -float("inf")
    val_records = []
    for epoch in range(start_epoch, args.epochs + 1):
        pbar = tqdm(loader, desc=f"CycleGAN epoch {epoch}/{args.epochs}")
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
            loss_g = args.lambda_adv * loss_adv + args.lambda_cycle * loss_cycle + args.lambda_identity * loss_identity + args.lambda_ssim * loss_ssim
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
            attn = model.G_AB.last_attention
            means = ["", "", "", ""]
            if attn is not None:
                vals = attn.mean(dim=(0, 2, 3)).detach().cpu().tolist()
                for i, value in enumerate(vals[:4]):
                    means[i] = value
            log.log({"epoch": epoch, "step": step, "loss_g": loss_g.item(), "loss_d": loss_d.item(), "loss_cycle": loss_cycle.item(), "loss_identity": loss_identity.item(), "loss_ssim": loss_ssim.item(), "attention_blue": means[0], "attention_green": means[1], "attention_lowlight": means[2], "attention_blur": means[3]})
            pbar.set_postfix(g=f"{loss_g.item():.3f}", d=f"{loss_d.item():.3f}")
            if step % args.sample_every == 0:
                save_comparison(Path(args.sample_dir) / f"step_{step}.png", [tensor_to_image(real_a), tensor_to_image(fake_b), tensor_to_image(real_b)], ["A", "G_AB(A)", "B"])
                if attn is not None:
                    weights = (attn[0].detach().cpu().numpy() * 255).astype("uint8")
                    for i, name in enumerate(_enabled_branches(args)[: weights.shape[0]]):
                        save_image_rgb(Path(args.sample_dir) / "attention" / f"step_{step}_{name}.png", __import__("numpy").stack([weights[i]] * 3, axis=-1))

        _save_checkpoint(Path(args.generator_dir) / f"epoch_{epoch}.pth", model, optim_g, optim_d, epoch, step, args)
        _save_checkpoint(Path(args.generator_dir) / "latest.pth", model, optim_g, optim_d, epoch, step, args)
        torch.save({"D_A": model.D_A.state_dict(), "D_B": model.D_B.state_dict()}, Path(args.discriminator_dir) / f"epoch_{epoch}.pth")
        if epoch % args.val_interval == 0:
            val_metrics = _evaluate_validation(model, args, device)
            val_records.append({
                "epoch": epoch,
                "step": step,
                **val_metrics,
            })

            Path(args.val_log_csv).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(val_records).to_csv(args.val_log_csv, index=False)

            print(
                f"[Validation] epoch={epoch} "
                f"PSNR={val_metrics['PSNR']:.4f} "
                f"SSIM={val_metrics['SSIM']:.4f} "
                f"UIQM={val_metrics['UIQM']:.4f} "
                f"UCIQE={val_metrics['UCIQE']:.4f} "
                f"score={val_metrics['score']:.4f}"
            )

            if val_metrics["score"] > best_score:
                best_score = val_metrics["score"]
                _save_checkpoint(
                    Path(args.best_dir) / "generator_best.pth",
                    model,
                    optim_g,
                    optim_d,
                    epoch,
                    step,
                    args,
                )
                print(f"[Best] Saved best model at epoch {epoch}, score={best_score:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Multi-Branch CycleGAN using UIEB only.")
    parser.add_argument("--uieb-raw-dir", default="data/raw_underwater/UIEB/raw-890")
    parser.add_argument("--uieb-reference-dir", default="data/raw_underwater/UIEB/reference-890")
    parser.add_argument("--physical-root", default="data/processed/physical_degradation")
    parser.add_argument("--physical-sample-ratio", type=float, default=1.0)
    parser.add_argument("--pretrained-branch-dir", default="checkpoints/pretrained_branches")
    parser.add_argument("--generator-dir", default="checkpoints/generator")
    parser.add_argument("--discriminator-dir", default="checkpoints/discriminator")
    parser.add_argument("--best-dir", default="checkpoints/best_model")
    parser.add_argument("--sample-dir", default="outputs/train_samples/cyclegan")
    parser.add_argument("--log-csv", default="logs/cyclegan_train_log.csv")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", default="")
    parser.add_argument("--sample-every", type=int, default=200)
    parser.add_argument("--lambda-adv", type=float, default=1.0)
    parser.add_argument("--lambda-cycle", type=float, default=10.0)
    parser.add_argument("--lambda-identity", type=float, default=5.0)
    parser.add_argument("--lambda-ssim", type=float, default=1.0)
    parser.add_argument("--fusion", choices=["attention", "concat", "average"], default="attention")
    parser.add_argument("--freeze-branches", action="store_true")
    parser.add_argument("--plain-cyclegan", action="store_true")
    parser.add_argument("--no-physical-degradation", action="store_true")
    parser.add_argument("--no-branch-pretrain", action="store_true")
    parser.add_argument("--no-attention", action="store_true")
    parser.add_argument("--disable-branch", action="append", choices=["blue", "green", "lowlight", "blur"])
    parser.add_argument("--ablation-name", default="")
    parser.add_argument("--val-input-dir", default="")
    parser.add_argument("--val-reference-dir", default="")
    parser.add_argument("--val-interval", type=int, default=5)
    parser.add_argument("--val-max-images", type=int, default=0)
    parser.add_argument("--val-ssim-weight", type=float, default=10.0)
    parser.add_argument("--val-log-csv", default="logs/validation_metrics.csv")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
