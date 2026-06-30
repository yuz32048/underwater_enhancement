from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from datasets import build_uieb_pairs
from models.waternet import WaterNet, gamma_correction, gray_world_white_balance, lab_clahe
from utils.image_io import image_to_tensor, pil_loader, save_comparison, tensor_to_image
from utils.logger import CSVLogger


class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11):
        super().__init__()
        self.avg = nn.AvgPool2d(window_size, stride=1, padding=window_size // 2)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        mu_x = self.avg(x)
        mu_y = self.avg(y)
        sigma_x = self.avg(x * x) - mu_x * mu_x
        sigma_y = self.avg(y * y) - mu_y * mu_y
        sigma_xy = self.avg(x * y) - mu_x * mu_y
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
            (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2) + 1e-8
        )
        return 1.0 - ssim.mean()


class WaterNetDataset(Dataset):
    def __init__(self, input_dir: str | Path, reference_dir: str | Path, image_size: int = 256):
        self.pairs = build_uieb_pairs(input_dir, reference_dir)
        if not self.pairs:
            raise FileNotFoundError(f"No paired UIEB images found in {input_dir} and {reference_dir}")
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inp_path, ref_path = self.pairs[idx]
        inp = np.array(pil_loader(inp_path, self.image_size))
        ref = np.array(pil_loader(ref_path, self.image_size))
        wb = gray_world_white_balance(inp)
        ce = lab_clahe(inp)
        gc = gamma_correction(inp)
        return (
            image_to_tensor(inp),
            image_to_tensor(wb),
            image_to_tensor(ce),
            image_to_tensor(gc),
            image_to_tensor(ref),
        )


def _device(name: str) -> torch.device:
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else ("cpu" if name == "auto" else name))


def _save_checkpoint(
    path: str | Path,
    model: WaterNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    args: argparse.Namespace,
    best_loss: float,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "best_loss": best_loss,
            "args": vars(args),
        },
        path,
    )


def _load_checkpoint(
    path: str | Path,
    model: WaterNet,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, int, float]:
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=False)
    if isinstance(checkpoint, dict) and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("epoch", 0)) + 1, int(checkpoint.get("step", 0)), float(checkpoint.get("best_loss", float("inf")))


def _evaluate(model: WaterNet, loader: DataLoader, args: argparse.Namespace, device: torch.device) -> float:
    model.eval()
    l1 = nn.L1Loss()
    ssim = SSIMLoss().to(device)
    losses = []
    max_batches = args.val_max_batches
    with torch.no_grad():
        for batch_idx, (_, wb, ce, gc, target) in enumerate(tqdm(loader, desc="WaterNet validation", leave=False), start=1):
            wb = wb.to(device)
            ce = ce.to(device)
            gc = gc.to(device)
            target = target.to(device)
            out = model(wb, ce, gc)
            loss_l1 = l1(out, target)
            loss_ssim = ssim(out, target)
            losses.append((args.lambda_l1 * loss_l1 + args.lambda_ssim * loss_ssim).item())
            if max_batches > 0 and batch_idx >= max_batches:
                break
    model.train()
    return float(sum(losses) / max(len(losses), 1))


def train(args: argparse.Namespace) -> None:
    device = _device(args.device)
    train_dataset = WaterNetDataset(args.uieb_raw_dir, args.uieb_reference_dir, args.image_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )

    val_input_dir = args.val_input_dir or args.uieb_raw_dir
    val_reference_dir = args.val_reference_dir or args.uieb_reference_dir
    val_dataset = WaterNetDataset(val_input_dir, val_reference_dir, args.image_size)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    model = WaterNet(channels=args.channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    l1 = nn.L1Loss()
    ssim = SSIMLoss().to(device)
    log = CSVLogger(args.log_csv, ["epoch", "step", "loss", "l1", "ssim", "val_loss"])

    save_dir = Path(args.save_dir)
    sample_dir = Path(args.sample_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    step = 0
    best_loss = float("inf")
    if args.resume:
        start_epoch, step, best_loss = _load_checkpoint(args.resume, model, optimizer, device)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"WaterNet epoch {epoch}/{args.epochs}")
        for raw, wb, ce, gc, target in pbar:
            raw = raw.to(device)
            wb = wb.to(device)
            ce = ce.to(device)
            gc = gc.to(device)
            target = target.to(device)

            out = model(wb, ce, gc)
            loss_l1 = l1(out, target)
            loss_ssim = ssim(out, target)
            loss = args.lambda_l1 * loss_l1 + args.lambda_ssim * loss_ssim

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            log.log({
                "epoch": epoch,
                "step": step,
                "loss": loss.item(),
                "l1": loss_l1.item(),
                "ssim": loss_ssim.item(),
            })

            if step % args.sample_every == 0:
                save_comparison(
                    sample_dir / f"step_{step}.png",
                    [tensor_to_image(raw), tensor_to_image(out), tensor_to_image(target)],
                    ["input", "waternet", "reference"],
                )

        val_loss = ""
        if epoch % args.val_interval == 0:
            val_value = _evaluate(model, val_loader, args, device)
            val_loss = val_value
            print(f"[WaterNet validation] epoch={epoch} loss={val_value:.6f}")
            if val_value < best_loss:
                best_loss = val_value
                _save_checkpoint(save_dir / "waternet_best.pth", model, optimizer, epoch, step, args, best_loss)
                torch.save({"model": model.state_dict(), "args": vars(args)}, save_dir / "waternet.pth")
                print(f"[Best] Saved WaterNet best checkpoint at epoch {epoch}")

        log.log({"epoch": epoch, "step": step, "val_loss": val_loss})
        _save_checkpoint(save_dir / f"epoch_{epoch}.pth", model, optimizer, epoch, step, args, best_loss)
        _save_checkpoint(save_dir / "latest.pth", model, optimizer, epoch, step, args, best_loss)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train WaterNet on UIEB raw/reference pairs.")
    parser.add_argument("--uieb-raw-dir", default="data/raw_underwater/UIEB/raw-890")
    parser.add_argument("--uieb-reference-dir", default="data/raw_underwater/UIEB/reference-890")
    parser.add_argument("--val-input-dir", default="")
    parser.add_argument("--val-reference-dir", default="")
    parser.add_argument("--save-dir", default="checkpoints/waternet")
    parser.add_argument("--sample-dir", default="outputs/train_samples/waternet")
    parser.add_argument("--log-csv", default="logs/waternet_train_log.csv")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", default="")
    parser.add_argument("--lambda-l1", type=float, default=1.0)
    parser.add_argument("--lambda-ssim", type=float, default=0.5)
    parser.add_argument("--sample-every", type=int, default=200)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--val-max-batches", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
