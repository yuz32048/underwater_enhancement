from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import BranchPretrainDataset
from models.branches import branch_from_name
from train.losses import SSIMLoss
from utils.image_io import save_comparison, tensor_to_image
from utils.logger import CSVLogger


BRANCHES = ["blue", "green", "lowlight", "blur"]
WEIGHT_NAMES = {
    "blue": "blue_branch.pth",
    "green": "green_branch.pth",
    "lowlight": "lowlight_branch.pth",
    "blur": "blur_branch.pth",
}


def _device(name: str) -> torch.device:
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else ("cpu" if name == "auto" else name))


def train_one(branch: str, args: argparse.Namespace) -> None:
    device = _device(args.device)
    dataset = BranchPretrainDataset(
        branch=branch,
        classified_root=args.classified_root,
        physical_root=args.physical_root,
        reference_dir=args.reference_dir,
        mapping_csv=args.mapping_csv,
        image_size=args.image_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)
    model = branch_from_name(branch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    l1 = nn.L1Loss()
    ssim = SSIMLoss().to(device)
    log = CSVLogger(args.log_csv, ["branch", "epoch", "step", "loss", "l1", "ssim"])
    save_dir = Path(args.save_dir)
    sample_dir = Path(args.sample_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    step = 0

    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(loader, desc=f"{branch} branch epoch {epoch}/{args.epochs}")
        for inp, target in pbar:
            inp = inp.to(device)
            target = target.to(device)
            out = model(inp)
            loss_l1 = l1(out, target)
            loss_ssim = ssim(out, target)
            loss = args.lambda_l1 * loss_l1 + args.lambda_ssim * loss_ssim
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            log.log({"branch": branch, "epoch": epoch, "step": step, "loss": loss.item(), "l1": loss_l1.item(), "ssim": loss_ssim.item()})
            if step % args.sample_every == 0:
                save_comparison(sample_dir / f"{branch}_step_{step}.png", [tensor_to_image(inp), tensor_to_image(out), tensor_to_image(target)], ["input", "output", "reference"])

    torch.save({"model": model.state_dict(), "branch": branch, "epochs": args.epochs}, save_dir / WEIGHT_NAMES[branch])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised pretraining for the four expert branches.")
    parser.add_argument("--branch", choices=BRANCHES + ["all"], default="all")
    parser.add_argument("--classified-root", default="data/processed/UIEB_classified")
    parser.add_argument("--physical-root", default="data/processed/physical_degradation")
    parser.add_argument("--reference-dir", default="data/raw_underwater/UIEB/reference-890")
    parser.add_argument("--mapping-csv", default="results/physical_degradation_mapping.csv")
    parser.add_argument("--save-dir", default="checkpoints/pretrained_branches")
    parser.add_argument("--sample-dir", default="outputs/train_samples/branches")
    parser.add_argument("--log-csv", default="logs/branch_train_log.csv")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lambda-l1", type=float, default=1.0)
    parser.add_argument("--lambda-ssim", type=float, default=0.5)
    parser.add_argument("--sample-every", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    branches = BRANCHES if args.branch == "all" else [args.branch]
    for branch_name in branches:
        train_one(branch_name, args)

