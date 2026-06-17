import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.cyclegan import CycleGAN
from train.dataset import UnpairedImageDataset
from train.losses import SSIMLoss, gan_loss
from utils.image_io import load_config, save_comparison, tensor_to_image
from utils.logger import CSVLogger, setup_logger


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _save_checkpoint(path: Path, model: CycleGAN, optim_g, optim_d, epoch: int, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "step": step,
        "model": model.state_dict(),
        "optim_g": optim_g.state_dict(),
        "optim_d": optim_d.state_dict(),
    }, path)


def _load_checkpoint(path: Path, model: CycleGAN, optim_g, optim_d, device: torch.device) -> tuple[int, int]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optim_g.load_state_dict(ckpt["optim_g"])
    optim_d.load_state_dict(ckpt["optim_d"])
    return int(ckpt.get("epoch", 0)) + 1, int(ckpt.get("step", 0))


def train(cfg: dict, root: str | Path = ".") -> None:
    root = Path(root)
    tc = cfg["training"]
    paths = cfg["paths"]
    device = _device(tc.get("device", "auto"))
    logger = setup_logger("train", root / paths["log_dir"] / "train.log")

    dataset = UnpairedImageDataset(root / paths["degraded_dir"], root / paths["clean_dir"], int(tc["image_size"]))
    loader = DataLoader(dataset, batch_size=int(tc["batch_size"]), shuffle=True, num_workers=int(tc.get("num_workers", 2)), drop_last=True)

    model = CycleGAN().to(device)
    optim_g = torch.optim.Adam(list(model.G_AB.parameters()) + list(model.G_BA.parameters()), lr=float(tc["lr"]), betas=(float(tc.get("beta1", 0.5)), 0.999))
    optim_d = torch.optim.Adam(list(model.D_A.parameters()) + list(model.D_B.parameters()), lr=float(tc["lr"]), betas=(float(tc.get("beta1", 0.5)), 0.999))
    adv_criterion = nn.MSELoss()
    l1 = nn.L1Loss()
    ssim_loss = SSIMLoss().to(device)
    csv_logger = CSVLogger(root / paths["log_dir"] / "training_log.csv", ["epoch", "step", "loss_g", "loss_d", "loss_cycle", "loss_identity", "loss_ssim"])

    start_epoch, global_step = 0, 0
    if tc.get("resume"):
        start_epoch, global_step = _load_checkpoint(root / tc["resume"], model, optim_g, optim_d, device)
        logger.info("Resumed from %s at epoch %s", tc["resume"], start_epoch)

    for epoch in range(start_epoch, int(tc["epochs"])):
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{tc['epochs']}")
        for real_a, real_b in pbar:
            real_a = real_a.to(device)
            real_b = real_b.to(device)

            fake_b = model.G_AB(real_a)
            rec_a = model.G_BA(fake_b)
            fake_a = model.G_BA(real_b)
            rec_b = model.G_AB(fake_a)
            id_a = model.G_BA(real_a)
            id_b = model.G_AB(real_b)

            loss_adv = gan_loss(model.D_B(fake_b), True, adv_criterion) + gan_loss(model.D_A(fake_a), True, adv_criterion)
            loss_cycle = l1(rec_a, real_a) + l1(rec_b, real_b)
            loss_identity = l1(id_a, real_a) + l1(id_b, real_b)
            loss_ssim = ssim_loss(rec_a, real_a) + ssim_loss(rec_b, real_b)
            loss_g = (
                float(tc["lambda_adv"]) * loss_adv
                + float(tc["lambda_cycle"]) * loss_cycle
                + float(tc["lambda_identity"]) * loss_identity
                + float(tc["lambda_ssim"]) * loss_ssim
            )
            optim_g.zero_grad()
            loss_g.backward()
            optim_g.step()

            loss_d_a = 0.5 * (gan_loss(model.D_A(real_a), True, adv_criterion) + gan_loss(model.D_A(fake_a.detach()), False, adv_criterion))
            loss_d_b = 0.5 * (gan_loss(model.D_B(real_b), True, adv_criterion) + gan_loss(model.D_B(fake_b.detach()), False, adv_criterion))
            loss_d = loss_d_a + loss_d_b
            optim_d.zero_grad()
            loss_d.backward()
            optim_d.step()

            global_step += 1
            pbar.set_postfix(loss_g=f"{loss_g.item():.3f}", loss_d=f"{loss_d.item():.3f}")
            csv_logger.log({
                "epoch": epoch + 1,
                "step": global_step,
                "loss_g": loss_g.item(),
                "loss_d": loss_d.item(),
                "loss_cycle": loss_cycle.item(),
                "loss_identity": loss_identity.item(),
                "loss_ssim": loss_ssim.item(),
            })
            if global_step % int(tc.get("save_sample_every", 200)) == 0:
                save_comparison(root / paths["sample_dir"] / f"step_{global_step}.png", [tensor_to_image(real_a), tensor_to_image(fake_b), tensor_to_image(real_b)], ["input", "enhanced", "clean"])

        if (epoch + 1) % int(tc.get("checkpoint_every_epoch", 1)) == 0:
            _save_checkpoint(root / paths["checkpoint_dir"] / f"epoch_{epoch + 1}.pth", model, optim_g, optim_d, epoch, global_step)
            _save_checkpoint(root / paths["checkpoint_dir"] / "latest.pth", model, optim_g, optim_d, epoch, global_step)
            logger.info("Saved checkpoint for epoch %s", epoch + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    train(load_config(config_path), config_path.parent)
