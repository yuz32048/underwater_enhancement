import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.metrics import psnr, ssim, uciqe, uiqm
from models.cyclegan import CycleGAN
from utils.image_io import image_to_tensor, list_images, load_config, pil_loader, read_image_rgb, save_comparison, save_image_rgb, tensor_to_image


def run_test(cfg: dict, root: str | Path = ".") -> pd.DataFrame:
    root = Path(root)
    tc = cfg["testing"]
    train_cfg = cfg["training"]
    device = torch.device("cuda" if torch.cuda.is_available() and train_cfg.get("device", "auto") != "cpu" else "cpu")
    ckpt_path = root / (tc.get("checkpoint") or (Path(cfg["paths"]["checkpoint_dir"]) / "latest.pth"))
    out_dir = root / tc.get("output_dir", cfg["paths"]["enhanced_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    model = CycleGAN().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    image_size = int(tc["image_size"])

    rows: List[Dict] = []
    input_dir = root / tc["input_dir"]
    target_dir = root / tc.get("target_dir", "")
    with torch.no_grad():
        for path in tqdm(list_images(input_dir), desc="Enhancing test images"):
            pil = pil_loader(path).resize((image_size, image_size))
            tensor = image_to_tensor(__import__("numpy").array(pil)).unsqueeze(0).to(device)
            enhanced = model.G_AB(tensor)
            enhanced_img = tensor_to_image(enhanced)
            out_path = out_dir / f"{path.stem}_enhanced.png"
            save_image_rgb(out_path, enhanced_img)
            before = tensor_to_image(tensor)
            save_comparison(out_dir / "comparisons" / f"{path.stem}_comparison.png", [before, enhanced_img], ["before", "after"])

            row = {"image_path": path.relative_to(input_dir).as_posix(), "enhanced_path": out_path.relative_to(out_dir).as_posix(), "UIQM": uiqm(enhanced_img), "UCIQE": uciqe(enhanced_img)}
            target = target_dir / path.name
            if target.exists():
                target_img = read_image_rgb(target)
                target_img = __import__("cv2").resize(target_img, (enhanced_img.shape[1], enhanced_img.shape[0]))
                row["PSNR"] = psnr(enhanced_img, target_img)
                row["SSIM"] = ssim(enhanced_img, target_img)
            else:
                row["PSNR"] = ""
                row["SSIM"] = ""
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "evaluation_metrics.csv", index=False)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    run_test(load_config(config_path), config_path.parent)
