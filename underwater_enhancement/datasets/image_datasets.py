from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.image_io import image_to_tensor, list_images, pil_loader


BRANCH_DIRS = {
    "blue": "blue_cast",
    "green": "green_cast",
    "lowlight": "low_light",
    "low_light": "low_light",
    "blur": "blur",
}


def _key(path: Path) -> str:
    return path.stem.lower()


def build_uieb_pairs(raw_dir: str | Path, reference_dir: str | Path) -> list[tuple[Path, Path]]:
    raw = list_images(raw_dir)
    refs = list_images(reference_dir)
    ref_by_name = {_key(p): p for p in refs}
    pairs: list[tuple[Path, Path]] = []
    for idx, raw_path in enumerate(raw):
        ref = ref_by_name.get(_key(raw_path))
        if ref is None and idx < len(refs):
            ref = refs[idx]
        if ref is not None:
            pairs.append((raw_path, ref))
    return pairs


def _resize_tensor(path: Path, image_size: int) -> torch.Tensor:
    return image_to_tensor(__import__("numpy").array(pil_loader(path, image_size)))


class PairedImageDataset(Dataset):
    def __init__(self, pairs: Iterable[tuple[Path, Path]], image_size: int = 256):
        self.pairs = list(pairs)
        if not self.pairs:
            raise FileNotFoundError("No paired images found.")
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        a, b = self.pairs[idx]
        return _resize_tensor(a, self.image_size), _resize_tensor(b, self.image_size)


class CycleGANDataset(Dataset):
    def __init__(self, domain_a_dirs: Iterable[str | Path], domain_b_dir: str | Path, image_size: int = 256):
        self.domain_a = []
        for d in domain_a_dirs:
            self.domain_a.extend(list_images(d))
        self.domain_b = list_images(domain_b_dir)
        if not self.domain_a:
            raise FileNotFoundError("No images found for CycleGAN Domain A.")
        if not self.domain_b:
            raise FileNotFoundError("No images found for CycleGAN Domain B.")
        self.image_size = image_size

    def __len__(self) -> int:
        return max(len(self.domain_a), len(self.domain_b))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.domain_a[idx % len(self.domain_a)]
        b = self.domain_b[idx % len(self.domain_b)]
        return _resize_tensor(a, self.image_size), _resize_tensor(b, self.image_size)


class BranchPretrainDataset(Dataset):
    def __init__(
        self,
        branch: str,
        classified_root: str | Path,
        physical_root: str | Path,
        reference_dir: str | Path,
        mapping_csv: str | Path,
        image_size: int = 256,
    ):
        self.branch = BRANCH_DIRS.get(branch, branch)
        self.image_size = image_size
        self.samples: list[tuple[Path, Path]] = []
        refs = {_key(p): p for p in list_images(reference_dir)}
        ref_list = list_images(reference_dir)

        real_dir = Path(classified_root) / self.branch
        for img in list_images(real_dir):
            ref = refs.get(_key(img))
            if ref is None and ref_list:
                ref = refs.get(_key(Path(img.name))) or ref_list[min(len(self.samples), len(ref_list) - 1)]
            if ref is not None:
                self.samples.append((img, ref))

        mapping_path = Path(mapping_csv)
        physical_dir = Path(physical_root) / self.branch
        if mapping_path.exists():
            with mapping_path.open("r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("degradation_type") != self.branch:
                        continue
                    inp = Path(row["degraded_image"])
                    if not inp.is_absolute():
                        inp = mapping_path.parent.parent / inp
                    ref = Path(row.get("source_reference_path", ""))
                    if not ref.is_absolute():
                        ref = mapping_path.parent.parent / ref
                    if inp.exists() and ref.exists():
                        self.samples.append((inp, ref))
        else:
            for img in list_images(physical_dir):
                stem = img.stem
                for suffix in (f"_{self.branch}", "_blue", "_green", "_lowlight", "_blur"):
                    stem = stem.removesuffix(suffix)
                ref = refs.get(stem.lower())
                if ref is not None:
                    self.samples.append((img, ref))

        if not self.samples:
            raise FileNotFoundError(f"No pretraining pairs found for branch {self.branch}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        inp, ref = self.samples[idx]
        return _resize_tensor(inp, self.image_size), _resize_tensor(ref, self.image_size)


class TestImageDataset:
    def __init__(self, name: str, input_dir: str | Path, reference_dir: str | Path | None = None):
        self.name = name
        self.input_dir = Path(input_dir)
        self.reference_dir = Path(reference_dir) if reference_dir else None
        self.images = list_images(self.input_dir)
        refs = list_images(self.reference_dir) if self.reference_dir else []
        self.refs = {_key(p): p for p in refs}

    def reference_for(self, image: Path) -> Path | None:
        if not self.reference_dir:
            return None
        ref = self.refs.get(_key(image))
        if ref is not None:
            return ref
        candidate = self.reference_dir / image.name
        return candidate if candidate.exists() else None


def save_average_metrics(metrics_csv: str | Path, average_csv: str | Path) -> None:
    df = pd.read_csv(metrics_csv)
    numeric = df[["PSNR", "SSIM", "UIQM", "UCIQE"]].apply(pd.to_numeric, errors="coerce")
    grouped = numeric.join(df["dataset"]).groupby("dataset").mean(numeric_only=True)
    overall = pd.DataFrame([numeric.mean(numeric_only=True)], index=["overall"])
    out = pd.concat([grouped, overall])
    Path(average_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(average_csv)

