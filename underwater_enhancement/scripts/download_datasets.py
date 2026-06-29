
from __future__ import annotations

import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


UIEB_URL = "https://huggingface.co/datasets/yuz32048/underwater_img/resolve/main/UIEB.zip"
EUVP_URL = "https://huggingface.co/datasets/yuz32048/underwater_img/resolve/main/EUVP.zip"

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw_underwater"


def download_file(url: str, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(save_path, "wb") as f, tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        desc=save_path.name,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def extract_zip(zip_path: Path, target_dir: Path):
    print(f"Extracting {zip_path.name} ...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)

    print(f"Extracted to {target_dir}")


def download_and_extract(name: str, url: str):
    zip_path = DATA_DIR / f"{name}.zip"

    print(f"\nDownloading {name}...")
    download_file(url, zip_path)

    print(f"\nExtracting {name}...")
    extract_zip(zip_path, DATA_DIR)

    print(f"Removing {zip_path.name}")
    zip_path.unlink(missing_ok=True)

    print(f"{name} completed.\n")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    download_and_extract("UIEB", UIEB_URL)
    download_and_extract("EUVP", EUVP_URL)

    print("All datasets downloaded successfully.")


if __name__ == "__main__":
    main()
