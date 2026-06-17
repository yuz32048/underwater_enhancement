import argparse
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]


DATASETS: Dict[str, Dict] = {
    "uieb": {
        "homepage": "https://li-chongyi.github.io/proj_benchmark.html",
        "license_note": "UIEB is for academic/non-commercial use. Redistribution is forbidden by the dataset page.",
        "files": [
            {
                "name": "UIEB_raw.zip",
                "google_id": "12W_kkblc2Vryb9zHQ6BfGQ_NKUfXYk13",
                "password": "1234567",
            },
            {
                "name": "UIEB_reference.zip",
                "google_id": "1cA-8CzajnVEL4feBRKdBxjEe6hwql6Z7",
                "password": "8901234",
            },
        ],
    },
    "uieb_challenging": {
        "homepage": "https://li-chongyi.github.io/proj_benchmark.html",
        "license_note": "UIEB challenging set is for academic/non-commercial use. Redistribution is forbidden by the dataset page.",
        "files": [
            {
                "name": "UIEB_challenging.zip",
                # The official page should be used if this id becomes stale.
                "google_id": "",
                "password": "5678901",
            },
        ],
    },
    "euvp": {
        "homepage": "https://irvlab.cs.umn.edu/resources/euvp-dataset",
        "license_note": "Use EUVP according to the terms stated by the IRVLab/FUnIE-GAN dataset maintainers.",
        "google_folder": "1ZEql33CajGfHHzPe1vFxUFCMcP0YbZb3",
    },
}


def _require_gdown():
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install with `pip install gdown` or `pip install -r requirements.txt`.") from exc
    return gdown


def download_google_file(file_id: str, output_dir: Path, name: str, quiet: bool = False) -> Path:
    if not file_id:
        raise ValueError(f"No Google Drive file id configured for {name}. Please download it from the dataset homepage manually.")
    gdown = _require_gdown()
    output_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    output = output_dir / name
    try:
        downloaded = gdown.download(url, str(output), quiet=quiet, fuzzy=True)
    except TypeError:
        downloaded = gdown.download(url, str(output), quiet=quiet)
    if downloaded is None:
        raise RuntimeError(f"Download failed for {name}. The Google Drive link may require browser confirmation or may be stale.")
    return Path(downloaded)


def download_google_folder(folder_id: str, output_dir: Path, quiet: bool = False) -> Path:
    gdown = _require_gdown()
    output_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    try:
        files = gdown.download_folder(url=url, output=str(output_dir), quiet=quiet, use_cookies=False)
    except TypeError:
        files = gdown.download_folder(url=url, output=str(output_dir), quiet=quiet)
    if not files:
        raise RuntimeError("Google Drive folder download failed. Try opening the official dataset page in a browser.")
    return output_dir


def _safe_extract_zip(zip_path: Path, dst: Path, password: Optional[str] = None) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    pwd = password.encode("utf-8") if password else None
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = (dst / member.filename).resolve()
            if not str(target).startswith(str(dst.resolve())):
                raise RuntimeError(f"Unsafe path in archive: {member.filename}")
        zf.extractall(dst, pwd=pwd)


def _safe_extract_tar(tar_path: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tf:
        for member in tf.getmembers():
            target = (dst / member.name).resolve()
            if not str(target).startswith(str(dst.resolve())):
                raise RuntimeError(f"Unsafe path in archive: {member.name}")
        tf.extractall(dst)


def extract_archive(path: Path, dst: Path, password: Optional[str] = None) -> bool:
    suffixes = "".join(path.suffixes).lower()
    if path.suffix.lower() == ".zip":
        _safe_extract_zip(path, dst, password)
        return True
    if suffixes.endswith(".tar.gz") or suffixes.endswith(".tgz") or path.suffix.lower() == ".tar":
        _safe_extract_tar(path, dst)
        return True
    return False


def organize_uieb(root: Path) -> None:
    raw_dir = root / "raw"
    ref_dir = root / "reference"
    raw_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        lower = folder.name.lower()
        if "raw" in lower:
            _move_images(folder, raw_dir)
        elif "ref" in lower or "reference" in lower or "best" in lower:
            _move_images(folder, ref_dir)


def _move_images(src: Path, dst: Path) -> None:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    for img in src.rglob("*"):
        if img.suffix.lower() not in exts:
            continue
        target = dst / img.name
        if target.exists():
            target = dst / f"{img.stem}_{abs(hash(str(img))) % 100000}{img.suffix}"
        shutil.copy2(img, target)


def download_uieb(output_root: Path, extract: bool, quiet: bool) -> None:
    spec = DATASETS["uieb"]
    target = output_root / "UIEB"
    archives = target / "archives"
    print(f"UIEB homepage: {spec['homepage']}")
    print(spec["license_note"])
    for item in spec["files"]:
        archive = download_google_file(item["google_id"], archives, item["name"], quiet=quiet)
        print(f"Downloaded: {archive}")
        if extract:
            extract_dir = target / Path(item["name"]).stem
            extracted = extract_archive(archive, extract_dir, item.get("password"))
            print(f"Extracted: {extracted} -> {extract_dir}")
    if extract:
        organize_uieb(target)
        print(f"Organized UIEB folders: {target / 'raw'} and {target / 'reference'}")


def download_euvp(output_root: Path, quiet: bool) -> None:
    spec = DATASETS["euvp"]
    target = output_root / "EUVP"
    print(f"EUVP homepage: {spec['homepage']}")
    print(spec["license_note"])
    download_google_folder(spec["google_folder"], target, quiet=quiet)
    print(f"Downloaded EUVP folder to: {target}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download UIEB and EUVP underwater image datasets.")
    parser.add_argument("--datasets", nargs="+", default=["uieb", "euvp"], choices=["uieb", "euvp"], help="Datasets to download.")
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "data" / "external"), help="Directory where datasets will be saved.")
    parser.add_argument("--no-extract", action="store_true", help="Keep downloaded archives without extracting them.")
    parser.add_argument("--quiet", action="store_true", help="Reduce gdown output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    for name in args.datasets:
        if name == "uieb":
            download_uieb(output_root, extract=not args.no_extract, quiet=args.quiet)
        elif name == "euvp":
            download_euvp(output_root, quiet=args.quiet)


if __name__ == "__main__":
    main()
