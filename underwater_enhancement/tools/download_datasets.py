from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Default Hugging Face dataset repositories.
# These are intentionally centralized so users can replace them if their Hugging
# Face mirror uses a different owner/name while keeping the script unchanged.
HF_DATASETS = {
    "uieb": {
        "repo_id": "Hikari0608/UIEB",
        "url": "https://huggingface.co/datasets/Hikari0608/UIEB",
        "target_dir": "UIEB",
        "expected_dirs": ["raw-890", "reference-890", "challenging-60"],
    },
    "euvp": {
        "repo_id": "Ken1053/EUVP",
        "url": "https://huggingface.co/datasets/Ken1053/EUVP",
        "target_dir": "EUVP",
        "expected_dirs": ["EUVP_Paired", "Unpaired", "test_samples", "eval_data"],
    },
}

ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz")
SKIP_NAMES = {".git", ".gitattributes", "README.md", "README.txt"}


def _require_snapshot_download():
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install with `pip install huggingface_hub` or `pip install -r requirements.txt`.") from exc
    return snapshot_download


def _is_archive(path: Path) -> bool:
    suffixes = "".join(path.suffixes).lower()
    return path.suffix.lower() == ".zip" or suffixes.endswith((".tar.gz", ".tgz")) or path.suffix.lower() == ".tar"


def _safe_extract_zip(path: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        for member in zf.infolist():
            target = (dst / member.filename).resolve()
            if not str(target).startswith(str(dst.resolve())):
                raise RuntimeError(f"Unsafe path in archive: {member.filename}")
        zf.extractall(dst)


def _safe_extract_tar(path: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path) as tf:
        for member in tf.getmembers():
            target = (dst / member.name).resolve()
            if not str(target).startswith(str(dst.resolve())):
                raise RuntimeError(f"Unsafe path in archive: {member.name}")
        tf.extractall(dst)


def extract_archive(path: Path, dst: Path) -> None:
    if path.suffix.lower() == ".zip":
        _safe_extract_zip(path, dst)
        return
    _safe_extract_tar(path, dst)


def copy_item(src: Path, dst: Path, overwrite: bool) -> None:
    if src.name in SKIP_NAMES:
        return
    if src.is_dir():
        if dst.exists() and overwrite:
            shutil.rmtree(dst)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        return
    if _is_archive(src):
        extract_archive(src, dst.parent)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return
    shutil.copy2(src, dst)


def materialize_snapshot(snapshot_dir: Path, target_dir: Path, overwrite: bool) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for item in snapshot_dir.iterdir():
        if item.name in SKIP_NAMES:
            continue
        # Some HF mirrors include a top-level UIEB/EUVP folder. In that case,
        # place its children directly under the project target directory.
        if item.is_dir() and item.name.lower() == target_dir.name.lower():
            for child in item.iterdir():
                copy_item(child, target_dir / child.name, overwrite)
        else:
            copy_item(item, target_dir / item.name, overwrite)


def download_hf_dataset(name: str, output_root: Path, cache_root: Path, token: str | None, revision: str | None, overwrite: bool) -> Path:
    spec = HF_DATASETS[name]
    snapshot_download = _require_snapshot_download()
    snapshot_dir = cache_root / spec["target_dir"]
    target_dir = output_root / spec["target_dir"]
    print(f"Downloading {name.upper()} from Hugging Face: {spec['url']}")
    downloaded = Path(
        snapshot_download(
            repo_id=spec["repo_id"],
            repo_type="dataset",
            revision=revision,
            token=token,
            local_dir=str(snapshot_dir),
            local_dir_use_symlinks=False,
        )
    )
    materialize_snapshot(downloaded, target_dir, overwrite)
    print(f"Saved {name.upper()} to: {target_dir}")
    missing = [d for d in spec["expected_dirs"] if not (target_dir / d).exists()]
    if missing:
        print(f"Warning: {target_dir} is missing expected directories: {', '.join(missing)}")
        print("If your Hugging Face mirror already uses a different but valid structure, update the project paths or HF_DATASETS constants.")
    return target_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download UIEB and EUVP from Hugging Face into data/raw_underwater.")
    parser.add_argument("--datasets", nargs="+", choices=["uieb", "euvp"], default=["uieb", "euvp"], help="Datasets to download.")
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "data" / "raw_underwater"), help="Directory used by the project.")
    parser.add_argument("--cache-root", default=str(PROJECT_ROOT / "data" / "_hf_downloads"), help="Local Hugging Face snapshot cache for this project.")
    parser.add_argument("--revision", default=None, help="Optional Hugging Face branch, tag, or commit hash.")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN", ""), help="Hugging Face token. Defaults to HF_TOKEN env var.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files in data/raw_underwater.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    cache_root = Path(args.cache_root).resolve()
    token = args.token or None
    for name in args.datasets:
        download_hf_dataset(name, output_root, cache_root, token, args.revision, args.overwrite)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
