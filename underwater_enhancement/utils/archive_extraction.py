import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional

ARCHIVE_EXTENSIONS = {".zip", ".rar", ".tar", ".tgz", ".gz", ".bz2", ".xz", ".7z"}


def prepare_archives(input_dir: str | Path, cfg: dict) -> List[Path]:
    archive_cfg = cfg.get("archive_extraction", {})
    if not archive_cfg.get("enabled", True):
        return []

    input_dir = Path(input_dir)
    if not input_dir.exists():
        return []

    extract_root = input_dir / archive_cfg.get("extract_dir_name", "_extracted_archives")
    overwrite = bool(archive_cfg.get("overwrite", False))
    passwords = archive_cfg.get("passwords", [""])
    extracted_dirs: List[Path] = []

    for archive in sorted(input_dir.iterdir()):
        if not archive.is_file() or not _is_archive(archive):
            continue
        target = extract_root / archive.stem
        marker = target / ".extracted"
        if marker.exists() and not overwrite:
            extracted_dirs.append(target)
            continue
        target.mkdir(parents=True, exist_ok=True)
        _extract_archive(archive, target, passwords=passwords, overwrite=overwrite)
        marker.write_text(f"source={archive.name}\n", encoding="utf-8")
        extracted_dirs.append(target)
    return extracted_dirs


def _is_archive(path: Path) -> bool:
    name = path.name.lower()
    if name.endswith((".tar.gz", ".tar.bz2", ".tar.xz")):
        return True
    return path.suffix.lower() in ARCHIVE_EXTENSIONS


def _extract_archive(archive: Path, target: Path, passwords: Iterable[str], overwrite: bool) -> None:
    suffixes = "".join(archive.suffixes).lower()
    suffix = archive.suffix.lower()
    if suffix == ".zip":
        _extract_zip(archive, target, passwords)
        return
    if suffix == ".rar":
        _extract_rar(archive, target, passwords, overwrite=overwrite)
        return
    if suffixes.endswith((".tar.gz", ".tar.bz2", ".tar.xz")) or suffix in {".tar", ".tgz"}:
        _extract_tar(archive, target)
        return
    if suffix == ".7z":
        _extract_with_external_tool(archive, target, passwords, overwrite=overwrite)
        return
    raise RuntimeError(f"Unsupported archive type: {archive}")


def _extract_zip(archive: Path, target: Path, passwords: Iterable[str]) -> None:
    last_error: Optional[Exception] = None
    with zipfile.ZipFile(archive) as zf:
        _validate_zip_members(zf, target)
        for password in passwords:
            pwd = password.encode("utf-8") if password else None
            try:
                zf.extractall(target, pwd=pwd)
                return
            except RuntimeError as exc:
                last_error = exc
    raise RuntimeError(f"Failed to extract zip archive {archive}. Check archive password.") from last_error


def _validate_zip_members(zf: zipfile.ZipFile, target: Path) -> None:
    root = target.resolve()
    for member in zf.infolist():
        dst = (target / member.filename).resolve()
        if not str(dst).startswith(str(root)):
            raise RuntimeError(f"Unsafe path in archive: {member.filename}")


def _extract_tar(archive: Path, target: Path) -> None:
    root = target.resolve()
    with tarfile.open(archive) as tf:
        for member in tf.getmembers():
            dst = (target / member.name).resolve()
            if not str(dst).startswith(str(root)):
                raise RuntimeError(f"Unsafe path in archive: {member.name}")
        tf.extractall(target)


def _extract_rar(archive: Path, target: Path, passwords: Iterable[str], overwrite: bool) -> None:
    try:
        import rarfile
    except ImportError:
        _extract_with_external_tool(archive, target, passwords, overwrite=overwrite)
        return

    try:
        last_error: Optional[Exception] = None
        with rarfile.RarFile(archive) as rf:
            _validate_rar_members(rf, target)
            for password in passwords:
                try:
                    rf.extractall(target, pwd=password or None)
                    return
                except Exception as exc:
                    last_error = exc
        raise RuntimeError(f"Failed to extract rar archive {archive}. Check archive password.") from last_error
    except rarfile.RarCannotExec:
        _extract_with_external_tool(archive, target, passwords, overwrite=overwrite)
    except rarfile.RarUnknownError:
        _extract_with_external_tool(archive, target, passwords, overwrite=overwrite)


def _validate_rar_members(rf, target: Path) -> None:
    root = target.resolve()
    for member in rf.infolist():
        dst = (target / member.filename).resolve()
        if not str(dst).startswith(str(root)):
            raise RuntimeError(f"Unsafe path in archive: {member.filename}")


def _extract_with_external_tool(archive: Path, target: Path, passwords: Iterable[str], overwrite: bool) -> None:
    tool = _find_archive_tool()
    if tool is None:
        raise RuntimeError(
            f"Cannot extract {archive.name}. Install 7-Zip/unrar, or install Python package `rarfile` with a RAR backend."
        )
    mode = "-aoa" if overwrite else "-aos"
    last_output = ""
    for password in passwords:
        cmd = [tool, "x", str(archive), f"-o{target}", mode, "-y"]
        cmd.append(f"-p{password}" if password else "-p")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if result.returncode == 0:
            return
        last_output = result.stdout
    raise RuntimeError(f"External extraction failed for {archive}. Output:\n{last_output}")


def _find_archive_tool() -> Optional[str]:
    for name in ["7z", "7za", "7zr", "unrar"]:
        path = shutil.which(name)
        if path:
            return path
    return None
