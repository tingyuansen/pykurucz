"""Disk cache for convert_atm_to_npz outputs."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


def npz_cache_key(
    *,
    atm_path: Path,
    atlas_tables: Path,
    molecules: Path,
    converter_script: Path,
) -> str:
    """SHA256 key from atm content + dependency mtimes/sizes."""
    h = hashlib.sha256()
    h.update(atm_path.read_bytes())
    for path in (atlas_tables, molecules, converter_script):
        if path.is_file():
            st = path.stat()
            h.update(str(path).encode())
            h.update(str(st.st_mtime_ns).encode())
            h.update(str(st.st_size).encode())
    return h.hexdigest()


def default_npz_cache_dir(repo_root: Path) -> Path:
    return repo_root / "results" / "synthe_npz_cache"


def resolve_cached_npz(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.npz"


def copy_cached_npz(cached: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    shutil.copy2(cached, dest)


def store_npz_cache(cache_dir: Path, key: str, src: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = resolve_cached_npz(cache_dir, key)
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    return dest
