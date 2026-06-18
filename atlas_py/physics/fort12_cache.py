"""Fort.12 (SELECTLINES output) disk cache keyed by line-catalog identity."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Iterable, Optional

from .kapcont import build_waveset


def _catalog_fingerprint(paths: Iterable[Optional[Path]]) -> str:
    """Hash resolved catalog paths with mtime+size for cache invalidation."""
    h = hashlib.sha256()
    for path in paths:
        if path is None:
            h.update(b"<none>")
            continue
        p = Path(path).resolve()
        h.update(str(p).encode())
        if p.exists():
            st = p.stat()
            h.update(str(st.st_mtime_ns).encode())
            h.update(str(st.st_size).encode())
        else:
            h.update(b"<missing>")
    return h.hexdigest()


def fort12_cache_key(
    *,
    teff: float,
    fort11_path: Optional[Path],
    fort111_path: Optional[Path],
    fort21_path: Optional[Path],
    fort31_path: Optional[Path],
    fort41_path: Optional[Path],
    fort51_path: Optional[Path],
    fort61_path: Optional[Path],
) -> str:
    """Stable cache key: catalog fingerprints + teff-bucketed waveset start."""
    wave_set, _ = build_waveset(float(teff))
    nustart_bucket = int(round(float(wave_set[0]) * 1000.0))
    cat_hash = _catalog_fingerprint(
        (
            fort11_path,
            fort111_path,
            fort21_path,
            fort31_path,
            fort41_path,
            fort51_path,
            fort61_path,
        )
    )
    return f"{cat_hash}_ns{nustart_bucket}"


def resolve_fort12_cache_path(
    cache_dir: Path,
    cache_key: str,
) -> Path:
    return cache_dir / f"fort12_{cache_key}.bin"


def load_or_prepare_fort12_cache(
    *,
    cache_dir: Optional[Path],
    cache_key: str,
    generated_path: Path,
) -> Path:
    """Return cached fort.12 path if present; else store *generated_path* in cache."""
    if cache_dir is None:
        return generated_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = resolve_fort12_cache_path(cache_dir, cache_key)
    if cached.exists():
        return cached
    if generated_path.exists() and generated_path.resolve() != cached.resolve():
        shutil.copy2(generated_path, cached)
    return cached if cached.exists() else generated_path
