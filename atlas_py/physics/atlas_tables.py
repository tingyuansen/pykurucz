"""
Loader for hard-coded Kurucz ATLAS lookup tables extracted from ``atlas7v.for``.

The extractor script writes all supported DATA blocks into an ``atlas_tables.npz``
archive. This module provides a thin caching layer around that archive so the
physics kernels can pull the constants they need without re-reading the file
each time.

Default search (when ``path`` is omitted): ``atlas_py/data/atlas_tables.npz``,
then the sibling ``synthe_py/data/atlas_tables.npz`` under the same repository
root (no import of ``synthe_py``; filesystem fallback only).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np


def _default_archive_path() -> Path:
    """Prefer ``atlas_py/data``; fall back to ``synthe_py/data`` if present."""
    here = Path(__file__).resolve()
    atlas_pkg = here.parents[1]
    primary = atlas_pkg / "data" / "atlas_tables.npz"
    if primary.exists():
        return primary
    repo_root = here.parents[2]
    fallback = repo_root / "synthe_py" / "data" / "atlas_tables.npz"
    if fallback.exists():
        return fallback
    return primary


_DEFAULT_ARCHIVE = _default_archive_path()

# Cached copy of the most recent load to avoid repeatedly touching the file.
_CACHE: Optional[Dict[str, np.ndarray]] = None


class AtlasTablesMissing(RuntimeError):
    """Raised when the atlas tables archive is absent or incomplete."""


def _load_archive(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise AtlasTablesMissing(
            f"Atlas tables archive not found at {path}. "
            "Run tools/extract_atlas_tables.py to generate it."
        )
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def load_atlas_tables(path: Path | None = None, *, force_reload: bool = False) -> Dict[str, np.ndarray]:
    """
    Return a mapping of table name -> NumPy array.

    Parameters
    ----------
    path:
        Optional explicit path to the ``atlas_tables.npz`` archive.  Defaults to
        ``atlas_py/data/atlas_tables.npz``, or ``synthe_py/data/atlas_tables.npz``
        when the former is missing (same repo layout).
    force_reload:
        When ``True`` the archive is re-read even if it was previously cached.
    """
    global _CACHE
    archive_path = path or _default_archive_path()
    if force_reload or _CACHE is None:
        _CACHE = _load_archive(archive_path)
    return _CACHE


def get_table(name: str, *, path: Path | None = None) -> np.ndarray:
    """Convenience accessor for a single table."""
    tables = load_atlas_tables(path)
    try:
        return tables[name]
    except KeyError as exc:
        raise AtlasTablesMissing(
            f"Table '{name}' not present in atlas archive {path or _default_archive_path()}"
        ) from exc


def ensure_tables(names: Iterable[str], *, path: Path | None = None) -> None:
    """
    Validate that the requested table names are present in the archive.

    Useful for startup checks; raises ``AtlasTablesMissing`` if any are absent.
    """
    tables = load_atlas_tables(path)
    missing = [name for name in names if name not in tables]
    if missing:
        raise AtlasTablesMissing(
            f"Missing atlas tables: {', '.join(missing)}. "
            "Re-run the extractor if needed."
        )
