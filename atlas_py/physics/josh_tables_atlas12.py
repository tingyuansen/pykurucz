"""ATLAS12 JOSH/BLOCKJ/BLOCKH table loader."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

_DEFAULT_ARCHIVE = Path(__file__).resolve().parents[1] / "data" / "josh_tables_atlas12.npz"
_CACHE: Optional[Dict[str, np.ndarray]] = None


class JoshTablesMissing(RuntimeError):
    """Raised when the ATLAS12 JOSH table archive is missing."""


def load_josh_tables(
    path: Path | None = None, *, force_reload: bool = False
) -> Dict[str, np.ndarray]:
    """Load ATLAS12 JOSH tables from archive."""
    global _CACHE
    p = path or _DEFAULT_ARCHIVE
    if force_reload or _CACHE is None:
        if not p.exists():
            raise JoshTablesMissing(
                f"Missing ATLAS12 JOSH table archive: {p}. "
                "Run atlas_py.tools.extract_josh_tables_atlas12 first."
            )
        with np.load(p) as data:
            _CACHE = {k: data[k] for k in data.files}
    return _CACHE
