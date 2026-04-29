"""Loader for molecular equilibrium reference tables (atlas_py)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "nmolec_tables.npz"
_CACHE: Optional[Dict[str, np.ndarray]] = None

_REQUIRED_KEYS = (
    "_ATMASS",
    "_H2_PF",
)


class NmolechTablesMissing(RuntimeError):
    """Raised when the nmolec table archive is missing or incomplete."""


def load_nmolec_tables(*, force_reload: bool = False) -> Dict[str, np.ndarray]:
    global _CACHE
    if _CACHE is not None and not force_reload:
        return _CACHE
    if not _DATA_PATH.exists():
        raise NmolechTablesMissing(
            f"nmolec table archive not found at {_DATA_PATH}. "
            "Run atlas_py/tools/extract_atlas_physics_tables.py."
        )
    with np.load(_DATA_PATH, allow_pickle=False) as data:
        missing = [key for key in _REQUIRED_KEYS if key not in data.files]
        if missing:
            raise NmolechTablesMissing(
                f"Missing nmolec arrays in {_DATA_PATH}: {', '.join(missing)}"
            )
        _CACHE = {key: np.asarray(data[key]) for key in _REQUIRED_KEYS}
    return _CACHE
