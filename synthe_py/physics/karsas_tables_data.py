"""Loader for Karsas hydrogen cross-section tables."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "karsas_tables.npz"
_CACHE: Optional[Dict[str, np.ndarray]] = None


class KarsasTablesMissing(RuntimeError):
    """Raised when Karsas table archive is missing or incomplete."""


def load_karsas_tables(*, force_reload: bool = False) -> Dict[str, np.ndarray]:
    global _CACHE
    if _CACHE is not None and not force_reload:
        return _CACHE
    if not _DATA_PATH.exists():
        raise KarsasTablesMissing(
            f"Karsas table archive not found at {_DATA_PATH}. "
            "Run synthe_py/tools/extract_synthe_physics_tables.py."
        )
    with np.load(_DATA_PATH, allow_pickle=False) as data:
        required = ("FREQ_LOG", "XN_LOG", "XL_LOG_ARRAY", "EKARSAS")
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KarsasTablesMissing(
                f"Missing Karsas arrays in {_DATA_PATH}: {', '.join(missing)}"
            )
        _CACHE = {key: np.asarray(data[key]) for key in required}
    return _CACHE

