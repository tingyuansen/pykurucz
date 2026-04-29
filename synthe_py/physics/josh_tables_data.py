"""Loader for JOSH solver tables extracted from atlas7v-derived constants."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "josh_tables.npz"
_CACHE: Optional[Dict[str, np.ndarray]] = None


class JoshTablesMissing(RuntimeError):
    """Raised when the packaged JOSH table archive is missing/incomplete."""


def load_josh_tables(*, force_reload: bool = False) -> Dict[str, np.ndarray]:
    global _CACHE
    if _CACHE is not None and not force_reload:
        return _CACHE
    if not _DATA_PATH.exists():
        raise JoshTablesMissing(
            f"JOSH table archive not found at {_DATA_PATH}. "
            "Run synthe_py/tools/extract_synthe_physics_tables.py."
        )
    with np.load(_DATA_PATH, allow_pickle=False) as data:
        required = ("CH_WEIGHTS", "CK_WEIGHTS", "XTAU_GRID", "COEFJ_MATRIX", "COEFH_MATRIX")
        missing = [key for key in required if key not in data.files]
        if missing:
            raise JoshTablesMissing(
                f"Missing JOSH arrays in {_DATA_PATH}: {', '.join(missing)}"
            )
        _CACHE = {key: np.asarray(data[key]) for key in required}
    return _CACHE

