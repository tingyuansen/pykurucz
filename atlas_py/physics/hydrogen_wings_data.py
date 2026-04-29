"""Loader for hydrogen wings reference tables (atlas_py)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "hydrogen_wings_tables.npz"
_CACHE: Optional[Dict[str, np.ndarray]] = None

_REQUIRED_KEYS = (
    "KNMTAB",
    "FSTARK",
    "HIGH_LEVEL_TERMS_ARRAY",
    "LOW_LEVEL_TERMS_ARRAY",
)


class HydrogenWingsTablesMissing(RuntimeError):
    """Raised when the hydrogen_wings table archive is missing or incomplete."""


def load_hydrogen_wings_tables(*, force_reload: bool = False) -> Dict[str, np.ndarray]:
    global _CACHE
    if _CACHE is not None and not force_reload:
        return _CACHE
    if not _DATA_PATH.exists():
        raise HydrogenWingsTablesMissing(
            f"hydrogen_wings table archive not found at {_DATA_PATH}. "
            "Run atlas_py/tools/extract_atlas_physics_tables.py."
        )
    with np.load(_DATA_PATH, allow_pickle=False) as data:
        missing = [key for key in _REQUIRED_KEYS if key not in data.files]
        if missing:
            raise HydrogenWingsTablesMissing(
                f"Missing hydrogen_wings arrays in {_DATA_PATH}: {', '.join(missing)}"
            )
        _CACHE = {key: np.asarray(data[key]) for key in _REQUIRED_KEYS}
    return _CACHE
