"""Loader for partition function reference tables (atlas_py)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "pfsaha_tables.npz"
_CACHE: Optional[Dict[str, np.ndarray]] = None

_REQUIRED_KEYS = (
    "LOCZ",
    "EHYD",
    "GHYD",
    "EHE1",
    "GHE1",
    "EHE2",
    "GHE2",
    "EC1",
    "GC1",
    "EC2",
    "GC2",
    "EMG1",
    "GMG1",
    "EMG2",
    "GMG2",
    "EAL1",
    "GAL1",
    "ESI1",
    "GSI1",
    "ESI2",
    "GSI2",
    "ENA1",
    "GNA1",
    "EO1",
    "GO1",
    "EB1",
    "GB1",
    "EK1",
    "GK1",
)


class PfsahaTablesMissing(RuntimeError):
    """Raised when the pfsaha table archive is missing or incomplete."""


def load_pfsaha_tables(*, force_reload: bool = False) -> Dict[str, np.ndarray]:
    global _CACHE
    if _CACHE is not None and not force_reload:
        return _CACHE
    if not _DATA_PATH.exists():
        raise PfsahaTablesMissing(
            f"pfsaha table archive not found at {_DATA_PATH}. "
            "Run atlas_py/tools/extract_atlas_physics_tables.py."
        )
    with np.load(_DATA_PATH, allow_pickle=False) as data:
        missing = [key for key in _REQUIRED_KEYS if key not in data.files]
        if missing:
            raise PfsahaTablesMissing(
                f"Missing pfsaha arrays in {_DATA_PATH}: {', '.join(missing)}"
            )
        _CACHE = {key: np.asarray(data[key]) for key in _REQUIRED_KEYS}
    return _CACHE
