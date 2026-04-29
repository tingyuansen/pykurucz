"""Loader for tabulated hydrogen profile constants."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "hydrogen_profile_tables.npz"
_CACHE: Optional[Dict[str, np.ndarray]] = None

_REQUIRED_KEYS = (
    "propbm",
    "c",
    "d",
    "pp",
    "beta",
    "stalph",
    "stwtal",
    "istal",
    "lnghal",
    "stcomp",
    "stcpwt",
    "lncomp",
    "cutoff_h2_plus",
    "cutoff_h2",
    "asum_lyman",
    "asum",
    "y1wtm",
    "xknmtb",
    "tabvi",
    "tabh1",
)


class HydrogenTablesMissing(RuntimeError):
    """Raised when hydrogen-profile archive is missing or incomplete."""


def load_hydrogen_profile_tables(*, force_reload: bool = False) -> Dict[str, np.ndarray]:
    global _CACHE
    if _CACHE is not None and not force_reload:
        return _CACHE
    if not _DATA_PATH.exists():
        raise HydrogenTablesMissing(
            f"Hydrogen profile archive not found at {_DATA_PATH}. "
            "Run synthe_py/tools/extract_synthe_physics_tables.py."
        )
    with np.load(_DATA_PATH, allow_pickle=False) as data:
        missing = [key for key in _REQUIRED_KEYS if key not in data.files]
        if missing:
            raise HydrogenTablesMissing(
                f"Missing hydrogen arrays in {_DATA_PATH}: {', '.join(missing)}"
            )
        _CACHE = {key: np.asarray(data[key]) for key in _REQUIRED_KEYS}
    return _CACHE

