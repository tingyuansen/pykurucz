"""Loader for large KAPP continuum reference tables (atlas_py)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "kapp_continuum_tables.npz"
_CACHE: Optional[Dict[str, np.ndarray]] = None

_REQUIRED_KEYS = (
    "COULFF_Z4LOG",
    "HMINOP_WBF",
    "HMINOP_BF",
    "HMINOP_WAVEK",
    "HMINOP_THETAFF",
    "HMINOP_FFBEG",
    "HMINOP_FFEND",
    "HRAYOP_GAVRILAM",
    "HRAYOP_GAVRILAMAB",
    "HRAYOP_GAVRILAMBC",
    "HRAYOP_GAVRILAMCD",
    "HRAYOP_GAVRILALYMANCONT",
    "HRAYOP_FGAVRILALYMANCONT",
    "COULFF_A_TABLE",
    "HOTOP_TRANSITIONS",
    "_SI2OP_PEACH",
    "_SI2OP_FREQSI",
    "_SI2OP_FLOG",
    "_SI2OP_TLG",
    "_CH_PARTITION",
    "_OH_PARTITION",
    "_CH_CROSSSECT",
    "_OH_CROSSSECT",
    "_H2_COLL_H2H2",
    "_H2_COLL_H2HE",
    "H_ENERGY_CM",
    "H_STAT_WEIGHT",
)


class KappContinuumTablesMissing(RuntimeError):
    """Raised when the kapp_continuum table archive is missing or incomplete."""


def load_kapp_continuum_tables(*, force_reload: bool = False) -> Dict[str, np.ndarray]:
    global _CACHE
    if _CACHE is not None and not force_reload:
        return _CACHE
    if not _DATA_PATH.exists():
        raise KappContinuumTablesMissing(
            f"kapp_continuum table archive not found at {_DATA_PATH}. "
            "Run atlas_py/tools/extract_atlas_physics_tables.py."
        )
    with np.load(_DATA_PATH, allow_pickle=False) as data:
        missing = [key for key in _REQUIRED_KEYS if key not in data.files]
        if missing:
            raise KappContinuumTablesMissing(
                f"Missing kapp_continuum arrays in {_DATA_PATH}: {', '.join(missing)}"
            )
        _CACHE = {key: np.asarray(data[key]) for key in _REQUIRED_KEYS}
    return _CACHE
