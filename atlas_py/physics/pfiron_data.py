"""Loader for PFIRON partition-function table.

Fortran reference: ``SUBROUTINE PFIRON`` in ``atlas12.for`` (line 16070).
The table is PFTAB(7, 56, 10, 9) in Fortran column-major order where:
  dim 0 (size 7): Debye potential-lowering levels
  dim 1 (size 56): temperature bins (log10-spaced)
  dim 2 (size 10): ionization stages (0 = neutral)
  dim 3 (size 9):  elements Ca..Ni (atomic numbers 20..28; index = Z - 20)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "pfiron_tables.npz"
_CACHE: Optional[Dict[str, np.ndarray]] = None
_REQUIRED_KEYS = ("PFTAB",)


class PfironTablesMissing(RuntimeError):
    """Raised when the pfiron_tables.npz data file is absent or incomplete."""


def load_pfiron_tables(*, force_reload: bool = False) -> Dict[str, np.ndarray]:
    """Return the cached PFIRON tables dict, loading from disk on first call.

    Returns
    -------
    dict with key ``"PFTAB"`` → ndarray of shape (7, 56, 10, 9).
    """
    global _CACHE
    if _CACHE is not None and not force_reload:
        return _CACHE
    if not _DATA_PATH.exists():
        raise PfironTablesMissing(
            f"PFIRON table file not found: {_DATA_PATH}. "
            "Run atlas_py/tools/extract_atlas_physics_tables.py to generate it."
        )
    with np.load(_DATA_PATH, allow_pickle=False) as data:
        missing = [k for k in _REQUIRED_KEYS if k not in data.files]
        if missing:
            raise PfironTablesMissing(
                f"PFIRON table file is missing keys: {missing}. "
                "Re-run extract_atlas_physics_tables.py."
            )
        _CACHE = {k: np.asarray(data[k]) for k in _REQUIRED_KEYS}
    return _CACHE
