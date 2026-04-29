"""
PFSAHA ionization potentials loaded from the Fortran `atlas7v.for` tables.

The data in `pfsaha_ion_pots.npz` is generated automatically by
`synthe_py/tools/extract_pfsaha_ion_pots.py`, by parsing the POT* DATA
statements that PFSAHA uses (POTH, POTHe, POTLi, ..., POTNi, ...).

Each entry is an array of ionization potentials in eV:
    IONIZATION_POTENTIALS['NI'][0] = first ionization potential of Ni
    IONIZATION_POTENTIALS['NI'][1] = second ionization potential of Ni
    ...

This ensures the Python Saha implementation uses the exact same IP values
as the Fortran PFSAHA code, without re-encoding them by hand.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np


_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "pfsaha_ion_pots.npz"


def _load_ion_pots() -> Dict[str, np.ndarray]:
    if not _DATA_PATH.exists():
        raise FileNotFoundError(
            f"PFSAHA ionization potentials file not found: {_DATA_PATH}.\n"
            "Run synthe_py/tools/extract_pfsaha_ion_pots.py to generate it "
            "from src/atlas7v.for."
        )
    data = np.load(_DATA_PATH, allow_pickle=False)
    out: Dict[str, np.ndarray] = {}
    for key in data.files:
        # Keys are element symbols, arrays are 1D eV values
        out[key] = np.asarray(data[key], dtype=np.float64)
    return out


IONIZATION_POTENTIALS: Dict[str, np.ndarray] = _load_ion_pots()


