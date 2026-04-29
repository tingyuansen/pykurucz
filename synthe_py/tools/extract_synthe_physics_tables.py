#!/usr/bin/env python3
"""Extract large immutable SYNTHE physics tables into package NPZ archives.

This script centralizes the table-key contract used by runtime loaders. It is
intentionally deterministic: each output archive is built from a fixed symbol
list and written to ``synthe_py/data``.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthe_py.physics import josh_tables
from synthe_py.physics import kapp
from synthe_py.physics import karsas_tables
from synthe_py.physics.profiles.hydrogen import hydrogen_tables


JOSH_TABLE_KEYS = (
    "CH_WEIGHTS",
    "CK_WEIGHTS",
    "XTAU_GRID",
    "COEFJ_MATRIX",
    "COEFH_MATRIX",
)

KARSAS_TABLE_KEYS = (
    "FREQ_LOG",
    "XN_LOG",
    "XL_LOG_ARRAY",
    "EKARSAS",
)

KAPP_TABLE_KEYS = (
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

HYDROGEN_TABLE_KEYS = (
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


def _collect_symbols(module: object, keys: Iterable[str]) -> dict[str, np.ndarray]:
    tables: dict[str, np.ndarray] = {}
    for key in keys:
        value = getattr(module, key)
        tables[key] = np.asarray(value)
    return tables


def _write_archive(path: Path, tables: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **tables)
    keys = ", ".join(sorted(tables))
    print(f"Wrote {path} ({len(tables)} arrays: {keys})")


def main() -> None:
    data_dir = Path(__file__).resolve().parents[1] / "data"

    _write_archive(
        data_dir / "josh_tables.npz",
        _collect_symbols(josh_tables, JOSH_TABLE_KEYS),
    )
    _write_archive(
        data_dir / "karsas_tables.npz",
        _collect_symbols(karsas_tables, KARSAS_TABLE_KEYS),
    )
    _write_archive(
        data_dir / "kapp_continuum_tables.npz",
        _collect_symbols(kapp, KAPP_TABLE_KEYS),
    )

    hydrogen = hydrogen_tables()
    hydrogen_tables_map = {
        key: np.asarray(getattr(hydrogen, key)) for key in HYDROGEN_TABLE_KEYS
    }
    _write_archive(data_dir / "hydrogen_profile_tables.npz", hydrogen_tables_map)


if __name__ == "__main__":
    main()
