#!/usr/bin/env python3
"""Extract large immutable ATLAS12 physics tables into package NPZ archives.

This is a one-time bootstrap script. Run it once (while the source modules
still contain inline numpy array literals) to generate the NPZ files under
``atlas_py/data/``. After that the source modules load tables from those
archives via ``*_data.py`` companion modules.

Usage::

    python -m atlas_py.tools.extract_atlas_physics_tables
    # or
    python atlas_py/tools/extract_atlas_physics_tables.py
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Table key manifests (contract between extractor and runtime loaders)
# ---------------------------------------------------------------------------

KAPP_CONTINUUM_TABLE_KEYS = (
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

KARSAS_TABLE_KEYS = (
    "FREQ_LOG",
    "XN_LOG",
    "XL_LOG_ARRAY",
    "EKARSAS",
)

LINE_OPACITY_TABLE_KEYS = (
    "_TABVI",
    "_TABH1",
)

HYDROGEN_WINGS_TABLE_KEYS = (
    "KNMTAB",
    "FSTARK",
    "HIGH_LEVEL_TERMS_ARRAY",
    "LOW_LEVEL_TERMS_ARRAY",
)

NMOLEC_TABLE_KEYS = (
    "_ATMASS",
    "_H2_PF",
)

PFSAHA_TABLE_KEYS = (
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_symbols(module: object, keys: Iterable[str]) -> dict[str, np.ndarray]:
    tables: dict[str, np.ndarray] = {}
    for key in keys:
        value = getattr(module, key)
        tables[key] = np.asarray(value, dtype=np.float64)
    return tables


def _write_archive(path: Path, tables: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **tables)
    keys = ", ".join(sorted(tables))
    print(f"Wrote {path} ({len(tables)} arrays: {keys})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    data_dir = Path(__file__).resolve().parents[1] / "data"

    # kapp_continuum
    from atlas_py.physics import kapp_continuum

    _write_archive(
        data_dir / "kapp_continuum_tables.npz",
        _collect_symbols(kapp_continuum, KAPP_CONTINUUM_TABLE_KEYS),
    )

    # karsas_tables
    from atlas_py.physics import karsas_tables

    _write_archive(
        data_dir / "karsas_tables.npz",
        _collect_symbols(karsas_tables, KARSAS_TABLE_KEYS),
    )

    # line_opacity
    from atlas_py.physics import line_opacity

    _write_archive(
        data_dir / "line_opacity_tables.npz",
        _collect_symbols(line_opacity, LINE_OPACITY_TABLE_KEYS),
    )

    # hydrogen_wings
    from atlas_py.physics import hydrogen_wings

    _write_archive(
        data_dir / "hydrogen_wings_tables.npz",
        _collect_symbols(hydrogen_wings, HYDROGEN_WINGS_TABLE_KEYS),
    )

    # nmolec
    from atlas_py.physics import nmolec

    _write_archive(
        data_dir / "nmolec_tables.npz",
        _collect_symbols(nmolec, NMOLEC_TABLE_KEYS),
    )

    # pfsaha
    from atlas_py.physics import pfsaha

    _write_archive(
        data_dir / "pfsaha_tables.npz",
        _collect_symbols(pfsaha, PFSAHA_TABLE_KEYS),
    )

    # pfiron: extracted directly from atlas12.for Fortran source DATA statements.
    # Requires atlas12.for to be present at kurucz/src/atlas12.for relative to the
    # pykurucz package root.  The generated pfiron_tables.npz contains
    # PFTAB(7,56,10,9) for Ca–Ni iron-group partition functions.
    _write_pfiron_table(data_dir)

    print("\nAll atlas_py physics tables written to", data_dir)


def _write_pfiron_table(data_dir: Path) -> None:
    """Parse PFIRON DATA blocks from atlas12.for and write pfiron_tables.npz."""
    import re

    # data_dir is atlas_py/data; repo_root is two levels up
    repo_root = data_dir.parents[1]
    # Prefer self-contained data/src/; fall back to sibling kurucz/src/
    fortran_src = repo_root / "data" / "src" / "atlas12.for"
    if not fortran_src.exists():
        fortran_src = repo_root.parent / "kurucz" / "src" / "atlas12.for"
    if not fortran_src.exists():
        print(f"  WARNING: atlas12.for not found; skipping pfiron_tables.npz")
        return

    with open(fortran_src) as fh:
        all_lines = fh.readlines()

    section = all_lines[16069:22107]  # PFIRON subroutine body
    pf_arrays: dict[str, list[float]] = {}
    current_name: str | None = None
    current_buf = ""

    for line in section:
        raw = line.rstrip()
        if not raw or raw[0].upper() in ("C", "c"):
            continue
        if len(raw) > 5 and raw[5] not in (" ", "0", ""):
            current_buf += raw[6:] if len(raw) > 6 else ""
            continue
        stmt = raw[5:].strip() if len(raw) > 5 else raw.strip()
        if current_name and current_buf:
            vs = re.sub(r"DATA\s+PF\d+\s*/\s*", "", current_buf, flags=re.IGNORECASE)
            vs = re.sub(r"/.*$", "", vs)
            pf_arrays[current_name].extend(
                float(v) for v in re.split(r"[,\s]+", vs) if v.strip()
            )
            current_name = None
            current_buf = ""
        m = re.match(r"\s*DATA\s+(PF\d+)\s*/(.*)$", stmt, re.IGNORECASE)
        if m:
            current_name = m.group(1).upper()
            pf_arrays[current_name] = []
            current_buf = "DATA " + current_name + " /" + m.group(2)
        elif current_name:
            current_buf += stmt

    if current_name and current_buf:
        vs = re.sub(r"DATA\s+PF\d+\s*/\s*", "", current_buf, flags=re.IGNORECASE)
        vs = re.sub(r"/.*$", "", vs)
        pf_arrays[current_name].extend(
            float(v) for v in re.split(r"[,\s]+", vs) if v.strip()
        )

    p63 = np.zeros((63, 560), dtype=np.float64)
    for k in range(1, 561):
        p63[:, k - 1] = pf_arrays[f"PF{k:03d}"]

    # Reconstruct PFTAB(7,56,10,9) from P63(63,560) using Fortran column-major
    # equivalence: both arrays share the same memory layout.
    pftab = p63.ravel(order="F").reshape((7, 56, 10, 9), order="F")

    out_path = data_dir / "pfiron_tables.npz"
    np.savez_compressed(str(out_path), PFTAB=pftab)
    print(f"  Written {out_path}  ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
