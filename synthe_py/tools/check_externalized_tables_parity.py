#!/usr/bin/env python3
"""Parity checks for externalized SYNTHE table payloads."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthe_py.physics import josh_tables, karsas_tables, kapp
from synthe_py.physics.profiles.hydrogen import hydrogen_tables


def _max_rel_diff(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.maximum(np.abs(b), 1e-300)
    return float(np.max(np.abs(a - b) / denom))


def _check_group(npz_path: Path, module_map: dict[str, np.ndarray]) -> list[str]:
    issues: list[str] = []
    with np.load(npz_path, allow_pickle=False) as data:
        for key, arr in module_map.items():
            if key not in data.files:
                issues.append(f"{npz_path.name}: missing key '{key}'")
                continue
            ref = np.asarray(data[key])
            got = np.asarray(arr)
            if ref.shape != got.shape:
                issues.append(
                    f"{npz_path.name}:{key} shape mismatch {got.shape} != {ref.shape}"
                )
                continue
            if not np.array_equal(ref, got, equal_nan=True):
                max_abs = float(np.max(np.abs(got - ref)))
                max_rel = _max_rel_diff(got, ref)
                issues.append(
                    f"{npz_path.name}:{key} value mismatch (max_abs={max_abs:.3e}, max_rel={max_rel:.3e})"
                )
    return issues


def main() -> int:
    data_dir = ROOT / "synthe_py" / "data"
    hyd = hydrogen_tables()

    issues: list[str] = []
    issues.extend(
        _check_group(
            data_dir / "josh_tables.npz",
            {
                "CH_WEIGHTS": josh_tables.CH_WEIGHTS,
                "CK_WEIGHTS": josh_tables.CK_WEIGHTS,
                "XTAU_GRID": josh_tables.XTAU_GRID,
                "COEFJ_MATRIX": josh_tables.COEFJ_MATRIX,
                "COEFH_MATRIX": josh_tables.COEFH_MATRIX,
            },
        )
    )
    issues.extend(
        _check_group(
            data_dir / "karsas_tables.npz",
            {
                "FREQ_LOG": karsas_tables.FREQ_LOG,
                "XN_LOG": karsas_tables.XN_LOG,
                "XL_LOG_ARRAY": karsas_tables.XL_LOG_ARRAY,
                "EKARSAS": karsas_tables.EKARSAS,
            },
        )
    )
    issues.extend(
        _check_group(
            data_dir / "kapp_continuum_tables.npz",
            {
                "COULFF_Z4LOG": kapp.COULFF_Z4LOG,
                "HMINOP_WBF": kapp.HMINOP_WBF,
                "HMINOP_BF": kapp.HMINOP_BF,
                "HMINOP_WAVEK": kapp.HMINOP_WAVEK,
                "HMINOP_THETAFF": kapp.HMINOP_THETAFF,
                "HMINOP_FFBEG": kapp.HMINOP_FFBEG,
                "HMINOP_FFEND": kapp.HMINOP_FFEND,
                "HRAYOP_GAVRILAM": kapp.HRAYOP_GAVRILAM,
                "HRAYOP_GAVRILAMAB": kapp.HRAYOP_GAVRILAMAB,
                "HRAYOP_GAVRILAMBC": kapp.HRAYOP_GAVRILAMBC,
                "HRAYOP_GAVRILAMCD": kapp.HRAYOP_GAVRILAMCD,
                "HRAYOP_GAVRILALYMANCONT": kapp.HRAYOP_GAVRILALYMANCONT,
                "HRAYOP_FGAVRILALYMANCONT": kapp.HRAYOP_FGAVRILALYMANCONT,
                "COULFF_A_TABLE": kapp.COULFF_A_TABLE,
                "HOTOP_TRANSITIONS": kapp.HOTOP_TRANSITIONS,
                "_SI2OP_PEACH": kapp._SI2OP_PEACH,
                "_SI2OP_FREQSI": kapp._SI2OP_FREQSI,
                "_SI2OP_FLOG": kapp._SI2OP_FLOG,
                "_SI2OP_TLG": kapp._SI2OP_TLG,
                "_CH_PARTITION": kapp._CH_PARTITION,
                "_OH_PARTITION": kapp._OH_PARTITION,
                "_CH_CROSSSECT": kapp._CH_CROSSSECT,
                "_OH_CROSSSECT": kapp._OH_CROSSSECT,
                "_H2_COLL_H2H2": kapp._H2_COLL_H2H2,
                "_H2_COLL_H2HE": kapp._H2_COLL_H2HE,
                "H_ENERGY_CM": kapp.H_ENERGY_CM,
                "H_STAT_WEIGHT": kapp.H_STAT_WEIGHT,
            },
        )
    )
    issues.extend(
        _check_group(
            data_dir / "hydrogen_profile_tables.npz",
            {
                "propbm": hyd.propbm,
                "c": hyd.c,
                "d": hyd.d,
                "pp": hyd.pp,
                "beta": hyd.beta,
                "stalph": hyd.stalph,
                "stwtal": hyd.stwtal,
                "istal": hyd.istal,
                "lnghal": hyd.lnghal,
                "stcomp": hyd.stcomp,
                "stcpwt": hyd.stcpwt,
                "lncomp": hyd.lncomp,
                "cutoff_h2_plus": hyd.cutoff_h2_plus,
                "cutoff_h2": hyd.cutoff_h2,
                "asum_lyman": hyd.asum_lyman,
                "asum": hyd.asum,
                "y1wtm": hyd.y1wtm,
                "xknmtb": hyd.xknmtb,
                "tabvi": hyd.tabvi,
                "tabh1": hyd.tabh1,
            },
        )
    )

    if issues:
        print("Parity check FAILED:")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    print("Parity check PASSED: module tables are bitwise-equal to NPZ payloads.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
