"""Compare internal state dumps produced by `--debug-state`."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _summary(name: str, ref: np.ndarray, test: np.ndarray) -> str:
    diff = test - ref
    frac = diff / np.maximum(np.abs(ref), 1e-300)
    mean = float(np.mean(frac))
    rms = float(np.sqrt(np.mean(frac * frac)))
    max_abs = float(np.max(np.abs(frac)))
    return f"{name:12s} mean={mean:+.3e} rms={rms:.3e} max={max_abs:.3e}"


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: np.asarray(data[k]) for k in data.files}


def _first_present(d: dict[str, np.ndarray], names: list[str]) -> tuple[str, np.ndarray] | None:
    for name in names:
        if name in d:
            return name, d[name]
    return None


def main() -> int:
    p = argparse.ArgumentParser(description="Compare two atlas_py debug-state npz files")
    p.add_argument("ref", type=Path)
    p.add_argument("test", type=Path)
    args = p.parse_args()

    a = _load(args.ref)
    b = _load(args.test)
    fields: list[tuple[str, list[str]]] = [
        ("P", ["p"]),
        ("XNE", ["xne"]),
        ("XNATOM", ["xnatom"]),
        ("RHO", ["rho"]),
        ("CHARGESQ", ["chargesq"]),
        ("ABROSS", ["abross", "abross_out"]),
        ("ACCRAD", ["accrad", "accrad_out"]),
        ("PRAD", ["prad", "prad_out"]),
        ("FLXRAD", ["flxrad", "flxrad_out"]),
        ("RJMINS", ["rjmins", "rjmins_out"]),
        ("RDABH", ["rdabh", "rdabh_out"]),
        ("RDIAGJ", ["rdiagj", "rdiagj_out"]),
        ("FLXERR", ["flxerr", "flxerr_out"]),
        ("FLXDRV", ["flxdrv", "flxdrv_out"]),
        ("DTFLUX", ["dtflux", "dtflux_out"]),
        ("DTLAMB", ["dtlamb", "dtlamb_out"]),
        ("T1", ["t1", "t1_out"]),
        ("T", ["temperature"]),
        ("RHOX", ["rhox"]),
    ]
    for label, aliases in fields:
        ka = _first_present(a, aliases)
        kb = _first_present(b, aliases)
        if ka is None or kb is None:
            print(f"{label.lower():12s} missing in one file")
            continue
        _, va = ka
        _, vb = kb
        if va.shape != vb.shape:
            print(f"{label.lower():12s} shape mismatch {va.shape} vs {vb.shape}")
            continue
        print(_summary(label, va.astype(np.float64), vb.astype(np.float64)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

