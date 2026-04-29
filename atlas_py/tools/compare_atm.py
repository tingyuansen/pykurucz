"""Field-by-field comparison for two ATLAS `.atm` files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from atlas_py.io.atmosphere import load_atm


def _summary(name: str, ref: np.ndarray, test: np.ndarray) -> str:
    diff = test - ref
    frac = diff / np.maximum(np.abs(ref), 1e-300)
    mean = float(np.mean(frac))
    rms = float(np.sqrt(np.mean(frac * frac)))
    max_abs = float(np.max(np.abs(frac)))
    return f"{name:10s} mean={mean:+.3e} rms={rms:.3e} max={max_abs:.3e}"


def _worst_depths(name: str, ref: np.ndarray, test: np.ndarray, n: int) -> str:
    frac = np.abs((test - ref) / np.maximum(np.abs(ref), 1e-300))
    order = np.argsort(frac)[::-1][:n]
    parts = [f"{name} worst depths:"]
    for idx in order:
        parts.append(
            f"#{idx+1}: frac={frac[idx]:.3e} ref={ref[idx]:.3e} test={test[idx]:.3e}"
        )
    return " | ".join(parts)


def main() -> int:
    p = argparse.ArgumentParser(description="Compare two .atm files")
    p.add_argument("ref", type=Path)
    p.add_argument("test", type=Path)
    p.add_argument(
        "--show-worst",
        type=int,
        default=0,
        help="Show top-N worst depth rows per field",
    )
    args = p.parse_args()

    a = load_atm(args.ref)
    b = load_atm(args.test)
    if a.layers != b.layers:
        raise ValueError(f"Layer count mismatch: {a.layers} vs {b.layers}")

    print(_summary("RHOX", a.rhox, b.rhox))
    print(_summary("T", a.temperature, b.temperature))
    print(_summary("P", a.gas_pressure, b.gas_pressure))
    print(_summary("XNE", a.electron_density, b.electron_density))
    print(_summary("ABROSS", a.abross, b.abross))
    print(_summary("ACCRAD", a.accrad, b.accrad))
    print(_summary("VTURB", a.vturb, b.vturb))
    if args.show_worst > 0:
        n = int(args.show_worst)
        print(_worst_depths("P", a.gas_pressure, b.gas_pressure, n))
        print(_worst_depths("XNE", a.electron_density, b.electron_density, n))
        print(_worst_depths("T", a.temperature, b.temperature, n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

