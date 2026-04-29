"""checkconv-style atmosphere convergence diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from atlas_py.io.atmosphere import load_atm


def _max_frac(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs((b - a) / np.maximum(np.abs(a), 1e-300))))


def main() -> int:
    p = argparse.ArgumentParser(description="Compute simple convergence metrics for two .atm files")
    p.add_argument("prev", type=Path)
    p.add_argument("curr", type=Path)
    args = p.parse_args()

    a = load_atm(args.prev)
    b = load_atm(args.curr)
    if a.layers != b.layers:
        raise ValueError("Layer-count mismatch")

    print(f"max|dT/T|      = {_max_frac(a.temperature, b.temperature):.6e}")
    print(f"max|dP/P|      = {_max_frac(a.gas_pressure, b.gas_pressure):.6e}")
    print(f"max|dXNE/XNE|  = {_max_frac(a.electron_density, b.electron_density):.6e}")
    print(f"max|dABR/ABR|  = {_max_frac(a.abross, b.abross):.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

