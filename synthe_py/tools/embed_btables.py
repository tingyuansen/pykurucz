#!/usr/bin/env python3
"""
Merge depth-dependent B tables into an atmosphere cache.

Usage
-----
    python embed_btables.py atmosphere.npz btables.npz [--output merged.npz]

The script reads the existing atmosphere archive, replaces (or injects) the
``bhyd``, ``bc1``, ``bc2``, ``bsi1`` and ``bsi2`` arrays with the values from
``btables.npz`` and writes the result back to disk.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    """Load a NumPy archive into a plain dictionary."""

    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed BFUDGE B tables into an atmosphere archive.")
    parser.add_argument("atmosphere", type=Path, help="Input atmosphere .npz file to update.")
    parser.add_argument("btables", type=Path, help="btables .npz file produced by convert_btables.py.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. Defaults to overwriting the atmosphere archive.",
    )
    args = parser.parse_args()

    atmosphere_data = load_npz(args.atmosphere)
    btables_data = load_npz(args.btables)

    required = {"bhyd", "bc1", "bc2", "bsi1", "bsi2"}
    missing = required - btables_data.keys()
    if missing:
        raise KeyError(f"btables archive missing required arrays: {', '.join(sorted(missing))}")

    for key in required:
        atmosphere_data[key] = btables_data[key]
    if "bmin" in btables_data:
        atmosphere_data["bmin"] = btables_data["bmin"]

    target = args.output or args.atmosphere
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, **atmosphere_data)
    print(f"Updated {target} with BFUDGE tables from {args.btables}")


if __name__ == "__main__":
    main()


