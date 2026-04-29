#!/usr/bin/env python3
"""Extract PFIRON lookup tables from atlas7v.for into pfiron_data.npz.

This parser reads the fixed-form Fortran DATA blocks:
  DATA PF001/.../
  ...
  DATA PF560/.../

and rebuilds Fortran arrays:
  P63(63,560)
  PFTAB(7,56,10,9) via EQUIVALENCE memory layout.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


_DATA_START_RE = re.compile(r"\s*DATA\s+PF(\d{3})\s*/", re.IGNORECASE)
_FLOAT_RE = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][+-]?\d+)?")


def _fixed_form_payload(line: str) -> str:
    """Return fixed-form statement payload (columns 7+)."""
    return line[6:] if len(line) > 6 else ""


def _parse_pf_blocks(atlas_lines: list[str]) -> dict[int, list[float]]:
    """Parse DATA PFxxx blocks into a map: PF index -> 63 floats."""
    pf_columns: dict[int, list[float]] = {}
    i = 0
    n_lines = len(atlas_lines)

    while i < n_lines:
        payload = _fixed_form_payload(atlas_lines[i])
        match = _DATA_START_RE.match(payload)
        if not match:
            i += 1
            continue

        pf_idx = int(match.group(1))
        statement = payload.strip()

        # Collect continuation lines until closing slash is reached.
        while statement.count("/") < 2 and i + 1 < n_lines:
            i += 1
            statement += " " + _fixed_form_payload(atlas_lines[i]).strip()

        raw_values = statement.split("/", 1)[1].rsplit("/", 1)[0]
        normalized = raw_values.replace("D", "E").replace("d", "e")
        values = [float(tok) for tok in _FLOAT_RE.findall(normalized)]

        if len(values) != 63:
            raise ValueError(f"PF{pf_idx:03d} has {len(values)} values (expected 63)")

        pf_columns[pf_idx] = values
        i += 1

    if len(pf_columns) != 560:
        raise ValueError(f"Parsed {len(pf_columns)} PF blocks (expected 560)")

    return pf_columns


def extract_pfiron_data(atlas7v_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pftab, potlo, potlolog) exactly matching Fortran PFIRON."""
    lines = atlas7v_path.read_text(encoding="latin-1").splitlines()
    pf_cols = _parse_pf_blocks(lines)

    # Reconstruct P63(63,560): column index is PF block number.
    p63 = np.zeros((63, 560), dtype=np.float64)
    for col_idx in range(1, 561):
        p63[:, col_idx - 1] = pf_cols[col_idx]

    # Match Fortran EQUIVALENCE (PFTAB(1,1,1,1),P63(1,1)).
    flat = p63.ravel(order="F")
    pftab = flat.reshape((7, 56, 10, 9), order="F")

    potlo = np.array([500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0, 32000.0])
    potlolog = np.array([2.69897, 3.0, 3.30103, 3.60206, 3.90309, 4.20412, 4.50515])
    return pftab, potlo, potlolog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--atlas7v",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "src" / "atlas7v.for",
        help="Path to atlas7v.for",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "pfiron_data.npz",
        help="Output .npz path",
    )
    args = parser.parse_args()

    pftab, potlo, potlolog = extract_pfiron_data(args.atlas7v)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, pftab=pftab, potlo=potlo, potlolog=potlolog)
    print(f"Wrote {args.output}")
    print(f"PFTAB shape={pftab.shape}, min={pftab.min():.6g}, max={pftab.max():.6g}")


if __name__ == "__main__":
    main()
