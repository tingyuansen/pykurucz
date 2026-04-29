#!/usr/bin/env python3
"""
Parse the auxiliary `btables.dat` dump produced by the patched `spectrv`
executable and convert it into a NumPy archive.

The resulting `.npz` file can then be merged into an atmosphere cache so that
the Python BFUDGE implementation uses the same depth-dependent departure
coefficients as legacy SYNTHE.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np


def _finalise_table(
    label: str | None,
    rows: List[List[float]],
    store: Dict[str, np.ndarray],
) -> None:
    """Move the accumulated rows for ``label`` into the result store."""

    if not label or not rows:
        return

    data = np.asarray(rows, dtype=np.float64)
    label = label.upper()

    if label.startswith("BHYD"):
        if data.shape[1] != 9:
            raise ValueError(f"BHYD table expected 9 columns (8 + BMIN), found {data.shape[1]}")
        store["bhyd"] = data[:, :8]
        store["bmin"] = data[:, 8]
    elif label.startswith("BC1"):
        store["bc1"] = data
    elif label.startswith("BC2"):
        store["bc2"] = data
    elif label.startswith("BSI1"):
        store["bsi1"] = data
    elif label.startswith("BSI2"):
        store["bsi2"] = data
    else:
        raise ValueError(f"Unrecognised table label '{label}' in btables dump")


def parse_btables(path: Path) -> Dict[str, np.ndarray]:
    """Parse the ASCII dump emitted by the patched Fortran code."""

    store: Dict[str, np.ndarray] = {}
    current_label: str | None = None
    current_rows: List[List[float]] = []

    with path.open("r", encoding="ascii") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                # Flush the previous table before handling the new header.
                _finalise_table(current_label, current_rows, store)
                current_rows = []

                header = line[1:].strip()
                upper_header = header.upper()

                if upper_header.startswith("NRHOX"):
                    parts = upper_header.split()
                    if len(parts) < 2:
                        raise ValueError(f"Malformed NRHOX header: '{line}'")
                    store["nrhox"] = np.array(int(parts[-1]), dtype=np.int64)
                    current_label = None
                elif upper_header.startswith("PH1_PC1_PSI1"):
                    parts = header.split()
                    if len(parts) < 4:
                        raise ValueError(f"Malformed PH1_PC1_PSI1 header: '{line}'")
                    try:
                        ph1, pc1, psi1 = (float(parts[-3]), float(parts[-2]), float(parts[-1]))
                    except ValueError as exc:
                        raise ValueError(f"Could not decode PH1/PC1/PSI1 from '{line}'") from exc
                    store["ph1"] = np.array(ph1, dtype=np.float64)
                    store["pc1"] = np.array(pc1, dtype=np.float64)
                    store["psi1"] = np.array(psi1, dtype=np.float64)
                    current_label = None
                else:
                    current_label = header
                continue

            if current_label is None:
                continue

            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Malformed data row '{line}' for table '{current_label}'")

            try:
                # Discard the leading depth index; only the coefficients are needed.
                values = [float(item) for item in parts[1:]]
            except ValueError as exc:
                raise ValueError(f"Failed to parse numeric row '{line}' for table '{current_label}'") from exc

            current_rows.append(values)

    _finalise_table(current_label, current_rows, store)

    expected_keys = {"bhyd", "bc1", "bc2", "bsi1", "bsi2"}
    missing = expected_keys - store.keys()
    if missing:
        raise ValueError(f"btables dump missing expected tables: {', '.join(sorted(missing))}")

    return store


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert btables.dat into a NumPy archive.")
    parser.add_argument("source", type=Path, help="Path to btables.dat produced by spectrv.")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path("btables.npz"),
        help="Destination .npz path (default: btables.npz in current directory).",
    )
    args = parser.parse_args()

    tables = parse_btables(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **tables)
    print(f"Wrote {len(tables)} arrays to {args.output}")


if __name__ == "__main__":
    main()


