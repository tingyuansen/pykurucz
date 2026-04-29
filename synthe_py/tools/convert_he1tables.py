#!/usr/bin/env python3
"""Convert the SYNTHE he1tables.dat file into a structured NumPy archive."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy as np


TEMPERATURES = np.array([5000.0, 10000.0, 20000.0, 40000.0], dtype=np.float64)


def _next_data_line(lines: Iterator[str]) -> str:
    for line in lines:
        stripped = line.strip()
        if stripped:
            return stripped
    raise StopIteration


def _read_block(
    lines: Iterator[str],
    n_ne: int,
    n_dlam: int,
    species: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    dlam = np.zeros(n_dlam, dtype=np.float64)
    ne_values = np.zeros(n_ne, dtype=np.float64)
    phi = np.zeros((species, TEMPERATURES.size, n_ne, n_dlam), dtype=np.float64)

    for il in range(n_dlam):
        for ne_idx in range(n_ne):
            parts = _next_data_line(lines).split()
            if len(parts) < 2 + species * TEMPERATURES.size:
                raise ValueError("Unexpected he1tables.dat row format")
            ne = float(parts[0])
            dwl = float(parts[1])
            values = np.array(parts[2: 2 + species * TEMPERATURES.size], dtype=np.float64)
            values = values.reshape(species, TEMPERATURES.size)
            phi[:, :, ne_idx, il] = values
            if il == 0:
                ne_values[ne_idx] = ne
        dlam[il] = dwl
    result: Dict[str, np.ndarray] = {}
    if species == 2:
        result["phi_h_plus"] = phi[0]
        result["phi_he_plus"] = phi[1]
    else:
        result["phi"] = phi[0]
    return ne_values, dlam, result


def convert(input_path: Path, output_path: Path) -> None:
    with input_path.open("r", encoding="ascii", errors="ignore") as fh:
        lines = iter(fh.readlines())
        data: Dict[str, np.ndarray] = {}
        while True:
            try:
                line = _next_data_line(lines)
            except StopIteration:
                break
            tag = line.split()[0]
            if tag == "4471":
                _ = _next_data_line(lines)  # column header
                ne, dlam, block = _read_block(lines, n_ne=7, n_dlam=142, species=2)
                data["line_4471_ne"] = ne
                data["line_4471_dlam"] = dlam
                data["line_4471_phi_h_plus"] = block["phi_h_plus"]
                data["line_4471_phi_he_plus"] = block["phi_he_plus"]
            elif tag == "4026":
                _ = _next_data_line(lines)
                ne, dlam, block = _read_block(lines, n_ne=8, n_dlam=196, species=1)
                data["line_4026_ne"] = ne
                data["line_4026_dlam"] = dlam
                data["line_4026_phi"] = block["phi"]
            elif tag == "4387":
                _ = _next_data_line(lines)
                ne, dlam, block = _read_block(lines, n_ne=8, n_dlam=204, species=1)
                data["line_4387_ne"] = ne
                data["line_4387_dlam"] = dlam
                data["line_4387_phi"] = block["phi"]
            elif tag.startswith("492"):
                _ = _next_data_line(lines)
                ne, dlam, block = _read_block(lines, n_ne=7, n_dlam=142, species=2)
                data["line_4921_ne"] = ne
                data["line_4921_dlam"] = dlam
                data["line_4921_phi_h_plus"] = block["phi_h_plus"]
                data["line_4921_phi_he_plus"] = block["phi_he_plus"]
            else:
                raise ValueError(f"Unexpected section tag '{tag}' in he1tables.dat")

    data["temperatures"] = TEMPERATURES
    np.savez(output_path, **data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert he1tables.dat to NPZ")
    parser.add_argument("input", type=Path, help="Path to he1tables.dat")
    parser.add_argument("output", type=Path, help="Destination .npz file")
    args = parser.parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
