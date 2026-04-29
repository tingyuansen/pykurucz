#!/usr/bin/env python3
"""Convert SYNTHE fort.29 (ASYNTH) to a NumPy archive."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def convert_fort29(input_path: Path, output_path: Path, nrhox: int = 80) -> None:
    wavelengths: list[float] = []
    asynth: list[np.ndarray] = []

    with input_path.open("r", encoding="ascii") as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            wave = float(line)
            values: list[float] = []
            while len(values) < nrhox:
                data_line = fh.readline()
                if not data_line:
                    raise ValueError("Unexpected EOF while reading fort.29")
                values.extend(float(x) for x in data_line.split())
            wavelengths.append(wave)
            asynth.append(np.asarray(values[:nrhox], dtype=np.float32))

    np.savez(
        output_path,
        wavelength=np.asarray(wavelengths, dtype=np.float64),
        asynth=np.vstack(asynth) if asynth else np.zeros((0, nrhox), dtype=np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert fort.29 to NPZ")
    parser.add_argument("fort29", type=Path, help="Path to fort.29 text file")
    parser.add_argument("output", type=Path, help="Destination .npz file")
    parser.add_argument("--nrhox", type=int, default=80, help="Number of depth layers")
    args = parser.parse_args()
    convert_fort29(args.fort29, args.output, nrhox=args.nrhox)


if __name__ == "__main__":
    main()
