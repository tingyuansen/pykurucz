#!/usr/bin/env python3
"""Convert SYNTHE fort.20 (line core coefficients) to a NumPy archive."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


def _read_record(handle) -> bytes:
    header = handle.read(4)
    if not header:
        raise EOFError
    (nbytes,) = struct.unpack("<i", header)
    payload = handle.read(nbytes)
    trailer = handle.read(4)
    if len(payload) != nbytes or len(trailer) != 4:
        raise ValueError("Truncated fort.20 record")
    (check,) = struct.unpack("<i", trailer)
    if check != nbytes:
        raise ValueError("fort.20 record length mismatch")
    return payload


def convert_fort20(input_path: Path, output_path: Path) -> None:
    wavelengths: list[float] = []
    nblo: list[int] = []
    nbup: list[int] = []
    nelion: list[int] = []
    aline: list[np.ndarray] = []

    with input_path.open("rb") as fh:
        while True:
            try:
                payload = _read_record(fh)
            except EOFError:
                break

            if len(payload) != 224:
                raise ValueError(f"Unexpected record size {len(payload)} in fort.20")

            offset = 0
            lindat8 = struct.unpack("<14d", payload[offset : offset + 14 * 8])
            offset += 14 * 8
            wavelength = lindat8[11]
            wavelengths.append(wavelength)

            floats = struct.unpack("<28f", payload[offset:])
            nblo.append(int(round(floats[13])))
            nbup.append(int(round(floats[14])))
            nelion.append(int(round(floats[16])))
            aline.append(np.asarray(floats[20:], dtype=np.float32))

    np.savez(
        output_path,
        wavelength=np.asarray(wavelengths, dtype=np.float64),
        nblo=np.asarray(nblo, dtype=np.int16),
        nbup=np.asarray(nbup, dtype=np.int16),
        nelion=np.asarray(nelion, dtype=np.int16),
        aline=np.vstack(aline) if aline else np.zeros((0, 0), dtype=np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert fort.20 tape to NPZ")
    parser.add_argument("fort20", type=Path, help="Path to fort.20 binary")
    parser.add_argument("output", type=Path, help="Destination .npz file")
    args = parser.parse_args()
    convert_fort20(args.fort20, args.output)


if __name__ == "__main__":
    main()
