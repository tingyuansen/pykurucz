#!/usr/bin/env python3
"""Convert SYNTHE fort.9 binary to a structured NumPy archive."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path
from typing import Tuple

import numpy as np


def _read_record(handle) -> bytes:
    header = handle.read(4)
    if not header:
        raise EOFError
    (nbytes,) = struct.unpack("<i", header)
    payload = handle.read(nbytes)
    trailer = handle.read(4)
    if len(payload) != nbytes or len(trailer) != 4:
        raise ValueError("Truncated fort.9 record")
    (check,) = struct.unpack("<i", trailer)
    if check != nbytes:
        raise ValueError("fort.9 record length mismatch")
    return payload


def _parse_header(record: bytes) -> Tuple[float, float, float, int, int, int, float, int]:
    if len(record) != 44:
        raise ValueError(f"Unexpected header length {len(record)}")
    wlbeg, resolu, wlend = struct.unpack("<3d", record[:24])
    length, nrhox, linout = struct.unpack("<3i", record[24:36])
    (turbv,) = struct.unpack("<f", record[36:40])
    (ifvac,) = struct.unpack("<i", record[40:44])
    return wlbeg, resolu, wlend, length, nrhox, linout, turbv, ifvac


def convert_fort9(input_path: Path, output_path: Path) -> None:
    with input_path.open("rb") as fh:
        header = _read_record(fh)
        wlbeg, resolu, wlend, length, nrhox, linout, turbv, ifvac = _parse_header(header)

        # Continuum edge metadata (currently unused) -- skip but store raw payload.
        edge_payload = _read_record(fh)

        # Skip the ASYNTH blocks (one per wavelength point).
        stride = None
        for _ in range(length):
            payload = _read_record(fh)
            if stride is None:
                stride = len(payload) // 4
            if len(payload) != stride * 4:
                raise ValueError("Inconsistent ASYNTH record size")

        nlines_payload = _read_record(fh)
        if len(nlines_payload) != 4:
            raise ValueError("Unexpected NLINES record size")
        (n_lines,) = struct.unpack("<i", nlines_payload)

        if stride is None:
            raise ValueError("No ASYNTH records encountered in fort.9")

        lindat8 = np.empty((n_lines, 14), dtype=np.float64)
        lindat4 = np.empty((n_lines, 28), dtype=np.float32)
        alinec = np.empty((n_lines, nrhox), dtype=np.float32)

        for i in range(n_lines):
            payload = _read_record(fh)
            expected = 14 * 8 + 28 * 4 + stride * 4
            if len(payload) != expected:
                raise ValueError(
                    f"Unexpected line record size {len(payload)} (expected {expected})"
                )
            lindat8[i] = np.frombuffer(payload[:14 * 8], dtype="<f8")
            lindat4[i] = np.frombuffer(payload[14 * 8 : 14 * 8 + 28 * 4], dtype="<f4")
            full_alinec = np.frombuffer(payload[14 * 8 + 28 * 4 :], dtype="<f4", count=stride)
            alinec[i] = full_alinec[:nrhox]

    wavelength = lindat8[:, 11].copy()  # WLVAC in the Fortran layout

    np.savez(
        output_path,
        wlbeg=wlbeg,
        wlend=wlend,
        resolu=resolu,
        length=length,
        nrhox=nrhox,
        linout=linout,
        turbv=turbv,
        ifvac=ifvac,
        stride=stride,
        continuum_edges=edge_payload,
        wavelength=wavelength,
        lindat8=lindat8,
        lindat4=lindat4,
        alinec=alinec,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert fort.9 binary to NPZ")
    parser.add_argument("fort9", type=Path, help="Path to fort.9 binary file")
    parser.add_argument("output", type=Path, help="Destination .npz file")
    args = parser.parse_args()
    convert_fort9(args.fort9, args.output)


if __name__ == "__main__":
    main()
