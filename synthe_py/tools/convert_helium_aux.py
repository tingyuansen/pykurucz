#!/usr/bin/env python3
"""Extract embedded helium wing tables from synthe.for into an NPZ archive."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

FORTRAN_FILE = Path(__file__).resolve().parents[2] / "src" / "synthe.for"


def _extract_blocks(label: str, fortran_lines: List[str]) -> List[str]:
    entries: List[str] = []
    in_block = False
    for idx, line in enumerate(fortran_lines):
        if f"DATA ({label}(I)" in line and not line.lstrip().startswith("C"):
            in_block = True
            continue
        if in_block:
            stripped = line.strip()
            if not stripped:
                continue
            if "'" in line:
                try:
                    content = line.split("'")[1]
                except IndexError:  # pragma: no cover - defensive
                    continue
                entries.append(content)
            if stripped.endswith("'/"):
                in_block = False
    return entries


_FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?")


def _parse_griem(entries: List[str]) -> Dict[str, np.ndarray]:
    if len(entries) % 5 != 0:
        raise ValueError("Unexpected number of GRIEM entries")
    n_records = len(entries) // 5
    code = np.zeros(n_records, dtype=np.float64)
    wavelength = np.zeros(n_records, dtype=np.float64)
    log_ne = np.zeros(n_records, dtype=np.float64)
    ttab = np.zeros((n_records, 4), dtype=np.float64)
    width = np.zeros_like(ttab)
    shift = np.zeros_like(ttab)
    alpha = np.zeros_like(ttab)
    beta = np.zeros_like(ttab)

    for i in range(n_records):
        block = entries[i * 5 : (i + 1) * 5]
        header_nums = _FLOAT_RE.findall(block[0])
        if len(header_nums) < 3:
            raise ValueError(f"Failed to parse GRIEM header: {block[0]}")
        code[i] = float(header_nums[0])
        wavelength[i] = float(header_nums[1])
        log_ne[i] = float(header_nums[2])
        for row in range(4):
            numbers = [float(x) for x in _FLOAT_RE.findall(block[row + 1])]
            if len(numbers) < 5:
                raise ValueError("Incomplete GRIEM data row")
            ttab[i, row] = numbers[0]
            width[i, row] = numbers[1]
            shift[i, row] = numbers[2]
            alpha[i, row] = numbers[3]
            beta[i, row] = numbers[4]

    return {
        "griem_code": code,
        "griem_wavelength": wavelength,
        "griem_log_ne": log_ne,
        "griem_ttab": ttab,
        "griem_width": width,
        "griem_shift": shift,
        "griem_alpha": alpha,
        "griem_beta": beta,
    }


def _parse_dimitri(entries: List[str]) -> Dict[str, np.ndarray]:
    if len(entries) % 5 != 0:
        raise ValueError("Unexpected number of DIMITRI entries")
    n_records = len(entries) // 5
    code = np.zeros(n_records, dtype=np.float64)
    wavelength = np.zeros(n_records, dtype=np.float64)
    log_ne = np.zeros(n_records, dtype=np.float64)
    ttab = np.zeros((n_records, 4), dtype=np.float64)
    width = np.zeros_like(ttab)
    shift = np.zeros_like(ttab)
    width_p = np.zeros_like(ttab)
    shift_p = np.zeros_like(ttab)
    width_he = np.zeros_like(ttab)
    shift_he = np.zeros_like(ttab)

    for i in range(n_records):
        block = entries[i * 5 : (i + 1) * 5]
        header_nums = _FLOAT_RE.findall(block[0])
        if len(header_nums) < 3:
            raise ValueError(f"Failed to parse DIMITRI header: {block[0]}")
        code[i] = float(header_nums[0])
        wavelength[i] = float(header_nums[1])
        log_ne[i] = float(header_nums[2])
        for row in range(4):
            numbers = [float(x) for x in _FLOAT_RE.findall(block[row + 1])]
            if len(numbers) < 7:
                raise ValueError("Incomplete DIMITRI data row")
            ttab[i, row] = numbers[0]
            width[i, row] = numbers[1]
            shift[i, row] = numbers[2]
            width_p[i, row] = numbers[3]
            shift_p[i, row] = numbers[4]
            width_he[i, row] = numbers[5]
            shift_he[i, row] = numbers[6]

    return {
        "dimitri_code": code,
        "dimitri_wavelength": wavelength,
        "dimitri_log_ne": log_ne,
        "dimitri_ttab": ttab,
        "dimitri_width": width,
        "dimitri_shift": shift,
        "dimitri_width_p": width_p,
        "dimitri_shift_p": shift_p,
        "dimitri_width_he": width_he,
        "dimitri_shift_he": shift_he,
    }


def convert(output_path: Path) -> None:
    lines = FORTRAN_FILE.read_text(encoding="ascii", errors="ignore").splitlines()
    griem_entries = _extract_blocks("GRIEM0200", lines)
    dimitri_entries = _extract_blocks("DIMITRI0200", lines)

    data: Dict[str, np.ndarray] = {}
    data.update(_parse_griem(griem_entries))
    data.update(_parse_dimitri(dimitri_entries))

    np.savez(output_path, **data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract helium auxiliary tables from synthe.for")
    parser.add_argument("output", type=Path, help="Destination .npz file")
    args = parser.parse_args()
    convert(args.output)


if __name__ == "__main__":  # pragma: no cover
    main()
