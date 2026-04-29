"""Extract EOS-like depth tables from ATLAS12 stdout log.

Two formats are supported:
1) NELECT debug rows (`atlas12.for` around line ~3132):
   I3,1PE15.7,0PF10.1,1P6E12.3
   -> J, RHOX, T, P, XNE, XNATOM, WTMOLE, RHO, CHARGESQ
2) Main atmosphere table (RHOX/T/P/XNE/ABROSS/PRAD/VTURB) printed in run logs.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

_NUM = r"[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][+\-]?\d+)?"
_PAT_NELECT = re.compile(
    rf"^\s*(\d+)\s+({_NUM})\s+({_NUM})\s+({_NUM})\s+({_NUM})\s+({_NUM})\s+({_NUM})\s+({_NUM})\s+({_NUM})\s*$"
)


def _to_float(x: str) -> float:
    return float(x.replace("D", "E").replace("d", "e"))


def _parse_main_table_row(line: str) -> list[float] | None:
    parts = line.strip().split()
    if len(parts) < 8:
        return None
    # Main table begins with integer depth index and scientific-notation RHOX.
    if not parts[0].isdigit():
        return None
    if "E" not in parts[1].upper() and "D" not in parts[1].upper():
        return None
    try:
        # idx is parts[0], then: rhox, t, p, xne, abross, prad, vturb
        return [_to_float(tok) for tok in parts[1:8]]
    except ValueError:
        return None


def _parse_xnatom_table_row(line: str) -> list[float] | None:
    parts = line.strip().split()
    if len(parts) < 7:
        return None
    if not parts[0].isdigit():
        return None
    if "E" not in parts[1].upper() and "D" not in parts[1].upper():
        return None
    try:
        # idx is parts[0], then: rhox, t, p, xne, xnatom, rho
        return [_to_float(tok) for tok in parts[1:7]]
    except ValueError:
        return None


def main() -> int:
    p = argparse.ArgumentParser(description="Extract EOS table rows from Fortran log")
    p.add_argument("log", type=Path)
    p.add_argument("--output", type=Path, required=True, help="Output .npz path")
    args = p.parse_args()

    nelect_rows: list[list[float]] = []
    atm_blocks: list[list[list[float]]] = []
    current_atm_block: list[list[float]] = []
    xnatom_blocks: list[list[list[float]]] = []
    current_xnatom_block: list[list[float]] = []
    in_main_table = False
    in_xnatom_table = False
    for line in args.log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _PAT_NELECT.match(line)
        if m is not None:
            vals = [_to_float(g) for g in m.groups()[1:]]
            nelect_rows.append(vals)
            continue
        if "RHOX" in line and "XNATOM" in line and "RHO/" in line:
            if current_xnatom_block:
                xnatom_blocks.append(current_xnatom_block)
                current_xnatom_block = []
            in_xnatom_table = True
            continue
        if "RHOX" in line and "ABROSS" in line and "VTURB" in line:
            if current_atm_block:
                atm_blocks.append(current_atm_block)
                current_atm_block = []
            in_main_table = True
            continue
        if in_xnatom_table:
            row = _parse_xnatom_table_row(line)
            if row is not None:
                current_xnatom_block.append(row)
                continue
            if current_xnatom_block:
                xnatom_blocks.append(current_xnatom_block)
                current_xnatom_block = []
            in_xnatom_table = False
        if in_main_table:
            row = _parse_main_table_row(line)
            if row is not None:
                current_atm_block.append(row)
                continue
            if current_atm_block:
                atm_blocks.append(current_atm_block)
                current_atm_block = []
            in_main_table = False

    if current_xnatom_block:
        xnatom_blocks.append(current_xnatom_block)
    if current_atm_block:
        atm_blocks.append(current_atm_block)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if nelect_rows:
        arr = np.asarray(nelect_rows, dtype=np.float64)
        np.savez(
            args.output,
            rhox=arr[:, 0],
            temperature=arr[:, 1],
            p=arr[:, 2],
            xne=arr[:, 3],
            xnatom=arr[:, 4],
            wtmole=arr[:, 5],
            rho=arr[:, 6],
            chargesq=arr[:, 7],
        )
        print(f"Extracted {arr.shape[0]} NELECT rows -> {args.output}")
        return 0

    if atm_blocks:
        arr = np.asarray(atm_blocks[-1], dtype=np.float64)
        out: dict[str, np.ndarray] = {
            "rhox": arr[:, 0],
            "temperature": arr[:, 1],
            "p": arr[:, 2],
            "xne": arr[:, 3],
            "abross": arr[:, 4],
            "prad": arr[:, 5],
            "vturb": arr[:, 6],
        }
        if xnatom_blocks:
            xarr = np.asarray(xnatom_blocks[-1], dtype=np.float64)
            if xarr.shape[0] == arr.shape[0]:
                out["xnatom"] = xarr[:, 4]
                out["rho"] = xarr[:, 5]
        np.savez(
            args.output,
            **out,
        )
        print(
            f"Extracted {arr.shape[0]} atmosphere rows "
            f"(block {len(atm_blocks)}/{len(atm_blocks)}) -> {args.output}"
        )
        return 0

    raise ValueError("No EOS-like table rows matched in log")


if __name__ == "__main__":
    raise SystemExit(main())

