#!/usr/bin/env python3
"""Extract PFSAHA NNN tables from `atlas12.for`.

ATLAS12 declares:
    DIMENSION NNN(6,365)
    DIMENSION NNN01..NNN40 plus NNN67
with EQUIVALENCE onto NNN.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

import numpy as np


def _default_fortran_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    data_path = repo_root / "data" / "src" / "atlas12.for"
    if data_path.exists():
        return data_path
    return repo_root.parent / "kurucz" / "src" / "atlas12.for"


FORTRAN_PATH = _default_fortran_path()
OUTPUT_PATH = Path("atlas_py/data/pfsaha_levels_atlas12.npz")


def _parse_int(token: str) -> int:
    token = token.strip().rstrip(",/")
    if not token:
        raise ValueError("empty token")
    return int(token)


def extract_nnn(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="ascii", errors="ignore").splitlines()
    vals: List[int] = []

    i = 0
    while i < len(lines):
        m = re.match(r"\s*DATA\s+NNN(\d+)\s*/", lines[i])
        if m is None:
            i += 1
            continue
        idx = int(m.group(1))
        if not (1 <= idx <= 40 or idx == 67):
            i += 1
            continue

        j = i + 1
        while j < len(lines):
            line = lines[j]
            if re.match(r"\s*DATA\s+NNN\d+\s*/", line):
                break
            if not line.strip() or line.strip().upper().startswith("C"):
                j += 1
                continue
            has_slash = "/" in line
            content = line.split("/", 1)[0] if has_slash else line
            row = re.match(r"\s*(\d+)\s+(.*)$", content)
            rest = row.group(2) if row else content

            ints: List[int] = []
            for seg in rest.split(","):
                seg = seg.strip()
                if not seg:
                    continue
                m_int = re.search(r"(-?\d+)", seg)
                if m_int is None:
                    continue
                ints.append(_parse_int(m_int.group(1)))

            if len(ints) >= 6:
                vals.extend(ints[:6])
            j += 1
            if has_slash:
                break
        i = j

    expected = 6 * 365
    if len(vals) != expected:
        raise ValueError(f"Expected {expected} entries, found {len(vals)}")
    return np.asarray(vals, dtype=np.int64).reshape(6, 365, order="F")


def main() -> None:
    if not FORTRAN_PATH.exists():
        raise SystemExit(f"Fortran source not found: {FORTRAN_PATH}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    nnn = extract_nnn(FORTRAN_PATH)
    np.savez(OUTPUT_PATH, NNN=nnn)
    print(f"Wrote {OUTPUT_PATH} with shape {nnn.shape}")


if __name__ == "__main__":
    main()

