#!/usr/bin/env python3
"""Extract IONPOTS data tables from `atlas12.for`."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import numpy as np


def _default_fortran_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    data_path = repo_root / "data" / "src" / "atlas12.for"
    if data_path.exists():
        return data_path
    return repo_root.parent / "kurucz" / "src" / "atlas12.for"


FORTRAN_PATH = _default_fortran_path()
OUTPUT_PATH = Path("atlas_py/data/ionpots_atlas12.npz")


def _norm_name(name: str) -> str:
    return re.sub(r"\s+", "", name).upper()


def _parse_float_token(tok: str) -> float:
    tok = tok.strip().rstrip(",/")
    tok = tok.replace("D", "E").replace("d", "e")
    return float(tok)


def _extract_equivalence(lines: List[str]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    pat = re.compile(r"EQUIVALENCE\s+\(POTION\(\s*(\d+)\s*\)\s*,\s*([A-Za-z0-9 ]+)\(1\)\)")
    for line in lines:
        m = pat.search(line)
        if m is None:
            continue
        start = int(m.group(1))
        name = _norm_name(m.group(2))
        mapping[name] = start
    return mapping


def _extract_data_blocks(lines: List[str]) -> Dict[str, List[float]]:
    data: Dict[str, List[float]] = {}
    i = 0
    start_pat = re.compile(r"\s*DATA\s+([A-Za-z0-9 ]+)\s*/")
    while i < len(lines):
        m = start_pat.match(lines[i])
        if m is None:
            i += 1
            continue
        raw_name = m.group(1)
        name = _norm_name(raw_name)
        if not name.startswith("POT"):
            i += 1
            continue
        vals: List[float] = []
        j = i
        first = True
        while j < len(lines):
            line = lines[j]
            if first:
                # Parse from slash onward on DATA line.
                parts = line.split("/", 1)
                content = parts[1] if len(parts) > 1 else ""
                first = False
            else:
                content = line
            has_end = "/" in content
            content = content.split("/", 1)[0]
            # Drop Fortran continuation label if present (e.g. "     1 ...")
            content = re.sub(r"^\s*\d+\s+", "", content)
            for seg in content.split(","):
                seg = seg.strip()
                if not seg:
                    continue
                try:
                    vals.append(_parse_float_token(seg))
                except ValueError:
                    pass
            j += 1
            if has_end:
                break
        data[name] = vals
        i = j
    return data


def build_potion_array(equiv: Dict[str, int], blocks: Dict[str, List[float]]) -> np.ndarray:
    potion = np.zeros(999, dtype=np.float64)
    for name, vals in blocks.items():
        start = equiv.get(name)
        if start is None:
            continue
        idx0 = start - 1
        for k, v in enumerate(vals):
            if idx0 + k < potion.size:
                potion[idx0 + k] = v
    return potion


def main() -> None:
    if not FORTRAN_PATH.exists():
        raise SystemExit(f"Fortran source not found: {FORTRAN_PATH}")
    lines = FORTRAN_PATH.read_text(encoding="ascii", errors="ignore").splitlines()
    equiv = _extract_equivalence(lines)
    blocks = _extract_data_blocks(lines)
    potion = build_potion_array(equiv, blocks)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUTPUT_PATH, POTION=potion)
    print(f"Wrote {OUTPUT_PATH} with {np.count_nonzero(potion)} non-zero entries")


if __name__ == "__main__":
    main()

