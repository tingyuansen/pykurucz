#!/usr/bin/env python3
"""
Find the first DEQ divergence between Fortran and Python by comparing
all tracked DEQ columns (1, 8, 9, 16) across all SOLVIT calls.
"""

import re
from pathlib import Path
from typing import Dict, Tuple, Optional

EntryKey = Tuple[int, int, int, int]  # (layer, iter, call, col, row)
EntryMap = Dict[Tuple[int, int, int, int, int], float]

HEADER_RE = re.compile(
    r"(?:FT|PY)_NMOLEC: Selected DEQ columns before SOLVIT \(LAYER=\s*(?P<layer>\d+)\s+ITER=\s*(?P<iter>\d+)\s+CALL=\s*(?P<call>\d+)\)"
)
COLUMN_RE = re.compile(
    r"Column\s+(?P<col>\d+)\s+values\s+\(EQ=\s*(?P<eq>[+-]?\d\.\d+E[+-]\d+)\)"
)
ROW_RE = re.compile(r"Row\s+(?P<row>\d+)\s*->\s*(?P<value>[+-]?\d\.\d+E[+-]\d+)")


def parse_entries(path: Path, prefix: str) -> EntryMap:
    entries: EntryMap = {}
    if not path.exists():
        print(f"Warning: {path} not found")
        return entries

    current_layer: Optional[int] = None
    current_iter: Optional[int] = None
    current_call: Optional[int] = None
    current_col: Optional[int] = None

    with path.open() as fh:
        for line in fh:
            header_match = HEADER_RE.search(line)
            if header_match:
                current_layer = int(header_match.group("layer"))
                current_iter = int(header_match.group("iter"))
                current_call = int(header_match.group("call"))
                current_col = None
                continue

            col_match = COLUMN_RE.search(line)
            if (
                col_match
                and current_layer is not None
                and current_iter is not None
                and current_call is not None
            ):
                current_col = int(col_match.group("col"))
                continue

            row_match = ROW_RE.search(line)
            if (
                row_match
                and current_layer is not None
                and current_iter is not None
                and current_call is not None
                and current_col is not None
            ):
                row = int(row_match.group("row"))
                value = float(row_match.group("value"))
                entries[
                    (current_layer, current_iter, current_call, current_col, row)
                ] = value
                continue

    return entries


def main():
    ft_log = Path("synthe/stmp_at12_aaaaa/nmolec_debug.log")
    py_log = Path("logs/nmolec_debug_python.log")

    ft_entries = parse_entries(ft_log, "FT")
    py_entries = parse_entries(py_log, "PY")

    print(f"Fortran entries: {len(ft_entries)}")
    print(f"Python entries: {len(py_entries)}")

    # Tracked columns
    tracked_cols = [1, 8, 9, 16]

    # Find all unique (layer, call) combinations (ignore iter for alignment)
    # Group by layer and call number
    ft_by_call: Dict[Tuple[int, int, int, int], float] = {}
    py_by_call: Dict[Tuple[int, int, int, int], float] = {}

    for (l, i, c, col, r), val in ft_entries.items():
        if col in tracked_cols:
            ft_by_call[(l, c, col, r)] = val

    for (l, i, c, col, r), val in py_entries.items():
        if col in tracked_cols:
            py_by_call[(l, c, col, r)] = val

    # Sort by layer, call, col, row
    all_keys = sorted(set(ft_by_call.keys()) | set(py_by_call.keys()))

    print(f"\nComparing {len(all_keys)} entries...")
    print(
        f"{'Layer':>5} {'Call':>4} {'Col':>3} {'Row':>3} {'Fortran':>18} {'Python':>18} {'RelDiff':>12} {'Status':>6}"
    )
    print("-" * 90)

    rel_tol = 1e-6
    first_diff = None
    shown = 0
    max_show = 20

    for layer, call, col, row in all_keys:
        ft_val = ft_by_call.get((layer, call, col, row))
        py_val = py_by_call.get((layer, call, col, row))

        if ft_val is None or py_val is None:
            continue

        if ft_val == py_val:
            rel_diff = 0.0
        else:
            denom = max(abs(ft_val), abs(py_val), 1e-300)
            rel_diff = abs(ft_val - py_val) / denom

        status = "ok" if rel_diff <= rel_tol else "DIFF"

        if status == "DIFF":
            if first_diff is None:
                first_diff = (layer, call, col, row, ft_val, py_val, rel_diff)
            if shown < max_show:
                print(
                    f"{layer:5d} {call:4d} {col:3d} {row:3d} "
                    f"{ft_val:18.9E} {py_val:18.9E} {rel_diff:12.3E} {status:>6}"
                )
                shown += 1

    if first_diff:
        layer, call, col, row, ft_val, py_val, rel_diff = first_diff
        print(f"\n{'='*90}")
        print(f"FIRST DIVERGENCE FOUND:")
        print(f"  Location: Layer {layer}, Call {call}, Column {col}, Row {row}")
        print(f"  Fortran:  {ft_val:.12E}")
        print(f"  Python:   {py_val:.12E}")
        print(f"  Rel Diff: {rel_diff:.6E}")
        print(f"{'='*90}")

        # Now check the previous call to see if it was already diverging
        if call > 1:
            prev_key = (layer, call - 1, col, row)
            ft_prev = ft_by_call.get(prev_key)
            py_prev = py_by_call.get(prev_key)
            if ft_prev is not None and py_prev is not None:
                prev_rel_diff = abs(ft_prev - py_prev) / max(
                    abs(ft_prev), abs(py_prev), 1e-300
                )
                print(f"\nPrevious call (call {call-1}):")
                print(f"  Fortran:  {ft_prev:.12E}")
                print(f"  Python:   {py_prev:.12E}")
                print(f"  Rel Diff: {prev_rel_diff:.6E}")
                if prev_rel_diff > rel_tol:
                    print(f"  → Already diverging at call {call-1}")
                else:
                    print(f"  → Divergence starts at call {call}")
    else:
        print("\nNo divergences found within tolerance!")


if __name__ == "__main__":
    main()
