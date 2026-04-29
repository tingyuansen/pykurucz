#!/usr/bin/env python3
"""
Extract PFSAHA level/partition-function tables (NNN arrays) from `src/atlas7v.for`.

In `atlas7v.for` PFSAHA declares:

    DIMENSION NNN(6,374)
    DIMENSION NNN01(54), NNN02(54), ..., NNN39(54), NNN40(12)
    EQUIVALENCE (NNN(   1),NNN01(1)), (NNN(  55),NNN02(1)), ...

and then populates these arrays with a series of DATA statements:

    DATA NNN01/ ... /
    DATA NNN02/ ... /
    ...
    DATA NNN40/ ... /

This script parses those DATA blocks from the Fortran source, concatenates the
values into a single NNN array of shape (6, 374), and saves them as a NumPy file
for consumption by the Python PFSAHA implementation.

Output:
    synthe_py/data/pfsaha_levels.npz
        - key "NNN": int64 array of shape (6, 374) matching the Fortran NNN array
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

import numpy as np


FORTRAN_PATH = Path("src/atlas7v.for")
OUTPUT_PATH = Path("synthe_py/data/pfsaha_levels.npz")


def _parse_fortran_int(token: str) -> int:
    token = token.strip()
    if not token:
        raise ValueError("Empty integer token")
    # Strip trailing commas or slashes if present
    token = token.rstrip(",/")
    return int(token)


def extract_nnn_from_atlas7v(path: Path) -> np.ndarray:
    """
    Extract NNN01..NNN40 DATA tables from atlas7v.for and reconstruct the NNN array.
    """
    text = path.read_text(encoding="ascii", errors="ignore")
    lines = text.splitlines()

    # Collect values from NNN01..NNN40 plus NNN67/NNN88 in order
    all_values: List[int] = []
    block_counts: dict[int, int] = {}

    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"\s*DATA\s+NNN(\d+)\s*/", line)
        if m is None:
            i += 1
            continue

        block_idx = int(m.group(1))
        # Main NNN array NNN(6,374) is built from NNN01..NNN40, plus NNN67 and NNN88
        if not (1 <= block_idx <= 40 or block_idx in (67, 88)):
            i += 1
            continue

        # For each DATA NNNxx/ block, subsequent lines contain rows like:
        #   1 200020001, 200020011, 201620881, 231228281, 378953411,  1359502, D+F 1.00
        # The first integer on the line is a row index; the next 6 integers are the
        # packed NNN entries we want. Anything after that (labels, REF codes) is ignored.
        j = i + 1
        while j < len(lines):
            l2 = lines[j]
            # Stop if we hit another DATA NNNxx declaration
            if re.match(r"\s*DATA\s+NNN\d+\s*/", l2):
                break

            # Check if this line contains any data rows (skip comments/blank).
            # Fortran comments can start with 'C' or 'c' in column 1.
            if l2.strip().upper().startswith("C") or not l2.strip():
                j += 1
                continue

            # If the previous block ended on the previous line (with '/'), we may
            # already be done – but since we break on next DATA, we can just proceed.

            # Find if this line closes the block
            has_slash = "/" in l2

            # Strip off everything after the trailing '/' for parsing
            content = l2
            if has_slash:
                content = content.split("/", 1)[0]

            # Expect: leading row index, then 6 integer fields
            match_row = re.match(r"\s*(\d+)\s+(.*)$", content)
            if match_row:
                # Skip leading row index, parse remaining fields
                rest = match_row.group(2)
            else:
                # Some continuation lines (e.g. in NNN67) may not have an explicit
                # row index in column 1; in that case, treat the whole content
                # as data and take the first 6 integer fields.
                rest = content

            # Split by commas, then extract first integer from each segment
            ints_on_line: List[int] = []
            for seg in rest.split(","):
                seg = seg.strip()
                if not seg:
                    continue
                # Look for the first integer anywhere in the segment (to handle
                # odd prefixes like 'T 112816481')
                m_int = re.search(r"(-?\d+)", seg)
                if not m_int:
                    continue
                try:
                    val = _parse_fortran_int(m_int.group(1))
                except ValueError:
                    continue
                ints_on_line.append(val)

            # We expect at least 6 integers (the NNN entries) after the row index
            if len(ints_on_line) >= 6:
                all_values.extend(ints_on_line[:6])
                block_counts[block_idx] = block_counts.get(block_idx, 0) + 6

            j += 1
            if has_slash:
                # End of this DATA block
                break

        i = j

    # Fortran declares NNN(6,374) => 6*374 = 2244 entries
    expected = 6 * 374
    if len(all_values) != expected:
        print("NNN block counts (actual vs expected per block):")
        for b in sorted(block_counts):
            print(f"  NNN{b}: {block_counts[b]} entries")
        raise ValueError(
            f"Expected {expected} NNN entries, found {len(all_values)}. "
            "Check parsing of NNN DATA blocks."
        )

    arr = np.asarray(all_values, dtype=np.int64).reshape(6, 374, order="F")
    return arr


def main() -> None:
    if not FORTRAN_PATH.exists():
        raise SystemExit(f"Fortran source not found: {FORTRAN_PATH}")

    out_dir = OUTPUT_PATH.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    nnn = extract_nnn_from_atlas7v(FORTRAN_PATH)

    np.savez(OUTPUT_PATH, NNN=nnn)
    print(f"Wrote PFSAHA NNN levels to {OUTPUT_PATH}")
    print("NNN shape:", nnn.shape)


if __name__ == "__main__":
    main()


