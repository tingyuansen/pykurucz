#!/usr/bin/env python3
"""
Extract PFSAHA ionization potential tables (POTH, POTHe, ..., POTNi, ...) from
the Fortran source `src/atlas7v.for` and save them as a structured NumPy file.

This lets the Python Saha/partition-function implementation use exactly the
same ionization potentials as Fortran, without re-encoding them by hand.

Output:
    synthe_py/data/pfsaha_ion_pots.npz
        - keys are element symbols (e.g. "H", "HE", "NI")
        - values are 1D float arrays of ionization potentials in eV
          (first entry = first ionization, second = second ionization, ...)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import numpy as np


FORTRAN_PATH = Path("src/atlas7v.for")
OUTPUT_PATH = Path("synthe_py/data/pfsaha_ion_pots.npz")


def _parse_fortran_real(token: str) -> float:
    """
    Parse a Fortran-style real literal into Python float, handling D/E exponents.
    """
    token = token.strip()
    if not token:
        raise ValueError("Empty numeric token")
    # Replace Fortran D exponent with E
    token = token.replace("D", "E").replace("d", "e")
    return float(token)


def extract_pots_from_atlas7v(path: Path) -> Dict[str, np.ndarray]:
    """
    Scan atlas7v.for for DATA POTX(...) / ... / blocks and build a mapping
    element_symbol -> array of ionization potentials in eV.
    """
    text = path.read_text(encoding="ascii", errors="ignore")
    lines = text.splitlines()

    pots: Dict[str, List[float]] = {}

    i = 0
    while i < len(lines):
        line = lines[i]
        # Look for a DATA POTXXX( or DATA POTXXX /
        m = re.match(r"\s*DATA\s+POT([A-Za-z0-9]+)\s*\(?.*?/([^/]*)$", line)
        if m is None:
            i += 1
            continue

        var_suffix = m.group(1)  # e.g. "Ni", "He", "Fe", "Mn"
        # Normalise element symbol: upper-case the suffix
        elem = var_suffix.upper()

        # Collect everything between the first '/' on this line and the closing '/'
        # Handles multi-line DATA statements.
        # Find first '/' on the current line
        start_idx = line.find("/")
        if start_idx < 0:
            i += 1
            continue

        data_str_parts = [line[start_idx + 1 :]]
        # Check if this line already has the closing '/'
        if "/" not in data_str_parts[0]:
            # Accumulate following lines until we hit a '/' terminator
            i += 1
            while i < len(lines):
                l2 = lines[i]
                if "/" in l2:
                    # Include up to the closing '/'
                    end_pos = l2.find("/")
                    data_str_parts.append(l2[:end_pos])
                    break
                else:
                    data_str_parts.append(l2)
                i += 1
        else:
            # Trim after the closing '/'
            first = data_str_parts[0]
            end_pos = first.find("/")
            data_str_parts[0] = first[:end_pos]

        data_str = " ".join(data_str_parts)
        # Split on commas and parse numbers
        tokens = [t for t in re.split(r"[,\s]+", data_str) if t]
        values: List[float] = []
        for tok in tokens:
            try:
                val = _parse_fortran_real(tok)
            except ValueError:
                continue
            values.append(val)

        if not values:
            i += 1
            continue

        # Store under element symbol; multiple DATA blocks for the same POT*
        # are concatenated (though in practice PFSAHA uses a single block).
        if elem not in pots:
            pots[elem] = []
        pots[elem].extend(values)

        i += 1

    # Convert to NumPy arrays and apply PFSAHA's eV conversion where needed.
    # In PFSAHA, IP(ION) = POTION(INDEX)/8065.479D0, i.e. POTION is in cm^-1.
    # The POT* arrays are slices of POTION, so divide by 8065.479 to get eV.
    out: Dict[str, np.ndarray] = {}
    for elem, vals in pots.items():
        arr_cm = np.asarray(vals, dtype=float)
        arr_ev = arr_cm / 8065.479
        out[elem] = arr_ev

    return out


def main() -> None:
    if not FORTRAN_PATH.exists():
        raise SystemExit(f"Fortran source not found: {FORTRAN_PATH}")

    out_dir = OUTPUT_PATH.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    pots = extract_pots_from_atlas7v(FORTRAN_PATH)

    # Save as .npz with one array per element symbol
    np.savez(OUTPUT_PATH, **pots)
    print(f"Wrote PFSAHA ionization potentials to {OUTPUT_PATH}")
    print("Elements extracted:", ", ".join(sorted(pots.keys())))


if __name__ == "__main__":
    main()


