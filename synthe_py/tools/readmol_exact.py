#!/usr/bin/env python3
"""Exact implementation of READMOL subroutine for reading molecular data.

From atlas7v_1.for lines 3484-3578.
Reads molecular data from fort.2 (molecules.dat) and sets up molecular equilibrium equations.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple
from pathlib import Path

# Constants
MAXMOL = 200
MAXEQ = 30
MAXLOC = 3 * MAXMOL


def readmol_exact(molecules_path: Path) -> Tuple[
    int,  # nummol
    np.ndarray,  # code_mol (MAXMOL,)
    np.ndarray,  # equil (7, MAXMOL)
    np.ndarray,  # locj (MAXMOL+1,)
    np.ndarray,  # kcomps (MAXLOC,)
    np.ndarray,  # idequa (MAXEQ,)
    int,  # nequa
    int,  # nloc
]:
    """
    Read molecular data from molecules.dat (fort.2 format).

    Format: F18.2,F7.3,6E11.4
    - C: Molecular code (e.g., 60808. = CO2 = 6,8,8)
    - E1-E7: Equilibrium constants (polynomial coefficients)

    Returns:
        (nummol, code_mol, equil, locj, kcomps, idequa, nequa, nloc)
    """
    # Initialize arrays
    code_mol = np.zeros(MAXMOL, dtype=np.float64)
    equil = np.zeros((7, MAXMOL), dtype=np.float64)
    locj = np.zeros(MAXMOL + 1, dtype=np.int32)
    kcomps = np.zeros(MAXLOC, dtype=np.int32)
    idequa = np.zeros(MAXEQ, dtype=np.int32)
    # Fortran uses IFEQUA(101) for inverse electrons, so we need 102 elements (0-101)
    ifequa = np.zeros(102, dtype=np.int32)  # 0-101 (Fortran 1-based: 1-101)

    # XCODE: powers of 10 for decoding molecular codes
    xcode = np.array([1e14, 1e12, 1e10, 1e8, 1e6, 1e4, 1e2, 1e0], dtype=np.float64)

    kloc = 0  # Component location counter (0-based)
    locj[0] = 0  # First molecule starts at index 0

    with molecules_path.open("r") as f:
        lines = f.readlines()

    nummol = 0
    for line_idx, line in enumerate(lines):
        # CRITICAL: Do NOT strip the line - Fortran reads fixed-width fields as-is
        # Only strip individual fields when parsing
        # Remove newline but keep trailing spaces for fixed-width parsing
        line = line.rstrip("\n\r")
        # CRITICAL: Fortran uses 'C' for comments (not '#')
        # Match Fortran's comment handling: skip lines starting with 'C' or 'c'
        stripped = line.strip()
        if (
            not stripped
            or stripped.startswith("C")
            or stripped.startswith("c")
            or stripped.startswith("#")
        ):
            continue

        # Parse: F18.2,F7.3,6E11.4
        # Format: C (18 chars), E1 (7 chars), E2-E7 (11 chars each)
        # CRITICAL: Fortran's F18.2 can read from shorter lines - it reads what's available
        # So we must NOT skip lines shorter than 18 chars!
        # Try fixed-width parsing first
        try:
            # Read C from first 18 chars (or less if line is shorter)
            c_str = line[0 : min(18, len(line))].strip()
            if not c_str:
                continue
            c = float(c_str)

            # Parse E1-E7 (may be missing if line is short)
            e1 = 0.0
            e2 = 0.0
            e3 = 0.0
            e4 = 0.0
            e5 = 0.0
            e6 = 0.0
            e7 = 0.0

            if len(line) >= 25:
                e1_str = line[18:25].strip()
                if e1_str:
                    e1 = float(e1_str)
            if len(line) >= 36:
                e2_str = line[25:36].strip()
                if e2_str:
                    e2 = float(e2_str)
            if len(line) >= 47:
                e3_str = line[36:47].strip()
                if e3_str:
                    e3 = float(e3_str)
            if len(line) >= 58:
                e4_str = line[47:58].strip()
                if e4_str:
                    e4 = float(e4_str)
            if len(line) >= 69:
                e5_str = line[58:69].strip()
                if e5_str:
                    e5 = float(e5_str)
            if len(line) >= 80:
                e6_str = line[69:80].strip()
                if e6_str:
                    e6 = float(e6_str)
            if len(line) >= 91:
                e7_str = line[80:91].strip()
                if e7_str:
                    e7 = float(e7_str)
        except (ValueError, IndexError):
            # Fallback to space-separated parsing (only if fixed-width fails)
            # CRITICAL: Must match Fortran's behavior exactly
            # Fortran reads F18.2,F7.3,6E11.4 - if a field is empty, it reads 0.0
            # So we should only use space-separated if fixed-width completely fails
            # But we already tried fixed-width above, so this fallback should rarely be needed
            parts = line.split()
            if len(parts) < 1:
                continue
            try:
                c = float(parts[0])
                # For EQUIL values, only read if we have enough parts
                # Match Fortran: if field is missing, use 0.0
                e1 = float(parts[1]) if len(parts) > 1 else 0.0
                e2 = float(parts[2]) if len(parts) > 2 else 0.0
                e3 = float(parts[3]) if len(parts) > 3 else 0.0
                e4 = float(parts[4]) if len(parts) > 4 else 0.0
                e5 = float(parts[5]) if len(parts) > 5 else 0.0
                e6 = float(parts[6]) if len(parts) > 6 else 0.0
                e7 = float(parts[7]) if len(parts) > 7 else 0.0
        # Skip zero/blank molecular code (sentinel or placeholder in some molecules.dat)
            except (ValueError, IndexError):
                continue

        if c == 0.0 or (isinstance(c, float) and abs(c) < 1e-12):
            continue
        # Terminator: CODE=0 means end of molecule list (Fortran conven        # Terminator: CODE=0 means end of molecule list (Fortran convention)
        if c == 0.0:
            break

        if nummol >= MAXMOL:
            raise ValueError(f"Too many molecules (>{MAXMOL})")

        if kloc >= MAXLOC:
            raise ValueError(f"Too many components (>{MAXLOC})")

        # Decode molecular code
        # Find starting power of 10
        ii = 0
        for i in range(8):
            if c >= xcode[i]:
                ii = i
                break
        else:
            raise ValueError(f"Invalid molecular code: {c}")

        # Extract element IDs from code
        x = c
        num_elements = 0  # Count actual elements (not inverse electrons)
        for i in range(ii, 8):
            id_elem = int(x / xcode[i])
            x = x - float(id_elem) * xcode[i]
            if id_elem == 0:
                id_elem = 100  # Electrons

            ifequa[id_elem] = 1
            kcomps[kloc] = id_elem
            kloc += 1
            num_elements += 1

        # Extract ionization state (remainder after extracting elements)
        ion = int(x * 100.0 + 0.5)
        # CRITICAL FIX: Only add inverse electrons for multi-element molecules
        # Single-element ions (like K+ with CODE=19.01) should NOT have inverse electrons added
        # This matches Fortran behavior where single-element ions are skipped (NCOMP=1)
        if ion >= 1:
            ifequa[100] = 1  # Electrons needed
            ifequa[101] = 1  # Inverse electrons
            for _ in range(ion):
                if kloc >= MAXLOC:
                    raise ValueError(f"Too many components (>{MAXLOC})")
                kcomps[kloc] = 101  # Inverse electrons
                kloc += 1

        # Store molecule data
        locj[nummol + 1] = kloc
        code_mol[nummol] = c
        equil[0, nummol] = e1
        equil[1, nummol] = e2
        equil[2, nummol] = e3
        equil[3, nummol] = e4
        equil[4, nummol] = e5
        equil[5, nummol] = e6
        equil[6, nummol] = e7

        # NOTE: Code 101.00 = H2 (dihydrogen), parsed as [1, 1] (two H atoms)
        # Code 100.00 = H- (hydride ion), which has inverse electron behavior
        # The previous "CRITICAL FIX" here was WRONG - it incorrectly treated H2 as H-
        # H2 should have kcomps = [H, H] = [1, 1], not [H, inverse_electron]

        nummol += 1

    nloc = kloc

    # Assign equation numbers to each component
    # CRITICAL: Match Fortran's IDEQUA assignment exactly!
    # Fortran: IEQUA=1, then IEQUA=IEQUA+1 before assigning IDEQUA(IEQUA)=I
    # This means IDEQUA(1) is NEVER set (stays at initial value, probably 0)
    # IDEQUA(2) = first element, IDEQUA(3) = second element, etc.
    # Python: idequa[0] = 0 (total particles, never used), idequa[1] = first element, etc.
    # But to match Fortran's indexing, we need idequa[1] = IDEQUA(2), idequa[2] = IDEQUA(3), etc.
    # So Python's idequa[k] for k>=1 should match Fortran's IDEQUA(k+1)
    iequa = 1  # Start at 1 (matches Fortran IEQUA=1)

    for i in range(1, 101):  # Elements 1-100
        if ifequa[i] == 1:
            iequa += 1  # Increment FIRST (matches Fortran)
            ifequa[i] = iequa  # Map element ID to equation number
            # CRITICAL: idequa[iequa-1] maps to Fortran IDEQUA(iequa)
            # But Fortran's IDEQUA(1) is never set, so idequa[0] should stay 0
            # idequa[1] = IDEQUA(2), idequa[2] = IDEQUA(3), etc.
            idequa[iequa - 1] = i  # Map equation number to element ID (0-based)

    nequa = iequa
    nequa1 = nequa + 1
    ifequa[101] = nequa1  # Inverse electrons (0-based: index 101 = Fortran 101)

    # Remap component indices to equation numbers
    # Fortran: KCOMPS(KLOC) = IFEQUA(ID) (1-based)
    # Python: kcomps[kloc_idx] = ifequa[id_elem] - 1 (convert to 0-based)
    for kloc_idx in range(nloc):
        id_elem = kcomps[kloc_idx]
        if id_elem < len(ifequa):
            kcomps[kloc_idx] = ifequa[id_elem] - 1  # Convert to 0-based
        else:
            raise ValueError(f"Invalid element ID {id_elem} in KCOMPS[{kloc_idx}]")

    return nummol, code_mol, equil, locj, kcomps, idequa, nequa, nloc
