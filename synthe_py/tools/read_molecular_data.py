#!/usr/bin/env python3
"""Read molecular data file (READMOL equivalent) matching Fortran exactly.

This implements the READMOL subroutine from atlas7v_1.for lines 3484-3578.
It reads molecular data from a file (Fortran unit 2) and sets up molecular equilibrium structures.

File format (line 3515-3516):
  READ(2,13)C,E1,E2,E3,E4,E5,E6,E7
  FORMAT(F18.2,F7.3,6E11.4)

Where:
  C: Molecule code (e.g., 60808. for CO2, 100. for H-, 101.01 for H2+)
  E1-E7: Equilibrium constants (7 values)
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass

# Constants matching Fortran
MAXMOL = 200
MAX1 = MAXMOL + 1
MAXEQ = 30
MAXLOC = 3 * MAXMOL

# XCODE array for decoding molecule codes (line 3493)
XCODE = np.array([1e14, 1e12, 1e10, 1e8, 1e6, 1e4, 1e2, 1e0], dtype=np.float64)


@dataclass
class MolecularData:
    """Molecular data structures matching Fortran COMMON blocks."""
    
    nummol: int  # Number of molecules
    nequa: int  # Number of equations
    nloc: int  # Number of components
    
    # From COMMON /IFEQUA/
    idequa: np.ndarray  # Shape: (NEQUA,) - element IDs for equations
    kcomps: np.ndarray  # Shape: (NLOC,) - component indices
    locj: np.ndarray  # Shape: (MAX1,) - molecule location indices
    
    # From COMMON /XNMOL/
    code_mol: np.ndarray  # Shape: (MAXMOL,) - molecule codes
    
    # From COMMON /IFEQUA/
    equil: np.ndarray  # Shape: (7, MAXMOL) - equilibrium constants
    
    # From COMMON /IFEQUA/ (computed)
    ifequa: np.ndarray  # Shape: (101,) - flag for which elements need equations


def read_molecular_data_file(mol_file_path: Path) -> MolecularData:
    """Read molecular data file matching Fortran READMOL exactly.
    
    Args:
        mol_file_path: Path to molecular data file (typically molecules.dat)
    
    Returns:
        MolecularData object with all molecular structures
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    if not mol_file_path.exists():
        raise FileNotFoundError(f"Molecular data file not found: {mol_file_path}")
    
    # Initialize arrays (matching Fortran lines 3504-3509)
    # Fortran IFEQUA(101) is 1-indexed, so indices 1-101 → Python needs size 102 (indices 0-101)
    code_mol = np.zeros(MAXMOL, dtype=np.float64)
    ifequa = np.zeros(102, dtype=np.int32)  # Elements 1-101 (Fortran 1-indexed → Python 0-101)
    kcomps = np.zeros(MAXLOC, dtype=np.int32)
    locj = np.zeros(MAX1, dtype=np.int32)
    equil = np.zeros((7, MAXMOL), dtype=np.float64)
    
    locj[0] = 1  # LOCJ(1) = 1 (line 3512)
    kloc = 1  # Component location counter (line 3511)
    
    # Read molecules from file (lines 3513-3547)
    with mol_file_path.open("r") as f:
        jmol = 0
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            # Parse line: C, E1, E2, E3, E4, E5, E6, E7
            # Format: F18.2, F7.3, 6E11.4
            # The file has two formats:
            # 1. Simple: just the code (e.g., "1.", "1.01", "2.")
            # 2. Full: code + 7 equilibrium constants
            
            parts = line.split()
            if len(parts) == 0:
                continue
            
            # Try to parse code
            try:
                c = float(parts[0])
            except ValueError:
                continue
            
            # Check if this is a simple code (no equilibrium constants) or full entry
            if len(parts) >= 8:
                # Full entry with equilibrium constants
                try:
                    e1 = float(parts[1])
                    e2 = float(parts[2])
                    e3 = float(parts[3])
                    e4 = float(parts[4])
                    e5 = float(parts[5])
                    e6 = float(parts[6])
                    e7 = float(parts[7])
                except (ValueError, IndexError):
                    # Try fixed-width parsing (Fortran format)
                    try:
                        if len(line) >= 91:
                            c = float(line[0:18])
                            e1 = float(line[18:25])
                            e2 = float(line[25:36])
                            e3 = float(line[36:47])
                            e4 = float(line[47:58])
                            e5 = float(line[58:69])
                            e6 = float(line[69:80])
                            e7 = float(line[80:91])
                        else:
                            # Simple code - set equilibrium constants to zero
                            e1 = e2 = e3 = e4 = e5 = e6 = e7 = 0.0
                    except (ValueError, IndexError):
                        # Simple code - set equilibrium constants to zero
                        e1 = e2 = e3 = e4 = e5 = e6 = e7 = 0.0
            else:
                # Simple code - set equilibrium constants to zero
                # These will be computed from PFSAHA if needed (line 3724-3730)
                e1 = e2 = e3 = e4 = e5 = e6 = e7 = 0.0
            
            # Check for end marker (line 3517)
            if c == 0.0:
                break
            
            # Check if too many molecules (line 3514)
            if kloc > MAXLOC:
                raise ValueError(f"Too many molecules/components: kloc={kloc} > MAXLOC={MAXLOC}")
            
            if jmol >= MAXMOL:
                raise ValueError(f"Too many molecules: jmol={jmol} >= MAXMOL={MAXMOL}")
            
            # Decode molecule code (lines 3520-3538)
            # Find which XCODE range C falls into
            x = c
            found_range = False
            for ii in range(8):
                if c >= XCODE[ii]:
                    found_range = True
                    break
            
            if not found_range:
                raise ValueError(f"Invalid molecule code: {c}")
            
            # Extract element IDs from code (lines 3524-3531)
            for i in range(ii, 8):
                id_elem = int(x / XCODE[i])
                x = x - float(id_elem) * XCODE[i]
                if id_elem == 0:
                    id_elem = 100  # Special marker for electrons
                
                ifequa[id_elem] = 1
                kcomps[kloc - 1] = id_elem  # Convert to 0-based
                kloc += 1
            
            # Extract ion count (lines 3532-3538)
            ion = int(x * 100.0 + 0.5)
            if ion >= 1:
                ifequa[100] = 1
                ifequa[101] = 1
                for i in range(ion):
                    kcomps[kloc - 1] = 101  # H+ ions
                    kloc += 1
            
            # Set LOCJ and CODE (lines 3539-3547)
            locj[jmol + 1] = kloc
            code_mol[jmol] = c
            equil[0, jmol] = e1
            equil[1, jmol] = e2
            equil[2, jmol] = e3
            equil[3, jmol] = e4
            equil[4, jmol] = e5
            equil[5, jmol] = e6
            equil[6, jmol] = e7
            
            jmol += 1
    
    nummol = jmol
    nloc = kloc - 1
    
    # Assign equation numbers to each component (lines 3552-3564)
    # The first equation is for total number of particles
    # The first variable is XNATOM
    # If any component is 100 or 101, variable NEQUA is XNE
    iequa = 1  # Start at 1 (equation 0 is for total particles)
    
    idequa = np.zeros(MAXEQ, dtype=np.int32)
    
    # Fortran: DO 25 I=1,100 (elements 1-100)
    # Python: range(1, 101) → indices 1-100 (Fortran elements 1-100)
    for i in range(1, 101):  # Elements 1-100 (Fortran 1-indexed → Python indices 1-100)
        if ifequa[i] == 1:
            iequa += 1
            ifequa[i] = iequa  # Store equation number
            idequa[iequa - 1] = i  # Store element ID (1-100)
    
    nequa = iequa
    nequa1 = nequa + 1
    ifequa[101] = nequa1  # NEQUA1 for inverse XNE (Fortran IFEQUA(101) → Python index 101)
    
    # Convert KCOMPS to equation numbers (lines 3569-3571)
    for kloc_idx in range(nloc):
        id_elem = kcomps[kloc_idx]
        kcomps[kloc_idx] = ifequa[id_elem]  # Convert to equation number
    
    # Trim arrays to actual sizes
    idequa = idequa[:nequa]
    kcomps = kcomps[:nloc]
    locj = locj[:nummol + 1]
    code_mol = code_mol[:nummol]
    equil = equil[:, :nummol]
    
    return MolecularData(
        nummol=nummol,
        nequa=nequa,
        nloc=nloc,
        idequa=idequa,
        kcomps=kcomps,
        locj=locj,
        code_mol=code_mol,
        equil=equil,
        ifequa=ifequa,
    )


def find_molecular_data_file(search_paths: Optional[list[Path]] = None) -> Optional[Path]:
    """Find molecular data file in common locations.
    
    Args:
        search_paths: Optional list of paths to search. If None, uses default locations.
    
    Returns:
        Path to molecular data file if found, None otherwise
    """
    if search_paths is None:
        _repo_root = Path(__file__).resolve().parents[2]
        # Default search paths — prefer self-contained data/lines/, then repo lines/,
        # then sibling kurucz/ (legacy layout)
        search_paths = [
            _repo_root / "data" / "lines" / "molecules.dat",
            _repo_root / "lines" / "molecules.dat",
            _repo_root.parent / "kurucz" / "lines" / "molecules.dat",
            Path("lines/molecules.dat"),
            Path("synthe/lines/molecules.dat"),
        ]
    
    for path in search_paths:
        if path.exists():
            return path
    
    return None

