"""READMOL parser for ATLAS12 molecular-equilibrium data.

Fortran reference: `atlas12.for` lines 2782-2888.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAXMOL = 200
MAXEQ = 35
MAXLOC = 3 * MAXMOL


@dataclass
class ReadMolData:
    """ATLAS12 READMOL structures."""

    nummol: int
    nequa: int
    nloc: int
    code_mol: np.ndarray  # (MAXMOL,)
    equil: np.ndarray  # (7, MAXMOL), row 6 unused for atlas12 but kept for compatibility
    locj: np.ndarray  # (MAXMOL+1,)
    kcomps: np.ndarray  # (MAXLOC,), equation indices (0-based)
    idequa: np.ndarray  # (MAXEQ,), element IDs
    ifequa: np.ndarray  # (102,)


def _parse_data_line(raw: str) -> tuple[float, float, float, float, float, float, float]:
    """Parse one READ(2,13) record: F18.2,F7.3,5E11.4."""
    line = raw.rstrip("\n\r")
    if not line:
        raise ValueError("blank")
    # Fixed-width parse first.
    c = float(line[0 : min(18, len(line))].strip())
    e1 = float(line[18:25].strip()) if len(line) >= 25 and line[18:25].strip() else 0.0
    e2 = float(line[25:36].strip()) if len(line) >= 36 and line[25:36].strip() else 0.0
    e3 = float(line[36:47].strip()) if len(line) >= 47 and line[36:47].strip() else 0.0
    e4 = float(line[47:58].strip()) if len(line) >= 58 and line[47:58].strip() else 0.0
    e5 = float(line[58:69].strip()) if len(line) >= 69 and line[58:69].strip() else 0.0
    e6 = float(line[69:80].strip()) if len(line) >= 80 and line[69:80].strip() else 0.0
    return c, e1, e2, e3, e4, e5, e6


def readmol_atlas12(path: Path) -> ReadMolData:
    """Read molecular equilibrium definitions from `molecules.new` / `molecules.dat`."""
    if not path.exists():
        raise FileNotFoundError(f"Molecular data file not found: {path}")

    code_mol = np.zeros(MAXMOL, dtype=np.float64)
    equil = np.zeros((7, MAXMOL), dtype=np.float64)
    locj = np.zeros(MAXMOL + 1, dtype=np.int32)
    kcomps = np.zeros(MAXLOC, dtype=np.int32)
    idequa = np.zeros(MAXEQ, dtype=np.int32)
    ifequa = np.zeros(102, dtype=np.int32)  # 1..101 in Fortran

    xcode = np.array([1e14, 1e12, 1e10, 1e8, 1e6, 1e4, 1e2, 1e0], dtype=np.float64)

    kloc = 1
    locj[0] = 1
    nummol = 0

    with path.open("r", encoding="ascii", errors="ignore") as f:
        for raw in f:
            stripped = raw.strip()
            if not stripped or stripped.startswith("C") or stripped.startswith("c") or stripped.startswith("#"):
                continue
            c, e1, e2, e3, e4, e5, e6 = _parse_data_line(raw)
            if c == 0.0:
                break
            if nummol >= MAXMOL:
                raise ValueError(f"Too many molecules (>{MAXMOL})")
            if kloc > MAXLOC:
                raise ValueError(f"Too many components (>{MAXLOC})")

            ii = -1
            for i in range(8):
                if c >= xcode[i]:
                    ii = i
                    break
            if ii < 0:
                raise ValueError(f"Invalid molecule code: {c}")

            x = c
            for i in range(ii, 8):
                ide = int(x / xcode[i] + 0.5)
                x = x - float(ide) * xcode[i]
                if ide == 0:
                    ide = 100
                ifequa[ide] = 1
                kcomps[kloc - 1] = ide
                kloc += 1

            ion = int(x * 100.0 + 0.5)
            if ion >= 1:
                ifequa[100] = 1
                ifequa[101] = 1
                for _ in range(ion):
                    if kloc > MAXLOC:
                        raise ValueError(f"Too many components (>{MAXLOC})")
                    kcomps[kloc - 1] = 101
                    kloc += 1

            locj[nummol + 1] = kloc
            code_mol[nummol] = c
            equil[0, nummol] = e1
            equil[1, nummol] = e2
            equil[2, nummol] = e3
            equil[3, nummol] = e4
            equil[4, nummol] = e5
            equil[5, nummol] = e6
            # equil[6,*] intentionally 0.0 for atlas12 (only 6 coeffs in READMOL)
            nummol += 1

    nloc = kloc - 1
    iequa = 1
    for i in range(1, 101):
        if ifequa[i] != 0:
            iequa += 1
            ifequa[i] = iequa
            if iequa > MAXEQ:
                raise ValueError(f"Too many equations NEQUA={iequa} > MAXEQ={MAXEQ}")
            idequa[iequa - 1] = i
    nequa = iequa
    nequa1 = nequa + 1
    ifequa[101] = nequa1

    # Convert components to equation indices (0-based for Python kernels).
    for k in range(nloc):
        ide = int(kcomps[k])
        kcomps[k] = int(ifequa[ide] - 1)

    return ReadMolData(
        nummol=nummol,
        nequa=nequa,
        nloc=nloc,
        code_mol=code_mol,
        equil=equil,
        locj=locj,
        kcomps=kcomps,
        idequa=idequa,
        ifequa=ifequa,
    )


def find_default_molecules_file() -> Path | None:
    """Search common local locations for ATLAS12 molecular input file.

    Search order:
    1. data/lines/ inside the pykurucz repo (self-contained layout after setup_data.sh)
    2. lines/ inside the pykurucz repo
    3. Sibling ../kurucz/lines/ (legacy layout)
    4. CWD-relative lines/
    """
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "data" / "lines" / "molecules.new",
        repo_root / "data" / "lines" / "molecules.dat",
        repo_root / "lines" / "molecules.new",
        repo_root / "lines" / "molecules.dat",
        repo_root.parent / "kurucz" / "lines" / "molecules.new",
        repo_root.parent / "kurucz" / "lines" / "molecules.dat",
        Path("lines/molecules.new"),
        Path("lines/molecules.dat"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


