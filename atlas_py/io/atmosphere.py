"""ATLAS12 `.atm` read/write utilities.

Units (Fortran atlas12.for):
- RHOX: g cm^-2
- T: K
- P: dyn cm^-2
- XNE: cm^-3
- ABROSS: cm^2 g^-1
- ACCRAD: cm s^-2
- VTURB: cm s^-1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Dict, List

import numpy as np


@dataclass
class AtlasAtmosphere:
    """Structured ATLAS12 atmosphere model."""

    rhox: np.ndarray
    temperature: np.ndarray
    gas_pressure: np.ndarray
    electron_density: np.ndarray
    abross: np.ndarray
    accrad: np.ndarray
    vturb: np.ndarray
    extra1: np.ndarray
    extra2: np.ndarray
    metadata: Dict[str, str] = field(default_factory=dict)
    abundances: Dict[int, float] = field(default_factory=dict)

    @property
    def layers(self) -> int:
        return int(self.rhox.size)

    @property
    def tk(self) -> np.ndarray:
        # Fortran: TK(J)=1.38054D-16*T(J) (atlas12.for line 943)
        return self.temperature * 1.38054e-16

    @property
    def hkt(self) -> np.ndarray:
        # Fortran TEMP common uses HKT = H / (K_B * T), not 1/(K_B * T).
        return 6.6256e-27 / (self.tk + 1e-300)

    @property
    def hckt(self) -> np.ndarray:
        return (6.6256e-27 * 2.99792458e10) / (self.tk + 1e-300)

    @property
    def tkev(self) -> np.ndarray:
        return self.temperature / 11604.5

    @property
    def tlog(self) -> np.ndarray:
        return np.log(np.maximum(self.temperature, 1e-300))


def _parse_abundance_change(line: str, out: Dict[int, float]) -> None:
    # Locate "ABUNDANCE CHANGE" keyword, then parse (z, val) pairs from the
    # tokens that follow it.  This correctly handles both standalone lines
    # ("ABUNDANCE CHANGE 3 -13.24 ...") and combined lines that start with
    # "ABUNDANCE SCALE  1.00000 ABUNDANCE CHANGE 1 0.92163 2 0.07837 ...".
    key = "ABUNDANCE CHANGE"
    pos = line.find(key)
    if pos < 0:
        return
    tail = line[pos + len(key):]
    parts = tail.split()
    if len(parts) < 2:
        return
    for idx in range(0, len(parts) - 1, 2):
        try:
            z = int(parts[idx])
            val = float(parts[idx + 1])
        except ValueError:
            break  # non-integer token signals end of (z, val) pairs
        out[z] = val


def load_atm(path: Path) -> AtlasAtmosphere:
    """Load an ATLAS-style atmosphere from `.atm` text."""

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    metadata: Dict[str, str] = {}
    abundances: Dict[int, float] = {}

    teff = None
    grav = None
    ifop = None
    title = ""

    deck_idx = -1
    deck_header_idx = -1
    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("TEFF"):
            metadata["teff_line"] = raw
            parts = line.split()
            for j, tok in enumerate(parts):
                if tok == "TEFF" and j + 1 < len(parts):
                    teff = float(parts[j + 1].rstrip("."))
                if tok == "GRAVITY" and j + 1 < len(parts):
                    grav = float(parts[j + 1])
                if tok == "G" and j >= 1 and parts[j - 1] == "LOG" and j + 1 < len(parts):
                    try:
                        grav = float(parts[j + 1])
                    except ValueError:
                        pass
        elif line.startswith("TITLE"):
            title = raw[5:].strip() if len(raw) > 5 else ""
        elif line.startswith("OPACITY IFOP"):
            ifop = raw.strip()
        elif line.startswith("PRESSURE"):
            parts = line.split()
            if len(parts) >= 2:
                metadata["ifpres"] = "1" if parts[1].upper() == "ON" else "0"
        elif "ABUNDANCE CHANGE" in line:
            _parse_abundance_change(line, abundances)
        elif line.startswith("READ DECK6"):
            deck_idx = i + 1
            deck_header_idx = i
            metadata["deck6_header"] = raw.strip()
            break

    if deck_idx < 0:
        # Fallback parser for atlas12 stdout dumps where unit 7 is preconnected
        # to stdout (no explicit READ DECK6 punch deck emitted).
        table_idx = -1
        for i, raw in enumerate(lines):
            line = raw.strip().upper()
            if "RHOX" in line and "XNE" in line and "ABROSS" in line and "VTURB" in line:
                table_idx = i + 1
                break
        if table_idx < 0:
            raise ValueError(f"READ DECK6 section not found in {path}")
        rows: List[List[float]] = []
        for raw in lines[table_idx:]:
            line = raw.strip()
            if not line:
                if rows:
                    break
                continue
            parts = line.split()
            # Expected format: depth, rhox, T, P, XNE, ABROSS, ACCRAD, VTURB, extra1, extra2
            if len(parts) < 8:
                continue
            try:
                _idx = int(parts[0])
                vals = [float(parts[k]) for k in range(1, 10)]
            except ValueError:
                if rows:
                    break
                continue
            rows.append(vals)
        if not rows:
            raise ValueError(f"No layer rows parsed from fallback table in {path}")
        arr = np.asarray(rows, dtype=np.float64)
        metadata["title"] = title
        if ifop is not None:
            metadata["ifop"] = ifop
        if teff is not None:
            metadata["teff"] = f"{teff:.6f}"
        if grav is not None:
            metadata["grav"] = f"{grav:.6f}"
        # Fallback table carries PRAD in column 6; we keep it in `accrad` slot
        # for compatibility with existing AtlasAtmosphere schema.
        return AtlasAtmosphere(
            rhox=arr[:, 0],
            temperature=arr[:, 1],
            gas_pressure=arr[:, 2],
            electron_density=arr[:, 3],
            abross=arr[:, 4],
            accrad=arr[:, 5],
            vturb=arr[:, 6],
            extra1=arr[:, 7],
            extra2=arr[:, 8],
            metadata=metadata,
            abundances=abundances,
        )

    if deck_header_idx >= 0:
        metadata["predeck_block"] = "\n".join(lines[: deck_header_idx + 1])

    rows: List[List[float]] = []
    row_stop_idx = deck_idx
    for i in range(deck_idx, len(lines)):
        raw = lines[i]
        line = raw.strip()
        if not line:
            continue
        # Fortran fixed-format READ DECK6 can emit touching fields (e.g., 4.131E+00-1.058E+07).
        line = re.sub(r"(?<=[0-9])([+-])(?=\d)", r" \1", line)
        parts = line.split()
        if len(parts) < 7:
            row_stop_idx = i
            break
        try:
            row7 = [float(parts[k]) for k in range(7)]
        except ValueError:
            row_stop_idx = i
            break
        e1 = 0.0
        e2 = 0.0
        if len(parts) > 7:
            try:
                e1 = float(parts[7])
            except ValueError:
                row_stop_idx = i
                break
        if len(parts) > 8:
            try:
                e2 = float(parts[8])
            except ValueError:
                row_stop_idx = i
                break
        rows.append(row7 + [e1, e2])
        row_stop_idx = i + 1

    if not rows:
        raise ValueError(f"No layer rows parsed from READ DECK6 in {path}")

    arr = np.asarray(rows, dtype=np.float64)
    metadata["title"] = title
    if ifop is not None:
        metadata["ifop"] = ifop
    if teff is not None:
        metadata["teff"] = f"{teff:.6f}"
    if grav is not None:
        metadata["grav"] = f"{grav:.6f}"
    for raw in lines[row_stop_idx:]:
        s = raw.strip()
        if not s:
            continue
        if s.startswith("PRADK"):
            metadata["pradk_line"] = raw.rstrip()
        elif s.startswith("BEGIN"):
            metadata["begin_line"] = raw.rstrip()
        elif s.startswith("END"):
            metadata["end_line"] = raw.rstrip()

    return AtlasAtmosphere(
        rhox=arr[:, 0],
        temperature=arr[:, 1],
        gas_pressure=arr[:, 2],
        electron_density=arr[:, 3],
        abross=arr[:, 4],
        accrad=arr[:, 5],
        vturb=arr[:, 6],
        extra1=arr[:, 7],
        extra2=arr[:, 8],
        metadata=metadata,
        abundances=abundances,
    )


def write_atm(model: AtlasAtmosphere, path: Path) -> None:
    """Write an ATLAS-compatible `.atm` file."""

    teff = model.metadata.get("teff", "0.0")
    grav = model.metadata.get("grav", "0.0")
    title = model.metadata.get("title", "atlas_py")
    ifop = model.metadata.get(
        "ifop", "OPACITY IFOP 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 0 0 0"
    )

    out: List[str] = []
    predeck = model.metadata.get("predeck_block")
    if predeck:
        out.extend(predeck.splitlines())
    else:
        out.append(f"TEFF   {float(teff):7.1f}  GRAVITY  {float(grav):6.4f} LTE ")
        out.append(f"TITLE {title:<74}")
        out.append(f" {ifop}")
        out.append(" CONVECTION ON   1.25 TURBULENCE OFF  0.00  0.00  0.00  0.00")
        h_abund = model.abundances.get(1, 0.92000)
        he_abund = model.abundances.get(2, 0.08000)
        out.append(
            f"ABUNDANCE SCALE   1.00000 ABUNDANCE CHANGE 1 {h_abund:.5f} 2 {he_abund:.5f}"
        )
        metals = {z: v for z, v in model.abundances.items() if z >= 3}
        if metals:
            entries = sorted(metals.items())
            line = " ABUNDANCE CHANGE"
            for z, v in entries:
                frag = f" {z:2d} {v:7.2f}"
                if len(line) + len(frag) > 78:
                    out.append(line)
                    line = " ABUNDANCE CHANGE"
                line += frag
            out.append(line)
        out.append("READ DECK6 80 RHOX,T,P,XNE,ABROSS,ACCRAD,VTURB")
    for i in range(model.layers):
        # Match Fortran atlas12.for FORMAT 554: (1PE15.8,0PF9.1,1P7E10.3)
        # 1PE10.3 for each of the 7 data values means field width=10, 3 decimal places.
        # Positive values: ' x.xxxE±xx' (1 leading space + 9 chars = 10 total).
        # Negative values: '-x.xxxE±xx' (no leading space, 10 chars exactly).
        # When P>0 is followed by XNE<0, the two fields concatenate as 'P.xxxE+yy-X.xxxE+zz',
        # which Fortran's FREEFF parser reads as P = abs(XNE) (discards P, returns |XNE|).
        # Python's SYNTHE reader (parse_atm_file) implements the same FREEFF behavior, so both
        # read identical P values from the .atm file.
        out.append(
            f"{model.rhox[i]:14.8E} {model.temperature[i]:8.1f}"
            f"{model.gas_pressure[i]:10.3E}{model.electron_density[i]:10.3E}"
            f"{model.abross[i]:10.3E}{model.accrad[i]:10.3E}"
            f"{model.vturb[i]:10.3E}{model.extra1[i]:10.3E}{model.extra2[i]:10.3E}"
        )
    pradk_line = model.metadata.get("pradk_line")
    if pradk_line:
        out.append(pradk_line)
    begin_line = model.metadata.get("begin_line", "BEGIN                    ITERATION  1 COMPLETED")
    out.append(begin_line)
    end_line = model.metadata.get("end_line")
    if end_line:
        out.append(end_line)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")

