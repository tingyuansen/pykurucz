"""Read Fortran unformatted binary fort.10 produced by xnfpelsyn.

Binary layout (Fortran REAL*8 with 4-byte record markers):
  Rec 1: NT(int), TEFF(f64), GLOG(f64), TITLE(74×f64)
  Rec 2: IN(int), (FRQEDG, WLEDGE, CMEDGE)×IN, IDMOL(mm), MOMASS(mm)
  Rec 3: NUMNU(int), FREQSET(NUMNU)
  Rec 4: T(NT), TKEV, TK, HKT, TLOG, HCKT, P, XNE, XNATOM, RHO, RHOX,
         VTURB, XNFH, XNFHE, XNFH2  — 15 arrays of NT f64 each
  Per-depth (NT times):
    Rec 5a: CONTINALL(1131)  — log10(total abs)
    Rec 5b: CONTABS(1131)    — log10(absorption)
    Rec 5c: CONTSCAT(1131)   — log10(scattering)
    Rec 5d: XNFPEL(6, mw), DOPPLE(6, mw) — populations & Doppler widths

Parameters: kw=99, mw=139, mm=100
"""
from __future__ import annotations

import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

KW = 99
MW = 139
MM = MW - 39  # 100


def _read_record(f) -> bytes:
    """Read one Fortran unformatted sequential record (4-byte markers)."""
    marker = f.read(4)
    if len(marker) < 4:
        raise EOFError("Unexpected end of file reading record marker")
    (nbytes,) = struct.unpack("<i", marker)
    data = f.read(nbytes)
    if len(data) < nbytes:
        raise EOFError(f"Expected {nbytes} bytes, got {len(data)}")
    end_marker = f.read(4)
    (nbytes2,) = struct.unpack("<i", end_marker)
    if nbytes != nbytes2:
        raise ValueError(f"Record markers mismatch: {nbytes} vs {nbytes2}")
    return data


@dataclass
class Fort10Data:
    nt: int = 0
    teff: float = 0.0
    glog: float = 0.0

    # Edge/frequency grids
    n_edges: int = 0
    frqedg: np.ndarray = field(default_factory=lambda: np.empty(0))
    wledge: np.ndarray = field(default_factory=lambda: np.empty(0))
    cmedge: np.ndarray = field(default_factory=lambda: np.empty(0))
    idmol: np.ndarray = field(default_factory=lambda: np.empty(0))
    momass: np.ndarray = field(default_factory=lambda: np.empty(0))

    numnu: int = 0
    freqset: np.ndarray = field(default_factory=lambda: np.empty(0))

    # Thermodynamic arrays — shape (NT,)
    temperature: np.ndarray = field(default_factory=lambda: np.empty(0))
    tkev: np.ndarray = field(default_factory=lambda: np.empty(0))
    tk: np.ndarray = field(default_factory=lambda: np.empty(0))
    hkt: np.ndarray = field(default_factory=lambda: np.empty(0))
    tlog: np.ndarray = field(default_factory=lambda: np.empty(0))
    hckt: np.ndarray = field(default_factory=lambda: np.empty(0))
    pressure: np.ndarray = field(default_factory=lambda: np.empty(0))
    xne: np.ndarray = field(default_factory=lambda: np.empty(0))
    xnatom: np.ndarray = field(default_factory=lambda: np.empty(0))
    rho: np.ndarray = field(default_factory=lambda: np.empty(0))
    rhox: np.ndarray = field(default_factory=lambda: np.empty(0))
    vturb: np.ndarray = field(default_factory=lambda: np.empty(0))
    xnfh: np.ndarray = field(default_factory=lambda: np.empty(0))
    xnfhe: np.ndarray = field(default_factory=lambda: np.empty(0))
    xnfh2: np.ndarray = field(default_factory=lambda: np.empty(0))

    # Per-depth population/Doppler data — shape (NT, 6, MW)
    xnfpel: np.ndarray = field(default_factory=lambda: np.empty(0))
    dopple: np.ndarray = field(default_factory=lambda: np.empty(0))

    # Continuum — shape (NT, NUMNU) or (NT, 1131)
    continall: Optional[np.ndarray] = None
    contabs: Optional[np.ndarray] = None
    contscat: Optional[np.ndarray] = None


def read_fort10(path: str | Path, skip_continua: bool = False) -> Fort10Data:
    """Parse a Fortran fort.10 binary file from xnfpelsyn.

    All Fortran arrays are dimensioned with kw=99 (max depths), but only
    the first NT entries are physically meaningful.  INTEGER variables
    (NT, IN, NUMNU) are Fortran default INTEGER*4.  Everything else is REAL*8.
    XNFHE is DIMENSION(kw,2), so the thermodynamic record has 16×kw f64 values.
    """
    path = Path(path)
    data = Fort10Data()

    with open(path, "rb") as f:
        # Record 1: NT(int4), TEFF(f8), GLOG(f8), TITLE(74×f8)
        rec = _read_record(f)
        data.nt = struct.unpack_from("<i", rec, 0)[0]
        tail = np.frombuffer(rec[4:], dtype="<f8")
        data.teff = tail[0]
        data.glog = tail[1]
        nt = data.nt

        # Record 2: IN(int4), (FRQEDG,WLEDGE,CMEDGE)×IN f8, IDMOL(MM) f8, MOMASS(MM) f8
        rec = _read_record(f)
        data.n_edges = struct.unpack_from("<i", rec, 0)[0]
        n_in = data.n_edges
        tail = np.frombuffer(rec[4:], dtype="<f8")
        data.frqedg = tail[0:n_in].copy()
        data.wledge = tail[n_in:2 * n_in].copy()
        data.cmedge = tail[2 * n_in:3 * n_in].copy()
        offset = 3 * n_in
        data.idmol = tail[offset:offset + MM].copy()
        data.momass = tail[offset + MM:offset + 2 * MM].copy()

        # Record 3: NUMNU(int4), FREQSET(NUMNU) f8
        rec = _read_record(f)
        data.numnu = struct.unpack_from("<i", rec, 0)[0]
        tail = np.frombuffer(rec[4:], dtype="<f8")
        data.freqset = tail[:data.numnu].copy()

        # Record 4: 12 thermodynamic arrays of kw f8, then XNFH(kw),
        # XNFHE(kw,2)=2*kw, XNFH2(kw) → total 16*kw f64
        rec = _read_record(f)
        buf = np.frombuffer(rec, dtype="<f8")
        off = 0
        data.temperature = buf[off:off + KW].copy()[:nt]; off += KW
        data.tkev = buf[off:off + KW].copy()[:nt]; off += KW
        data.tk = buf[off:off + KW].copy()[:nt]; off += KW
        data.hkt = buf[off:off + KW].copy()[:nt]; off += KW
        data.tlog = buf[off:off + KW].copy()[:nt]; off += KW
        data.hckt = buf[off:off + KW].copy()[:nt]; off += KW
        data.pressure = buf[off:off + KW].copy()[:nt]; off += KW
        data.xne = buf[off:off + KW].copy()[:nt]; off += KW
        data.xnatom = buf[off:off + KW].copy()[:nt]; off += KW
        data.rho = buf[off:off + KW].copy()[:nt]; off += KW
        data.rhox = buf[off:off + KW].copy()[:nt]; off += KW
        data.vturb = buf[off:off + KW].copy()[:nt]; off += KW
        data.xnfh = buf[off:off + KW].copy()[:nt]; off += KW
        data.xnfhe = buf[off:off + 2 * KW].copy()[:nt]; off += 2 * KW
        data.xnfh2 = buf[off:off + KW].copy()[:nt]; off += KW

        # Per-depth records (NT iterations)
        data.xnfpel = np.zeros((nt, 6, MW), dtype=np.float64)
        data.dopple = np.zeros((nt, 6, MW), dtype=np.float64)
        if not skip_continua:
            data.continall = np.zeros((nt, 1131), dtype=np.float64)
            data.contabs = np.zeros((nt, 1131), dtype=np.float64)
            data.contscat = np.zeros((nt, 1131), dtype=np.float64)

        for j in range(nt):
            # CONTINALL(1131), CONTABS(1131), CONTSCAT(1131)
            rec_ca = _read_record(f)
            rec_cb = _read_record(f)
            rec_cs = _read_record(f)
            if not skip_continua:
                data.continall[j] = np.frombuffer(rec_ca, dtype="<f8")[:1131]
                data.contabs[j] = np.frombuffer(rec_cb, dtype="<f8")[:1131]
                data.contscat[j] = np.frombuffer(rec_cs, dtype="<f8")[:1131]

            # XNFPEL(6, MW), DOPPLE(6, MW) — Fortran column-major
            rec = _read_record(f)
            buf = np.frombuffer(rec, dtype="<f8")
            xnfpel_flat = buf[:6 * MW]
            dopple_flat = buf[6 * MW:2 * 6 * MW]
            # Fortran stores (ION, NELEM) column-major: fastest index is ION
            data.xnfpel[j] = xnfpel_flat.reshape((MW, 6)).T.copy()
            data.dopple[j] = dopple_flat.reshape((MW, 6)).T.copy()

    return data


def print_species_summary(
    data: Fort10Data,
    nelem: int,
    label: str = "",
) -> None:
    """Print XNFPEL(6, nelem) and DOPPLE(6, nelem) for all depths."""
    ei = nelem - 1  # 0-based
    print(f"{'='*80}")
    print(f"Species: {label}  NELEM={nelem}  NELION={nelem*6}")
    print(f"{'='*80}")
    print(f"{'Depth':>5}  {'XNFPEL(1)':>12}  {'XNFPEL(6)':>12}  "
          f"{'DOPPLE(1)':>12}  {'DOPPLE(6)':>12}  {'RHO':>12}")
    print("-" * 80)
    for j in range(data.nt):
        xp1 = data.xnfpel[j, 0, ei]
        xp6 = data.xnfpel[j, 5, ei]
        d1 = data.dopple[j, 0, ei]
        d6 = data.dopple[j, 5, ei]
        rho = data.rho[j]
        print(f"{j+1:5d}  {xp1:12.4e}  {xp6:12.4e}  {d1:12.4e}  {d6:12.4e}  {rho:12.4e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Read and print fort.10 data")
    parser.add_argument("fort10", help="Path to fort.10 binary file")
    parser.add_argument("--nelem", type=int, nargs="*", default=[61, 62],
                        help="NELEM values to print (default: 61=TiO, 62=VO)")
    parser.add_argument("--skip-continua", action="store_true",
                        help="Skip reading continuum arrays for speed")
    args = parser.parse_args()

    d = read_fort10(args.fort10, skip_continua=args.skip_continua)
    print(f"NT={d.nt}, TEFF={d.teff:.1f}, GLOG={d.glog:.2f}")
    print(f"N_EDGES={d.n_edges}, NUMNU={d.numnu}")
    print()

    labels = {61: "TiO", 62: "VO", 19: "K I", 45: "CN+/CH13"}
    for ne in args.nelem:
        print_species_summary(d, ne, labels.get(ne, f"NELEM{ne}"))
        print()
