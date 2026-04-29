"""Molecular line-list compiler for the Python SYNTHE pipeline.

Ports the logic of three Kurucz Fortran programs:
  - rmolecasc.for : reads ASCII .dat/.asc molecular band files
  - rschwenk.for  : reads the packed-binary TiO (Schwenke) line list
  - rh2ofast.for  : reads the packed-binary H2O (Partridge-Schwenke) line list

All three compilers convert raw molecular data into the same internal tfort.12
format (NBUFF, CONGF, NELION, ELO, GAMRF, GAMSF, GAMWF), which we return here
as numpy arrays compatible with CompiledLineCatalog in compiler.py.

Public API
----------
compile_molecular_ascii(paths, wlbeg, wlend, resolution) -> dict
compile_tio_schwenke(bin_path, wlbeg, wlend, resolution)  -> dict
compile_h2o_partridge(bin_path, wlbeg, wlend, resolution) -> dict
merge_molecular_into_compiled(compiled, *mol_dicts)        -> CompiledLineCatalog
"""

from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .compiler import CompiledLineCatalog
from .fort19 import Fort19Data

# ---------------------------------------------------------------------------
# Physical constants (match Fortran values exactly)
# ---------------------------------------------------------------------------
# Fortran rmolecasc.for/rgfall.for: FREQ = 2.99792458D17 / WLVAC
# WLVAC is in nm (computed as 1e7/wavenumber_cm^-1, so units are nm).
# Python also stores wavelengths in nm, so the constant is the same: c in nm/s.
_C_LIGHT_NM = 2.99792458e17          # nm/s  (c in nm/s, matches Fortran FREQ convention)
_CGF_CONSTANT = 0.026538 / 1.77245   # same as compiler.py  (Phil Cargile constant / sqrt(pi))
_LOG10 = math.log(10.0)

# ---------------------------------------------------------------------------
# Vacuum-to-air wavelength correction  (VACAIR subroutine in the Fortran)
# Used when IFVAC != 1.  Both rschwenk and rh2ofast apply this.
# ---------------------------------------------------------------------------
def _vacair(wl_vac_nm: float) -> float:
    """Return air wavelength (nm) given vacuum wavelength (nm)."""
    waven = 1.0e7 / wl_vac_nm  # wavenumber cm^-1
    return wl_vac_nm / (
        1.0000834213
        + 2406030.0 / (1.30e10 - waven**2)
        + 15997.0 / (3.89e9 - waven**2)
    )


def _build_airshift_table(n: int = 100_000) -> np.ndarray:
    """Pre-build airshift[IWL] table where IWL = int(wl_nm * 10 + 0.5).
    Index 0..1999 are zero (UV); 2000..n-1 hold (air-vac) shifts in nm."""
    shift = np.zeros(n, dtype=np.float64)
    for i in range(2000, n):
        wv = i * 0.1
        shift[i] = _vacair(wv) - wv
    return shift


# ---------------------------------------------------------------------------
# NELION table for molecules (from rmolecasc.for dispatch table)
# Key: (code_int, iso) -> (NELION, ISO1, ISO2, X1, X2)
# ---------------------------------------------------------------------------
# Format of the Fortran input line (FORMAT 1111):
#   F10.4  WL         vacuum wavelength (Å in gfall convention -> nm if /10)
#   F7.3   GFLOG      log10(gf)
#   F5.1   XJ         lower J
#   F10.3  E          lower energy (cm^-1)
#   F5.1   XJP        upper J
#   F11.3  EP         upper energy (cm^-1)
#   I4     ICODE      integer species code (e.g. 106 -> CH, 607 -> CN, ...)
#   A8     LABEL      lower state label
#   A8     LABELP     upper state label
#   I2     ISO        isotopologue index (1-60 lookup in the Fortran jump table)
#   I4     LOGGR      log10(gamma_rad) * 100  (stored as integer)
#
# NOTE: WL in these files is in Angstroms (the Kurucz convention).  The code
# below converts to nm internally so that the pipeline stays in nm throughout.

# Dispatch: (CODE_int, ISO_int) -> (NELION, ISO1, ISO2, X1, X2)
# This is a direct encoding of the Fortran computed GO TO table.
_MOL_DISPATCH: Dict[Tuple[int, int], Tuple[int, int, int, float, float]] = {
    # H2
    (240, 1):  (240,  1,  1,  0.0,    -5.0),
    # HD
    (240, 2):  (240,  1,  2,  0.0,    -4.469),
    # CH  (CODE 106 -> ISO dispatch at label 1220/1320 for ^12C^1H / ^13C^1H)
    (106, 12): (246,  1, 12,  0.0,    -0.005),
    (106, 13): (246,  1, 13,  0.0,    -1.955),
    # NH  (CODE dispatched through ISO=14 or ISO=15; CODE=607 at ISO=14 is CN)
    (114, 14): (252,  1, 14,  0.0,    -0.002),
    (114, 15): (252,  1, 15,  0.0,    -2.444),
    # OH  (CODE 108 or 816 -> OH; CODE 813 at ISO=16 -> AlO)
    (108, 16): (258,  1, 16,  0.0,    -0.001),
    (108, 18): (258,  1, 18,  0.0,    -2.690),
    # CO  (CODE 608/816 -> CO; various isotopologues)
    (608, 12): (276, 12, 16, -0.005,  -0.001),
    (608, 13): (276, 13, 16, -1.955,  -0.001),
    (608, 16): (276, 12, 16, -0.005,  -0.001),  # fallback
    (608, 17): (276, 12, 17, -0.005,  -3.398),
    (608, 18): (276, 12, 18, -0.005,  -2.690),
    # CN  (CODE 607 -> CN; different isotopologues via ISO and CODE sub-dispatch)
    (607, 12): (270, 12, 14, -0.005,  -0.002),
    (607, 13): (270, 13, 14, -1.955,  -0.002),
    (607, 15): (270, 12, 15, -0.005,  -2.444),
    # C2  (CODE 606 -> C2)
    (606, 12): (264, 12, 12, -0.005,  -0.005),
    (606, 13): (264, 12, 13, -0.005,  -1.955),
    (606, 33): (264, 13, 13, -1.955,  -1.955),
    # MgH
    (112, 24): (300,  1, 24,  0.0,    -0.105),
    (112, 25): (300,  1, 25,  0.0,    -0.996),
    (112, 26): (300,  1, 26,  0.0,    -0.947),
    # SiH
    (114, 28): (312,  1, 28,  0.0,    -0.035),  # CODE 114 overloaded; SiH when ISO=28,29,30
    (114, 29): (312,  1, 29,  0.0,    -1.331),
    (114, 30): (312,  1, 30,  0.0,    -1.516),
    # NaH
    (123, 23): (492,  1, 23,  0.0,     0.0),
    # KH
    (119, 39): (498, 39,  1, -0.030,   0.0),
    (119, 41): (498, 41,  1, -1.172,   0.0),
    # CaH
    (120, 40): (342, 40,  1, -0.013,   0.0),
    (120, 42): (342, 42,  1, -2.189,   0.0),
    (120, 43): (342, 43,  1, -2.870,   0.0),
    (120, 44): (342, 44,  1, -1.681,   0.0),
    (120, 46): (342, 46,  1, -4.398,   0.0),
    (120, 48): (342, 48,  1, -2.728,   0.0),
    # TiO (CODE 822 used by rschwenk; rmolecasc uses CODE=(O_mass)(Ti_mass) patterns)
    (816, 46): (366, 16, 46,  0.0,    -1.101),
    (816, 47): (366, 16, 47,  0.0,    -1.138),
    (816, 48): (366, 16, 48,  0.0,    -0.131),
    (816, 49): (366, 16, 49,  0.0,    -1.259),
    (816, 50): (366, 16, 50,  0.0,    -1.272),
    # VO
    (816, 51): (372, 16, 51,  0.0,    -0.001),
    # CrH
    (124, 50): (432, 50,  1, -1.362,   0.0),
    (124, 52): (432, 52,  1, -0.077,   0.0),
    (124, 53): (432, 53,  1, -1.022,   0.0),
    (124, 54): (432, 54,  1, -1.626,   0.0),
    # FeH — fehfx.dat uses ICODE=156 (rmolecasc.for I4 field = ' 156')
    # Fortran dispatch: ISO drives the label (540/560/570/580 → NELION=444)
    # X1 values from rmolecasc.for labels 540(CODE≠124)/560/570/580.
    (156, 54): (444, 54,  1, -1.237,   0.0),
    (156, 56): (444, 56,  1, -0.038,   0.0),
    (156, 57): (444, 57,  1, -1.658,   0.0),
    (156, 58): (444, 58,  1, -2.553,   0.0),
    # Legacy entries with ICODE=126 kept for any hypothetical files using that code
    (126, 54): (444, 54,  1, -1.237,   0.0),
    (126, 56): (444, 56,  1, -0.038,   0.0),
    (126, 57): (444, 57,  1, -1.658,   0.0),
    (126, 58): (444, 58,  1, -2.553,   0.0),
    # AlO  (CODE 813: rmolecasc.for labels 1600/1700/1820 → NELION=324)
    (813, 16): (324, 27, 16,  0.0,    -0.001),
    (813, 17): (324, 27, 17,  0.0,    -3.398),
    (813, 18): (324, 27, 18,  0.0,    -2.690),
    # CoO  (CODE 827: IDMOL(57)=827 → NELION=576, NOT AlO)
    (827, 16): (576, 59, 16,  0.0,     0.0),
    (827, 17): (576, 59, 17,  0.0,    -3.398),
    (827, 18): (576, 59, 18,  0.0,    -2.690),
    # SiO
    (814, 16): (330, 28, 16, -0.035,  -0.001),
    (814, 17): (330, 29, 16, -1.328,  -0.001),
    (814, 18): (330, 30, 16, -1.510,  -0.001),
    (814, 28): (330, 28, 16, -0.035,  -0.001),
    (814, 29): (330, 29, 16, -1.328,  -0.001),
    (814, 30): (330, 30, 16, -1.510,  -0.001),
    # MgO
    (812, 24): (306, 24, 16,  0.0,    -0.105),
}

# The "principal" ISO for each CODE when the code alone identifies the molecule
# and the Fortran dispatches purely on CODE (not on ISO for the NELION assignment).
# This covers the rmolecasc computed-GO-TO at the TOP of the dispatch (ISO column).
# We build a secondary lookup: CODE_only -> (NELION, ISO1, ISO2, X1, X2)
# used when (CODE, ISO) is not in _MOL_DISPATCH.
_MOL_CODE_ONLY_DISPATCH: Dict[int, Tuple[int, int, int, float, float]] = {
    101: (240,  1,  1,  0.0,    -5.0),    # H2 (IDMOL(1)=101 → NELION=240)
    106: (246,  1, 12,  0.0,    -0.005),  # CH
    107: (252,  1, 14,  0.0,    -0.002),  # NH
    108: (258,  1, 16,  0.0,    -0.001),  # OH
    608: (276, 12, 16, -0.005,  -0.001),  # CO
    607: (270, 12, 14, -0.005,  -0.002),  # CN
    606: (264, 12, 12, -0.005,  -0.005),  # C2
    112: (300,  1, 24,  0.0,    -0.105),  # MgH
    113: (306,  1, 27,  0.0,     0.0),    # AlH (IDMOL(12)=113 → NELION=306)
    114: (312,  1, 28,  0.0,    -0.035),  # SiH
    111: (492,  1, 11,  0.0,     0.0),    # NaH (IDMOL(43)=111 → NELION=492)
    119: (498, 39,  1, -0.030,   0.0),    # KH
    120: (342, 40,  1, -0.013,   0.0),    # CaH
    123: (426,  1, 23,  0.0,     0.0),    # VH  (IDMOL(32)=123 → NELION=426)
    124: (432, 52,  1, -0.077,   0.0),    # CrH
    126: (444, 56,  1, -0.038,   0.0),    # FeH (legacy code)
    156: (444, 56,  1, -0.038,   0.0),    # FeH (actual ICODE=156 in fehfx.dat)
    822: (366, 48, 16,  0.0,    -0.131),  # TiO (IDMOL(22)=822 → NELION=366)
    816: (348, 16, 32,  0.0,     0.0),    # SO  (IDMOL(19)=816 → NELION=348)
    813: (324, 27, 16,  0.0,    -0.001),  # AlO (IDMOL(15)=813 → NELION=324)
    823: (372, 51, 16,  0.0,     0.0),    # VO  (IDMOL(23)=823 → NELION=372)
    827: (576, 59, 16,  0.0,     0.0),    # CoO (IDMOL(57)=827 → NELION=576)
    814: (330, 28, 16, -0.035,  -0.001),  # SiO
    812: (318, 24, 16,  0.0,    -0.105),  # MgO (IDMOL(14)=812 → NELION=318)
}


def _dispatch_molecule(code_int: int, iso: int) -> Optional[Tuple[int, int, int, float, float]]:
    """Return (NELION, ISO1, ISO2, X1, X2) for a molecular line or None to skip."""
    row = _MOL_DISPATCH.get((code_int, iso))
    if row is not None:
        return row
    # Fall back to CODE-only lookup
    return _MOL_CODE_ONLY_DISPATCH.get(code_int)


# ---------------------------------------------------------------------------
# ASCII molecular line compiler  (rmolecasc.for)
# ---------------------------------------------------------------------------
_ASCII_FMT_COLS = [
    # (start, end, type)  -- 0-based column indices, end exclusive
    (0,  10, "f"),   # WL (Angstrom)
    (10, 17, "f"),   # GFLOG
    (17, 22, "f"),   # XJ
    (22, 32, "f"),   # E (lower energy, cm^-1)
    (32, 37, "f"),   # XJP
    (37, 48, "f"),   # EP (upper energy, cm^-1)
    (48, 52, "i"),   # ICODE (integer)
    (52, 60, "s"),   # LABEL  (A8)
    (60, 68, "s"),   # LABELP (A8)
    (68, 70, "i"),   # ISO (I2)
    (70, 74, "i"),   # LOGGR (I4)
]


def _parse_field(line: str, start: int, end: int, typ: str):
    """Parse one Fortran-style fixed-width field from a text line."""
    s = line[start:end] if len(line) >= end else line[start:]
    s = s.strip()
    if not s:
        return 0 if typ == "i" else (0.0 if typ == "f" else "")
    if typ == "f":
        try:
            return float(s)
        except ValueError:
            return 0.0
    if typ == "i":
        try:
            return int(s)
        except ValueError:
            return 0
    return s  # "s" -> string


def _parse_ascii_line(line: str):
    """Parse one fixed-format Kurucz molecular ASCII record.

    Returns (wl_nm, gflog, xj, e, xjp, ep, icode, label, labelp, iso, loggr)
    with wavelength already converted from Angstrom to nm.
    """
    wl_aa  = _parse_field(line, 0,  10, "f")
    gflog  = _parse_field(line, 10, 17, "f")
    xj     = _parse_field(line, 17, 22, "f")
    e      = _parse_field(line, 22, 32, "f")
    xjp    = _parse_field(line, 32, 37, "f")
    ep     = _parse_field(line, 37, 48, "f")
    icode  = _parse_field(line, 48, 52, "i")
    label  = _parse_field(line, 52, 60, "s")
    labelp = _parse_field(line, 60, 68, "s")
    iso    = _parse_field(line, 68, 70, "i")
    loggr  = _parse_field(line, 70, 74, "i")
    return wl_aa, gflog, xj, e, xjp, ep, icode, label, labelp, iso, loggr


def _compute_wlvac_nm(wl_aa: float, e: float, ep: float, ifvac: int) -> float:
    """Compute vacuum wavelength in nm.

    Fortran rmolecasc.for lines 111-112:
      WLVAC = ABS(WL)
      IF(IFVAC.EQ.1) WLVAC = 1.E7 / ABS(ABS(EP) - ABS(E))

    Key: in the Kurucz molecular ASCII files WL is stored in **nm** (not Angstroms).
    rmolecasc.for uses START/STOP in nm and WLVAC directly (no /10 conversion).
    When IFVAC=1 (standard for these files), the wavelength is recomputed from the
    energy-level difference: 1e7 / ΔE_cm^-1 → nm.  No extra division by 10.
    """
    if ifvac == 1:
        delta = abs(abs(ep) - abs(e))
        if delta > 0.0:
            return 1.0e7 / delta  # cm^-1 -> nm  (Fortran: 1.E7/ABS(ABS(EP)-ABS(E)))
    return abs(wl_aa)  # Already in nm — no /10 conversion


def compile_molecular_ascii(
    paths: Sequence[Path],
    wlbeg: float,
    wlend: float,
    resolution: float,
    ifvac: int = 0,
    ifpred: int = 0,
) -> Dict[str, np.ndarray]:
    """Compile one or more Kurucz ASCII molecular data files.

    Parameters
    ----------
    paths:      list of .dat/.asc files
    wlbeg/end:  wavelength window in nm (vacuum)
    resolution: resolving power R = lambda/dlambda
    ifvac:      1 -> wavelengths stored as vacuum; 0 -> air (rmolecasc default = 0)
    ifpred:     1 -> include predicted lines (negative energies)

    Returns
    -------
    dict with keys:
      nbuff, cgf, nelion, elo_cm, gamma_rad, gamma_stark, gamma_vdw, limb
    (all 1-D numpy arrays, same semantics as CompiledLineCatalog fields)
    """
    ratio = 1.0 + 1.0 / resolution
    ratiolg = math.log(ratio)
    ixwlbeg = math.floor(math.log(wlbeg) / ratiolg)
    if math.exp(ixwlbeg * ratiolg) < wlbeg:
        ixwlbeg += 1

    stop = wlend + 0.1  # Fortran: STOP = WLEND + 1.  (in Å; here nm -> use 0.1)
    start = wlbeg - 0.01

    nbuffs:      List[int]   = []
    cgfs:        List[float] = []
    nelions:     List[int]   = []
    elos:        List[float] = []
    gamma_rads:  List[float] = []
    gamma_starks:List[float] = []
    gamma_vdws:  List[float] = []
    limbs:       List[int]   = []

    for path in paths:
        try:
            fh = open(path, "r", encoding="ascii", errors="replace")
        except OSError:
            continue
        with fh:
            for raw_line in fh:
                if len(raw_line) < 52:
                    continue
                (wl_aa, gflog, xj, e, xjp, ep,
                 icode, label, labelp, iso, loggr) = _parse_ascii_line(raw_line)

                if abs(wl_aa) == 0.0:
                    continue

                # Skip predicted lines if not requested
                if ifpred == 0 and (e < 0.0 or ep < 0.0):
                    continue

                wlvac = _compute_wlvac_nm(wl_aa, e, ep, ifvac)
                if wlvac < start:
                    continue
                if wlvac > stop:
                    # Files are sorted by wavelength; once past window we can stop
                    break

                # When ifvac=1, some lines get wavelengths far from their stored
                # file position (e.g. C2 A-X high-v lines stored at 1824-1910 nm
                # in the file map to ~1799 nm via energy levels).  tfort.12 was
                # compiled with the file wavelength as the window guard, so any
                # line whose file position is outside [wlbeg-10, wlend+10] nm
                # should be excluded to match tfort.12 behaviour.
                if ifvac == 1:
                    wl_file = abs(wl_aa)
                    if wl_file > stop + 10.0 or (wl_file > 0.0 and wl_file < start - 10.0):
                        continue

                mol = _dispatch_molecule(icode, iso)
                if mol is None:
                    continue

                nelion, iso1, iso2, x1, x2 = mol
                fudge = 0.0
                gf = math.exp((gflog + x1 + x2 + fudge) * _LOG10)
                elo = min(abs(e), abs(ep))

                ixwl = math.log(max(wlvac, 1e-30)) / ratiolg + 0.5
                nbuff_val = int(ixwl) - ixwlbeg + 1

                # Fortran rmolecasc.for: FREQ = 2.99792458D17 / WLVAC (WLVAC in nm)
                freq = _C_LIGHT_NM / max(wlvac, 1e-30)
                congf = _CGF_CONSTANT * gf / freq

                frq4pi = freq * 12.5664
                gammar = 10.0 ** (loggr * 0.01)
                # Fortran default guesses for molecular damping
                gammas = 3.0e-5
                gammaw = 1.0e-7
                # Vibrational-rotational lines (upper state starts with 'X')
                if labelp.strip().startswith("X"):
                    gammas = 3.0e-8
                    gammaw = 1.0e-8

                gamrf = gammar / frq4pi
                gamsf = gammas / frq4pi
                gamwf = gammaw / frq4pi

                nbuffs.append(nbuff_val)
                cgfs.append(congf)
                nelions.append(nelion)
                elos.append(elo)
                gamma_rads.append(gamrf)
                gamma_starks.append(gamsf)
                gamma_vdws.append(gamwf)
                limbs.append(7)  # molecular lines: LIM=7 (large line_size -> min(8-0,7)=7)

    return _arrays(nbuffs, cgfs, nelions, elos, gamma_rads, gamma_starks, gamma_vdws, limbs)


# ---------------------------------------------------------------------------
# TiO (Schwenke) binary compiler  (rschwenk.for)
# ---------------------------------------------------------------------------
# Binary fort.11 records: each is 4 INT16 values: IWL, IELION, IELO, IGFLOG
# WL packed as: WLVAC = exp(IWL * RATIOLOG) where RATIOLOG = log(1 + 1/2000000)
# Isotopologue: ISO = abs(IELION) - 8949  (gives 1..5 for 46..50TiO)
_TIO_RATIOLOG = math.log(1.0 + 1.0 / 2_000_000.0)
_TIO_XISO   = [0.0793, 0.0728, 0.7394, 0.0551, 0.0534]  # isotope abundances
_TIO_X2ISO  = [-1.101, -1.138, -0.131, -1.259, -1.272]  # log10 iso abundance correction
_TIO_ISO2   = [46, 47, 48, 49, 50]
_TIO_NELION = 366
_TABLOG_SIZE = 32768
_TABLOG_OFFSET = 16384


def _build_tablog() -> np.ndarray:
    """Build the 32768-entry log-to-linear lookup table used by Schwenke/H2O readers."""
    i = np.arange(_TABLOG_SIZE, dtype=np.float64)
    return (10.0 ** ((i - _TABLOG_OFFSET) * 0.001)).astype(np.float32)


_TABLOG: Optional[np.ndarray] = None


def _get_tablog() -> np.ndarray:
    global _TABLOG
    if _TABLOG is None:
        _TABLOG = _build_tablog()
    return _TABLOG



def compile_tio_schwenke(
    bin_path: Path,
    wlbeg: float,
    wlend: float,
    resolution: float,
    ifvac: int = 0,
) -> Dict[str, np.ndarray]:
    """Compile the Schwenke TiO packed-binary line list.

    Fortran rschwenk.for record layout — RECORDSIZE=4 (4-byte units) = 16 bytes/record,
    little-endian (VAX/x86 native byte order):
      IWL     int32  packed wavelength: WLVAC = exp(IWL * RATIOLOG)
      IELION  int16  encodes isotopologue: ISO = |IELION| - 8949, range 1..5 (46-50TiO)
      IELO    int16  packed lower energy via TABLOG
      IGFLOG  int16  packed log(gf) via TABLOG
      IGR     int16  packed gamma_rad via TABLOG (rschwenk line 182: GAMRF=TABLOG(IGR)/FRQ4PI)
      IGS     int16  overridden to 1 in Fortran; actual value is unused
      IGW     int16  overridden to 9384 in Fortran; actual value is unused

    Fortran COMMON /IIIIIII/IWL,IELION,IELO,IGFLOG,IGR,IGS,IGW with INTEGER*4 IIIIIII(4)
    maps to exactly 16 bytes = one direct-access record (RECORDSIZE=4 in 4-byte units).
    """
    tablog = _get_tablog()
    ratio = 1.0 + 1.0 / resolution
    ratiolg = math.log(ratio)
    ixwlbeg = math.floor(math.log(wlbeg) / ratiolg)
    if math.exp(ixwlbeg * ratiolg) < wlbeg:
        ixwlbeg += 1

    # IWL(i4) IELION(i2) IELO(i2) IGFLOG(i2) IGR(i2) IGS(i2) IGW(i2) = 16 bytes, little-endian
    _REC_DTYPE = np.dtype([
        ("iwl",    "<i4"),
        ("ielion", "<i2"),
        ("ielo",   "<i2"),
        ("igflog", "<i2"),
        ("igr",    "<i2"),
        ("igs",    "<i2"),
        ("igw",    "<i2"),
    ])
    try:
        arr = np.memmap(bin_path, mode="r", dtype=_REC_DTYPE)
    except OSError:
        return _arrays([], [], [], [], [], [], [], [])

    if arr.size == 0:
        return _arrays([], [], [], [], [], [], [], [])

    # Vectorised wavelength (rschwenk.for lines 158-161)
    wlvac = np.exp(arr["iwl"].astype(np.float64) * _TIO_RATIOLOG)
    if ifvac != 1:
        airshift = _build_airshift_table(60_000)
        kwl = np.clip((wlvac * 10.0 + 0.5).astype(np.int64), 0, len(airshift) - 1)
        wl_use = wlvac + airshift[kwl]          # air wavelength (rschwenk line 160-161)
    else:
        wl_use = wlvac

    # ISO = ABS(IELION) - 8949, valid range 1..5 (rschwenk line 151)
    iso_arr = np.abs(arr["ielion"].astype(np.int32)) - 8949
    mask = (
        (wl_use >= wlbeg - 1.0)
        & (wl_use <= wlend + 1.0)
        & (iso_arr >= 1)
        & (iso_arr <= 5)
    )
    n_sel = int(mask.sum())
    if n_sel == 0:
        return _arrays([], [], [], [], [], [], [], [])

    wl_s     = wl_use[mask]
    iso_s    = iso_arr[mask]
    ielo_s   = np.clip(arr["ielo"][mask].astype(np.int32),   0, _TABLOG_SIZE - 1)
    igflog_s = np.clip(arr["igflog"][mask].astype(np.int32), 0, _TABLOG_SIZE - 1)
    igr_s    = np.clip(arr["igr"][mask].astype(np.int32),    0, _TABLOG_SIZE - 1)

    # rschwenk.for line 165: FREQ = 2.99792458D17 / WLVAC  (post-air when IFVAC!=1)
    freq   = _C_LIGHT_NM / wl_s
    frq4pi = freq * 12.5664

    # rschwenk.for line 166: CONGF = 0.01502 * TABLOG(IGFLOG) / FREQ * XISO(ISO)
    xiso_s = np.array(_TIO_XISO, dtype=np.float64)[iso_s - 1]
    cgf    = 0.01502 * tablog[igflog_s] / freq * xiso_s

    # rschwenk.for line 162: ELO = TABLOG(IELO)
    elo = tablog[ielo_s].astype(np.float64)

    # rschwenk.for lines 180-184: IGS=1, IGW=9384 (Fortran overrides before GAMSF/GAMWF)
    # GAMRF = TABLOG(IGR) / FRQ4PI  (uses actual IGR from the binary record)
    gamrf = tablog[igr_s] / frq4pi
    gamsf = float(tablog[1])    / frq4pi
    gamwf = float(tablog[9384]) / frq4pi

    # rschwenk.for lines 163-164: NBUFF = IXWL - IXWLBEG + 1
    ixwl  = np.floor(np.log(np.maximum(wl_s, 1e-30)) / ratiolg + 0.5).astype(np.int32)
    nbuff = ixwl - ixwlbeg + 1

    nelion = np.full(n_sel, _TIO_NELION, dtype=np.int32)
    limb   = np.full(n_sel, 7, dtype=np.int16)

    return _arrays(nbuff, cgf, nelion, elo, gamrf, gamsf, gamwf, limb)


# ---------------------------------------------------------------------------
# H2O (Partridge-Schwenke) binary compiler  (rh2ofast.for)
# ---------------------------------------------------------------------------
# Binary fort.11 (h2ofastfix.bin) records: INT32 IWL + INT16 IELO + INT16 IGFLOG
# = 8 bytes / record.  WL packed with same RATIOLOG as TiO.
# Sign encoding for isotopologue (rh2ofast.for lines 144-149):
#   if IELO > 0 and IGFLOG > 0: ISO = 1  (1H1H16O)
#   elif IELO > 0:               ISO = 2  (1H1H17O)
#   elif IGFLOG > 0:             ISO = 3  (1H1H18O)
#   else:                        ISO = 4  (1H2H16O)
_H2O_RATIOLOG = math.log(1.0 + 1.0 / 2_000_000.0)
_H2O_XISO  = [0.9976, 0.0004, 0.0020, 0.00001]
_H2O_X2ISO = [-0.001, -3.398, -2.690, -5.000]
_H2O_NELION = 534


def compile_h2o_partridge(
    bin_path: Path,
    wlbeg: float,
    wlend: float,
    resolution: float,
    ifvac: int = 0,
) -> Dict[str, np.ndarray]:
    """Compile the Partridge-Schwenke H2O packed-binary line list.

    Fortran rh2ofast.for record layout — RECORDSIZE=2 (2-byte units) = 8 bytes/record,
    little-endian (VAX/x86 native byte order):
      IWL     int32  packed wavelength: WLVAC = exp(IWL * RATIOLOG)
      IELO    int16  sign encodes isotopologue; abs value = ELO in cm^-1 (direct integer)
      IGFLOG  int16  sign encodes isotopologue; abs value = TABLOG index for gf

    Key Fortran convention (rh2ofast.for lines 135-154):
      FREQ is computed BEFORE the air correction is applied to WLVAC (line 136).
      WLVAC used in NBUFF and GAMMAR is the post-air wavelength (lines 142, 165).
    """
    tablog = _get_tablog()
    ratio = 1.0 + 1.0 / resolution
    ratiolg = math.log(ratio)
    ixwlbeg = math.floor(math.log(wlbeg) / ratiolg)
    if math.exp(ixwlbeg * ratiolg) < wlbeg:
        ixwlbeg += 1

    # 8 bytes per record: INT32 IWL + INT16 IELO + INT16 IGFLOG, little-endian
    _REC_DTYPE = np.dtype([("iwl", "<i4"), ("ielo", "<i2"), ("igflog", "<i2")])
    try:
        arr = np.memmap(bin_path, mode="r", dtype=_REC_DTYPE)
    except OSError:
        return _arrays([], [], [], [], [], [], [], [])

    if arr.size == 0:
        return _arrays([], [], [], [], [], [], [], [])

    # Vectorised wavelength (rh2ofast.for lines 135-140)
    # FREQ uses the original vacuum WLVAC (line 136), computed before air correction
    wlvac_vac = np.exp(arr["iwl"].astype(np.float64) * _H2O_RATIOLOG)
    freq_vac  = _C_LIGHT_NM / wlvac_vac     # pre-air FREQ (rh2ofast convention)

    if ifvac != 1:
        airshift = _build_airshift_table(100_000)
        kwl    = np.clip((wlvac_vac * 10.0 + 0.5).astype(np.int64), 0, len(airshift) - 1)
        wl_use = wlvac_vac + airshift[kwl]  # air wavelength for window / NBUFF / GAMMAR
    else:
        wl_use = wlvac_vac

    mask  = (wl_use >= wlbeg - 1.0) & (wl_use <= wlend + 1.0)
    n_sel = int(mask.sum())
    if n_sel == 0:
        return _arrays([], [], [], [], [], [], [], [])

    ielo_raw   = arr["ielo"][mask].astype(np.int32)
    igflog_raw = arr["igflog"][mask].astype(np.int32)
    wl_s       = wl_use[mask]
    freq_s     = freq_vac[mask]          # pre-air FREQ used for CONGF (rh2ofast line 154)

    # Isotopologue from sign encoding (rh2ofast.for lines 143-149)
    # ISO=1: IELO>0 and IGFLOG>0; ISO=2: IELO>0 (IGFLOG<=0)
    # ISO=3: IGFLOG>0 (IELO<=0);  ISO=4: both <=0
    iso_idx = np.where(
        (ielo_raw > 0) & (igflog_raw > 0), 0,
        np.where(ielo_raw > 0, 1,
        np.where(igflog_raw > 0, 2, 3))
    )
    ielo_abs   = np.abs(ielo_raw)
    igflog_abs = np.clip(np.abs(igflog_raw), 0, _TABLOG_SIZE - 1)

    xiso_s = np.array(_H2O_XISO, dtype=np.float64)[iso_idx]
    frq4pi = freq_s * 12.5664

    # rh2ofast.for line 154: CONGF = 0.01502 * TABLOG(IGFLOG) / FREQ * XISO(ISO)
    cgf = 0.01502 * tablog[igflog_abs] / freq_s * xiso_s

    # rh2ofast.for line 151: ELO = ABS(IELO)  (direct cm^-1 value, not a TABLOG index)
    elo = ielo_abs.astype(np.float64)

    # rh2ofast.for lines 165-169: GAMMAR = 2.223e13 / WLVAC^2 * 0.001 (post-air WLVAC)
    # IGS=1, IGW=9384 (Fortran fixed overrides)
    gammar = 2.223e13 / np.maximum(wl_s, 1e-6) ** 2 * 0.001
    gamrf  = gammar / frq4pi
    gamsf  = float(tablog[1])    / frq4pi
    gamwf  = float(tablog[9384]) / frq4pi

    # NBUFF = INT(LOG(WL) / RATIOLG + 0.5) - IXWLBEG + 1  (post-air WL)
    ixwl  = np.floor(np.log(np.maximum(wl_s, 1e-30)) / ratiolg + 0.5).astype(np.int32)
    nbuff = ixwl - ixwlbeg + 1

    nelion = np.full(n_sel, _H2O_NELION, dtype=np.int32)
    limb   = np.full(n_sel, 7, dtype=np.int16)

    return _arrays(nbuff, cgf, nelion, elo, gamrf, gamsf, gamwf, limb)


# ---------------------------------------------------------------------------
# Merge helper
# ---------------------------------------------------------------------------
def merge_molecular_into_compiled(
    compiled: CompiledLineCatalog,
    *mol_dicts: Dict[str, np.ndarray],
) -> CompiledLineCatalog:
    """Append molecular line arrays into an existing CompiledLineCatalog.

    The catalog (fort.14-style) metadata is left unchanged; only the
    fort.12-style arrays (nbuff/cgf/nelion/elo_cm/gammas/limb) are extended.
    """
    all_nbuffs       = [compiled.nbuff]
    all_cgfs         = [compiled.cgf]
    all_nelions      = [compiled.nelion]
    all_elo_cms      = [compiled.elo_cm]
    all_gamma_rads   = [compiled.gamma_rad]
    all_gamma_starks = [compiled.gamma_stark]
    all_gamma_vdws   = [compiled.gamma_vdw]
    all_limbs        = [compiled.limb]

    for d in mol_dicts:
        if len(d.get("nbuff", [])) == 0:
            continue
        all_nbuffs.append(d["nbuff"])
        all_cgfs.append(d["cgf"])
        all_nelions.append(d["nelion"])
        all_elo_cms.append(d["elo_cm"])
        all_gamma_rads.append(d["gamma_rad"])
        all_gamma_starks.append(d["gamma_stark"])
        all_gamma_vdws.append(d["gamma_vdw"])
        all_limbs.append(d["limb"])

    return CompiledLineCatalog(
        catalog=compiled.catalog,
        fort19_data=compiled.fort19_data,
        nbuff=np.concatenate(all_nbuffs).astype(np.int32),
        cgf=np.concatenate(all_cgfs).astype(np.float32),
        nelion=np.concatenate(all_nelions).astype(np.int16),
        elo_cm=np.concatenate(all_elo_cms).astype(np.float32),
        gamma_rad=np.concatenate(all_gamma_rads).astype(np.float32),
        gamma_stark=np.concatenate(all_gamma_starks).astype(np.float32),
        gamma_vdw=np.concatenate(all_gamma_vdws).astype(np.float32),
        limb=np.concatenate(all_limbs).astype(np.int16),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _arrays(
    nbuffs, cgfs, nelions, elos, gamma_rads, gamma_starks, gamma_vdws, limbs
) -> Dict[str, np.ndarray]:
    return {
        "nbuff":       np.asarray(nbuffs,       dtype=np.int32),
        "cgf":         np.asarray(cgfs,         dtype=np.float32),
        "nelion":      np.asarray(nelions,       dtype=np.int16),
        "elo_cm":      np.asarray(elos,         dtype=np.float32),
        "gamma_rad":   np.asarray(gamma_rads,   dtype=np.float32),
        "gamma_stark": np.asarray(gamma_starks, dtype=np.float32),
        "gamma_vdw":   np.asarray(gamma_vdws,   dtype=np.float32),
        "limb":        np.asarray(limbs,         dtype=np.int16),
    }


__all__ = [
    "compile_molecular_ascii",
    "compile_tio_schwenke",
    "compile_h2o_partridge",
    "merge_molecular_into_compiled",
]
