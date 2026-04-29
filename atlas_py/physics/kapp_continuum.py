"""
Full KAPP implementation: Compute ACONT and SIGMAC from atlas_tables.

Continuum-opacity kernel for ``atlas_py`` (copied from ``synthe_py.physics.kapp``;
local name ``kapp_continuum`` to avoid colliding with ``atlas_py.physics.kapp``).

This module implements the Fortran KAPP subroutine (atlas12.for)
which computes continuum absorption (ACONT) and scattering (SIGMAC) from
precomputed populations and cross-section tables.

The KAPP subroutine:
1. Calls subroutines for each species (HOP, HE1OP, HE2OP, C1OP, etc.)
2. Sums contributions: ACONT = AHYD + AHMIN + AHE1 + AHE2 + AC1 + ...
3. Computes scattering: SIGMAC = SIGH + SIGHE + SIGEL + SIGH2 + SIGX
4. Computes source function: SCONT = weighted average of source terms
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Tuple, Optional

import numpy as np

try:
    import numba
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

from .karsas_tables import xkarsas, xkarsas_grid
from .hydrogen_wings import compute_hydrogen_continuum
from .hydrogen_profile import compute_xnh2 as _compute_xnh2_equilh2
from .tcorr import rosstab_eval as _rosstab_eval, TcorrState as _TcorrState
from .trace_runtime import trace_emit, trace_enabled, trace_in_focus

logger = logging.getLogger(__name__)

from .kapp_continuum_data import load_kapp_continuum_tables as _load_kc_tables

_KC_TABLES = _load_kc_tables()

# COULFF table for Coulomb free-free Gaunt factors (atlas12.for)
COULFF_Z4LOG = _KC_TABLES["COULFF_Z4LOG"]
# HMINOP tables (atlas12.for HMINOP subroutine)
HMINOP_WBF = _KC_TABLES["HMINOP_WBF"]
HMINOP_BF = _KC_TABLES["HMINOP_BF"]
HMINOP_WAVEK = _KC_TABLES["HMINOP_WAVEK"]
HMINOP_THETAFF = _KC_TABLES["HMINOP_THETAFF"]
HMINOP_FFBEG = _KC_TABLES["HMINOP_FFBEG"]
HMINOP_FFEND = _KC_TABLES["HMINOP_FFEND"]
# HRAYOP Gavrila tables (atlas12.for HRAYOP subroutine)
HRAYOP_GAVRILAM = _KC_TABLES["HRAYOP_GAVRILAM"]
HRAYOP_GAVRILAMAB = _KC_TABLES["HRAYOP_GAVRILAMAB"]
HRAYOP_GAVRILAMBC = _KC_TABLES["HRAYOP_GAVRILAMBC"]
HRAYOP_GAVRILAMCD = _KC_TABLES["HRAYOP_GAVRILAMCD"]
HRAYOP_GAVRILALYMANCONT = _KC_TABLES["HRAYOP_GAVRILALYMANCONT"]
HRAYOP_FGAVRILALYMANCONT = _KC_TABLES["HRAYOP_FGAVRILALYMANCONT"]
# Coulomb free-free table
COULFF_A_TABLE = _KC_TABLES["COULFF_A_TABLE"]
# HOTOP transition table
HOTOP_TRANSITIONS = _KC_TABLES["HOTOP_TRANSITIONS"]
# SI2OP Peach tables
_SI2OP_PEACH = _KC_TABLES["_SI2OP_PEACH"]
_SI2OP_FREQSI = _KC_TABLES["_SI2OP_FREQSI"]
_SI2OP_FLOG = _KC_TABLES["_SI2OP_FLOG"]
_SI2OP_TLG = _KC_TABLES["_SI2OP_TLG"]
# CH/OH partition and cross-section tables
_CH_PARTITION = _KC_TABLES["_CH_PARTITION"]
_OH_PARTITION = _KC_TABLES["_OH_PARTITION"]
_CH_CROSSSECT = _KC_TABLES["_CH_CROSSSECT"]
_OH_CROSSSECT = _KC_TABLES["_OH_CROSSSECT"]
# H2 collision-induced absorption tables
_H2_COLL_H2H2 = _KC_TABLES["_H2_COLL_H2H2"]
_H2_COLL_H2HE = _KC_TABLES["_H2_COLL_H2HE"]
# Hydrogen energy levels and statistical weights
H_ENERGY_CM = _KC_TABLES["H_ENERGY_CM"]
H_ENERGY_EV = H_ENERGY_CM / 8065.479
H_STAT_WEIGHT = _KC_TABLES["H_STAT_WEIGHT"]
H_MAX_LEVEL = 6


if TYPE_CHECKING:
    # Duck-typed: ``synthe_py`` AtmosphereModel or ``atlas_py`` KappAtmosphereAdapter.
    AtmosphereModel = Any

# Constants matching Fortran exactly
C_LIGHT_CM = 2.99792458e10  # cm/s
C_LIGHT_NM = 2.99792458e17  # nm/s
H_PLANCK = 6.62607015e-27  # erg * s
K_BOLTZ = 1.380649e-16  # erg / K
# CRITICAL: Match Fortran's TKEV calculation exactly (atlas7v.for line 1954: TKEV(J)=8.6171D-5*T(J))
KBOLTZ_EV = 8.6171e-5  # eV/K (matches Fortran: 8.6171D-5)
RYDBERG_CM = 109677.576  # cm^-1
LN10 = np.log(10.0)
_FORTRAN_LN10 = 2.30258509299405
_SI2OP12_XTAB: np.ndarray | None = None

# COULFF table for Coulomb free-free Gaunt factors (atlas7v.for line 4597-4612)
# HMINOP tables (atlas7v.for line 5228-5278)
# FFBEG: (11, 11) array - first 11 columns of FF
# FFEND: (11, 11) array - last 11 columns of FF
# HRAYOP: Gavrila tables for hydrogen Rayleigh scattering (atlas7v.for line 5351-5420)
# HOTOP transition table (atlas7v.for HOTOP DATA A1..A7, 60 entries × 7 fields):
# (freq0, xsect, alpha, power, multiplier, excitation_eV, xNfpId)

def _coulff(
    j: int,
    nz: int,
    freq: float,
    freqlg: float,
    temperature: np.ndarray,
    tlog: np.ndarray,
) -> float:
    """Compute Coulomb free-free Gaunt factor (atlas7v.for line 5057-5187).

    Parameters
    ----------
    j:
        Layer index (0-based)
    nz:
        Ion charge (1 for neutral, 2 for singly ionized, etc.)
    freq:
        Frequency in Hz
    freqlg:
        log10(frequency)
    temperature:
        Temperature array
    tlog:
        log10(temperature) array

    Returns
    -------
    coulff:
        Coulomb free-free Gaunt factor
    """
    if nz < 1 or nz > 6:
        return 1.0  # Default for unsupported charge states

    z4log = COULFF_Z4LOG[nz - 1]
    temp = temperature[j]
    tlog_j = tlog[j]

    # GAMLOG = log10(158000*Z*Z/T) * 2
    # GAMLOG = 10.39638 - TLOG(J)/1.15129 + Z4LOG(NZ)
    gamlog = 10.39638 - tlog_j / 1.15129 + z4log
    igam = max(1, min(int(gamlog + 7), 10))

    # HVKTLG = log10(h*nu/(k*T)) * 2
    # HVKTLG = (FREQLG - TLOG(J))/1.15129 - 20.63764
    hvktlg = (freqlg - tlog_j) / 1.15129 - 20.63764
    ihvkt = max(1, min(int(hvktlg + 9), 11))

    p = gamlog - float(igam - 7)
    q = hvktlg - float(ihvkt - 9)

    # Bilinear interpolation.
    # COULFF_A_TABLE shape is (12, 11): rows=IHVKT (0..11), cols=IGAM (0..10).
    # Fortran A(IGAM, IHVKT) = COULFF_A_TABLE[IHVKT-1, IGAM-1].
    # igam ∈ [1,10] → col index igam-1 ∈ [0,9]; igam (col+1) ∈ [1,10] < 11 ✓
    # ihvkt ∈ [1,11] → row index ihvkt-1 ∈ [0,10]; ihvkt (row+1) ∈ [1,11] < 12 ✓
    a_00 = COULFF_A_TABLE[ihvkt - 1, igam - 1]  # A(IGAM,   IHVKT)
    a_01 = COULFF_A_TABLE[ihvkt,     igam - 1]  # A(IGAM,   IHVKT+1)
    a_10 = COULFF_A_TABLE[ihvkt - 1, igam    ]  # A(IGAM+1, IHVKT)
    a_11 = COULFF_A_TABLE[ihvkt,     igam    ]  # A(IGAM+1, IHVKT+1)

    coulff = (1.0 - p) * ((1.0 - q) * a_00 + q * a_01) + p * (
        (1.0 - q) * a_10 + q * a_11
    )

    return coulff


def _coulff_grid(nz: int, freqlg: np.ndarray, tlog: np.ndarray) -> np.ndarray:
    """Vectorized COULFF over (layer, frequency_chunk) for HOTOP."""
    if nz < 1 or nz > 6:
        return np.ones((tlog.size, freqlg.size), dtype=np.float64)

    z4log = COULFF_Z4LOG[nz - 1]
    tlog_col = tlog[:, np.newaxis]
    freqlg_row = freqlg[np.newaxis, :]

    gamlog = 10.39638 - tlog_col / 1.15129 + z4log
    hvktlg = (freqlg_row - tlog_col) / 1.15129 - 20.63764

    igam = np.clip((gamlog + 7.0).astype(np.int64), 1, 10)
    ihvkt = np.clip((hvktlg + 9.0).astype(np.int64), 1, 11)

    p = gamlog - (igam - 7.0)
    q = hvktlg - (ihvkt - 9.0)

    ig = igam - 1   # Fortran IGAM-1: column index for COULFF_A_TABLE (0..9)
    ih = ihvkt - 1  # Fortran IHVKT-1: row index for COULFF_A_TABLE (0..10)
    # COULFF_A_TABLE shape (12, 11): rows=IHVKT (0..11), cols=IGAM (0..10).
    # Fortran A(IGAM, IHVKT) = COULFF_A_TABLE[IHVKT-1, IGAM-1] = COULFF_A_TABLE[ih, ig].
    # igam ∈ [1,10] → ig ∈ [0,9]; ig+1 ∈ [1,10] < 11 ✓
    # ihvkt ∈ [1,11] → ih ∈ [0,10]; ih+1 ∈ [1,11] < 12 ✓
    a00 = COULFF_A_TABLE[ih,     ig    ]  # A(IGAM,   IHVKT)
    a01 = COULFF_A_TABLE[ih + 1, ig    ]  # A(IGAM,   IHVKT+1)
    a10 = COULFF_A_TABLE[ih,     ig + 1]  # A(IGAM+1, IHVKT)
    a11 = COULFF_A_TABLE[ih + 1, ig + 1]  # A(IGAM+1, IHVKT+1)

    return (1.0 - p) * ((1.0 - q) * a00 + q * a01) + p * (
        (1.0 - q) * a10 + q * a11
    )


def _linter(xold: np.ndarray, yold: np.ndarray, xnew: np.ndarray) -> np.ndarray:
    """Linear interpolation/extrapolation (atlas7v.for line 6771-6784).

    CRITICAL: Fortran LINTER extrapolates beyond table boundaries using the
    nearest two points. It does NOT clamp to edge values!

    This is important for H- free-free opacity at wavelengths > 500nm (beyond
    the FF table), where extrapolation gives physically correct decreasing
    cross-sections.

    Parameters
    ----------
    xold:
        Sorted array of x values (increasing)
    yold:
        Corresponding y values
    xnew:
        New x values to interpolate at

    Returns
    -------
    ynew:
        Interpolated/extrapolated y values
    """
    nold = xold.size
    nnew = xnew.size
    ynew = np.zeros(nnew, dtype=np.float64)

    # Fortran LINTER algorithm (atlas7v.for lines 6771-6784):
    # IOLD=2
    # DO 2 INEW=1,NNEW
    # 1 IF(XNEW(INEW).LT.XOLD(IOLD))GO TO 2
    #   IF(IOLD.EQ.NOLD)GO TO 2
    #   IOLD=IOLD+1
    #   GO TO 1
    # 2 YNEW(INEW)=YOLD(IOLD-1)+(YOLD(IOLD)-YOLD(IOLD-1))/
    #      (XOLD(IOLD)-XOLD(IOLD-1))*(XNEW(INEW)-XOLD(IOLD-1))
    #
    # This always uses linear interpolation/extrapolation - NO clamping!

    iold = 1  # Start at index 1 (second element, 0-based = Fortran's IOLD=2)
    for inew in range(nnew):
        # Find position in xold (move iold forward until xnew[inew] < xold[iold])
        while iold < nold - 1 and xnew[inew] >= xold[iold]:
            iold += 1

        # Linear interpolation/extrapolation using points [iold-1, iold]
        # When xnew < xold[0]: iold stays at 1, uses [0,1] to extrapolate LEFT
        # When xnew > xold[-1]: iold at nold-1, uses [nold-2, nold-1] to extrapolate RIGHT
        # When within table: normal linear interpolation
        denom = xold[iold] - xold[iold - 1]
        if abs(denom) < 1e-40:
            ynew[inew] = yold[iold - 1]
        else:
            weight = (xnew[inew] - xold[iold - 1]) / denom
            ynew[inew] = yold[iold - 1] + (yold[iold] - yold[iold - 1]) * weight

    return ynew


if _NUMBA_AVAILABLE:
    _linter = numba.njit(cache=True)(_linter)


def _map1_simple(xold: np.ndarray, fold: np.ndarray, xnew: float) -> float:
    """MAP1 for single value interpolation (used by HMINOP and HRAYOP).

    Uses the full MAP1 implementation to ensure exact matching with Fortran.
    This is a wrapper around the full _map1 function from josh_solver.
    """
    from .josh_math import _map1

    # Convert scalar to array for full MAP1 implementation
    xnew_arr = np.array([xnew], dtype=np.float64)
    fnew_arr, _ = _map1(xold, fold, xnew_arr)
    return float(fnew_arr[0])


def _planck_nu(freq: float, temperature: np.ndarray) -> np.ndarray:
    """Compute Planck function B_nu(T) in erg/s/cm^2/Hz/steradian."""
    const_factor = 2 * H_PLANCK / C_LIGHT_CM**2
    hnu_over_kt = H_PLANCK * freq / (K_BOLTZ * temperature)

    # Handle very small hnu_over_kt (Rayleigh-Jeans limit)
    RJ_THRESHOLD = 1e-6
    bnu = np.zeros_like(temperature)

    rj_mask = hnu_over_kt < RJ_THRESHOLD
    bnu[rj_mask] = 2 * K_BOLTZ * temperature[rj_mask] * freq**2 / C_LIGHT_CM**2

    full_planck_mask = ~rj_mask
    bnu[full_planck_mask] = (
        const_factor * freq**3 / np.expm1(hnu_over_kt[full_planck_mask])
    )

    # Ensure no NaNs or Infs
    bnu[np.isnan(bnu)] = 0.0
    bnu[np.isinf(bnu)] = 0.0

    return bnu


# =============================================================================
# LUKEOP HELPER FUNCTIONS (atlas7v.for lines 8952-9259)
# =============================================================================


def _seaton(freq0: float, xsect: float, power: float, a: float, freq: float) -> float:
    """Seaton photoionization cross-section formula (atlas7v.for line 9252-9259).

    SEATON = XSECT * (A + (1-A)*(FREQ0/FREQ)) * SQRT((FREQ0/FREQ)**(2*POWER))

    Parameters
    ----------
    freq0 : float
        Threshold frequency (Hz)
    xsect : float
        Cross-section at threshold (cm²)
    power : float
        Power-law exponent
    a : float
        Asymptotic constant
    freq : float
        Frequency to evaluate at (Hz)

    Returns
    -------
    float
        Photoionization cross-section (cm²)
    """
    if freq < freq0:
        return 0.0
    ratio = freq0 / freq
    return xsect * (a + (1.0 - a) * ratio) * np.sqrt(ratio ** int(2.0 * power + 0.01))


def _build_si2op12_xtab() -> np.ndarray:
    """Build atlas12 SI2OP XTAB(200,51) cache once."""
    ryd = 109732.298
    z = 2.0
    elim = np.array(
        [
            131838.4, 131838.4, 131838.4, 131838.4, 131838.4, 131838.4, 131838.4, 131838.4, 131838.4, 131838.4, 131838.4,
            184563.09, 184563.09, 184563.09, 184563.09, 184563.09, 184563.09, 184563.09, 184563.09, 184563.09, 184563.09,
            184563.09, 184563.09, 184563.09, 184563.09, 184563.09, 184563.09, 184563.09, 184563.09, 184563.09, 184563.09,
            184563.09, 184563.09, 184563.09, 184563.09, 184563.09, 184563.09, 184563.09, 184563.09, 184563.09, 184563.09,
            254052.92, 254052.92, 254052.92, 131838.4, 184563.09,
        ],
        dtype=np.float64,
    )
    elev = np.array(
        [
            114177.4, 113760.48, 112394.92, 103877.34, 97972.35, 103556.36, 101024.09, 81231.57, 79348.67, 65500.73, 191.55,
            157396.6, 157188.8, 156838.9, 156836.9, 155663.4, 155593.7, 155555.0, 153523.1, 153147.2, 152977.0, 152480.7,
            151245.1, 149905.6, 140696.0, 134905.34, 134136.03, 132648.5, 132012.27, 131815.5, 126250.9, 124595.5, 124373.8,
            121541.76, 117058.95, 114415.54, 108804.1, 83937.09, 76665.61, 55319.11, 43002.27, 143990.0, 135300.5, 123033.6,
            119645.92, 167005.92,
        ],
        dtype=np.float64,
    )
    tlev = np.array(
        [
            17661.0, 18077.92, 19443.48, 27961.06, 33866.05, 28282.04, 30814.31, 50606.83, 52489.73, 66337.67, 131646.85,
            27166.49, 27374.29, 27724.19, 27726.19, 28899.69, 28969.39, 29008.09, 31039.99, 31415.89, 31586.09, 32082.39,
            33317.99, 34657.49, 43867.09, 49657.75, 50427.06, 51914.59, 52550.82, 52747.59, 58312.19, 59967.59, 60189.29,
            63021.33, 67504.14, 70147.55, 75758.99, 100526.0, 107897.48, 129243.98, 141560.82, 110052.92, 118752.42, 131019.32,
            12192.48, 17557.17,
        ],
        dtype=np.float64,
    )
    glev = np.array(
        [
            18.0, 14.0, 10.0, 6.0, 2.0, 14.0, 10.0, 6.0, 10.0, 1.0, 6.0, 20.0, 10.0, 18.0, 36.0, 28.0, 10.0, 10.0, 6.0, 12.0,
            2.0, 20.0, 28.0, 10.0, 10.0, 4.0, 12.0, 6.0, 20.0, 10.0, 6.0, 12.0, 20.0, 6.0, 12.0, 28.0, 10.0, 6.0, 2.0, 10.0,
            12.0, 6.0, 10.0, 4.0, 1.0, 9.0,
        ],
        dtype=np.float64,
    )
    nlev = np.array(
        [5, 5, 5, 5, 5, 4, 4, 4, 3, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 5],
        dtype=np.int64,
    )
    llev = np.array(
        [4, 3, 2, 1, 0, 3, 2, 1, 2, 0, 1, 3, 3, 3, 3, 3, 3, 2, 2, 2, 1, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 2, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        dtype=np.int64,
    )
    zeff2lev = (nlev[:44].astype(np.float64) ** 2 / ryd) * tlev[:44]
    hckttab = np.empty(51, dtype=np.float64)
    bolt3s2 = np.empty(51, dtype=np.float64)
    bolt3s3p = np.empty(51, dtype=np.float64)
    bolt = np.empty((44, 51), dtype=np.float64)
    for k in range(51):
        ttab = 10.0 ** (3.48 + (k + 1) * 0.02)
        hckttab[k] = 6.6256e-27 * 2.99792458e10 / 1.38054e-16 / ttab
        bolt3s2[k] = np.exp(-elim[0] * hckttab[k])
        bolt3s3p[k] = np.exp(-elim[11] * hckttab[k])
        bolt[:, k] = glev[:44] * np.exp(-elev[:44] * hckttab[k])
    xtab = np.empty((200, 51), dtype=np.float64)
    for nu in range(1, 201):
        wnotab = nu * 1000.0
        freqtab = wnotab * 2.99792458e10
        freq3 = 2.815e29 / (freqtab * freqtab * freqtab) * z**4
        x = np.zeros(44, dtype=np.float64)
        for i in range(0, 11):
            if wnotab < tlev[i]:
                break
            x[i] = xkarsas(freqtab, zeff2lev[i], int(nlev[i]), int(llev[i]))
        for i in range(11, 37):
            if wnotab < tlev[i]:
                break
            x[i] = xkarsas(freqtab, zeff2lev[i], int(nlev[i]), int(llev[i]))
        for i in range(37, 41):
            if wnotab < tlev[i]:
                break
            x[i] = 2.0 * xkarsas(freqtab, zeff2lev[i], int(nlev[i]), int(llev[i]))
        for i in range(41, 44):
            if wnotab < tlev[i]:
                break
            x[i] = 3.0 * xkarsas(freqtab, zeff2lev[i], int(nlev[i]), int(llev[i]))
        for k in range(51):
            h = freq3 * glev[44] / (ryd * z**2 * hckttab[k]) * (
                np.exp(-max(elev[44], elim[44] - wnotab) * hckttab[k]) - bolt3s2[k]
            )
            h = h + freq3 * glev[45] / (ryd * z**2 * hckttab[k]) * (
                np.exp(-max(elev[45], elim[45] - wnotab) * hckttab[k]) - bolt3s3p[k]
            )
            h = h + np.dot(x, bolt[:, k])
            xtab[nu - 1, k] = np.log(max(h, 1e-300))
    return xtab


# SI2OP Peach tables (atlas7v.for lines 9050-9073)

def _si2op_vectorized(
    freq: float, freqlg: float, temp: np.ndarray, tlog: np.ndarray
) -> np.ndarray:
    """Silicon II opacity (atlas7v.for lines 9043-9097).

    Returns cross-section * partition function for each layer.
    Uses Peach tables with temperature/frequency interpolation.

    Parameters
    ----------
    freq : float
        Frequency (Hz)
    freqlg : float
        Log of frequency
    temp : np.ndarray
        Temperature array (K), shape (n_layers,)
    tlog : np.ndarray
        Log of temperature array, shape (n_layers,)

    Returns
    -------
    np.ndarray
        SI2OP values for each layer (cm²), shape (n_layers,)
    """
    n_layers = temp.size

    # Temperature interpolation indices (atlas7v.for lines 9077-9080)
    # N = MAX(MIN(5, INT(T/2000) - 4), 1)
    nt = np.clip((temp / 2000.0).astype(int) - 4, 1, 5)
    dt = (tlog - _SI2OP_TLG[nt - 1]) / (_SI2OP_TLG[nt] - _SI2OP_TLG[nt - 1])

    # Frequency interpolation (atlas7v.for lines 9083-9093)
    n = 0
    for i in range(7):
        if freq > _SI2OP_FREQSI[i]:
            n = i + 1
            break
    else:
        n = 8

    # Adjust index based on Fortran logic
    d = (
        (freqlg - _SI2OP_FLOG[n - 1]) / (_SI2OP_FLOG[n] - _SI2OP_FLOG[n - 1])
        if n > 0 and n < 9
        else 0.0
    )

    # Map n to Peach table index
    if n > 2:
        n = 2 * n - 2
    n = min(n, 13)

    d1 = 1.0 - d

    # Interpolate in frequency (atlas7v.for lines 9092-9093)
    if n < 14:
        x = _SI2OP_PEACH[n] * d + _SI2OP_PEACH[n - 1] * d1 if n > 0 else _SI2OP_PEACH[0]
    else:
        x = _SI2OP_PEACH[13]

    # Interpolate in temperature and compute final value (atlas7v.for lines 9094-9095)
    result = np.zeros(n_layers, dtype=np.float64)
    for j in range(n_layers):
        nj = nt[j] - 1  # 0-indexed
        if nj < 5:
            val = x[nj] * (1.0 - dt[j]) + x[nj + 1] * dt[j]
        else:
            val = x[5]
        result[j] = np.exp(val) * 6.0

    return result


# =============================================================================
# MOLECULAR OPACITY FUNCTIONS: CHOP, OHOP, H2COLLOP
# =============================================================================
# These contribute to COOLOP for cool stars (T < 9000K)
# ACOOL = AC1 + AMG1 + AAL1 + ASI1 + AFE1 + CHOP*XNFPCH + OHOP*XNFPOH + AH2COLL

# CH Partition function (atlas12.for CHOP, DATA PARTCH lines 7576-7623)
# OH Partition function (atlas12.for OHOP, DATA PARTOH)
# CH cross-section table (atlas12.for CHOP, DATA C1..C11 lines 7406-7625)
# Shape: (106, 15) - rows 0..105 where row k = Fortran CROSSCH(IT, k) (N=1..105)
#   Row 0 is a padding row; rows 1..105 cover 0.1–10.5 eV in 0.1 eV steps
# Second dimension: 15 temperature points (2000K to 9000K in 500K steps)
# Values stored as log10(cross-section * partition_function)

def _chop_opacity(freq: float, temp: np.ndarray) -> np.ndarray:
    """CH molecular opacity (atlas12.for FUNCTION CHOP lines 7388-7651).

    Returns cross-section * partition function for each layer.
    Only active for T < 9000K and energy 2.0-10.5 eV.

    Parameters
    ----------
    freq : float
        Frequency (Hz)
    temp : np.ndarray
        Temperature array (K), shape (n_layers,)

    Returns
    -------
    np.ndarray
        CHOP values for each layer (cm²), shape (n_layers,)
    """
    n_layers = temp.size
    result = np.zeros(n_layers, dtype=np.float64)

    # Convert frequency to energy in eV
    waveno = freq / 2.99792458e10  # cm^-1
    evolt = waveno / 8065.479  # eV

    # Energy index (0.1 eV bins, atlas12.for CHOP lines 7630-7636)
    # Fortran: N = int(EVOLT*10), then CROSSCH(IT, N) and CROSSCH(IT, N+1)
    # Python _CH_CROSSSECT has shape (106,15): row 0 is padding, row k = Fortran N=k (1-based)
    n = int(evolt * 10)
    if n < 20 or n >= 105:  # Fortran: IF(N.LT.20)RETURN / IF(N.GE.105)RETURN
        return result

    en = float(n) * 0.1  # Fortran: EN=FLOAT(N)*.1

    # idx = n so that _CH_CROSSSECT[n, it] == Fortran CROSSCH(IT, N) (1-based N → Python row N)
    idx = n
    if idx < 0 or idx >= 105:
        return result

    # Cross-section at each temperature (interpolated in energy)
    crosscht = np.zeros(15, dtype=np.float64)
    for it in range(15):
        crosscht[it] = (
            _CH_CROSSSECT[idx, it]
            + (_CH_CROSSSECT[idx + 1, it] - _CH_CROSSSECT[idx, it]) * (evolt - en) / 0.1
        )

    # For each layer, interpolate in temperature
    for j in range(n_layers):
        t_j = temp[j]
        if t_j >= 9000.0:
            continue

        # Partition function interpolation (200K grid starting at 1000K)
        it_part = int((t_j - 1000.0) / 200.0)
        it_part = max(0, min(it_part, 39))
        tn_part = float(it_part) * 200.0 + 1000.0
        part = (
            _CH_PARTITION[it_part]
            + (_CH_PARTITION[it_part + 1] - _CH_PARTITION[it_part])
            * (t_j - tn_part)
            / 200.0
        )

        # Cross-section interpolation (500K grid starting at 2000K)
        it_cross = int((t_j - 2000.0) / 500.0)
        it_cross = max(0, min(it_cross, 13))
        tn_cross = float(it_cross) * 500.0 + 2000.0

        log_xsect = (
            crosscht[it_cross]
            + (crosscht[it_cross + 1] - crosscht[it_cross]) * (t_j - tn_cross) / 500.0
        )

        # Convert from log10 to linear
        result[j] = np.exp(log_xsect * 2.30258509299405) * part

    return result


def _chop_opacity_grid(freq: np.ndarray, temp: np.ndarray) -> np.ndarray:
    """Vectorized CHOP over all frequencies, preserving `_chop_opacity` math."""

    freq_arr = np.asarray(freq, dtype=np.float64)
    temp_arr = np.asarray(temp, dtype=np.float64)
    n_layers = temp_arr.size
    out = np.zeros((n_layers, freq_arr.size), dtype=np.float64)

    evolt = (freq_arr / 2.99792458e10) / 8065.479
    n = np.asarray(evolt * 10.0, dtype=np.int64)
    valid_freq = (n >= 20) & (n < 105)
    cool = temp_arr < 9000.0
    if not np.any(valid_freq) or not np.any(cool):
        return out

    idx = n[valid_freq]
    en = idx.astype(np.float64) * 0.1
    frac_e = (evolt[valid_freq] - en) / 0.1
    cross = _CH_CROSSSECT[idx, :] + (
        _CH_CROSSSECT[idx + 1, :] - _CH_CROSSSECT[idx, :]
    ) * frac_e[:, np.newaxis]

    it_part = np.clip(np.asarray((temp_arr - 1000.0) / 200.0, dtype=np.int64), 0, 39)
    tn_part = it_part.astype(np.float64) * 200.0 + 1000.0
    part = _CH_PARTITION[it_part] + (
        _CH_PARTITION[it_part + 1] - _CH_PARTITION[it_part]
    ) * (temp_arr - tn_part) / 200.0

    it_cross = np.clip(np.asarray((temp_arr - 2000.0) / 500.0, dtype=np.int64), 0, 13)
    tn_cross = it_cross.astype(np.float64) * 500.0 + 2000.0
    frac_t = (temp_arr - tn_cross) / 500.0
    log_xsect = cross[:, it_cross] + (
        cross[:, it_cross + 1] - cross[:, it_cross]
    ) * frac_t[np.newaxis, :]

    vals = np.exp(log_xsect * 2.30258509299405).T * part[:, np.newaxis]
    out[:, valid_freq] = vals
    out[~cool, :] = 0.0
    return out


# OH cross-section table (atlas12.for OHOP, DATA C1..C13 lines 7673-7940)
# Shape: (130, 15) - 130 energy bins (2.1-15.0 eV), row k = Fortran CROSSOH(IT, k) (N=1..130)
# Second dimension: 15 temperature points (2000K to 9000K in 500K steps)
# Values stored as log10(cross-section * partition_function)

def _ohop_opacity(freq: float, temp: np.ndarray) -> np.ndarray:
    """OH molecular opacity (atlas12.for FUNCTION OHOP lines 7653-7968).

    Returns cross-section * partition function for each layer.
    Only active for T < 9000K and energy 2.1-15.0 eV.

    Parameters
    ----------
    freq : float
        Frequency (Hz)
    temp : np.ndarray
        Temperature array (K), shape (n_layers,)

    Returns
    -------
    np.ndarray
        OHOP values for each layer (cm²), shape (n_layers,)
    """
    n_layers = temp.size
    result = np.zeros(n_layers, dtype=np.float64)

    # Convert frequency to energy in eV
    waveno = freq / 2.99792458e10  # cm^-1
    evolt = waveno / 8065.479  # eV

    # Energy index (0.1 eV bins starting at 2.1 eV)
    n = int(evolt * 10) - 20  # Shifted by 2.0 eV
    if n <= 0 or n >= 130:  # Energy range 2.1-15.0 eV
        return result

    en = float(n) * 0.1 + 2.0

    # Interpolate cross-section in energy (atlas7v.for lines 8683-8685)
    idx = n - 1  # 0-based index
    if idx < 0 or idx >= 129:
        return result

    crossoht = np.zeros(15, dtype=np.float64)
    for it in range(15):
        crossoht[it] = (
            _OH_CROSSSECT[idx, it]
            + (_OH_CROSSSECT[idx + 1, it] - _OH_CROSSSECT[idx, it]) * (evolt - en) / 0.1
        )

    # For each layer, interpolate in temperature
    for j in range(n_layers):
        t_j = temp[j]
        if t_j >= 9000.0:
            continue

        # Partition function interpolation (200K grid starting at 1000K)
        it_part = int((t_j - 1000.0) / 200.0)
        it_part = max(0, min(it_part, 39))
        tn_part = float(it_part) * 200.0 + 1000.0
        part = (
            _OH_PARTITION[it_part]
            + (_OH_PARTITION[it_part + 1] - _OH_PARTITION[it_part])
            * (t_j - tn_part)
            / 200.0
        )

        # Cross-section interpolation (500K grid starting at 2000K)
        it_cross = int((t_j - 2000.0) / 500.0)
        it_cross = max(0, min(it_cross, 13))
        tn_cross = float(it_cross) * 500.0 + 2000.0

        log_xsect = (
            crossoht[it_cross]
            + (crossoht[it_cross + 1] - crossoht[it_cross]) * (t_j - tn_cross) / 500.0
        )

        # Convert from log10 to linear
        result[j] = np.exp(log_xsect * 2.30258509299405) * part

    return result


def _ohop_opacity_grid(freq: np.ndarray, temp: np.ndarray) -> np.ndarray:
    """Vectorized OHOP over all frequencies, preserving `_ohop_opacity` math."""

    freq_arr = np.asarray(freq, dtype=np.float64)
    temp_arr = np.asarray(temp, dtype=np.float64)
    n_layers = temp_arr.size
    out = np.zeros((n_layers, freq_arr.size), dtype=np.float64)

    evolt = (freq_arr / 2.99792458e10) / 8065.479
    n = np.asarray(evolt * 10.0, dtype=np.int64) - 20
    valid_freq = (n > 0) & (n < 130)
    cool = temp_arr < 9000.0
    if not np.any(valid_freq) or not np.any(cool):
        return out

    idx = n[valid_freq] - 1
    en = n[valid_freq].astype(np.float64) * 0.1 + 2.0
    frac_e = (evolt[valid_freq] - en) / 0.1
    cross = _OH_CROSSSECT[idx, :] + (
        _OH_CROSSSECT[idx + 1, :] - _OH_CROSSSECT[idx, :]
    ) * frac_e[:, np.newaxis]

    it_part = np.clip(np.asarray((temp_arr - 1000.0) / 200.0, dtype=np.int64), 0, 39)
    tn_part = it_part.astype(np.float64) * 200.0 + 1000.0
    part = _OH_PARTITION[it_part] + (
        _OH_PARTITION[it_part + 1] - _OH_PARTITION[it_part]
    ) * (temp_arr - tn_part) / 200.0

    it_cross = np.clip(np.asarray((temp_arr - 2000.0) / 500.0, dtype=np.int64), 0, 13)
    tn_cross = it_cross.astype(np.float64) * 500.0 + 2000.0
    frac_t = (temp_arr - tn_cross) / 500.0
    log_xsect = cross[:, it_cross] + (
        cross[:, it_cross + 1] - cross[:, it_cross]
    ) * frac_t[np.newaxis, :]

    vals = np.exp(log_xsect * 2.30258509299405).T * part[:, np.newaxis]
    out[:, valid_freq] = vals
    out[~cool, :] = 0.0
    return out


# H2 collision-induced absorption tables (atlas7v.for lines 8733-8912)
# H2-H2 and H2-He tables: 7 temperature points × 81 wavenumber points
# Temperature grid: 1000K to 7000K in 1000K steps
# Wavenumber grid: 0 to 20000 cm^-1 in 250 cm^-1 steps

# Complete H2-H2 collision-induced absorption table (atlas7v.for lines 8733-8822)
# 81 wavenumber bins (0-20000 cm^-1 in 250 cm^-1 steps) × 7 temperature points (1000-7000K)
# Values are log10(absorption coefficient in cm^5)
# Complete H2-He collision-induced absorption table (atlas7v.for lines 8823-8912)
# 81 wavenumber bins × 7 temperature points

def _h2_collision_opacity(
    freq: float,
    temp: np.ndarray,
    xnfph1: np.ndarray,
    bhyd1: np.ndarray,
    xnfhe1: np.ndarray,
    rho: np.ndarray,
    tkev: np.ndarray,
    tlog: np.ndarray,
    stim: np.ndarray,
) -> np.ndarray:
    """H2 collision-induced absorption (atlas12.for SUBROUTINE H2COLLOP lines 7970-8225).

    Computes H2-H2 and H2-He collision-induced dipole absorption.
    Based on Borysow, Jorgensen, and Zheng (1997) A&A 324, 185-195.

    Only active for wavenumber < 20000 cm^-1.

    XNH2 is computed via the EQUILH2 tabulated equilibrium constant
    (atlas12.for line 9684), matching the shared XNH2 COMMON /XNF/ array
    populated by H2RAOP before H2COLLOP is called.

    Parameters
    ----------
    freq : float
        Frequency (Hz)
    temp : np.ndarray
        Temperature array (K), shape (n_layers,)
    xnfph1 : np.ndarray
        Ground-state hydrogen population (atoms/cm³)
    bhyd1 : np.ndarray
        Hydrogen partition function ground state
    xnfhe1 : np.ndarray
        Helium I population (atoms/cm³)
    rho : np.ndarray
        Mass density (g/cm³)
    tkev : np.ndarray
        Temperature in eV
    tlog : np.ndarray
        Log of temperature
    stim : np.ndarray
        Stimulated emission factor

    Returns
    -------
    np.ndarray
        H2 collision-induced opacity (cm²/g), shape (n_layers,)
    """
    n_layers = temp.size
    result = np.zeros(n_layers, dtype=np.float64)

    waveno = freq / 2.99792458e10  # cm^-1
    if waveno > 20000.0:
        return result

    # Compute H2 number density (cm^-3) using EQUILH2 table lookup.
    # atlas12.for H2RAOP line 9684: XNH2(J) = (XNFP(J,1)*2*BHYD(J,1))^2 * EQUILH2(T(J))
    # H2COLLOP reuses XNH2 from COMMON /XNF/ which was filled by H2RAOP.
    # T > 20000K: skip (XNH2 stays 0).
    xnh2 = _compute_xnh2_equilh2(
        temperature_k=temp,
        xnfp_h1=xnfph1,
        bhyd1=bhyd1,
    )
    xnh2 = np.where(temp > 20000.0, 0.0, xnh2)

    # Wavenumber interpolation (atlas12.for H2COLLOP lines 8201-8210)
    # Fortran: NU=int(WAVENO/250.)+1 (1-based), DELNU=(WAVENO-250.*(NU-1))/250.
    # H2H2NU(IT) = H2H2(IT,NU)*(1-DELNU) + H2H2(IT,NU+1)*DELNU
    nu = int(waveno / 250.0)  # 0-based; Fortran NU = nu+1 (1-based)
    nu = min(79, nu)
    delnu = (waveno - 250.0 * nu) / 250.0

    # Interpolate tables in wavenumber first (atlas12.for lines 8208-8210)
    # Fortran: H2H2NU(IT) = H2H2(IT,NU)*(1-DELNU) + H2H2(IT,NU+1)*DELNU
    # Python rows map as: H2H2(IT,NU) = _H2_COLL_H2H2[NU-1, IT-1] = _H2_COLL_H2H2[nu, it]
    h2h2_nu = np.zeros(7, dtype=np.float64)
    h2he_nu = np.zeros(7, dtype=np.float64)

    for it in range(7):
        idx1 = min(nu, 80)
        idx2 = min(nu + 1, 80)
        h2h2_nu[it] = _H2_COLL_H2H2[idx1, it] * (1.0 - delnu) + _H2_COLL_H2H2[idx2, it] * delnu
        h2he_nu[it] = _H2_COLL_H2HE[idx1, it] * (1.0 - delnu) + _H2_COLL_H2HE[idx2, it] * delnu

    # For each layer, interpolate in temperature (atlas12.for H2COLLOP lines 8213-8222)
    # Fortran: IT=int(T/1000.), IT=MAX(1,MIN(6,IT)), DELT=(T-1000.*IT)/1000.
    # XH2H2=H2H2NU(IT)*DELT+H2H2NU(IT+1)*(1-DELT)
    for j in range(n_layers):
        t_j = temp[j]

        # Temperature bin index (1000K grid, Fortran 1-based → Python 0-based)
        # Fortran: IT=int(T/1000.), IT=MAX(1,MIN(6,IT))
        it = int(t_j / 1000.0)
        it = max(1, min(6, it))

        # Fractional temperature interpolation weight (atlas12.for line 8216)
        # Fortran: DELT=(T-1000.*IT)/1000. (note: formula uses IT*1000 as lower edge)
        delt = (t_j - 1000.0 * it) / 1000.0
        delt = max(0.0, min(1.0, delt))

        # Interpolate in temperature (atlas12.for lines 8218-8219)
        # Fortran: XH2H2 = H2H2NU(IT)*DELT + H2H2NU(IT+1)*(1-DELT)
        # Python: h2h2_nu[it-1] = Fortran H2H2NU(IT), h2h2_nu[it] = H2H2NU(IT+1)
        xh2h2 = h2h2_nu[it - 1] * delt + h2h2_nu[it] * (1.0 - delt)
        xh2he = h2he_nu[it - 1] * delt + h2he_nu[it] * (1.0 - delt)

        # atlas12.for H2COLLOP line 8221-8222:
        # AH2COLL(J)=(10.**XH2HE*XNF(J,3)+10.**XH2H2*XNH2(J))*XNH2(J)/RHO(J)*STIM(J)
        # XNF(J,3) = He I number density (neutral helium, collision partner)
        result[j] = (
            (10.0**xh2he * xnfhe1[j] + 10.0**xh2h2 * xnh2[j])
            * xnh2[j]
            / rho[j]
            * stim[j]
        )

    return result


def _h2_collision_opacity_grid(
    freq: np.ndarray,
    temp: np.ndarray,
    xnfph1: np.ndarray,
    bhyd1: np.ndarray,
    xnfhe1: np.ndarray,
    rho: np.ndarray,
    stim: np.ndarray,
) -> np.ndarray:
    """Vectorized H2 collision opacity over the frequency grid.

    This mirrors `_h2_collision_opacity` but computes the Fortran H2RAOP XNH2
    state once per KAPP call. In ATLAS12, XNH2 lives in COMMON /XNF/ and is
    reused by H2COLLOP for every frequency.
    """

    n_layers = temp.size
    nfreq = freq.size
    result = np.zeros((n_layers, nfreq), dtype=np.float64)

    waveno = freq / 2.99792458e10
    active = waveno <= 20000.0
    if not np.any(active):
        return result

    xnh2 = _compute_xnh2_equilh2(
        temperature_k=temp,
        xnfp_h1=xnfph1,
        bhyd1=bhyd1,
    )
    xnh2 = np.where(temp > 20000.0, 0.0, xnh2)

    wv = waveno[active]
    nu = np.asarray(wv / 250.0, dtype=np.int64)
    nu = np.minimum(nu, 79)
    delnu = (wv - 250.0 * nu) / 250.0
    idx1 = np.minimum(nu, 80)
    idx2 = np.minimum(nu + 1, 80)
    h2h2_nu = _H2_COLL_H2H2[idx1, :] * (1.0 - delnu[:, np.newaxis]) + _H2_COLL_H2H2[idx2, :] * delnu[:, np.newaxis]
    h2he_nu = _H2_COLL_H2HE[idx1, :] * (1.0 - delnu[:, np.newaxis]) + _H2_COLL_H2HE[idx2, :] * delnu[:, np.newaxis]

    it = np.asarray(temp / 1000.0, dtype=np.int64)
    it = np.clip(it, 1, 6)
    delt = np.clip((temp - 1000.0 * it) / 1000.0, 0.0, 1.0)

    active_idx = np.nonzero(active)[0]
    for j in range(n_layers):
        xh2h2 = h2h2_nu[:, it[j] - 1] * delt[j] + h2h2_nu[:, it[j]] * (1.0 - delt[j])
        xh2he = h2he_nu[:, it[j] - 1] * delt[j] + h2he_nu[:, it[j]] * (1.0 - delt[j])
        result[j, active_idx] = (
            (10.0**xh2he * xnfhe1[j] + 10.0**xh2h2 * xnh2[j])
            * xnh2[j]
            / rho[j]
            * stim[j, active_idx]
        )

    return result


# =============================================================================
# HYDROGEN PARTITION FUNCTION AND GROUND-STATE POPULATION
# =============================================================================
# To compute SIGH (hydrogen Rayleigh scattering) correctly, we need the
# ground-state population XNFPH, not total neutral hydrogen XNFH.
# Fortran computes XNFPH via POPS(1.01D0,11,XNFPH) but fort.10 only stores XNFH.
# We compute the ground-state fraction using the Boltzmann partition function.

# Hydrogen energy levels and statistical weights from Fortran atlas7v.for
# DATA EHYD/0.D0,82259.105D0,97492.302D0,102823.893D0,105291.651D0,106632.160D0/
# DATA GHYD/2.,8.,18.,32.,50.,72./
# Energy in cm^-1, convert to eV using 1 eV = 8065.479 cm^-1
# Maximum principal quantum number for partition function sum
# Fortran GHYD/EHYD arrays have 6 levels (n=1 to n=6)

def compute_hydrogen_partition_function(temperature: np.ndarray) -> np.ndarray:
    """Compute the hydrogen partition function U(T).

    Uses Fortran's EHYD and GHYD tables from atlas7v.for:
        U(T) = Σ g_n * exp(-E_n / kT)

    where g_n = 2n² (statistical weight) and E_n are from EHYD table.

    Parameters
    ----------
    temperature : np.ndarray
        Temperature in Kelvin (shape: (n_layers,))

    Returns
    -------
    partition_func : np.ndarray
        Hydrogen partition function U(T) (shape: (n_layers,))
    """
    # Boltzmann factor kT in eV
    kt_ev = KBOLTZ_EV * temperature  # eV

    # Initialize partition function
    partition_func = np.zeros_like(temperature, dtype=np.float64)

    # Sum over all levels (matching Fortran's 6 levels)
    for i in range(H_MAX_LEVEL):
        g_n = H_STAT_WEIGHT[i]
        e_n = H_ENERGY_EV[i]  # Already in eV

        # Boltzmann factor: exp(-E_n / kT)
        # Avoid overflow for low temperatures
        with np.errstate(over="ignore", invalid="ignore"):
            boltz = np.exp(-e_n / kt_ev)
            boltz = np.where(np.isfinite(boltz), boltz, 0.0)

        partition_func += g_n * boltz

    return partition_func


def compute_ground_state_hydrogen(
    xnf_h: np.ndarray, temperature: np.ndarray
) -> np.ndarray:
    """Compute ground-state hydrogen population from total neutral hydrogen.

    Fortran's PFSAHA with MODE=11 returns ionization_fraction / partition_function,
    while MODE=12 returns ionization_fraction.

    So: XNFPH(J,1) = XNFH(J) / U(T)

    where U(T) is the hydrogen partition function.

    This replicates what Fortran's POPS(1.01D0,11,XNFPH) computes.

    Parameters
    ----------
    xnf_h : np.ndarray
        Total neutral hydrogen number density (atoms/cm³), shape (n_layers,)
        This is what POPS(1.00D0,12,XNFH) returns.
    temperature : np.ndarray
        Temperature in Kelvin, shape (n_layers,)

    Returns
    -------
    xnfph : np.ndarray
        Ground-state hydrogen number density (atoms/cm³), shape (n_layers,)
        This is what POPS(1.01D0,11,XNFPH)(:,1) returns.
    """
    partition_func = compute_hydrogen_partition_function(temperature)
    # CRITICAL: Divide by partition function (not multiply by ground-state fraction!)
    # XNFPH = XNFH / U(T), matching Fortran's PFSAHA MODE=11 vs MODE=12
    return xnf_h / partition_func


# Default IFOP values from Fortran atlas7v.for line 2822:
# DATA IFOP/1,1,1,1,1,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0/
# Index:     1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
# IFOP(4)=1: HRAYOP (H Rayleigh)  - ENABLED by default
# IFOP(8)=0: HERAOP (He Rayleigh) - DISABLED by default
# IFOP(9)=1: COOLOP (C1,Mg1,Al1,Si1,Fe1 + molecules) - ENABLED by default
# IFOP(10)=0: LUKEOP (N1,O1,Mg2,Si2,Ca2) - DISABLED by default


# ---------------------------------------------------------------------------
# atlas12 HE1OP helper functions and data tables (atlas12.for lines 6128-6330)
# ---------------------------------------------------------------------------

# CROSSHE: Marr & West (1976) ground-state He I photoionization cross-section
# Cross-sections in Mb; multiply by 1e-18 to get cm².
_CROSSHE_X505 = np.array([
    7.58, 7.46, 7.33, 7.19, 7.06, 6.94, 6.81, 6.68, 6.55, 6.43,
    6.30, 6.18, 6.05, 5.93, 5.81, 5.69, 5.57, 5.45, 5.33, 5.21,
    5.10, 4.98, 4.87, 4.76, 4.64, 4.53, 4.42, 4.31, 4.20, 4.09,
    4.00, 3.88, 3.78, 3.68, 3.57, 3.47, 3.37, 3.27, 3.18, 3.08,
    2.98, 2.89, 2.80, 2.70, 2.61, 2.52, 2.44, 2.35, 2.26, 2.18,
    2.10, 2.02, 1.94, 1.86, 1.78, 1.70, 1.63, 1.55, 1.48, 1.41,
    1.34, 1.28, 1.21, 1.14, 1.08, 1.02, .961, .903, .847, .792,
    .738, .687, .637, .588, .542, .497, .454, .412, .373, .335,
    .299, .265, .233, .202, .174, .147, .123, .100, .0795, .0609,
    .0443, .0315,
])
_CROSSHE_X50 = np.array([
    .0315, .0282, .0250, .0220, .0193, .0168, .0145, .0124, .0105,
    .00885, .00736, .00604, .00489, .00389, .00303, .00231,
])
_CROSSHE_X20 = np.array([
    .00231, .00199, .00171, .00145, .00122, .00101, .000832,
    .000673, .000535, .000417, .000318,
])
_CROSSHE_X10 = np.array([
    .000318, .000274, .000235, .000200, .000168, .000139, .000115,
    .000093, .000074, .000057, .000044, .000032, .000023, .000016, .000010,
    .000006, .000003, .000001, .0000006, .0000003, 0.,
])


def _crosshe(freq: float) -> float:
    """Ground-state He I photoionization cross-section (atlas12.for CROSSHE)."""
    if freq < 5.945209e15:
        return 0.0
    wave = 2.99792458e18 / freq  # Angstroms
    if wave > 50.0:
        i = int(93.0 - (wave - 50.0) / 5.0)
        i = min(92, max(2, i))
        return (
            (wave - (92 - i) * 5 - 50) / 5.0
            * (_CROSSHE_X505[i - 2] - _CROSSHE_X505[i - 1])
            + _CROSSHE_X505[i - 1]
        ) * 1.0e-18
    if wave > 20.0:
        i = int(17.0 - (wave - 20.0) / 2.0)
        i = min(16, max(2, i))
        return (
            (wave - (16 - i) * 2 - 20) / 2.0
            * (_CROSSHE_X50[i - 2] - _CROSSHE_X50[i - 1])
            + _CROSSHE_X50[i - 1]
        ) * 1.0e-18
    if wave > 10.0:
        i = int(12.0 - (wave - 10.0) / 1.0)
        i = min(11, max(2, i))
        return (
            (wave - (11 - i) * 1 - 10) / 1.0
            * (_CROSSHE_X20[i - 2] - _CROSSHE_X20[i - 1])
            + _CROSSHE_X20[i - 1]
        ) * 1.0e-18
    i = int(22.0 - wave / 0.5)
    i = min(21, max(2, i))
    return (
        (wave - (21 - i) * 0.5) / 0.5
        * (_CROSSHE_X10[i - 2] - _CROSSHE_X10[i - 1])
        + _CROSSHE_X10[i - 1]
    ) * 1.0e-18


# HE12S1S: 1s2s ^1S cross-section with autoionization (atlas12.for lines 6208-6239)
_HE12S1S_FREQ = np.array([
    15.947182, 15.913654, 15.877320, 15.837666, 15.794025,
    15.745503, 15.690869, 15.628361, 15.555317, 15.467455,
    15.357189, 15.289399, 15.251073, 15.209035, 15.162487,
    14.982421,
])
_HE12S1S_X = np.array([
    -19.635557, -19.159345, -18.958474, -18.809535, -18.676481,
    -18.546006, -18.410962, -18.264821, -18.100205, -17.909165,
    -17.684370, -17.557867, -17.490360, -17.417876, -17.349386,
    -17.084441,
])


def _he12s1s(freq: float) -> float:
    """1s2s ^1S cross-section (atlas12.for HE12S1S)."""
    if freq < 32033.214 * C_LIGHT_CM:
        return 0.0
    if freq > 2.4 * 109722.267 * C_LIGHT_CM:
        waveno = freq / C_LIGHT_CM
        ek = (waveno - 32033.214) / 109722.267
        eps = 2.0 * (ek - 2.612316) / 0.00322
        return (
            0.008175 * (484940.0 / waveno) ** 2.71 * 8.067e-18
            * (eps + 76.21) ** 2 / (1.0 + eps ** 2)
        )
    freqlg = np.log10(freq)
    idx = 15  # fallback
    for i in range(1, 16):
        if freqlg > _HE12S1S_FREQ[i]:
            idx = i
            break
    x = (
        (freqlg - _HE12S1S_FREQ[idx])
        / (_HE12S1S_FREQ[idx - 1] - _HE12S1S_FREQ[idx])
        * (_HE12S1S_X[idx - 1] - _HE12S1S_X[idx])
        + _HE12S1S_X[idx]
    )
    return 10.0 ** x


# HE12S3S: 1s2s ^3S cross-section with autoionization (atlas12.for lines 6240-6271)
_HE12S3S_FREQ = np.array([
    15.956523, 15.923736, 15.888271, 15.849649, 15.807255,
    15.760271, 15.707580, 15.647601, 15.577992, 15.495055,
    15.392451, 15.330345, 15.295609, 15.257851, 15.216496,
    15.061770,
])
_HE12S3S_X = np.array([
    -18.426022, -18.610700, -18.593051, -18.543304, -18.465513,
    -18.378707, -18.278574, -18.164329, -18.033346, -17.882435,
    -17.705542, -17.605584, -17.553459, -17.500667, -17.451318,
    -17.266686,
])


def _he12s3s(freq: float) -> float:
    """1s2s ^3S cross-section (atlas12.for HE12S3S)."""
    if freq < 38454.691 * C_LIGHT_CM:
        return 0.0
    if freq > 2.4 * 109722.267 * C_LIGHT_CM:
        waveno = freq / C_LIGHT_CM
        ek = (waveno - 38454.691) / 109722.267
        eps = 2.0 * (ek - 2.47898) / 0.000780
        return (
            0.01521 * (470310.0 / waveno) ** 3.12 * 8.067e-18
            * (eps - 122.4) ** 2 / (1.0 + eps ** 2)
        )
    freqlg = np.log10(freq)
    idx = 15
    for i in range(1, 16):
        if freqlg > _HE12S3S_FREQ[i]:
            idx = i
            break
    x = (
        (freqlg - _HE12S3S_FREQ[idx])
        / (_HE12S3S_FREQ[idx - 1] - _HE12S3S_FREQ[idx])
        * (_HE12S3S_X[idx - 1] - _HE12S3S_X[idx])
        + _HE12S3S_X[idx]
    )
    return 10.0 ** x


# HE12P1P: 1s2p ^1P cross-section with autoionization (atlas12.for lines 6272-6305)
_HE12P1P_FREQ = np.array([
    15.939981, 15.905870, 15.868850, 15.828377, 15.783742,
    15.733988, 15.677787, 15.613218, 15.537343, 15.445346,
    15.328474, 15.255641, 15.214064, 15.168081, 15.116647,
    14.911002,
])
_HE12P1P_X = np.array([
    -18.798876, -19.685922, -20.011664, -20.143030, -20.091354,
    -19.908333, -19.656788, -19.367745, -19.043016, -18.674484,
    -18.240861, -17.989700, -17.852015, -17.702677, -17.525347,
    -16.816344,
])


def _he12p1p(freq: float) -> float:
    """1s2p ^1P cross-section with autoionization (atlas12.for HE12P1P)."""
    if freq < 27175.76 * C_LIGHT_CM:
        return 0.0
    if freq > 2.4 * 109722.267 * C_LIGHT_CM:
        waveno = freq / C_LIGHT_CM
        ek = (waveno - 27175.76) / 109722.267
        eps1s = 2.0 * (ek - 2.446534) / 0.01037
        eps1d = 2.0 * (ek - 2.59427) / 0.00538
        return (
            0.0009487 * (466750.0 / waveno) ** 3.69 * 8.067e-18
            * ((eps1s - 29.30) ** 2 / (1.0 + eps1s ** 2)
               + (eps1d + 172.4) ** 2 / (1.0 + eps1d ** 2))
        )
    freqlg = np.log10(freq)
    idx = 15
    for i in range(1, 16):
        if freqlg > _HE12P1P_FREQ[i]:
            idx = i
            break
    x = (
        (freqlg - _HE12P1P_FREQ[idx])
        / (_HE12P1P_FREQ[idx - 1] - _HE12P1P_FREQ[idx])
        * (_HE12P1P_X[idx - 1] - _HE12P1P_X[idx])
        + _HE12P1P_X[idx]
    )
    return 10.0 ** x


# HE12P3P: 1s2p ^3P cross-section (atlas12.for lines 6306-6330)
_HE12P3P_FREQ = np.array([
    15.943031, 15.909169, 15.872441, 15.832318, 15.788107,
    15.738880, 15.683351, 15.619667, 15.545012, 15.454805,
    15.340813, 15.270195, 15.230054, 15.185821, 15.136567,
    14.942557,
])
_HE12P3P_X = np.array([
    -19.791021, -19.697886, -19.591421, -19.471855, -19.337053,
    -19.183958, -19.009750, -18.807990, -18.570571, -18.288361,
    -17.943476, -17.738737, -17.624154, -17.497163, -17.403183,
    -17.032999,
])


def _he12p3p(freq: float) -> float:
    """1s2p ^3P cross-section (atlas12.for HE12P3P)."""
    if freq < 29223.753 * C_LIGHT_CM:
        return 0.0
    freqlg = np.log10(freq)
    idx = 15
    for i in range(1, 16):
        if freqlg > _HE12P3P_FREQ[i]:
            idx = i
            break
    x = (
        (freqlg - _HE12P3P_FREQ[idx])
        / (_HE12P3P_FREQ[idx - 1] - _HE12P3P_FREQ[idx])
        * (_HE12P3P_X[idx - 1] - _HE12P3P_X[idx])
        + _HE12P3P_X[idx]
    )
    return 10.0 ** x


def compute_kapp_continuum(
    atmosphere: "AtmosphereModel",
    freq: np.ndarray,
    atlas_tables: dict[str, np.ndarray],
    ifop: list[int],
    tcst: "_TcorrState | None" = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ACONT and SIGMAC using full KAPP logic.

    Parameters
    ----------
    atmosphere:
        The atmosphere model with populations (xnfph, xnf_he1, xnf_he2, etc.)
    freq:
        Frequency array in Hz (shape: (nfreq,))
    atlas_tables:
        Dictionary of B-tables (bhyd, bhe1, bhe2, bc1, bmg1, bal1, bsi1, bfe1, etc.)
        Shape: (n_layers, n_levels)
    ifop:
        20-element IFOP array read directly from the .atm file or control deck
        (atlas12.for DATA IFOP, line 1709). Must always come from the model input —
        never guessed from temperature or other physics.

    Returns
    -------
    acont:
        Continuum absorption coefficient (shape: (n_layers, nfreq)) in cm²/g
    sigmac:
        Continuum scattering coefficient (shape: (n_layers, nfreq)) in cm²/g
    scont:
        Continuum source function (shape: (n_layers, nfreq)) in erg/s/cm²/Hz/ster
    """
    n_layers = atmosphere.layers
    nfreq = freq.size

    logger.debug("Computing KAPP continuum: %d layers × %d frequencies", n_layers, nfreq)

    # Initialize arrays
    acont = np.zeros((n_layers, nfreq), dtype=np.float64)
    sigmac = np.zeros((n_layers, nfreq), dtype=np.float64)
    scont = np.zeros((n_layers, nfreq), dtype=np.float64)

    temp = np.asarray(atmosphere.temperature, dtype=np.float64)

    # Compute frequency-dependent quantities
    wavelength_nm = C_LIGHT_NM / np.maximum(freq, 1e-30)
    # hkt is per layer (shape: (n_layers,))
    hkt = H_PLANCK / (K_BOLTZ * temp)
    # hckt = hkt * c (cm²/s) - used in HE1OP, HE2OP, etc. (atlas7v.for line 81: HCKT(J)=HKT(J)*2.99792458D10)
    hckt = hkt * C_LIGHT_CM
    # ehvkt and stim are per layer and frequency (shape: (n_layers, nfreq))
    hnu_over_kt = H_PLANCK * freq[None, :] / (K_BOLTZ * temp[:, None])
    ehvkt = np.exp(-hnu_over_kt)
    stim = 1.0 - ehvkt
    # Compute Planck functions for all frequencies. This is algebraically the
    # same as `_planck_nu` but avoids a Python loop over the full ATLAS grid.
    bnu_all = np.zeros((n_layers, nfreq), dtype=np.float64)
    rj_mask = hnu_over_kt < 1.0e-6
    if np.any(rj_mask):
        bnu_all[rj_mask] = (
            2.0
            * K_BOLTZ
            * temp[:, np.newaxis]
            * (freq[np.newaxis, :] ** 2)
            / C_LIGHT_CM**2
        )[rj_mask]
    full_planck_mask = ~rj_mask
    if np.any(full_planck_mask):
        bnu_full = (
            (2.0 * H_PLANCK / C_LIGHT_CM**2)
            * (freq[np.newaxis, :] ** 3)
            / np.expm1(hnu_over_kt)
        )
        bnu_all[full_planck_mask] = bnu_full[full_planck_mask]
    bnu_all[~np.isfinite(bnu_all)] = 0.0
    waveno = freq / C_LIGHT_CM

    # Initialize component arrays
    ahyd = np.zeros((n_layers, nfreq), dtype=np.float64)
    ahmin = np.zeros((n_layers, nfreq), dtype=np.float64)
    ah2p = np.zeros((n_layers, nfreq), dtype=np.float64)
    ahe1 = np.zeros((n_layers, nfreq), dtype=np.float64)
    ahe2 = np.zeros((n_layers, nfreq), dtype=np.float64)
    ahemin = np.zeros((n_layers, nfreq), dtype=np.float64)
    ac1 = np.zeros((n_layers, nfreq), dtype=np.float64)
    amg1 = np.zeros((n_layers, nfreq), dtype=np.float64)
    aal1 = np.zeros((n_layers, nfreq), dtype=np.float64)
    asi1 = np.zeros((n_layers, nfreq), dtype=np.float64)
    afe1 = np.zeros((n_layers, nfreq), dtype=np.float64)
    acool = np.zeros((n_layers, nfreq), dtype=np.float64)
    aluke = np.zeros((n_layers, nfreq), dtype=np.float64)
    ahot = np.zeros((n_layers, nfreq), dtype=np.float64)
    axcont = np.zeros((n_layers, nfreq), dtype=np.float64)

    shyd = np.zeros((n_layers, nfreq), dtype=np.float64)
    shmin = np.zeros((n_layers, nfreq), dtype=np.float64)
    she1 = np.zeros((n_layers, nfreq), dtype=np.float64)
    she2 = np.zeros((n_layers, nfreq), dtype=np.float64)
    sc1 = np.zeros((n_layers, nfreq), dtype=np.float64)
    smg1 = np.zeros((n_layers, nfreq), dtype=np.float64)
    sal1 = np.zeros((n_layers, nfreq), dtype=np.float64)
    ssi1 = np.zeros((n_layers, nfreq), dtype=np.float64)
    sfe1 = np.zeros((n_layers, nfreq), dtype=np.float64)
    sxcont = np.zeros((n_layers, nfreq), dtype=np.float64)

    # XCONOP (IFOP(19)): external continuum from Rosseland mean-opacity table.
    # Fortran atlas12.for line 5252: IF(IFOP(19).EQ.1)CALL XCONOP
    # AXCONT(J) = ROSSTAB(T(J),P(J),VTURB(J)) -- frequency-independent per-layer opacity.
    # SXCONT(J) = 5.667D-5/12.5664 * T(J)**4 * 4.  -- thermal source function.
    if ifop[18] == 1 and tcst is not None:  # IFOP(19) in Fortran = ifop[18] in Python
        gas_pressure = np.asarray(atmosphere.gas_pressure, dtype=np.float64)
        if gas_pressure is not None:  # always true; kept for structural clarity
            axcont_1d = np.array(
                [_rosstab_eval(tcst, float(temp[j]), float(gas_pressure[j])) for j in range(n_layers)],
                dtype=np.float64,
            )
            sxcont_1d = 5.667e-5 / 12.5664 * temp ** 4 * 4.0  # (n_layers,)
            axcont[:, :] = axcont_1d[:, None]
            sxcont[:, :] = sxcont_1d[:, None]

    # Initialize scattering arrays (will be computed below)
    sigh = np.zeros((n_layers, nfreq), dtype=np.float64)
    sighe = np.zeros((n_layers, nfreq), dtype=np.float64)
    sigel = np.zeros((n_layers, nfreq), dtype=np.float64)
    sigh2 = np.zeros((n_layers, nfreq), dtype=np.float64)
    sigx = np.zeros((n_layers, nfreq), dtype=np.float64)

    rho = np.maximum(np.asarray(atmosphere.mass_density, dtype=np.float64), 1e-30)

    # Define xne at function scope (Fortran has it in COMMON, always accessible)
    if atmosphere.electron_density is not None:
        xne = np.asarray(atmosphere.electron_density, dtype=np.float64)
    else:
        xne = np.zeros(n_layers, dtype=np.float64)

    # HOP: Hydrogen opacity (`atlas12.for` lines 5268-5312).
    if ifop[0] == 1 and atmosphere.xnfph is not None:
        logger.debug("Computing HOP (hydrogen opacity)...")
        xnfph = np.asarray(atmosphere.xnfph, dtype=np.float64)
        if xnfph.ndim == 1:
            xnfph = xnfph[:, np.newaxis]
        xnfph1 = xnfph[:, 0] if xnfph.shape[1] > 0 else np.zeros(n_layers, dtype=np.float64)
        xnf_h_ionized = np.asarray(atmosphere.xnf_h_ionized, dtype=np.float64)

        bhyd = np.asarray(atmosphere.bhyd, dtype=np.float64)
        if bhyd.ndim == 1:
            bhyd = bhyd[:, np.newaxis]
        if bhyd.shape[1] < 6:
            pad = np.ones((n_layers, 6 - bhyd.shape[1]), dtype=np.float64)
            bhyd = np.hstack([bhyd, pad])

        tkev = np.maximum(temp * KBOLTZ_EV, 1e-300)
        bolt = np.zeros((n_layers, 8), dtype=np.float64)
        for n in range(1, 9):
            bolt[:, n - 1] = (
                np.exp(-(13.595 - 13.595 / float(n * n)) / tkev)
                * 2.0
                * float(n * n)
                * xnfph1
                / rho
            )
            if n <= 6:
                bolt[:, n - 1] *= bhyd[:, n - 1]
        freet = xne * xnf_h_ionized / rho / np.sqrt(np.maximum(temp, 1e-300))
        xr = xnfph1 * (tkev / 13.595) / rho
        boltex = np.exp(-13.427 / tkev) * xr
        exlim = np.exp(-13.595 / tkev) * xr
        tlog_arr = np.log(np.maximum(temp, 1e-300))
        coulff_h = _coulff_grid(1, np.log(np.maximum(freq, 1e-300)), tlog_arr)
        cont_h = np.vstack(
            [xkarsas_grid(freq, 1.0, n, n) for n in range(1, 9)]
        )

        freq3 = np.maximum(freq * freq * freq, 1e-300)
        cfree = 3.6919e8 / freq3
        cterm = 2.815e29 / freq3
        ex = np.broadcast_to(boltex[:, np.newaxis], (n_layers, nfreq)).copy()
        low_freq = freq < 4.05933e13
        if np.any(low_freq):
            ex[:, low_freq] = exlim[:, np.newaxis] / np.maximum(ehvkt[:, low_freq], 1e-300)

        h = (
            cont_h[6, :][np.newaxis, :] * bolt[:, 6][:, np.newaxis]
            + cont_h[7, :][np.newaxis, :] * bolt[:, 7][:, np.newaxis]
            + (ex - exlim[:, np.newaxis]) * cterm[np.newaxis, :]
            + coulff_h * freet[:, np.newaxis] * cfree[np.newaxis, :]
        ) * stim
        s = h * bnu_all
        for n in range(6):
            bh = np.maximum(bhyd[:, n], 1e-300)
            term = cont_h[n, :][np.newaxis, :] * bolt[:, n][:, np.newaxis]
            h = h + term * (1.0 - ehvkt / bh[:, np.newaxis])
            s = s + term * bnu_all * stim / bh[:, np.newaxis]
        ahyd[:, :] = h
        shyd[:, :] = np.where(h > 0.0, s / np.maximum(h, 1e-300), bnu_all)

    # H2PLOP: H2+ opacity (atlas7v.for line 5189-5211)
    if ifop[1] == 1 and atmosphere.xnfph is not None:
        logger.debug("Computing H2PLOP (H2+ opacity)...")
        xnfph_arr = np.asarray(atmosphere.xnfph, dtype=np.float64)
        bhyd = atlas_tables.get("bhyd", np.ones((n_layers, 8), dtype=np.float64))
        tkev = np.asarray(atmosphere.temperature, dtype=np.float64) * KBOLTZ_EV

        if xnfph_arr.shape[1] >= 2:
            active = freq <= 3.28805e15
            if np.any(active):
                f_active = freq[active]
                freqlg = np.log(f_active)
                freq15 = f_active / 1.0e15

                # FR = polynomial in FREQLG (atlas7v.for line 5200-5201)
                fr = (
                    -3.0233e3
                    + (
                        3.7797e2
                        + (-1.82496e1 + (3.9207e-1 - 3.1672e-3 * freqlg) * freqlg) * freqlg
                    )
                    * freqlg
                )

                # ES = polynomial in FREQ15 (atlas7v.for line 5203-5204)
                es = (
                    -7.342e-3
                    + (
                        -2.409e0
                        + (
                            1.028e0
                            + (-4.230e-1 + (1.224e-1 - 1.351e-2 * freq15) * freq15) * freq15
                        )
                        * freq15
                    )
                    * freq15
                )

                xnfph1 = xnfph_arr[:, 0]
                xnfph2 = xnfph_arr[:, 1]
                bhyd1 = (
                    bhyd[:, 0]
                    if bhyd.shape[1] > 0
                    else np.ones(n_layers, dtype=np.float64)
                )
                ah2p[:, active] = (
                    np.exp(
                        -es[np.newaxis, :] / tkev[:, np.newaxis]
                        + fr[np.newaxis, :]
                        + np.log(np.maximum(xnfph1, 1e-40))[:, np.newaxis]
                    )
                    * 2.0
                    * bhyd1[:, np.newaxis]
                    * xnfph2[:, np.newaxis]
                    / rho[:, np.newaxis]
                    * stim[:, active]
                )

    # HE1OP: Helium I opacity (atlas12.for lines 6007-6127)
    if ifop[4] == 1 and atmosphere.xnf_he1 is not None:
        logger.debug("Computing HE1OP (Helium I opacity, atlas12)...")
        # XNFP(J,3): He I partition-function-weighted number density
        xnfp_he1 = np.asarray(atmosphere.xnf_he1, dtype=np.float64)
        if xnfp_he1.ndim > 1:
            xnfp_he1 = xnfp_he1[:, 0]

        # XNF(J,4): He II raw number density (for free-free)
        xnf_all_he1 = np.asarray(atmosphere.xnf_all, dtype=np.float64)
        he2_xnf_raw = (
            xnf_all_he1[:, 3]
            if xnf_all_he1.shape[1] > 3
            else np.zeros(n_layers, dtype=np.float64)
        )

        tkev_he1 = KBOLTZ_EV * temp
        ryd_he1 = 109722.273 * C_LIGHT_CM  # atlas12.for line 6031

        # Level data (atlas12.for lines 6022-6027)
        _he1_g = np.array([1., 3., 1., 9., 3., 3., 1., 9., 20., 3.])
        _he1_chi = np.array([
            0., 19.819, 20.615, 20.964, 21.217,
            22.718, 22.920, 23.006, 23.073, 23.086,
        ])
        _he1_hefreq = np.array([
            5.945209e15, 1.152844e15, 0.9603331e15, 0.8761076e15,
            0.8147104e15, 0.4519048e15, 0.4030971e15, 0.3821191e15,
            0.3660215e15, 0.3627891e15,
        ])

        # BOLT(J,N) for N=1..10 (atlas12.for line 6034)
        he1_bolt = np.zeros((n_layers, 10), dtype=np.float64)
        for _n in range(10):
            he1_bolt[:, _n] = (
                np.exp(-_he1_chi[_n] / tkev_he1) * _he1_g[_n] * xnfp_he1 / rho
            )

        # BOLTN(J,N) for N=4..27 (atlas12.for lines 6035-6037)
        he1_boltn = np.zeros((n_layers, 28), dtype=np.float64)
        for _n in range(4, 28):
            he1_boltn[:, _n] = (
                np.exp(-24.587 * (1.0 - 1.0 / (_n * _n)) / tkev_he1)
                * 4.0 * _n * _n * xnfp_he1 / rho
            )

        # FREET(J) = XNE*XNF(J,4)/RHO/SQRT(T) (atlas12.for line 6038)
        he1_freet = xne * he2_xnf_raw / rho / np.sqrt(temp)

        # XR, BOLTEX, EXLIM (atlas12.for lines 6039-6041)
        he1_xr = xnfp_he1 * (4.0 / 2.0 / 13.595) * tkev_he1 / rho
        he1_boltex = np.exp(-23.730 / tkev_he1) * he1_xr
        he1_exlim = np.exp(-24.587 / tkev_he1) * he1_xr

        he1_tlog = np.log(np.maximum(temp, 1e-10))
        he1_coulff = _coulff_grid(1, np.log(np.maximum(freq, 1e-300)), he1_tlog)
        he1_xk_trans = {
            5: xkarsas_grid(freq, 1.236439, 3, 0),
            6: xkarsas_grid(freq, 1.102898, 3, 0),
            7: xkarsas_grid(freq, 1.045499, 3, 1),
            8: xkarsas_grid(freq, 1.001427, 3, 2),
            9: xkarsas_grid(freq, 0.9926, 3, 1),
        }
        he1_auto_grids: dict[float, np.ndarray] = {}
        for elim, levels in (
            (527490.06, (171135.000, 169087.0, 166277.546, 159856.069)),
            (588451.59, (186209.471, 186101.0, 185564.0, 184864.0, 183236.0)),
        ):
            for level in levels:
                freqhe_grid = (elim - level) * C_LIGHT_CM
                he1_auto_grids[freqhe_grid] = xkarsas_grid(freq, freqhe_grid / ryd_he1, 1, 0)
        he1_transn = np.zeros((28, nfreq), dtype=np.float64)
        for _n in range(4, 28):
            he1_transn[_n, :] = xkarsas_grid(freq, 4.0 - 3.0 / (_n * _n), 1, 0)

        for j in range(nfreq):
            f = freq[j]
            stim_j = stim[:, j]
            ehvkt_j = ehvkt[:, j]

            freq3 = f ** 3
            cfree = 3.6919e8 / freq3
            c_const = 2.815e29 / freq3

            # Find IMIN: first level with ionization freq <= current freq
            # (atlas12.for lines 6045-6049)
            imin = 0
            for _n in range(10):
                if _he1_hefreq[_n] <= f:
                    imin = _n + 1  # Fortran 1-based
                    break

            # Cross-sections TRANS[0..9] via fall-through (atlas12.for 6050-6065)
            trans = np.zeros(10, dtype=np.float64)
            if imin >= 1:
                if imin <= 1:
                    trans[0] = _crosshe(f)
                if imin <= 2:
                    trans[1] = _he12s3s(f)
                if imin <= 3:
                    trans[2] = _he12s1s(f)
                if imin <= 4:
                    trans[3] = _he12p3p(f)
                if imin <= 5:
                    trans[4] = _he12p1p(f)
                if imin <= 6:
                    trans[5] = he1_xk_trans[5][j]
                if imin <= 7:
                    trans[6] = he1_xk_trans[6][j]
                if imin <= 8:
                    trans[7] = he1_xk_trans[7][j]
                if imin <= 9:
                    trans[8] = he1_xk_trans[8][j]
                if imin <= 10:
                    trans[9] = he1_xk_trans[9][j]

                # He II n=2 autoionization (atlas12.for lines 6067-6084)
                elim2 = 527490.06
                freqhe = (elim2 - 171135.000) * C_LIGHT_CM
                if f >= freqhe:
                    trans[4] += he1_auto_grids[freqhe][j]
                    freqhe = (elim2 - 169087.0) * C_LIGHT_CM
                    if f >= freqhe:
                        trans[3] += he1_auto_grids[freqhe][j]
                        freqhe = (elim2 - 166277.546) * C_LIGHT_CM
                        if f >= freqhe:
                            trans[2] += he1_auto_grids[freqhe][j]
                            freqhe = (elim2 - 159856.069) * C_LIGHT_CM
                            if f >= freqhe:
                                trans[1] += he1_auto_grids[freqhe][j]

                # He II n=3 autoionization (atlas12.for lines 6086-6106)
                elim3 = 588451.59
                freqhe = (elim3 - 186209.471) * C_LIGHT_CM
                if f >= freqhe:
                    trans[9] += he1_auto_grids[freqhe][j]
                    freqhe = (elim3 - 186101.0) * C_LIGHT_CM
                    if f >= freqhe:
                        trans[8] += he1_auto_grids[freqhe][j]
                        freqhe = (elim3 - 185564.0) * C_LIGHT_CM
                        if f >= freqhe:
                            trans[7] += he1_auto_grids[freqhe][j]
                            freqhe = (elim3 - 184864.0) * C_LIGHT_CM
                            if f >= freqhe:
                                trans[6] += he1_auto_grids[freqhe][j]
                                freqhe = (elim3 - 183236.0) * C_LIGHT_CM
                                if f >= freqhe:
                                    trans[5] += he1_auto_grids[freqhe][j]

            # High-n levels N=4..27 (atlas12.for lines 6107-6110)
            transn = np.zeros(28, dtype=np.float64)
            if f >= 1.25408e16:
                for _n in range(4, 28):
                    transn[_n] = he1_transn[_n, j]

            # Per-layer computation (atlas12.for lines 6111-6123)
            if f < 2.055e14:
                ex = he1_exlim / ehvkt_j
            else:
                ex = he1_boltex.copy()
            he1_val = (ex - he1_exlim) * c_const

            if imin >= 1:
                for _n in range(imin - 1, 10):
                    he1_val = he1_val + trans[_n] * he1_bolt[:, _n]

            if f >= 1.25408e16:
                for _n in range(4, 28):
                    he1_val = he1_val + transn[_n] * he1_boltn[:, _n]

            # Free-free: COULFF(J,1)*FREET(J)*CFREE (atlas12.for line 6123)
            coulff_arr = he1_coulff[:, j]
            ahe1[:, j] = (he1_val + coulff_arr * he1_freet * cfree) * stim_j
            she1[:, j] = bnu_all[:, j]

    # HE2OP: Helium II opacity (atlas12.for lines 6331-6375)
    # Uses XKARSAS (Karzas & Latter 1960) cross-sections, matching atlas12 exactly.
    if ifop[5] == 1 and atmosphere.xnf_he2 is not None:
        logger.debug("Computing HE2OP (Helium II opacity, atlas12 XKARSAS)...")
        xnfp_he2 = np.asarray(atmosphere.xnf_he2, dtype=np.float64)  # XNFP(J,4)
        xnf_he3 = np.asarray(atmosphere.xnf_all[:, 4], dtype=np.float64)  # XNF(J,5)

        tkev = KBOLTZ_EV * temp

        # BOLT(J,N) = EXP(-(54.403 - 54.403/N²)/TKEV) * 2*N² * XNFP(J,4)/RHO
        bolt = np.zeros((n_layers, 9), dtype=np.float64)
        for n in range(1, 10):
            bolt[:, n - 1] = (
                np.exp(-(54.403 - 54.403 / (n * n)) / tkev)
                * 2.0 * (n * n)
                * xnfp_he2 / rho
            )

        # FREET(J) = XNE * XNF(J,5) / RHO / SQRT(T)
        freet = xne * xnf_he3 / rho / np.sqrt(temp)

        # XR = XNFP(J,4) * (2/2/13.595) * TKEV / RHO
        xr = xnfp_he2 * (1.0 / 13.595) * tkev / rho
        boltex = np.exp(-53.859 / tkev) * xr
        exlim = np.exp(-54.403 / tkev) * xr

        tlog_arr = np.log(np.maximum(temp, 1e-10))
        he2_coulff = _coulff_grid(2, np.log(np.maximum(freq, 1e-300)), tlog_arr)
        he2_cont = np.vstack(
            [xkarsas_grid(freq, 4.0, n, n) for n in range(1, 10)]
        )

        for j in range(nfreq):
            f = freq[j]

            cont = he2_cont[:, j]

            freq3 = f ** 3
            cfree = 3.6919e8 / freq3 * 4.0
            c_const = 2.815e29 * 4.0 / freq3

            if f < 1.31522e14:
                ex = exlim / ehvkt[:, j]
            else:
                ex = boltex
            he2_val = (ex - exlim) * c_const

            for n in range(9):
                he2_val = he2_val + cont[n] * bolt[:, n]

            coulff_arr = he2_coulff[:, j]

            ahe2[:, j] = (he2_val + coulff_arr * cfree * freet) * stim[:, j]
            she2[:, j] = bnu_all[:, j]

    # HEMIOP: He- opacity (atlas7v.for line 7296-7318)
    # Fortran evidence:
    #   AHEMIN(J)=(A*T(J)+B+C/T(J))/1.D15*XNE(J)/1.D15*XNFPHE(J,1)/1.D15/RHO(J)
    # where A, B, C are frequency-dependent polynomials in 1/FREQ.
    if ifop[6] == 1 and atmosphere.xnf_he1 is not None and atmosphere.electron_density is not None:
        logger.debug("Computing HEMIOP (He- opacity)...")
        xnfphe = np.asarray(atmosphere.xnf_he1, dtype=np.float64)
        if xnfphe.ndim == 1:
            xnfphe = xnfphe[:, np.newaxis]
        xnfphe1 = xnfphe[:, 0] if xnfphe.shape[1] > 0 else np.ones(n_layers)
        xne = np.asarray(atmosphere.electron_density, dtype=np.float64)

        a_coeff = 3.397e-01 + (-5.216e14 + 7.039e30 / freq) / freq
        b_coeff = -4.116e03 + (1.067e19 + 8.135e34 / freq) / freq
        c_coeff = 5.081e08 + (-8.724e22 - 5.659e37 / freq) / freq
        ahemin[:, :] = (
            (
                a_coeff[np.newaxis, :] * temp[:, np.newaxis]
                + b_coeff[np.newaxis, :]
                + c_coeff[np.newaxis, :] / temp[:, np.newaxis]
            )
            / 1.0e15
            * xne[:, np.newaxis]
            / 1.0e15
            * xnfphe1[:, np.newaxis]
            / 1.0e15
            / rho[:, np.newaxis]
        )

    # HMINOP: H- opacity (atlas7v.for line 5212-5316)
    if ifop[2] == 1 and atmosphere.xnfph is not None and atmosphere.electron_density is not None:
        logger.debug("Computing HMINOP (H- opacity)...")
        xnfph_arr = np.asarray(atmosphere.xnfph, dtype=np.float64)
        xne = np.asarray(atmosphere.electron_density, dtype=np.float64)
        bhyd = atlas_tables.get("bhyd", np.ones((n_layers, 8), dtype=np.float64))
        bmin = atlas_tables.get("bmin", np.ones((n_layers,), dtype=np.float64))
        tkev = temp * KBOLTZ_EV

        # Pre-compute XHMIN (atlas7v.for line 5298-5299) - per layer, not per frequency
        # Fortran uses XNFPH(J,1) from POPS (mode=11). Prefer the explicit XNFPH array.
        if xnfph_arr.shape[1] > 0:
            xnfph1 = xnfph_arr[:, 0]
        elif atmosphere.xnf_h is not None:
            xnfph1 = compute_ground_state_hydrogen(
                np.asarray(atmosphere.xnf_h, dtype=np.float64), temp
            )
        else:
            xnfph1 = np.ones(n_layers)
        bhyd1 = bhyd[:, 0] if bhyd.shape[1] > 0 else np.ones(n_layers)
        xhmin = (
            np.exp(0.754209 / tkev)
            / (2.0 * 2.4148e15 * temp * np.sqrt(temp))
            * bmin
            * bhyd1
            * xnfph1
            * xne
        )

        # Pre-compute THETA = 5040/T (atlas7v.for line 5296) - per layer
        theta = 5040.0 / temp

        # Pre-compute WFFLOG = log(91.134/WAVEK) (atlas7v.for line 5284)
        wfflog = np.log(91.134 / HMINOP_WAVEK)

        # Pre-compute FFLOG (atlas7v.for line 5290-5291) - once for all frequencies
        nwavek = HMINOP_WAVEK.size
        nthetaff = HMINOP_THETAFF.size
        # FF is (11, 22) array: first 11 columns from FFBEG, last 11 from FFEND
        ff_full = np.zeros((nthetaff, 22), dtype=np.float64)
        for it in range(nthetaff):
            for iw in range(22):
                # Fortran tables are column-major; index as [iw, it] to match.
                if iw < 11:
                    ff_full[it, iw] = HMINOP_FFBEG[iw, it]
                else:
                    ff_full[it, iw] = HMINOP_FFEND[iw - 11, it]

        # Pre-compute FFLOG = log(FF/THETAFF * 5040 * K_BOLTZ)
        fflog = np.zeros((22, nthetaff), dtype=np.float64)
        for iw in range(22):
            for it in range(nthetaff):
                ff_val = ff_full[it, iw]
                fflog[iw, it] = np.log(ff_val / HMINOP_THETAFF[it] * 5040.0 * K_BOLTZ)

        # WAVE = 2.99792458e17 / FREQ (atlas7v.for line 5300), wavelength in nm.
        # WAVELOG matches Fortran exactly because WFFLOG = log(91.134/WAVEK) = log(wave_nm).
        wavelog = np.log(wavelength_nm)

        # Interpolate FFLOG over wavelength once for every THETA table row.
        fftt = np.empty((nthetaff, nfreq), dtype=np.float64)
        for it in range(nthetaff):
            fftt[it, :] = np.exp(_linter(wfflog, fflog[:, it], wavelog))

        # Interpolate FFTT over THETA for each layer.  The theta bracket is
        # layer-only, so reuse it across the full frequency grid.
        fftheta = np.empty((n_layers, nfreq), dtype=np.float64)
        for layer_idx in range(n_layers):
            iold = int(np.searchsorted(HMINOP_THETAFF, theta[layer_idx], side="right"))
            iold = max(1, min(iold, nthetaff - 1))
            denom = HMINOP_THETAFF[iold] - HMINOP_THETAFF[iold - 1]
            if abs(denom) < 1e-40:
                fftheta[layer_idx, :] = fftt[iold - 1, :]
            else:
                weight = (theta[layer_idx] - HMINOP_THETAFF[iold - 1]) / denom
                fftheta[layer_idx, :] = fftt[iold - 1, :] + (fftt[iold, :] - fftt[iold - 1, :]) * weight

        # HMINBF from MAP1 (atlas7v.for line 5306)
        hminbf = np.zeros(nfreq, dtype=np.float64)
        hminbf_active = freq > 1.82365e14
        if np.any(hminbf_active):
            from .josh_math import _map1

            hminbf[hminbf_active], _ = _map1(
                HMINOP_WBF,
                HMINOP_BF,
                wavelength_nm[hminbf_active],
            )

        # Compute H- opacity (atlas7v.for line 5309-5313)
        # HMINFF = FFTETA * XNFPH(J,1) * 2. * BHYD(J,1) * XNE(J) / RHO(J) * 1e-26
        hminff = fftheta * (xnfph1 * 2.0 * bhyd1 * xne / rho * 1e-26)[:, np.newaxis]

        # H = HMINBF * 1e-18 * (1. - EHVKT(J)/BMIN(J)) * XHMIN(J) / RHO(J)
        h_bf = (
            hminbf[np.newaxis, :]
            * 1e-18
            * (1.0 - ehvkt / np.maximum(bmin, 1e-40)[:, np.newaxis])
            * xhmin[:, np.newaxis]
            / rho[:, np.newaxis]
        )

        ahmin[:, :] = h_bf + hminff

        # Source function (atlas7v.for line 5313-5314)
        # SHMIN = (H * BNU(J) * STIM(J) / (BMIN(J) - EHVKT(J)) + HMINFF * BNU(J)) / AHMIN(J)
        denom = bmin[:, np.newaxis] - ehvkt
        h_bf_src = h_bf * bnu_all * stim / np.maximum(denom, 1e-40)
        shmin[:, :] = np.where(
            ahmin > 0, (h_bf_src + hminff * bnu_all) / ahmin, bnu_all
        )

    # Scattering subroutines
    # ELECOP: Electron scattering (atlas7v.for line 7806-7817) - Simple!
    if ifop[11] == 1 and atmosphere.electron_density is not None:
        logger.debug("Computing ELECOP (electron scattering)...")
        xne = np.asarray(atmosphere.electron_density, dtype=np.float64)
        # SIGEL = 0.6653e-24 * XNE / RHO (atlas7v.for line 7815)
        sigel[:, :] = (0.6653e-24 * xne / rho)[:, np.newaxis]

    # HRAYOP: Hydrogen Rayleigh scattering (atlas12.for lines 5989-6006)
    # atlas12 uses a simple polynomial cross-section, clamping freq at Lyman-alpha.
    xnfph1 = None
    if ifop[3] == 1 and atmosphere.xnf_h is not None:
        logger.debug("Computing HRAYOP (hydrogen Rayleigh scattering, atlas12)...")
        xnf_h_total = np.asarray(atmosphere.xnf_h, dtype=np.float64)
        xnfph1 = compute_ground_state_hydrogen(xnf_h_total, temp)
    elif ifop[3] == 1 and atmosphere.xnfph is not None:
        xnfph_arr = np.asarray(atmosphere.xnfph, dtype=np.float64)
        xnfph1 = xnfph_arr[:, 0] if xnfph_arr.shape[1] > 0 else np.ones(n_layers)

    if xnfph1 is not None:
        bhyd = atlas_tables.get("bhyd", np.ones((n_layers, 8), dtype=np.float64))
        bhyd1 = bhyd[:, 0] if bhyd.shape[1] > 0 else np.ones(n_layers)
        pop_hray = xnfph1 * 2.0 * bhyd1 / rho

        # atlas12.for line 6000: W = c_AA / DMIN1(FREQ, 2.463D15)
        w = 2.99792458e18 / np.minimum(freq, 2.463e15)
        ww = w * w
        sig = (5.799e-13 + 1.422e-6 / ww + 2.784 / (ww * ww)) / (ww * ww)
        sigh[:, :] = pop_hray[:, np.newaxis] * sig[np.newaxis, :]

    # HERAOP: Helium Rayleigh scattering (atlas7v.for line 5818-5832)
    # CRITICAL: Fortran only calls HERAOP if IFOP(8) == 1 (atlas7v.for line 4046)
    # Fortran's default is IFOP(8) = 0 (atlas7v.for line 2822: DATA IFOP/...,0,0,.../)
    # This means HERAOP is DISABLED by default in Fortran!
    if (
        ifop[7] == 1 and atmosphere.xnf_he1 is not None
    ):  # IFOP(8) in Fortran = ifop[7] in Python (0-indexed)
        logger.debug("Computing HERAOP (helium Rayleigh scattering)...")
        xnfphe = np.asarray(atmosphere.xnf_he1, dtype=np.float64)
        if xnfphe.ndim == 1:
            xnfphe = xnfphe[:, np.newaxis]  # Make it 2D
        bhe1 = atlas_tables.get("bhe1", np.ones((n_layers, 29), dtype=np.float64))

        # WAVE = 2.99792458e18 / min(FREQ, 5.15e15) (atlas7v.for line 5826)
        wave = 2.99792458e18 / np.minimum(freq, 5.15e15)
        ww = wave**2
        # SIG = 5.484e-14 / WW / WW * (1. + (2.44e5 + 5.94e10 / (WW - 2.90e5)) / WW)^2 (atlas7v.for line 5828)
        sig = (
            5.484e-14
            / (ww * ww)
            * (1.0 + (2.44e5 + 5.94e10 / np.maximum(ww - 2.90e5, 1e-10)) / ww) ** 2
        )
        xnfphe1 = xnfphe[:, 0] if xnfphe.shape[1] > 0 else np.ones(n_layers)
        bhe1_1 = bhe1[:, 0] if bhe1.shape[1] > 0 else np.ones(n_layers)
        sighe[:, :] = (xnfphe1 / rho * bhe1_1)[:, np.newaxis] * sig[np.newaxis, :]
    else:
        logger.debug("Skipping HERAOP (helium Rayleigh scattering) - IFOP(8)=0")

    # H2RAOP: H2 Rayleigh scattering (atlas12.for lines 9659-9694)
    # CRITICAL: Fortran only calls H2RAOP if IFOP(13) == 1
    if ifop[12] == 1 and xnfph1 is not None:
        logger.debug("Computing H2RAOP (H2 Rayleigh scattering)...")
        bhyd1 = bhyd[:, 0] if bhyd.shape[1] > 0 else np.ones(n_layers)

        # Compute XNH2 (H2 number density cm^-3) using the EQUILH2 tabulated
        # equilibrium constant, matching atlas12.for line 9684:
        #   XNH2(J) = (XNFP(J,1) * 2 * BHYD(J,1))^2 * EQUILH2(T(J))
        # The old polynomial from atlas7v.for is commented out in atlas12.for
        # (lines 9677-9683) and must NOT be used.
        xnh2 = _compute_xnh2_equilh2(
            temperature_k=temp,
            xnfp_h1=xnfph1,
            bhyd1=bhyd1,
        )
        # Zero out layers where T > 20000K (Fortran: IF(T(J).GT.20000.) GO TO 11)
        xnh2 = np.where(temp > 20000.0, 0.0, xnh2)

        # Wave in Angstrom, capped at frequency 2.922e15 Hz
        wave = 2.99792458e18 / np.minimum(freq, 2.922e15)
        ww = wave**2

        # Cross-section formula (atlas12.for line 9688)
        sig = (8.14e-13 + 1.28e-6 / ww + 1.61 / (ww * ww)) / (ww * ww)

        # atlas12.for line 9692: SIGH2(J) = SIG * XNH2(J) / RHO(J)
        sigh2[:, :] = (xnh2 / rho)[:, np.newaxis] * sig[np.newaxis, :]

        logger.debug("  SIGH2[0] at first freq: %.6e", sigh2[0, 0])
    else:
        logger.debug("Skipping H2RAOP (H2 Rayleigh scattering) - IFOP(13)=0 or no XNFPH")

    # XSOP: Dummy scattering (atlas7v.for line 8083-8091) - does nothing
    # sigx remains zeros

    # Metal opacities
    # C1OP: Carbon I opacity — atlas12.for lines 6441-6759.
    # Fortran: XNFP(J,21), 1-based → Python index 20.
    xnfpc = np.asarray(atmosphere.xnfp_all[:, 20], dtype=np.float64)
    if ifop[8] == 1 and np.any(xnfpc > 0):
        logger.debug("Computing C1OP (Carbon I opacity)...")
        _RYD = 109732.298
        _c1_elev = np.array([
            79314.86, 78731.27, 78529.62, 78309.76, 78226.35,
            77679.82, 73975.91, 72610.72, 71374.90, 70743.95,
            69722.00, 68856.33, 61981.82, 60373.00, 21648.01,
            10192.63, 43.42, 16.42, 0.00, 119878.,
            105798.7, 97878., 75254.93, 64088.85, 33735.20,
        ], dtype=np.float64)
        _c1_glev = np.array([
            9., 3., 7., 15., 21., 5., 1., 5., 9., 3., 15., 3.,
            3., 9., 1., 5., 5., 3., 1., 3., 3., 5., 12., 15., 5.,
        ], dtype=np.float64)

        # Boltzmann factors: bolt[i, k] = glev[i] * exp(-elev[i] * hckt[k])
        # hckt is shape (n_layers,) — from atmosphere
        bolt = _c1_glev[:, None] * np.exp(-_c1_elev[:, None] * hckt[None, :])

        for j in range(nfreq):
            f = freq[j]
            wno = waveno[j]
            stim_j = stim[:, j]

            # Fortran: IF(FREQ.GT.3.28805D15)RETURN  (Lyman limit guard)
            if f > 3.28805e15:
                continue

            z = 1.0
            freq3 = 2.815e29 / f / f / f * z**4

            # Compute cross-sections x[0..24]
            x = np.zeros(25)

            # --- Group 1: C II 2P average limit (levels 1-14) ---
            elim1 = 90862.70
            # Levels 1-6: 2s2 2p3d states, XKARSAS(FREQ, ZEFF2, 3, 2)
            for i in range(6):
                if wno < elim1 - _c1_elev[i]:
                    break
                zeff2 = 9.0 / _RYD * (elim1 - _c1_elev[i])
                x[i] = xkarsas(f, zeff2, 3, 2)
            # Levels 7-12: 2s2 2p3p states, XKARSAS(FREQ, ZEFF2, 3, 1)
            for i in range(6, 12):
                if wno < elim1 - _c1_elev[i]:
                    break
                zeff2 = 9.0 / _RYD * (elim1 - _c1_elev[i])
                x[i] = xkarsas(f, zeff2, 3, 1)
            # Levels 13-14: 2s2 2p3s states, XKARSAS(FREQ, ZEFF2, 3, 0)
            for i in range(12, 14):
                if wno < elim1 - _c1_elev[i]:
                    break
                zeff2 = 9.0 / _RYD * (elim1 - _c1_elev[i])
                x[i] = xkarsas(f, zeff2, 3, 0)

            # --- Group 2: C II 2P1/2 limit (levels 15-19), Luo & Pradhan ---
            elim2 = 90820.42
            # Level 15 (1S): Luo & Pradhan + Burke & Taylor resonance
            if wno >= elim2 - _c1_elev[14]:
                xs0 = 10.0 ** (-16.80 - (wno - elim2 + _c1_elev[14]) / 3.0 / _RYD)
                eps = (wno - 97700.0) * 2.0 / 2743.0
                xs1 = (68e-18 * eps + 118e-18) / (eps**2 + 1.0)
                x[14] = (xs0 + xs1) * 1.0 / 3.0
            # Level 16 (1D): Luo & Pradhan + two Burke & Taylor resonances
            if wno >= elim2 - _c1_elev[15]:
                xd0 = 10.0 ** (-16.80 - (wno - elim2 + _c1_elev[15]) / 3.0 / _RYD)
                eps1 = (wno - 93917.0) * 2.0 / 9230.0
                xd1 = (22e-18 * eps1 + 26e-18) / (eps1**2 + 1.0)
                eps2 = (wno - 111130.0) * 2.0 / 2743.0
                xd2 = (-10.5e-18 * eps2 + 46e-18) / (eps2**2 + 1.0)
                x[15] = (xd0 + xd1 + xd2) * 1.0 / 3.0
            # Levels 17-19 (3P2, 3P1, 3P0): Luo & Pradhan
            for i in range(16, 19):
                if wno >= elim2 - _c1_elev[i]:
                    x[i] = 10.0 ** (-16.80 - (wno - elim2 + _c1_elev[i]) / 3.0 / _RYD) * 1.0 / 3.0

            # --- Group 2b: C II 2P3/2 limit (levels 15-19), 2/3 weight added ---
            elim2b = 90820.42 + 63.42
            if wno >= elim2b - _c1_elev[14]:
                xs0 = 10.0 ** (-16.80 - (wno - elim2b + _c1_elev[14]) / 3.0 / _RYD)
                eps = (wno - 97700.0) * 2.0 / 2743.0
                xs1 = (68e-18 * eps + 118e-18) / (eps**2 + 1.0)
                x[14] += (xs0 + xs1) * 2.0 / 3.0
            if wno >= elim2b - _c1_elev[15]:
                xd0 = 10.0 ** (-16.80 - (wno - elim2b + _c1_elev[15]) / 3.0 / _RYD)
                eps1 = (wno - 93917.0) * 2.0 / 9230.0
                xd1 = (22e-18 * eps1 + 26e-18) / (eps1**2 + 1.0)
                eps2 = (wno - 111130.0) * 2.0 / 2743.0
                xd2 = (-10.5e-18 * eps2 + 46e-18) / (eps2**2 + 1.0)
                x[15] += (xd0 + xd1 + xd2) * 2.0 / 3.0
            for i in range(16, 19):
                if wno >= elim2b - _c1_elev[i]:
                    x[i] += 10.0 ** (-16.80 - (wno - elim2b + _c1_elev[i]) / 3.0 / _RYD) * 2.0 / 3.0

            # --- Group 3: C II 4P1/2 limit (levels 20-25) ---
            elim3 = 90820.42 + 43003.3
            degen = 3.0
            for i in range(19, 25):
                if wno < elim3 - _c1_elev[i]:
                    break
                zeff2 = 4.0 / _RYD * (elim3 - _c1_elev[i])
                x[i] = xkarsas(f, zeff2, 2, 1) * degen

            # --- Kramers-Gaunt n>=4 free-free approximation ---
            elim_ff = 90820.42
            gfactor = 6.0
            ryd_z2_over_16 = _RYD * z**2 / 16.0
            kramers_exp_arg_min = max(elim_ff - ryd_z2_over_16, elim_ff - wno)
            # h_kramers[k] = freq3*gfactor*2/2/(RYD*Z^2*HCKT(k)) * (exp(-...) - exp(-elim*HCKT(k)))
            h_kramers = (freq3 * gfactor / (_RYD * z**2 * hckt)
                         * (np.exp(-kramers_exp_arg_min * hckt)
                            - np.exp(-elim_ff * hckt)))

            # Sum: H = kramers + sum(x[i] * bolt[i, :])
            h = h_kramers + np.dot(x, bolt)

            # Final: AC1(J) = H * XNFP(J,21) * STIM(J) / RHO(J)
            ac1[:, j] = h * xnfpc * stim_j / rho
            sc1[:, j] = bnu_all[:, j]

    # MG1OP: Magnesium I opacity — atlas12.for lines 6768-6896.
    # Fortran: XNFP(J,78), 1-based → Python index 77.
    xnfpmg = np.asarray(atmosphere.xnfp_all[:, 77], dtype=np.float64)
    if ifop[8] == 1 and np.any(xnfpmg > 0):
        logger.debug("Computing MG1OP (Magnesium I opacity)...")
        _RYD_MG = 109732.298
        _mg1_elev = np.array([
            54676.710, 54676.438, 54192.284, 53134.642, 49346.729,
            47957.034, 47847.797, 46403.065, 43503.333, 41197.043,
            35051.264, 21919.178, 21870.464, 21850.405, 0.,
        ], dtype=np.float64)
        _mg1_glev = np.array([
            21., 7., 15., 5., 3., 15., 9., 5., 1., 3., 3., 5., 3., 1., 1.,
        ], dtype=np.float64)

        bolt_mg = _mg1_glev[:, None] * np.exp(-_mg1_elev[:, None] * hckt[None, :])

        for j in range(nfreq):
            f = freq[j]
            wno = waveno[j]
            stim_j = stim[:, j]

            if f > 3.28805e15:
                continue

            z = 1.0
            freq3 = 2.815e29 / f / f / f * z**4

            x = np.zeros(15)
            elim = 61671.02

            # Levels 1-2: 3s4f states, XKARSAS(FREQ, ZEFF2, 4, 3)
            for i in range(2):
                if wno < elim - _mg1_elev[i]:
                    break
                zeff2 = 16.0 / _RYD_MG * (elim - _mg1_elev[i])
                x[i] = xkarsas(f, zeff2, 4, 3)
            # Levels 3-4: 3s4d states, XKARSAS(FREQ, ZEFF2, 4, 2)
            for i in range(2, 4):
                if wno < elim - _mg1_elev[i]:
                    break
                zeff2 = 16.0 / _RYD_MG * (elim - _mg1_elev[i])
                x[i] = xkarsas(f, zeff2, 4, 2)
            # Level 5: 3s4p 1P, XKARSAS(FREQ, ZEFF2, 4, 1)
            if wno >= elim - _mg1_elev[4]:
                zeff2 = 16.0 / _RYD_MG * (elim - _mg1_elev[4])
                x[4] = xkarsas(f, zeff2, 4, 1)
            # Level 6: 3s3d 3D — analytical
            if wno >= elim - _mg1_elev[5]:
                x[5] = 25e-18 * (13713.986 / wno) ** 2.7
            # Level 7: 3s4p 3P — analytical
            if wno >= elim - _mg1_elev[6]:
                x[6] = 33.8e-18 * (13823.223 / wno) ** 2.8
            # Level 8: 3s3d 1D — analytical
            if wno >= elim - _mg1_elev[7]:
                x[7] = 45e-18 * (15267.955 / wno) ** 2.7
            # Level 9: 3s4s 1S — analytical
            if wno >= elim - _mg1_elev[8]:
                x[8] = 0.43e-18 * (18167.687 / wno) ** 2.6
            # Level 10: 3s4s 3S — analytical
            if wno >= elim - _mg1_elev[9]:
                x[9] = 2.1e-18 * (20473.617 / wno) ** 2.6
            # Level 11: 3s3p 1P — analytical (two-term)
            if wno >= elim - _mg1_elev[10]:
                x[10] = 16e-18 * (26619.756 / wno) ** 2.1 - 7.8e-18 * (26619.756 / wno) ** 9.5
            # Levels 12-14: 3s3p 3P (three J sub-levels) — analytical with MAX
            for i in range(11, 14):
                if wno >= elim - _mg1_elev[i]:
                    xval = 20e-18 * (39759.842 / wno) ** 2.7
                    x[i] = max(xval, 40e-18 * (39759.842 / wno) ** 14)
            # Level 15: 3s2 1S — analytical
            if wno >= elim - _mg1_elev[14]:
                x[14] = 1.1e-18 * ((elim - _mg1_elev[14]) / wno) ** 10

            # Kramers-Gaunt n>=5 free-free
            gfactor = 2.0
            ryd_z2_over_25 = _RYD_MG * z**2 / 25.0
            kff_arg = max(elim - ryd_z2_over_25, elim - wno)
            h_kramers = (freq3 * gfactor / (_RYD_MG * z**2 * hckt)
                         * (np.exp(-kff_arg * hckt) - np.exp(-elim * hckt)))

            h = h_kramers + np.dot(x, bolt_mg)

            amg1[:, j] = h * xnfpmg * stim_j / rho
            smg1[:, j] = bnu_all[:, j]

    # FE1OP: Iron I opacity (atlas7v.for line 6623-6665) - simpler structure
    # Fortran: FE1OP uses XNFP(J,351) — POPSALL slot 351, 1-based → index 350.
    xnfpfe = np.asarray(atmosphere.xnfp_all[:, 350], dtype=np.float64)
    if ifop[8] == 1 and np.any(xnfpfe > 0):
        logger.debug("Computing FE1OP (Iron I opacity)...")
        bfe1 = atlas_tables.get("bfe1", np.ones((n_layers, 15), dtype=np.float64))
        bsi1 = atlas_tables.get("bsi1", np.ones((n_layers, 11), dtype=np.float64))

        # FE1OP uses arrays for transitions (atlas7v.for line 6635-6650)
        fe1_g = np.array(
            [
                25.0,
                35.0,
                21.0,
                15.0,
                9.0,
                35.0,
                33.0,
                21.0,
                27.0,
                49.0,
                9.0,
                21.0,
                27.0,
                9.0,
                9.0,
                25.0,
                33.0,
                15.0,
                35.0,
                3.0,
                5.0,
                11.0,
                15.0,
                13.0,
                15.0,
                9.0,
                21.0,
                15.0,
                21.0,
                25.0,
                35.0,
                9.0,
                5.0,
                45.0,
                27.0,
                21.0,
                15.0,
                21.0,
                15.0,
                25.0,
                21.0,
                35.0,
                5.0,
                15.0,
                45.0,
                35.0,
                55.0,
                25.0,
            ],
            dtype=np.float64,
        )

        fe1_e = np.array(
            [
                500.0,
                7500.0,
                12500.0,
                17500.0,
                19000.0,
                19500.0,
                19500.0,
                21000.0,
                22000.0,
                23000.0,
                23000.0,
                24000.0,
                24000.0,
                24500.0,
                24500.0,
                26000.0,
                26500.0,
                26500.0,
                27000.0,
                27500.0,
                28500.0,
                29000.0,
                29500.0,
                29500.0,
                29500.0,
                30000.0,
                31500.0,
                31500.0,
                33500.0,
                33500.0,
                34000.0,
                34500.0,
                34500.0,
                35000.0,
                35500.0,
                37000.0,
                37000.0,
                37000.0,
                38500.0,
                40000.0,
                40000.0,
                41000.0,
                41000.0,
                43000.0,
                43000.0,
                43000.0,
                43000.0,
                44000.0,
            ],
            dtype=np.float64,
        )

        fe1_wno = np.array(
            [
                63500.0,
                58500.0,
                53500.0,
                59500.0,
                45000.0,
                44500.0,
                44500.0,
                43000.0,
                58000.0,
                41000.0,
                54000.0,
                40000.0,
                40000.0,
                57500.0,
                55500.0,
                38000.0,
                57500.0,
                57500.0,
                37000.0,
                54500.0,
                53500.0,
                55000.0,
                34500.0,
                34500.0,
                34500.0,
                34000.0,
                32500.0,
                32500.0,
                32500.0,
                32500.0,
                32000.0,
                29500.0,
                29500.0,
                31000.0,
                30500.0,
                29000.0,
                27000.0,
                54000.0,
                27500.0,
                24000.0,
                47000.0,
                23000.0,
                44000.0,
                42000.0,
                42000.0,
                21000.0,
                42000.0,
                42000.0,
            ],
            dtype=np.float64,
        )

        # FE1OP processes all frequencies; individual transitions are checked inside.
        active = waveno >= 21000.0
        if np.any(active):
            h = np.zeros((n_layers, nfreq), dtype=np.float64)
            # Sum contributions from all transitions (atlas7v.for line 6655-6660)
            for i in range(len(fe1_wno)):
                use = fe1_wno[i] <= waveno
                if not np.any(use):
                    continue
                xsect = np.zeros(nfreq, dtype=np.float64)
                xsect[use] = 3e-18 / (
                    1.0 + ((fe1_wno[i] + 3000.0 - waveno[use]) / fe1_wno[i] / 0.1) ** 4
                )
                h += (
                    fe1_g[i]
                    * np.exp(-fe1_e[i] * hckt)[:, np.newaxis]
                    * xsect[np.newaxis, :]
                )

            # Fortran: COOLOP assembles FE1OP(J)*XNFP(J,351)*STIM(J)/RHO(J)
            afe1[:, active] = (
                h[:, active]
                * stim[:, active]
                * xnfpfe[:, np.newaxis]
                / rho[:, np.newaxis]
            )

            # BFUDGE = BSI1(J,1) (atlas7v.for line 6652)
            bfudge = bsi1[:, 0] if bsi1.shape[1] > 0 else np.ones(n_layers)
            sfe1[:, active] = (
                bnu_all[:, active]
                * stim[:, active]
                / np.maximum(bfudge[:, np.newaxis] - ehvkt[:, active], 1e-40)
            )

    # AL1OP: Aluminum I opacity — atlas12.for lines 6897-6912.
    # In atlas12, AL1OP is a FUNCTION (not subroutine), returning cross-section * partition.
    # COOLOP multiplies by XNFP(J,91)*STIM(J)/RHO(J).
    # Fortran: XNFP(J,91), 1-based → Python index 90.
    xnfpal = np.asarray(atmosphere.xnfp_all[:, 90], dtype=np.float64)
    if ifop[8] == 1 and np.any(xnfpal > 0):
        logger.debug("Computing AL1OP (Aluminum I opacity)...")
        _al1_elim = 48278.37

        active = freq <= 3.28805e15
        if np.any(active):
            al1op_val = np.zeros(nfreq, dtype=np.float64)
            # 3s2 3p 2P3/2 edge at (ELIM - 112.061) cm^-1
            use = active & (waveno >= _al1_elim - 112.061)
            al1op_val[use] = 6.5e-17 * ((_al1_elim - 112.061) / waveno[use]) ** 5 * 4.0
            # 3s2 3p 2P1/2 edge at ELIM cm^-1
            use = active & (waveno >= _al1_elim)
            al1op_val[use] += 6.5e-17 * (_al1_elim / waveno[use]) ** 5 * 2.0

            # COOLOP: AAL1(J) = AL1OP(J) * XNFP(J,91) * STIM(J) / RHO(J)
            aal1[:, active] = (
                xnfpal[:, np.newaxis]
                * stim[:, active]
                / rho[:, np.newaxis]
                * al1op_val[active][np.newaxis, :]
            )
            sal1[:, active] = bnu_all[:, active]

    # SI1OP: Silicon I opacity — atlas12.for lines 6913-7342.
    # Fortran: XNFP(J,105), 1-based → Python index 104.
    xnfpsi = np.asarray(atmosphere.xnfp_all[:, 104], dtype=np.float64)
    if ifop[8] == 1 and np.any(xnfpsi > 0):
        logger.debug("Computing SI1OP (Silicon I opacity)...")
        _RYD_SI = 109732.298
        _si1_elev = np.array([
            59962.284, 59100., 59077.112, 58893.40, 58801.529,
            58777., 57488.974, 56503.346, 54225.621, 53387.34,
            53362.24, 51612.012, 50533.424, 50189.389, 49965.894,
            49399.670, 49128.131, 48161.459, 47351.554, 47284.061,
            40991.884, 39859.920, 15394.370, 6298.850, 223.157,
            77.115, 0.000, 94000., 79664.0, 72000.,
            56698.738, 45303.310, 33326.053,
        ], dtype=np.float64)
        _si1_glev = np.array([
            9., 56., 15., 7., 3., 28., 21., 5., 15., 3., 7., 1., 9., 5., 21.,
            3., 9., 15., 5., 3., 3., 9., 1., 5., 5., 3., 1., 3., 3., 5., 12., 15., 5.,
        ], dtype=np.float64)
        # XKARSAS (n, l) per level: see atlas12.for lines 6949-7080
        _si1_nl = [
            (4, 2), (4, 3), (4, 2), (4, 2), (4, 2),  # 1-5
            (4, 3), (4, 2), (4, 2),                     # 6-8
            (3, 2), (3, 2), (3, 2),                     # 9-11
            (4, 1), (3, 2), (4, 1), (3, 2),             # 12-15
            (4, 1), (4, 1), (4, 1), (3, 2), (4, 1),    # 16-20
            (4, 0), (4, 0),                              # 21-22
        ]
        _si1_zeff_fac = [
            16., 16., 16., 16., 16.,  # 1-5
            16., 16., 16.,            # 6-8
            9., 9., 9.,              # 9-11
            16., 9., 16., 9.,        # 12-15
            16., 16., 16., 9., 16.,  # 16-20
            16., 16.,                 # 21-22
        ]

        bolt_si = _si1_glev[:, None] * np.exp(-_si1_elev[:, None] * hckt[None, :])

        for j in range(nfreq):
            f = freq[j]
            wno = waveno[j]
            stim_j = stim[:, j]

            if f > 3.28805e15:
                continue

            z = 1.0
            freq3 = 2.815e29 / f / f / f * z**4

            x = np.zeros(33)

            # --- Group 1: Si II 2P avg limit (levels 1-22) — XKARSAS ---
            elim1 = 65939.18
            for i in range(22):
                if wno < elim1 - _si1_elev[i]:
                    break
                zeff2 = _si1_zeff_fac[i] / _RYD_SI * (elim1 - _si1_elev[i])
                n_qn, l_qn = _si1_nl[i]
                x[i] = xkarsas(f, zeff2, n_qn, l_qn)

            # --- Group 2: Si II 2P1/2 limit (levels 23-27) — Nahar & Pradhan ---
            elim2 = 65747.55
            # Level 23 (1S): Nahar & Pradhan + resonance
            if wno >= elim2 - _si1_elev[22]:
                eps = (wno - 70000.0) * 2.0 / 6500.0
                reson1 = (97e-18 * eps + 94e-18) / (eps**2 + 1.0)
                x[22] = (37e-18 * (50353.180 / wno) ** 2.40 + reson1) / 3.0
            # Level 24 (1D): Nahar & Pradhan + resonance
            if wno >= elim2 - _si1_elev[23]:
                eps = (wno - 78600.0) * 2.0 / 13000.0
                reson1 = (-10e-18 * eps + 77e-18) / (eps**2 + 1.0)
                x[23] = (24.5e-18 * (59448.700 / wno) ** 1.85 + reson1) / 3.0
            # Level 25 (3P2): Nahar & Pradhan (two-part power law)
            if wno >= elim2 - _si1_elev[24]:
                if wno <= 74000.0:
                    x[24] = 72e-18 * (65524.393 / wno) ** 1.90 / 3.0
                else:
                    x[24] = 93e-18 * (65524.393 / wno) ** 4.00 / 3.0
            # Level 26 (3P1)
            if wno >= elim2 - _si1_elev[25]:
                if wno <= 74000.0:
                    x[25] = 72e-18 * (65524.393 / wno) ** 1.90 * 2.0 / 3.0
                else:
                    x[25] = 93e-18 * (65524.393 / wno) ** 4.00 * 2.0 / 3.0
            # Level 27 (3P0)
            if wno >= elim2 - _si1_elev[26]:
                if wno <= 74000.0:
                    x[26] = 72e-18 * (65524.393 / wno) ** 1.90 / 3.0
                else:
                    x[26] = 93e-18 * (65524.393 / wno) ** 4.00 / 3.0

            # --- Group 2b: Si II 2P3/2 limit (levels 23-27), 2/3 weight ---
            elim2b = 65747.55 + 287.45
            if wno >= elim2b - _si1_elev[22]:
                eps = (wno - 70000.0) * 2.0 / 6500.0
                reson1 = (97e-18 * eps + 94e-18) / (eps**2 + 1.0)
                x[22] += (37e-18 * (50353.180 / wno) ** 2.40 + reson1) * 2.0 / 3.0
            if wno >= elim2b - _si1_elev[23]:
                eps = (wno - 78600.0) * 2.0 / 13000.0
                reson1 = (-10e-18 * eps + 77e-18) / (eps**2 + 1.0)
                x[23] += (24.5e-18 * (59448.700 / wno) ** 1.85 + reson1) * 2.0 / 3.0
            if wno >= elim2b - _si1_elev[24]:
                if wno <= 74000.0:
                    x[24] += 72e-18 * (65524.393 / wno) ** 1.90 * 2.0 / 3.0
                else:
                    x[24] += 93e-18 * (65524.393 / wno) ** 4.00 * 2.0 / 3.0
            if wno >= elim2b - _si1_elev[25]:
                if wno <= 74000.0:
                    x[25] += 72e-18 * (65524.393 / wno) ** 1.90 * 2.0 / 3.0
                else:
                    x[25] += 93e-18 * (65524.393 / wno) ** 4.00 * 2.0 / 3.0
            if wno >= elim2b - _si1_elev[26]:
                if wno <= 74000.0:
                    x[26] += 72e-18 * (65524.393 / wno) ** 1.90 * 2.0 / 3.0
                else:
                    x[26] += 93e-18 * (65524.393 / wno) ** 4.00 * 2.0 / 3.0

            # --- Group 3: Si II 4P1/2 limit (levels 28-33) — XKARSAS ---
            elim3 = 65747.5 + 42824.35
            degen_si = 3.0
            for i in range(27, 33):
                if wno < elim3 - _si1_elev[i]:
                    break
                zeff2 = 9.0 / _RYD_SI * (elim3 - _si1_elev[i])
                x[i] = xkarsas(f, zeff2, 3, 1) * degen_si

            # Kramers-Gaunt n>=5 free-free
            elim_ff = 65747.55
            gfactor = 6.0
            ryd_z2_25 = _RYD_SI * z**2 / 25.0
            kff_arg = max(elim_ff - ryd_z2_25, elim_ff - wno)
            h_kramers = (freq3 * gfactor / (_RYD_SI * z**2 * hckt)
                         * (np.exp(-kff_arg * hckt) - np.exp(-elim_ff * hckt)))

            h = h_kramers + np.dot(x, bolt_si)

            asi1[:, j] = h * xnfpsi * stim_j / rho
            ssi1[:, j] = bnu_all[:, j]

    # LUKEOP: Lukewarm star opacity (atlas7v.for line 8952-8977)
    # Computes: N1OP, O1OP, MG2OP, SI2OP, CA2OP
    # Only computed if IFOP(10) = 1
    if ifop[9] == 1:  # IFOP(10) in Fortran = ifop[9] in Python (0-indexed)
        logger.debug("Computing LUKEOP (N1, O1, Mg2, Si2, Ca2 opacity)...")
        xnfp_all = np.asarray(atmosphere.xnfp_all, dtype=np.float64)

        def _xnfp_stage(start_1based: int, ion_stage_1based: int) -> np.ndarray:
            idx = start_1based + ion_stage_1based - 2  # to 0-based
            if 0 <= idx < xnfp_all.shape[1]:
                return xnfp_all[:, idx]
            return np.zeros(n_layers, dtype=np.float64)

        # atlas12 POPSALL slots (1-based):
        # N:28, O:36, Mg:78, Si:105, Ca:210
        global _SI2OP12_XTAB
        if _SI2OP12_XTAB is None:
            _SI2OP12_XTAB = _build_si2op12_xtab()

        xnfpn = _xnfp_stage(28, 1)
        xnfpo = _xnfp_stage(36, 1)
        xnfpc2 = _xnfp_stage(21, 2)
        xnfpmg2 = _xnfp_stage(78, 2)
        xnfpsi2 = _xnfp_stage(105, 2)
        xnfpca2 = _xnfp_stage(210, 2)

        tkev = KBOLTZ_EV * temp  # eV (matches Fortran TKEV(J) = 8.6171D-5 * T(J))
        # MG2OP constants from atlas12.for SUBROUTINE MG2OP.
        mg2_elev = np.array(
            [
                112197.0,
                108900.0,
                103705.66,
                103689.89,
                103419.82,
                97464.32,
                92790.51,
                93799.70,
                93310.80,
                80639.85,
                69804.95,
                71490.54,
                35730.36,
                0.0,
            ],
            dtype=np.float64,
        )
        mg2_glev = np.array(
            [98.0, 72.0, 18.0, 14.0, 10.0, 6.0, 2.0, 14.0, 10.0, 6.0, 2.0, 10.0, 6.0, 2.0],
            dtype=np.float64,
        )
        mg2_nl = [
            (7, 7),
            (6, 6),
            (5, 4),
            (5, 3),
            (5, 2),
            (5, 1),
            (5, 0),
            (4, 3),
            (4, 2),
            (4, 1),  # Fortran fix (was 4,2 in old code)
            (4, 0),
            (3, 2),
            (3, 1),
        ]
        mg2_zeff_num = np.array([49.0, 36.0, 25.0, 25.0, 25.0, 25.0, 25.0, 16.0, 16.0, 16.0, 16.0, 9.0, 9.0])
        mg2_elim = 121267.61
        mg2_ryd = 109732.298
        mg2_z = 2.0
        mg2_bolt = mg2_glev[:, None] * np.exp(-mg2_elev[:, None] * hckt[None, :])
        mg2_exp2 = np.exp(-mg2_elim * hckt)
        mg2_kthresh = mg2_elim - mg2_ryd * mg2_z**2 / (8.0**2)
        # C2OP constants from atlas12.for SUBROUTINE C2OP.
        c2_elev = np.array(
            [
                179073.05, 178955.94, 178495.47, 175292.30, 173347.84, 168978.34, 168124.17, 162522.34, 157234.07, 145550.1,
                131731.8, 116537.65, 42.28, 202188.07, 199965.31, 198856.92, 198431.96, 196572.80, 195786.71, 190000.0,
                188601.54, 186452.13, 184690.98, 182036.89, 181741.65, 177787.22, 167009.29, 110651.76, 96493.74, 74931.11,
                43035.8, 230407.2, 150464.6, 142027.1,
            ],
            dtype=np.float64,
        )
        c2_glev = np.array(
            [18.0, 14.0, 10.0, 6.0, 2.0, 14.0, 10.0, 6.0, 1.0, 10.0, 6.0, 1.0, 3.0, 6.0, 10.0, 12.0, 10.0, 20.0, 28.0, 2.0,
             10.0, 12.0, 4.0, 6.0, 20.0, 6.0, 12.0, 6.0, 2.0, 10.0, 12.0, 6.0, 10.0, 4.0],
            dtype=np.float64,
        )
        c2_bolt = c2_glev[:, None] * np.exp(-c2_elev[:, None] * hckt[None, :])
        c2_ryd = 109732.298
        c2_z = 2.0
        c2_freq_factor = 2.815e29 * c2_z**4
        c2_elim1 = 196664.7
        c2_elim2 = c2_elim1 + 52367.06
        c2_elim3 = c2_elim1 + 137425.70
        c2_a = np.exp(-c2_elim1 * hckt)
        c2_b = np.exp(-c2_elim2 * hckt)
        c2_k1 = c2_elim1 - c2_ryd * c2_z**2 / (6.0**2)
        c2_k2 = c2_elim2 - c2_ryd * c2_z**2 / (4.0**2)
        tlog10 = np.log(np.maximum(temp, 1e-300)) / _FORTRAN_LN10
        si2_it = np.clip(((tlog10 - 3.48) / 0.02).astype(np.int64), 1, 50)
        si2_tfrac = (tlog10 - 3.48 - si2_it * 0.02) / 0.02
        si2_boltn = (
            (1.0 * np.exp(-131838.4 * hckt) + 9.0 * np.exp(-184563.09 * hckt))
            / (109732.298 * 4.0 * hckt)
        )

        for j in range(nfreq):
            f = freq[j]
            freqlg = np.log(f)
            stim_j = stim[:, j]

            # N1OP: Nitrogen I opacity (atlas7v.for line 8978-9005)
            # Uses SEATON cross-sections at 3 edges: 853Å, 1020Å, 1130Å
            c1130 = 6.0 * np.exp(-3.575 / tkev)  # Level population factor
            c1020 = 10.0 * np.exp(-2.384 / tkev)

            x853 = 0.0
            x1020 = 0.0
            x1130 = 0.0

            if f >= 3.517915e15:  # 853 Å edge
                x853 = _seaton(3.517915e15, 1.142e-17, 2.0, 4.29, f)
            if f >= 2.941534e15:  # 1020 Å edge
                x1020 = _seaton(2.941534e15, 4.41e-18, 1.5, 3.85, f)
            if f >= 2.653317e15:  # 1130 Å edge
                x1130 = _seaton(2.653317e15, 4.2e-18, 1.5, 4.34, f)

            n1op = x853 * 4.0 + x1020 * c1020 + x1130 * c1130  # (n_layers,)

            # O1OP: Oxygen I opacity (atlas7v.for line 9006-9019)
            x911 = 0.0
            if f >= 3.28805e15:  # 911 Å edge
                x911 = _seaton(3.28805e15, 2.94e-18, 1.0, 2.66, f)
            o1op = x911 * 9.0  # scalar, broadcast to layers

            mg2_x = np.zeros(14, dtype=np.float64)
            for i in range(13):
                if waveno[j] < mg2_elim - mg2_elev[i]:
                    break
                zeff2 = mg2_zeff_num[i] / mg2_ryd * (mg2_elim - mg2_elev[i])
                n_qn, l_qn = mg2_nl[i]
                mg2_x[i] = xkarsas(f, zeff2, n_qn, l_qn)
            if waveno[j] >= mg2_elim - mg2_elev[13]:
                ratio = (mg2_elim - mg2_elev[13]) / max(waveno[j], 1e-300)
                mg2_x[13] = 0.14e-18 * (6.700 * ratio**4 - 5.700 * ratio**5)
            mg2_freq3 = 2.815e29 / max(f * f * f, 1e-300) * mg2_z**4
            mg2_pref = mg2_freq3 / (mg2_ryd * mg2_z**2 * hckt)
            mg2_h = mg2_pref * (
                np.exp(-np.maximum(mg2_kthresh, mg2_elim - waveno[j]) * hckt)
                - mg2_exp2
            )
            mg2_h = mg2_h + np.dot(mg2_x, mg2_bolt)
            mg2op = mg2_h * xnfpmg2 * stim_j / rho

            # C2OP: atlas12.for SUBROUTINE C2OP (lines 8297-8570)
            c2_x = np.zeros(34, dtype=np.float64)
            for i, (n_qn, l_qn) in enumerate([(5, 4), (5, 3), (5, 2), (5, 1), (5, 0), (4, 3), (4, 2), (4, 1), (4, 0), (3, 2), (3, 1), (3, 0)]):
                if waveno[j] < c2_elim1 - c2_elev[i]:
                    break
                zeff2 = (25.0 if n_qn == 5 else 16.0 if n_qn == 4 else 9.0) / c2_ryd * (c2_elim1 - c2_elev[i])
                c2_x[i] = xkarsas(f, zeff2, n_qn, l_qn)
            for i in range(13, 19):
                if waveno[j] < c2_elim2 - c2_elev[i]:
                    break
                c2_x[i] = xkarsas(f, 9.0 / c2_ryd * (c2_elim2 - c2_elev[i]), 3, 2)
            for i in range(19, 25):
                if waveno[j] < c2_elim2 - c2_elev[i]:
                    break
                c2_x[i] = xkarsas(f, 9.0 / c2_ryd * (c2_elim2 - c2_elev[i]), 3, 1)
            for i in range(25, 27):
                if waveno[j] < c2_elim2 - c2_elev[i]:
                    break
                c2_x[i] = xkarsas(f, 9.0 / c2_ryd * (c2_elim2 - c2_elev[i]), 3, 0)
            for i in range(31, 34):
                if waveno[j] < c2_elim3 - c2_elev[i]:
                    break
                c2_x[i] = 3.0 * xkarsas(f, 4.0 / c2_ryd * (c2_elim3 - c2_elev[i]), 2, 1)
            c2_freq3 = c2_freq_factor / max(f * f * f, 1e-300)
            c2_h = c2_freq3 / (c2_ryd * c2_z**2 * hckt) * (
                np.exp(-np.maximum(c2_k1, c2_elim1 - waveno[j]) * hckt) - c2_a
            )
            c2_h = c2_h + c2_freq3 * 9.0 / (c2_ryd * c2_z**2 * hckt) * (
                np.exp(-np.maximum(c2_k2, c2_elim2 - waveno[j]) * hckt) - c2_b
            )
            c2_h = c2_h + np.dot(c2_x, c2_bolt)
            c2op = c2_h * xnfpc2 * stim_j / rho

            # SI2OP: atlas12.for SUBROUTINE SI2OP (lines 8745-9177)
            if waveno[j] >= 12192.48:
                iw = int(waveno[j] * 0.001)
                iw = max(min(iw, 199), 1)
                wfrac = (waveno[j] - iw * 1000.0) / 1000.0
                i0 = iw - 1  # Fortran IW (1-based) -> Python 0-based
                i1 = iw      # Fortran IW+1
                it0 = si2_it - 1  # Fortran IT
                it1 = si2_it      # Fortran IT+1
                h00 = _SI2OP12_XTAB[i0, it0]
                h01 = _SI2OP12_XTAB[i0, it1]
                h10 = _SI2OP12_XTAB[i1, it0]
                h11 = _SI2OP12_XTAB[i1, it1]
                h0 = h00 * (1.0 - si2_tfrac) + h01 * si2_tfrac
                h1 = h10 * (1.0 - si2_tfrac) + h11 * si2_tfrac
                si2op = np.exp(h0 * (1.0 - wfrac) + h1 * wfrac) * xnfpsi2 * stim_j / rho
            else:
                si2_freq3 = 2.815e29 * (2.0**4) / max(f * f * f, 1e-300)
                si2_h = si2_freq3 * (1.0 / np.maximum(ehvkt[:, j], 1e-300) - 1.0) * si2_boltn
                si2op = si2_h * xnfpsi2 * stim_j / rho

            # CA2OP: Calcium II opacity (atlas7v.for line 9098-9122)
            c1218 = 10.0 * np.exp(-1.697 / tkev)
            c1420 = 6.0 * np.exp(-3.142 / tkev)
            x1044 = 0.0
            x1218 = 0.0
            x1420 = 0.0
            if f >= 2.870454e15:  # 1044 Å edge
                x1044 = 5.4e-20 * (2.870454e15 / f) ** 3
            if f >= 2.460127e15:  # 1218 Å edge
                x1218 = 1.64e-17 * np.sqrt(2.460127e15 / f)
            if f >= 2.110779e15:  # 1420 Å edge
                x1420 = _seaton(2.110779e15, 4.13e-18, 3.0, 0.69, f)
            ca2op = x1044 * 2.0 + x1218 * c1218 + x1420 * c1420  # (n_layers,)

            # Match Fortran LUKEOP weighting by ion populations.
            luke_n1 = n1op * xnfpn * stim_j / rho
            luke_o1 = o1op * xnfpo * stim_j / rho
            luke_ca2 = ca2op * xnfpca2 * stim_j / rho
            luke_c2 = c2op
            luke_mg2 = mg2op
            luke_si2 = si2op
            aluke[:, j] = luke_n1 + luke_o1 + luke_ca2 + luke_c2 + luke_mg2 + luke_si2
            if trace_enabled():
                wv = float(wavelength_nm[j])
                iwlk = int(np.log(max(wv, 1e-300)) / np.log(1.0 + 1.0 / 2_000_000.0) + 0.5)
                for k in range(n_layers):
                    if not trace_in_focus(wlvac_nm=wv, j0=k):
                        continue
                    trace_emit(
                        event="luke_terms1",
                        iter_num=1,
                        line_num_1b=0,
                        depth_1b=k + 1,
                        nu_1b=iwlk,
                        type_code=0,
                        wlvac_nm=wv,
                        center=float(luke_n1[k]),
                        adamp=float(luke_o1[k]),
                        cv=float(luke_ca2[k]),
                        tabcont=float(aluke[k, j]),
                        branch="lukeop",
                        reason="n1_o1_ca2",
                    )
                    trace_emit(
                        event="luke_terms2",
                        iter_num=1,
                        line_num_1b=0,
                        depth_1b=k + 1,
                        nu_1b=iwlk,
                        type_code=0,
                        wlvac_nm=wv,
                        center=float(luke_c2[k]),
                        adamp=float(luke_mg2[k]),
                        cv=float(luke_si2[k]),
                        tabcont=float(aluke[k, j]),
                        branch="lukeop",
                        reason="c2_mg2_si2",
                    )
                    trace_emit(
                        event="luke_pop",
                        iter_num=1,
                        line_num_1b=0,
                        depth_1b=k + 1,
                        nu_1b=iwlk,
                        type_code=0,
                        wlvac_nm=wv,
                        center=float(xnfp_all[k, 21] if xnfp_all.shape[1] > 21 else 0.0),
                        adamp=float(xnfpmg2[k]),
                        cv=float(xnfpsi2[k]),
                        tabcont=float(rho[k]),
                        branch="lukeop",
                        reason="xnfp_c2_mg2_si2",
                    )
    else:
        logger.debug("Skipping LUKEOP - IFOP(10)=0")

    # HOTOP: Hot star opacity (atlas7v.for line 9124-9251)
    # Free-free from C, N, O, Ne, Mg, Si, S, Fe ionization stages I-V
    # Only computed if IFOP(11) = 1
    if ifop[10] == 1:  # IFOP(11) in Fortran = ifop[10] in Python (0-indexed)
        logger.debug("Computing HOTOP (hot star opacity)...")
        xne = np.asarray(atmosphere.electron_density, dtype=np.float64)
        tlog_arr = np.log(np.maximum(temp, 1e-10))
        tkev = KBOLTZ_EV * temp

        # Build HOTOP population vectors matching Fortran POPS calls:
        # XNFP(1:4)=C I-IV, XNFP(5:9)=N I-V, XNFP(10:15)=O I-VI, XNFP(16:21)=Ne I-VI.
        hotop_xnfp = np.zeros((n_layers, 21), dtype=np.float64)
        xnf_sumqq = np.zeros((n_layers, 5), dtype=np.float64)
        xnfp_all = np.asarray(atmosphere.xnfp_all, dtype=np.float64)
        xnf_all = np.asarray(atmosphere.xnf_all, dtype=np.float64)

        def _copy_xnfp_block(dst: np.ndarray, dst_slice: slice, start_1based: int, nion: int) -> None:
            start0 = start_1based - 1
            end0 = min(start0 + nion, xnfp_all.shape[1])
            ncopy = max(0, end0 - start0)
            if ncopy > 0:
                dst[:, dst_slice.start : dst_slice.start + ncopy] = xnfp_all[:, start0:end0]

        # Match Fortran HOTOP staging:
        # C I-IV from slot 21, N I-V from 28, O I-VI from 36, Ne I-VI from 55.
        _copy_xnfp_block(hotop_xnfp, slice(0, 4), 21, 4)
        _copy_xnfp_block(hotop_xnfp, slice(4, 9), 28, 5)
        _copy_xnfp_block(hotop_xnfp, slice(9, 15), 36, 6)
        _copy_xnfp_block(hotop_xnfp, slice(15, 21), 55, 6)

        # XNFSUMQQ: sum over C,N,O,Ne,Mg,Si,S,Fe ions I..V.
        for start_1based in (21, 28, 36, 55, 78, 105, 136, 351):
            for iz in range(1, 6):
                idx = start_1based + iz - 1  # Fortran XNF(J, start+iz) → Python xnf_all[:, start+iz-1]
                if 0 <= idx < xnf_all.shape[1]:
                    xnf_sumqq[:, iz - 1] += (iz * iz) * xnf_all[:, idx]

        sqrt_temp = np.sqrt(np.maximum(temp, 1e-30))
        exp_hot = np.exp(
            -HOTOP_TRANSITIONS[:, 5][np.newaxis, :] / np.maximum(tkev[:, np.newaxis], 1e-30)
        )
        hot_id_idx = np.clip(HOTOP_TRANSITIONS[:, 6].astype(np.int64) - 1, 0, 20)
        chunk = 4096

        for i0 in range(0, nfreq, chunk):
            i1 = min(i0 + chunk, nfreq)
            f_chunk = freq[i0:i1]
            stim_chunk = stim[:, i0:i1]
            freqlg_chunk = np.log(f_chunk)

            # FREE = sum_q COULFF(q) * XNFSUMQQ(q), q=1..5 (atlas7v.for line 9286-9288)
            free = np.zeros((n_layers, f_chunk.size), dtype=np.float64)
            for q in range(1, 6):
                free += _coulff_grid(q, freqlg_chunk, tlog_arr) * xnf_sumqq[:, q - 1][:, np.newaxis]

            ahot_chunk = (
                free
                * (3.6919e8 / (f_chunk[np.newaxis, :] ** 3))
                * (xne[:, np.newaxis] / sqrt_temp[:, np.newaxis])
            )

            # Bound-free additions from HOTOP transition table (atlas7v.for line 9291-9302)
            for k in range(HOTOP_TRANSITIONS.shape[0]):
                freq0, xsect0, alpha0, power0, mult0, _, _ = HOTOP_TRANSITIONS[k]
                use = f_chunk >= freq0
                if not np.any(use):
                    continue
                ratio = freq0 / f_chunk[use]
                xsect = xsect0 * (
                    alpha0 + ratio - alpha0 * ratio
                ) * np.sqrt(ratio ** int(power0))
                xx = (
                    xsect[np.newaxis, :]
                    * hotop_xnfp[:, hot_id_idx[k]][:, np.newaxis]
                    * mult0
                )
                threshold = ahot_chunk[:, use] / 100.0
                ahot_chunk[:, use] += np.where(
                    xx > threshold,
                    xx * exp_hot[:, k][:, np.newaxis],
                    0.0,
                )

            ahot[:, i0:i1] = ahot_chunk * stim_chunk / rho[:, np.newaxis]
    else:
        logger.debug("Skipping HOTOP - IFOP(11)=0")

    # Molecular opacities for COOLOP (atlas12.for SUBROUTINE COOLOP lines 6411-6439)
    # ACOOL = AC1 + AMG1 + AAL1 + ASI1 + AFE1 + CHOP*XNFPCH + OHOP*XNFPOH + AH2COLL
    # C1, Mg1, Al1, Si1, Fe1 are already computed above
    # Now compute CHOP, OHOP, H2COLL for cool stars (T < 9000K)
    acool_mol = np.zeros((n_layers, nfreq), dtype=np.float64)

    # Check if we have cool temperatures (molecular opacities only matter for T < 9000K)
    t_min = temp.min()
    if t_min < 9000.0 and ifop[8] == 1:  # IFOP(9) = COOLOP enabled
        logger.debug("Computing molecular opacities (CHOP, OHOP, H2COLL) for COOLOP...")

        # CH/OH molecular populations come from POPSALL slots used by atlas12:
        # XNFP(:,846) for CH and XNFP(:,848) for OH (1-based indexing).
        xnfpch = np.asarray(atmosphere.xnfpch, dtype=np.float64)
        xnfpoh = np.asarray(atmosphere.xnfpoh, dtype=np.float64)

        # CHOP/OHOP: vectorized form of the Fortran per-frequency table
        # interpolation, scaled by molecular populations and stimulated emission.
        if xnfpch is not None:
            acool_mol += _chop_opacity_grid(freq, temp) * (xnfpch / rho)[:, np.newaxis] * stim
        if xnfpoh is not None:
            acool_mol += _ohop_opacity_grid(freq, temp) * (xnfpoh / rho)[:, np.newaxis] * stim

        # H2COLL: H2 collision-induced absorption. Fortran computes XNH2 once
        # into COMMON /XNF/ via H2RAOP and H2COLLOP reuses it for every frequency.
        if xnfph1 is not None:
            xnfhe1_arr = np.asarray(atmosphere.xnf_he1, dtype=np.float64)
            acool_mol += _h2_collision_opacity_grid(
                freq,
                temp,
                xnfph1,
                bhyd[:, 0] if bhyd.shape[1] > 0 else np.ones(n_layers),
                xnfhe1_arr,
                rho,
                stim,
            )

        logger.debug(
            "  Molecular opacity range: [%.6e, %.6e]",
            acool_mol.min(),
            acool_mol.max(),
        )

    # Sum ACONT/SCONT using atlas12-style dispatcher semantics:
    #   A = AH2P + AHE1 + AHE2 + AHEMIN + ACOOL + ALUKE + AHOT
    #   ACONT = A + AHYD + AHMIN + AXCONT
    #   SCONT = (A*BNU + AHYD*SHYD + AHMIN*SHMIN + AXCONT*SXCONT)/ACONT
    # where ACOOL bundles metal + molecular continuum terms.
    acool = ac1 + amg1 + aal1 + asi1 + afe1 + acool_mol
    a_base = ah2p + ahe1 + ahe2 + ahemin + acool + aluke + ahot
    acont = a_base + ahyd + ahmin + axcont

    # Compute SCONT (atlas12 KAPP weighting)
    scont = bnu_all.copy()
    mask = acont > 0
    numerator = (
        a_base * bnu_all
        + ahyd * shyd
        + ahmin * shmin
        + axcont * sxcont
    )
    scont[mask] = numerator[mask] / acont[mask]

    # Sum SIGMAC (atlas7v.for line 4584)
    sigmac = sigh + sighe + sigel + sigh2 + sigx

    if trace_enabled():
        ratiolg = np.log(1.0 + 1.0 / 2_000_000.0)
        for j in range(nfreq):
            wv = float(wavelength_nm[j])
            iwlk = int(np.log(max(wv, 1e-300)) / ratiolg + 0.5)
            for k in range(n_layers):
                if not trace_in_focus(wlvac_nm=wv, j0=k):
                    continue
                trace_emit(
                    event="kapp_comp1",
                    iter_num=1,
                    line_num_1b=0,
                    depth_1b=k + 1,
                    nu_1b=iwlk,
                    type_code=0,
                    wlvac_nm=wv,
                    center=float(ahyd[k, j]),
                    adamp=float(ahmin[k, j]),
                    cv=float(axcont[k, j]),
                    tabcont=float(acont[k, j]),
                    branch="kapp",
                    reason="ahyd_ahmin_axcont",
                )
                trace_emit(
                    event="kapp_comp2",
                    iter_num=1,
                    line_num_1b=0,
                    depth_1b=k + 1,
                    nu_1b=iwlk,
                    type_code=0,
                    wlvac_nm=wv,
                    center=float(ah2p[k, j]),
                    adamp=float(ahe1[k, j]),
                    cv=float(ahe2[k, j]),
                    tabcont=float(ahemin[k, j]),
                    branch="kapp",
                    reason="ah2p_he_terms",
                )
                trace_emit(
                    event="kapp_comp3",
                    iter_num=1,
                    line_num_1b=0,
                    depth_1b=k + 1,
                    nu_1b=iwlk,
                    type_code=0,
                    wlvac_nm=wv,
                    center=float(acool[k, j]),
                    adamp=float(aluke[k, j]),
                    cv=float(ahot[k, j]),
                    tabcont=float(a_base[k, j]),
                    branch="kapp",
                    reason="acool_aluke_ahot",
                )
                trace_emit(
                    event="kapp_scat1",
                    iter_num=1,
                    line_num_1b=0,
                    depth_1b=k + 1,
                    nu_1b=iwlk,
                    type_code=0,
                    wlvac_nm=wv,
                    center=float(sigh[k, j]),
                    adamp=float(sighe[k, j]),
                    cv=float(sigel[k, j]),
                    tabcont=float(sigmac[k, j]),
                    branch="kapp",
                    reason="sigh_sighe_sigel",
                )
                trace_emit(
                    event="kapp_scat2",
                    iter_num=1,
                    line_num_1b=0,
                    depth_1b=k + 1,
                    nu_1b=iwlk,
                    type_code=0,
                    wlvac_nm=wv,
                    center=float(sigh2[k, j]),
                    adamp=float(sigx[k, j]),
                    cv=0.0,
                    tabcont=float(sigmac[k, j]),
                    branch="kapp",
                    reason="sigh2_sigx",
                )

    logger.debug(
        "KAPP continuum computed: ACONT range [%.6e, %.6e]",
        acont.min(),
        acont.max(),
    )

    return acont, sigmac, scont
