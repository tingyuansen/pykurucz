"""
Full KAPP implementation: Compute ACONT and SIGMAC from atlas_tables.

This module implements the Fortran KAPP subroutine (atlas7v.for line 4479)
which computes continuum absorption (ACONT) and scattering (SIGMAC) from
precomputed B-tables and populations.

The KAPP subroutine:
1. Calls subroutines for each species (HOP, HE1OP, HE2OP, C1OP, etc.)
2. Sums contributions: ACONT = AHYD + AHMIN + AHE1 + AHE2 + AC1 + ...
3. Computes scattering: SIGMAC = SIGH + SIGHE + SIGEL + SIGH2 + SIGX
4. Computes source function: SCONT = weighted average of source terms
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Tuple, Optional

import numpy as np

_SCALE_HMINFF = float(os.environ.get("PY_SCALE_HMINFF", "1.0"))
_SCALE_HRAYOP = float(os.environ.get("PY_SCALE_HRAYOP", "1.0"))
_SCALE_H2RAOP = float(os.environ.get("PY_SCALE_H2RAOP", "1.0"))
_SCALE_ELECOP = float(os.environ.get("PY_SCALE_ELECOP", "1.0"))

from .karsas_tables import xkarsas
from .hydrogen_wings import compute_hydrogen_continuum
from .kapp_tables_data import load_kapp_tables

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..io.atmosphere import AtmosphereModel

# Constants matching Fortran exactly
C_LIGHT_CM = 2.99792458e10  # cm/s
C_LIGHT_NM = 2.99792458e17  # nm/s
H_PLANCK = 6.62607015e-27  # erg * s
K_BOLTZ = 1.380649e-16  # erg / K
# CRITICAL: Match Fortran's TKEV calculation exactly (atlas7v.for line 1954: TKEV(J)=8.6171D-5*T(J))
KBOLTZ_EV = 8.6171e-5  # eV/K (matches Fortran: 8.6171D-5)
RYDBERG_CM = 109677.576  # cm^-1
LN10 = np.log(10.0)

_KAPP_TABLES = load_kapp_tables()

# COULFF table for Coulomb free-free Gaunt factors (atlas7v.for line 4597-4612)
COULFF_Z4LOG = _KAPP_TABLES["COULFF_Z4LOG"]

# HMINOP tables (atlas7v.for line 5228-5278)
HMINOP_WBF = _KAPP_TABLES["HMINOP_WBF"]

HMINOP_BF = _KAPP_TABLES["HMINOP_BF"]

HMINOP_WAVEK = _KAPP_TABLES["HMINOP_WAVEK"]

HMINOP_THETAFF = _KAPP_TABLES["HMINOP_THETAFF"]

# FFBEG: (11, 11) array - first 11 columns of FF
HMINOP_FFBEG = _KAPP_TABLES["HMINOP_FFBEG"]

# FFEND: (11, 11) array - last 11 columns of FF
HMINOP_FFEND = _KAPP_TABLES["HMINOP_FFEND"]

# HRAYOP: Gavrila tables for hydrogen Rayleigh scattering (atlas7v.for line 5351-5420)
HRAYOP_GAVRILAM = _KAPP_TABLES["HRAYOP_GAVRILAM"]

HRAYOP_GAVRILAMAB = _KAPP_TABLES["HRAYOP_GAVRILAMAB"]

HRAYOP_GAVRILAMBC = _KAPP_TABLES["HRAYOP_GAVRILAMBC"]

HRAYOP_GAVRILAMCD = _KAPP_TABLES["HRAYOP_GAVRILAMCD"]

HRAYOP_GAVRILALYMANCONT = _KAPP_TABLES["HRAYOP_GAVRILALYMANCONT"]

HRAYOP_FGAVRILALYMANCONT = _KAPP_TABLES["HRAYOP_FGAVRILALYMANCONT"]
COULFF_A_TABLE = _KAPP_TABLES["COULFF_A_TABLE"]

# HOTOP transition table (atlas7v.for HOTOP DATA A1..A7, 60 entries × 7 fields):
# (freq0, xsect, alpha, power, multiplier, excitation_eV, xNfpId)
HOTOP_TRANSITIONS = _KAPP_TABLES["HOTOP_TRANSITIONS"]


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

    # Bilinear interpolation
    a_00 = COULFF_A_TABLE[igam - 1, ihvkt - 1]
    a_01 = COULFF_A_TABLE[igam - 1, ihvkt] if ihvkt < 11 else a_00
    a_10 = COULFF_A_TABLE[igam, ihvkt - 1] if igam < 10 else a_00
    a_11 = COULFF_A_TABLE[igam, ihvkt] if (igam < 10 and ihvkt < 11) else a_00

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

    ig = igam - 1
    ih = ihvkt - 1

    a00 = COULFF_A_TABLE[ig, ih]
    a01_raw = COULFF_A_TABLE[ig, np.minimum(ih + 1, 10)]
    a10_raw = COULFF_A_TABLE[np.minimum(ig + 1, 11), ih]
    a11_raw = COULFF_A_TABLE[np.minimum(ig + 1, 11), np.minimum(ih + 1, 10)]

    a01 = np.where(ihvkt < 11, a01_raw, a00)
    a10 = np.where(igam < 10, a10_raw, a00)
    a11 = np.where((igam < 10) & (ihvkt < 11), a11_raw, a00)

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


def _map1_simple(xold: np.ndarray, fold: np.ndarray, xnew: float) -> float:
    """MAP1 for single value interpolation (used by HMINOP and HRAYOP).

    Uses the full MAP1 implementation to ensure exact matching with Fortran.
    This is a wrapper around the full _map1 function from josh_solver.
    """
    from .josh_solver import _map1

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


# SI2OP Peach tables (atlas7v.for lines 9050-9073)
_SI2OP_PEACH = _KAPP_TABLES["_SI2OP_PEACH"]

_SI2OP_FREQSI = _KAPP_TABLES["_SI2OP_FREQSI"]

_SI2OP_FLOG = _KAPP_TABLES["_SI2OP_FLOG"]

_SI2OP_TLG = _KAPP_TABLES["_SI2OP_TLG"]


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

# CH Partition function (atlas7v.for line 8348-8355)
_CH_PARTITION = _KAPP_TABLES["_CH_PARTITION"]

# OH Partition function (atlas7v.for line 8665-8672)
_OH_PARTITION = _KAPP_TABLES["_OH_PARTITION"]

# CH cross-section table (atlas7v.for lines 8138-8347)
# Shape: (105, 15) - 105 energy bins (0.1-10.5 eV), 15 temperature points
# Stored as log10(cross-section * partition_function)
# Temperature grid: 2000K to 9000K in 500K steps (15 points)
_CH_CROSSSECT = _KAPP_TABLES["_CH_CROSSSECT"]


def _chop_opacity(freq: float, temp: np.ndarray) -> np.ndarray:
    """CH molecular opacity (atlas7v.for lines 8120-8384).

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

    # Energy index (0.1 eV bins starting at 0)
    n = int(evolt * 10)
    if n < 20 or n >= 105:  # Energy range 2.0-10.5 eV
        return result

    en = float(n) * 0.1

    # Interpolate cross-section in energy (index is n-2 for array starting at 0.2 eV)
    # Fortran data starts at 0.1 eV but first real values are at 0.2 eV
    idx = n - 2  # Adjust for 0-based indexing and 0.2 eV start
    if idx < 0 or idx >= 104:
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


# OH cross-section table (atlas7v.for lines 8405-8664)
# Shape: (130, 15) - 130 energy bins (2.1-15.0 eV), 15 temperature points
# Temperature grid: 2000K to 9000K in 500K steps (15 points)
# Stored as log10(cross-section * partition_function)
_OH_CROSSSECT = _KAPP_TABLES["_OH_CROSSSECT"]


def _ohop_opacity(freq: float, temp: np.ndarray) -> np.ndarray:
    """OH molecular opacity (atlas7v.for lines 8385-8701).

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


# H2 collision-induced absorption tables (atlas7v.for lines 8733-8912)
# H2-H2 and H2-He tables: 7 temperature points × 81 wavenumber points
# Temperature grid: 1000K to 7000K in 1000K steps
# Wavenumber grid: 0 to 20000 cm^-1 in 250 cm^-1 steps

# Complete H2-H2 collision-induced absorption table (atlas7v.for lines 8733-8822)
# 81 wavenumber bins (0-20000 cm^-1 in 250 cm^-1 steps) × 7 temperature points (1000-7000K)
# Values are log10(absorption coefficient in cm^5)
_H2_COLL_H2H2 = _KAPP_TABLES["_H2_COLL_H2H2"]

# Complete H2-He collision-induced absorption table (atlas7v.for lines 8823-8912)
# 81 wavenumber bins × 7 temperature points
_H2_COLL_H2HE = _KAPP_TABLES["_H2_COLL_H2HE"]


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
    """H2 collision-induced absorption (atlas7v.for lines 8702-8951).

    Computes H2-H2 and H2-He collision-induced dipole absorption.
    Based on Borysow, Jorgensen, and Zheng (1997) A&A 324, 185-195.

    Only active for wavenumber < 20000 cm^-1.

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

    # Compute H2 number density using equilibrium formula
    # XNH2 = (XNFPH1 * 2 * BHYD1)^2 * exp(...) / RHO
    # From atlas7v.for lines 8916-8923
    poly_t = (
        1.63660e-3
        + (
            -4.93992e-7
            + (
                1.11822e-10
                + (-1.49567e-14 + (1.06206e-18 - 3.08720e-23 * temp) * temp) * temp
            )
            * temp
        )
        * temp
    ) * temp

    exp_term = 4.478 / tkev - 46.4584 + poly_t - 1.5 * tlog
    exp_term = np.clip(exp_term, -100, 100)

    xnh2 = (xnfph1 * 2.0 * bhyd1) ** 2 * np.exp(exp_term)

    # Wavenumber interpolation (atlas7v.for lines 8932-8938)
    # NU = wavenumber bin index (0-79), DELNU = fractional interpolation weight
    nu = int(waveno / 250.0)
    nu = min(79, nu)
    delnu = (waveno - 250.0 * nu) / 250.0

    # Interpolate tables in wavenumber first (atlas7v.for line 8937-8938)
    # H2H2NU(IT) = H2H2(IT,NU+1)*DELNU + H2H2(IT,NU+2)*(1-DELNU)
    # Note: Fortran is 1-indexed, Python is 0-indexed
    # Also note: Fortran table is (7,81) = (temp, waveno), Python is (81,7) = (waveno, temp)
    h2h2_nu = np.zeros(7, dtype=np.float64)
    h2he_nu = np.zeros(7, dtype=np.float64)

    for it in range(7):
        # Interpolate between wavenumber bins nu and nu+1
        idx1 = min(nu, 80)
        idx2 = min(nu + 1, 80)
        h2h2_nu[it] = _H2_COLL_H2H2[idx1, it] * delnu + _H2_COLL_H2H2[idx2, it] * (
            1.0 - delnu
        )
        h2he_nu[it] = _H2_COLL_H2HE[idx1, it] * delnu + _H2_COLL_H2HE[idx2, it] * (
            1.0 - delnu
        )

    # For each layer, interpolate in temperature (atlas7v.for lines 8940-8948)
    for j in range(n_layers):
        t_j = temp[j]

        # Temperature bin index (1000K grid, 1-indexed in Fortran -> 0-indexed in Python)
        it = int(t_j / 1000.0)
        it = max(1, min(6, it))  # Clamp to valid range (Fortran: 1-6)

        # Fractional temperature interpolation weight
        delt = (t_j - 1000.0 * it) / 1000.0
        delt = max(0.0, min(1.0, delt))

        # Interpolate in temperature (atlas7v.for lines 8944-8945)
        # XH2H2 = H2H2NU(IT)*DELT + H2H2NU(IT+1)*(1-DELT)
        # Fortran IT is 1-6, maps to Python indices it-1 and it
        xh2h2 = h2h2_nu[it - 1] * delt + h2h2_nu[it] * (1.0 - delt)
        xh2he = h2he_nu[it - 1] * delt + h2he_nu[it] * (1.0 - delt)

        # Final opacity (atlas7v.for lines 8947-8948)
        # AH2COLL = (10^XH2HE * XNFHE + 10^XH2H2 * XNH2) * XNH2 / RHO * STIM
        result[j] = (
            (10.0**xh2he * xnfhe1[j] + 10.0**xh2h2 * xnh2[j])
            * xnh2[j]
            / rho[j]
            * stim[j]
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
H_ENERGY_CM = _KAPP_TABLES["H_ENERGY_CM"]
H_ENERGY_EV = H_ENERGY_CM / 8065.479  # Convert to eV
H_STAT_WEIGHT = _KAPP_TABLES["H_STAT_WEIGHT"]

# Maximum principal quantum number for partition function sum
# Fortran GHYD/EHYD arrays have 6 levels (n=1 to n=6)
H_MAX_LEVEL = 6


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
# IFOP(11)=0: HOTOP (hot star opacities) - DISABLED by default
# IFOP(13)=0: H2RAOP (H2 Rayleigh) - DISABLED by default
DEFAULT_IFOP = [1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]


def get_ifop_from_physics(
    atmosphere: "AtmosphereModel", match_fortran: bool = True
) -> list[int]:
    """
    Determine IFOP flags based on physical conditions of the atmosphere.

    IFOP flags control which opacity sources are included:
    - IFOP(1-3): Various bound-free opacities
    - IFOP(4): HRAYOP - Hydrogen Rayleigh scattering
    - IFOP(5-7): More bound-free opacities
    - IFOP(8): HERAOP - Helium Rayleigh scattering
    - IFOP(9-12): Various opacities
    - IFOP(13): H2RAOP - Molecular hydrogen Rayleigh scattering
    - IFOP(14-20): Specialized/experimental opacities

    Parameters
    ----------
    atmosphere : AtmosphereModel
        The atmosphere model with temperature and other properties
    match_fortran : bool
        If True (default), use Fortran defaults exactly (for validation).
        If False, use physics-based decisions that may differ from Fortran.

    Returns
    -------
    list[int]
        20-element IFOP array (0=off, 1=on) for each opacity source
    """
    # CRITICAL FIX: Start from Fortran defaults for matching
    # Previously started with [1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,0,0] which had
    # IFOP(8)=1 and IFOP(13)=1 enabled, but Fortran defaults have them disabled!
    # This caused Python to compute extra scattering (SIGHE, SIGH2) that Fortran doesn't.
    ifop = list(DEFAULT_IFOP)  # Copy to avoid modifying the constant

    if match_fortran:
        # Use exact Fortran defaults - do not modify anything
        logger.info(f"Using Fortran default IFOP: {ifop}")
        return ifop

    # Physics-based mode: enable additional opacity sources based on conditions
    # NOTE: This mode will NOT match Fortran exactly but may be more physically accurate
    t_char = float(np.median(atmosphere.temperature))

    # IFOP(4) = HRAYOP (H Rayleigh): Keep as default (enabled)
    # ifop[3] = 1  # Already set in DEFAULT_IFOP

    # IFOP(8) = HERAOP (He Rayleigh): Enable for physics accuracy
    # Fortran default is OFF, but He Rayleigh can be significant
    ifop[7] = 1
    logger.info(f"HERAOP enabled (physics mode): He Rayleigh scattering included")

    # IFOP(13) = H2RAOP (H2 Rayleigh): ON for cool stars where H2 forms
    # H2 becomes significant below ~5000-6000K due to molecular equilibrium
    if t_char < 6000.0:
        ifop[12] = 1
        logger.info(
            f"H2RAOP enabled: T_char={t_char:.0f}K < 6000K (cool star, H2 present)"
        )
    else:
        ifop[12] = 0
        logger.info(
            f"H2RAOP disabled: T_char={t_char:.0f}K >= 6000K (hot star, H2 dissociated)"
        )

    # IFOP(10) = LUKEOP (N1, O1, Mg2, Si2, Ca2): Enable for intermediate/hot stars
    # These opacities are important in the UV for stars with T > 5000K
    if t_char > 4500.0:
        ifop[9] = 1
        logger.info(
            f"LUKEOP enabled: T_char={t_char:.0f}K > 4500K (UV metal opacities)"
        )
    else:
        ifop[9] = 0
        logger.info(f"LUKEOP disabled: T_char={t_char:.0f}K <= 4500K (cool star)")

    # IFOP(11) = HOTOP (hot star free-free + bound-free): Enable for hot stars
    # These opacities are important for T > 8000K where metals are highly ionized
    if t_char > 8000.0:
        ifop[10] = 1
        logger.info(f"HOTOP enabled: T_char={t_char:.0f}K > 8000K (hot star opacities)")
    else:
        ifop[10] = 0
        logger.info(f"HOTOP disabled: T_char={t_char:.0f}K <= 8000K")

    logger.info(f"Physics-based IFOP: {ifop}")
    return ifop


def compute_kapp_continuum(
    atmosphere: "AtmosphereModel",
    freq: np.ndarray,
    atlas_tables: dict[str, np.ndarray],
    ifop: list[int] | None = None,
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
        Opacity flags controlling which opacity sources are included (1-indexed).
        Default: Fortran defaults [1,1,1,1,1,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0]
        IFOP(4)=1: HRAYOP (H Rayleigh)
        IFOP(8)=0: HERAOP (He Rayleigh) - DISABLED by default in Fortran
        IFOP(13)=0: H2RAOP (H2 Rayleigh) - DISABLED by default

    Returns
    -------
    acont:
        Continuum absorption coefficient (shape: (n_layers, nfreq)) in cm²/g
    sigmac:
        Continuum scattering coefficient (shape: (n_layers, nfreq)) in cm²/g
    scont:
        Continuum source function (shape: (n_layers, nfreq)) in erg/s/cm²/Hz/ster
    """
    # Auto-detect IFOP from physics if not provided (Fortran-independent!)
    if ifop is None:
        ifop = get_ifop_from_physics(atmosphere)
    n_layers = atmosphere.layers
    nfreq = freq.size

    logger.info(f"Computing KAPP continuum: {n_layers} layers × {nfreq} frequencies")

    # Initialize arrays
    acont = np.zeros((n_layers, nfreq), dtype=np.float64)
    sigmac = np.zeros((n_layers, nfreq), dtype=np.float64)
    scont = np.zeros((n_layers, nfreq), dtype=np.float64)

    # Compute Planck functions for all frequencies
    temp = np.asarray(atmosphere.temperature, dtype=np.float64)
    pop = getattr(atmosphere, "population_per_ion", None)
    has_pop_grid = (
        pop is not None
        and isinstance(pop, np.ndarray)
        and pop.ndim == 3
        and pop.shape[0] == n_layers
    )
    xnfphe_mode11 = None
    if has_pop_grid and pop.shape[1] > 2 and pop.shape[2] > 1:
        # Fortran XNFPHE from POPS(...,11): He I / He II / He III populations.
        xnfphe_mode11 = np.column_stack([pop[:, 0, 1], pop[:, 1, 1], pop[:, 2, 1]])
    bnu_all = np.zeros((n_layers, nfreq), dtype=np.float64)
    for i, f in enumerate(freq):
        bnu_all[:, i] = _planck_nu(f, temp)

    # Compute frequency-dependent quantities
    wavelength_nm = C_LIGHT_NM / np.maximum(freq, 1e-30)
    # hkt is per layer (shape: (n_layers,))
    hkt = H_PLANCK / (K_BOLTZ * temp)
    # hckt = hkt * c (cm²/s) - used in HE1OP, HE2OP, etc. (atlas7v.for line 81: HCKT(J)=HKT(J)*2.99792458D10)
    hckt = hkt * C_LIGHT_CM
    # ehvkt and stim are per layer and frequency (shape: (n_layers, nfreq))
    ehvkt = np.exp(-H_PLANCK * freq[None, :] / (K_BOLTZ * temp[:, None]))
    stim = 1.0 - ehvkt
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

    # HOP: Hydrogen opacity (atlas7v.for line 4596)
    if atmosphere.xnfph is not None:
        logger.info("Computing HOP (hydrogen opacity)...")
        xnfph = np.asarray(atmosphere.xnfph, dtype=np.float64)
        bhyd = atlas_tables.get("bhyd", np.ones((n_layers, 8), dtype=np.float64))

        xne = np.asarray(atmosphere.electron_density, dtype=np.float64)

        for j in range(nfreq):
            f = freq[j]
            freq3 = 2.815e29 / (f * f * f)
            wno = waveno[j]
            bnu_j = bnu_all[:, j]
            ehvkt_j = ehvkt[:, j]
            stim_j = stim[:, j]

            # H continuum computation (matching atlas7v.for HOP)
            # CRITICAL FIX: Fortran uses HCKT = HKT * C_LIGHT_CM in hydrogen opacity
            # From atlas7v_1.for line 1933: HCKT(J)=HKT(J)*2.99792458D10
            # And line 4045 uses: 109677.576D0*HCKT(J)
            hckt = hkt * C_LIGHT_CM
            # N=16 to infinity
            h = (
                freq3
                * 2.0
                / 2.0
                / (RYDBERG_CM * hckt)
                * (
                    np.exp(-np.maximum(109250.336, 109678.764 - wno) * hckt)
                    - np.exp(-109678.764 * hckt)
                )
                * stim_j
            )

            s = h * bnu_j

            # N=1 to 15 (add bound-free contributions)
            # For N=1-6, use BHYD departure coefficients
            # For N=7-15, use standard formula

            # N=15
            if wno >= 487.456:
                x = xkarsas(f, 1.0, 15, 15)
                a = x * 450.0 * np.exp(-109191.313 * hckt) * stim_j
                h = h + a
                s = s + a * bnu_j

            # N=14
            if wno >= 559.579:
                x = xkarsas(f, 1.0, 14, 14)
                a = x * 392.0 * np.exp(-109119.188 * hckt) * stim_j
                h = h + a
                s = s + a * bnu_j

            # N=13
            if wno >= 648.980:
                x = xkarsas(f, 1.0, 13, 13)
                a = x * 338.0 * np.exp(-109029.789 * hckt) * stim_j
                h = h + a
                s = s + a * bnu_j

            # N=12
            if wno >= 761.649:
                x = xkarsas(f, 1.0, 12, 12)
                a = x * 288.0 * np.exp(-108917.117 * hckt) * stim_j
                h = h + a
                s = s + a * bnu_j

            # N=11
            if wno >= 906.426:
                x = xkarsas(f, 1.0, 11, 11)
                a = x * 242.0 * np.exp(-108772.336 * hckt) * stim_j
                h = h + a
                s = s + a * bnu_j

            # N=10
            if wno >= 1096.776:
                x = xkarsas(f, 1.0, 10, 10)
                a = x * 200.0 * np.exp(-108581.992 * hckt) * stim_j
                h = h + a
                s = s + a * bnu_j

            # N=9
            if wno >= 1354.044:
                x = xkarsas(f, 1.0, 9, 9)
                a = x * 162.0 * np.exp(-108324.719 * hckt) * stim_j
                h = h + a
                s = s + a * bnu_j

            # N=8
            if wno >= 1713.713:
                x = xkarsas(f, 1.0, 8, 8)
                a = x * 128.0 * np.exp(-107965.051 * hckt) * stim_j
                h = h + a
                s = s + a * bnu_j

            # N=7
            if wno >= 2238.320:
                x = xkarsas(f, 1.0, 7, 7)
                a = x * 98.0 * np.exp(-107440.444 * hckt) * stim_j
                h = h + a
                s = s + a * bnu_j

            # N=6 (uses BHYD departure coefficient)
            if wno >= 3046.604:
                x = xkarsas(f, 1.0, 6, 6)
                bhyd_6 = (
                    bhyd[:, 5]
                    if bhyd.shape[1] > 5
                    else np.ones(n_layers, dtype=np.float64)
                )
                a = x * 72.0 * np.exp(-106632.160 * hckt) * (bhyd_6 - ehvkt_j)
                h = h + a
                s = s + a * bnu_j * stim_j / np.maximum(bhyd_6 - ehvkt_j, 1e-40)

            # N=5
            if wno >= 4387.113:
                x = xkarsas(f, 1.0, 5, 5)
                bhyd_5 = (
                    bhyd[:, 4]
                    if bhyd.shape[1] > 4
                    else np.ones(n_layers, dtype=np.float64)
                )
                a = x * 50.0 * np.exp(-105291.651 * hckt) * (bhyd_5 - ehvkt_j)
                h = h + a
                s = s + a * bnu_j * stim_j / np.maximum(bhyd_5 - ehvkt_j, 1e-40)

            # N=4
            if wno >= 6854.871:
                x = xkarsas(f, 1.0, 4, 4)
                bhyd_4 = (
                    bhyd[:, 3]
                    if bhyd.shape[1] > 3
                    else np.ones(n_layers, dtype=np.float64)
                )
                a = x * 32.0 * np.exp(-102823.893 * hckt) * (bhyd_4 - ehvkt_j)
                h = h + a
                s = s + a * bnu_j * stim_j / np.maximum(bhyd_4 - ehvkt_j, 1e-40)

            # N=3
            if wno >= 12186.462:
                x = xkarsas(f, 1.0, 3, 3)
                bhyd_3 = (
                    bhyd[:, 2]
                    if bhyd.shape[1] > 2
                    else np.ones(n_layers, dtype=np.float64)
                )
                a = x * 18.0 * np.exp(-97492.302 * hckt) * (bhyd_3 - ehvkt_j)
                h = h + a
                s = s + a * bnu_j * stim_j / np.maximum(bhyd_3 - ehvkt_j, 1e-40)

            # N=2
            if wno >= 27419.659:
                x = xkarsas(f, 1.0, 2, 2)
                bhyd_2 = (
                    bhyd[:, 1]
                    if bhyd.shape[1] > 1
                    else np.ones(n_layers, dtype=np.float64)
                )
                a = x * 8.0 * np.exp(-82259.105 * hckt) * (bhyd_2 - ehvkt_j)
                h = h + a
                s = s + a * bnu_j * stim_j / np.maximum(bhyd_2 - ehvkt_j, 1e-40)

            # N=1
            if wno >= 109678.764:
                x = xkarsas(f, 1.0, 1, 1)
                bhyd_1 = (
                    bhyd[:, 0]
                    if bhyd.shape[1] > 0
                    else np.ones(n_layers, dtype=np.float64)
                )
                a = x * 2.0 * 1.0 * (bhyd_1 - ehvkt_j)
                h = h + a
                s = s + a * bnu_j * stim_j / np.maximum(bhyd_1 - ehvkt_j, 1e-40)

            # Multiply by populations and normalize by density
            # H=H*XNFPH(J,1)/RHO(J)  (atlas7v.for line 4706)
            if xnfph.shape[1] > 0:
                xnfph1 = xnfph[:, 0]
                h = h * xnfph1 / rho
                s = s * xnfph1 / rho
            else:
                h = h / rho
                s = s / rho

            # Free-free contribution (atlas7v.for line 4709-4711)
            # A=3.6919E8/SQRT(T(J))*COULFF(J,1)/FREQ*XNE(J)/FREQ*XNFPH(J,2)/FREQ*STIM(J)/RHO(J)
            freqlg = np.log(f)
            tlog_arr = np.log(np.maximum(temp, 1e-10))
            coulff_arr = np.array(
                [
                    _coulff(j_idx, 1, f, freqlg, temp, tlog_arr)
                    for j_idx in range(n_layers)
                ]
            )

            if xnfph.shape[1] > 1:
                xnfph2 = xnfph[:, 1]
                a_ff = (
                    3.6919e8
                    / np.sqrt(temp)
                    * coulff_arr
                    / f
                    * xne
                    / f
                    * xnfph2
                    / f
                    * stim_j
                    / rho
                )
            else:
                a_ff = (
                    3.6919e8
                    / np.sqrt(temp)
                    * coulff_arr
                    / f
                    * xne
                    / f
                    / f
                    * stim_j
                    / rho
                )

            h = h + a_ff
            s = s + a_ff * bnu_j

            ahyd[:, j] = h
            shyd[:, j] = np.where(h > 0, s / h, bnu_j)

    # H2PLOP: H2+ opacity (atlas7v.for line 5189-5211)
    if atmosphere.xnfph is not None:
        logger.info("Computing H2PLOP (H2+ opacity)...")
        xnfph_arr = np.asarray(atmosphere.xnfph, dtype=np.float64)
        bhyd = atlas_tables.get("bhyd", np.ones((n_layers, 8), dtype=np.float64))
        tkev = np.asarray(atmosphere.temperature, dtype=np.float64) * KBOLTZ_EV

        for j in range(nfreq):
            f = freq[j]
            if f > 3.28805e15:
                continue
            wno = waveno[j]
            freqlg = np.log(f)
            freq15 = f / 1.0e15

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

            # AH2P = EXP(-ES/TKEV + FR + LOG(XNFPH(J,1))) * 2. * BHYD(J,1) * XNFPH(J,2) / RHO(J) * STIM(J)
            if xnfph_arr.shape[1] >= 2:
                xnfph1 = xnfph_arr[:, 0]
                xnfph2 = xnfph_arr[:, 1]
                bhyd1 = (
                    bhyd[:, 0]
                    if bhyd.shape[1] > 0
                    else np.ones(n_layers, dtype=np.float64)
                )
                stim_j = stim[:, j]

                ah2p_val = (
                    np.exp(-es / tkev + fr + np.log(np.maximum(xnfph1, 1e-40)))
                    * 2.0
                    * bhyd1
                    * xnfph2
                    / rho
                    * stim_j
                )
                ah2p[:, j] = ah2p_val

    # HE1OP: Helium I opacity (atlas7v.for line 5499-5704)
    if atmosphere.xnf_he1 is not None:
        logger.info("Computing HE1OP (Helium I opacity)...")
        if xnfphe_mode11 is not None:
            xnfphe = xnfphe_mode11
        else:
            xnfphe = np.asarray(atmosphere.xnf_he1, dtype=np.float64)
            if xnfphe.ndim == 1:
                xnfphe = xnfphe[:, np.newaxis]  # fallback only

        # XNFHE is POPS(...,12): mode-12 helium populations used in HE1 free-free term.
        if not hasattr(atmosphere, "xnf_he2") or atmosphere.xnf_he2 is None:
            raise ValueError(
                "Atmosphere model missing He II populations required by KAPP"
            )
        he1_mode12 = np.asarray(atmosphere.xnf_he1, dtype=np.float64)
        he2_mode12 = np.asarray(atmosphere.xnf_he2, dtype=np.float64)
        if he1_mode12.ndim > 1:
            he1_mode12 = he1_mode12[:, 0]
        if he2_mode12.ndim > 1:
            he2_mode12 = he2_mode12[:, 0]
        xnfhe = np.column_stack([he1_mode12, he2_mode12])

        bhe1 = atlas_tables.get("bhe1", np.ones((n_layers, 29), dtype=np.float64))
        bhe2 = atlas_tables.get("bhe2", np.ones((n_layers, 6), dtype=np.float64))

        for j in range(nfreq):
            f = freq[j]
            wno = waveno[j]
            freqlg = np.log(f)
            freq3 = 2.815e29 / (f * f * f)
            bnu_j = bnu_all[:, j]
            ehvkt_j = ehvkt[:, j]
            stim_j = stim[:, j]
            wl_nm = C_LIGHT_NM / f

            # N=6 to infinity (atlas7v.for line 5513-5516)
            # CRITICAL FIX: Fortran uses HCKT (cm²/s), not HKT (cm)!
            # HCKT = HKT * C_LIGHT_CM (atlas7v.for line 81)
            rydberg_he = 109722.267
            h = (
                freq3
                * 4.0
                / 2.0
                / (rydberg_he * hckt)
                * (
                    np.exp(-np.maximum(195262.919, 198310.76 - wno) * hckt)
                    - np.exp(-198310.76 * hckt)
                )
                * stim_j
                * (bhe2[:, 0] if bhe2.shape[1] > 0 else np.ones(n_layers))
            )
            s = h * bnu_j

            # Add bound-free contributions (N=5 down to N=1) - all 29 levels
            # BHE1 indices: 29 (5P 1P) down to 1 (1S 1S), BHE2 index: 1 (He II ground state)
            bhe2_1 = bhe2[:, 0] if bhe2.shape[1] > 0 else np.ones(n_layers)

            # N=5 levels (BHE1 indices 29 down to 20)
            if wno >= 4368.190 and bhe1.shape[1] > 28:  # 5P 1P
                x = freq3 / 3125.0
                a = (
                    x
                    * 3.0
                    * np.exp(-193942.57 * hckt)
                    * (bhe1[:, 28] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 28] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 4388.260 and bhe1.shape[1] > 27:  # 5G 1G
                x = freq3 / 3125.0
                a = (
                    x
                    * 9.0
                    * np.exp(-193922.5 * hckt)
                    * (bhe1[:, 27] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 27] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 4388.260 and bhe1.shape[1] > 26:  # 5G 3G
                x = freq3 / 3125.0
                a = (
                    x
                    * 27.0
                    * np.exp(-193922.5 * hckt)
                    * (bhe1[:, 26] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 26] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 4389.390 and bhe1.shape[1] > 25:  # 5F 1F
                x = freq3 / 3125.0
                a = (
                    x
                    * 7.0
                    * np.exp(-193921.37 * hckt)
                    * (bhe1[:, 25] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 25] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 4389.450 and bhe1.shape[1] > 24:  # 5F 3F
                x = freq3 / 3125.0
                a = (
                    x
                    * 15.0
                    * np.exp(-193921.31 * hckt)
                    * (bhe1[:, 24] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 24] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 4392.369 and bhe1.shape[1] > 23:  # 5D 1D
                x = freq3 / 3125.0
                a = (
                    x
                    * 5.0
                    * np.exp(-193918.391 * hckt)
                    * (bhe1[:, 23] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 23] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 4393.515 and bhe1.shape[1] > 22:  # 5D 3D
                x = freq3 / 3125.0
                a = (
                    x
                    * 15.0
                    * np.exp(-193917.245 * hckt)
                    * (bhe1[:, 22] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 22] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 4509.980 and bhe1.shape[1] > 21:  # 5P 3P
                x = freq3 / 3125.0
                a = (
                    x
                    * 9.0
                    * np.exp(-193800.78 * hckt)
                    * (bhe1[:, 21] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 21] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 4647.133 and bhe1.shape[1] > 20:  # 5S 1S
                x = freq3 / 3125.0
                a = (
                    x
                    * 1.0
                    * np.exp(-193663.627 * hckt)
                    * (bhe1[:, 20] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 20] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 4963.671 and bhe1.shape[1] > 19:  # 5S 3S
                x = freq3 / 3125.0
                a = (
                    x
                    * 3.0
                    * np.exp(-193347.089 * hckt)
                    * (bhe1[:, 19] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 19] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # N=4 levels (BHE1 indices 19 down to 12)
            if wno >= 6817.943 and bhe1.shape[1] > 18:  # 4P 1P
                x = freq3 / 1024.0
                a = (
                    x
                    * 3.0
                    * np.exp(-191492.817 * hckt)
                    * (bhe1[:, 18] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 18] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 6858.680 and bhe1.shape[1] > 17:  # 4F 1F
                x = freq3 / 1024.0
                a = (
                    x
                    * 7.0
                    * np.exp(-191452.08 * hckt)
                    * (bhe1[:, 17] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 17] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 6858.960 and bhe1.shape[1] > 16:  # 4F 3F
                x = freq3 / 1024.0
                a = (
                    x
                    * 21.0
                    * np.exp(-191451.80 * hckt)
                    * (bhe1[:, 16] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 16] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 6864.201 and bhe1.shape[1] > 15:  # 4D 1D
                x = freq3 / 1024.0
                a = (
                    x
                    * 5.0
                    * np.exp(-191446.559 * hckt)
                    * (bhe1[:, 15] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 15] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 6866.172 and bhe1.shape[1] > 14:  # 4D 3D
                x = freq3 / 1024.0
                a = (
                    x
                    * 15.0
                    * np.exp(-191444.588 * hckt)
                    * (bhe1[:, 14] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 14] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 7093.620 and bhe1.shape[1] > 13:  # 4P 3P
                x = freq3 / 1024.0
                a = (
                    x
                    * 9.0
                    * np.exp(-191217.14 * hckt)
                    * (bhe1[:, 13] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 13] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 7370.429 and bhe1.shape[1] > 12:  # 4S 1S
                x = freq3 / 1024.0
                a = (
                    x
                    * 1.0
                    * np.exp(-190940.331 * hckt)
                    * (bhe1[:, 12] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 12] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 8012.550 and bhe1.shape[1] > 11:  # 4S 3S
                x = freq3 / 1024.0
                a = (
                    x
                    * 3.0
                    * np.exp(-190298.210 * hckt)
                    * (bhe1[:, 11] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 11] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # N=3 levels (BHE1 indices 11 down to 6)
            if wno >= 12101.289 and bhe1.shape[1] > 10:  # 3P 1P
                x = np.exp(58.81 - 2.89 * freqlg)
                a = (
                    x
                    * 3.0
                    * np.exp(-186209.471 * hckt)
                    * (bhe1[:, 10] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 10] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 12205.695 and bhe1.shape[1] > 9:  # 3D 1D
                x = np.exp(85.20 - 3.69 * freqlg)
                a = (
                    x
                    * 5.0
                    * np.exp(-186105.065 * hckt)
                    * (bhe1[:, 9] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 9] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 12209.106 and bhe1.shape[1] > 8:  # 3D 3D
                x = np.exp(85.20 - 3.69 * freqlg)
                a = (
                    x
                    * 15.0
                    * np.exp(-186101.654 * hckt)
                    * (bhe1[:, 8] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 8] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 12746.066 and bhe1.shape[1] > 7:  # 3P 3P
                x = np.exp(49.30 - 2.60 * freqlg)
                a = (
                    x
                    * 9.0
                    * np.exp(-185564.694 * hckt)
                    * (bhe1[:, 7] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 7] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 13445.824 and bhe1.shape[1] > 6:  # 3S 1S
                x = np.exp(23.85 - 1.86 * freqlg)
                a = (
                    x
                    * 1.0
                    * np.exp(-184864.936 * hckt)
                    * (bhe1[:, 6] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 6] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 15073.868 and bhe1.shape[1] > 5:  # 3S 3S
                x = np.exp(12.69 - 1.54 * freqlg)
                a = (
                    x
                    * 3.0
                    * np.exp(-183236.892 * hckt)
                    * (bhe1[:, 5] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 5] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # N=2 levels (BHE1 indices 5 down to 2)
            if wno >= 27175.760 and bhe1.shape[1] > 4:  # 2P 1P
                x = np.exp(81.35 - 3.5 * freqlg)
                a = (
                    x
                    * 3.0
                    * np.exp(-171135.000 * hckt)
                    * (bhe1[:, 4] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 4] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 29223.753 and bhe1.shape[1] > 3:  # 2P 3P
                x = np.exp(61.21 - 2.9 * freqlg)
                a = (
                    x
                    * 9.0
                    * np.exp(-169087.007 * hckt)
                    * (bhe1[:, 3] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 3] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 32033.214 and bhe1.shape[1] > 2:  # 2S 1S
                x = np.exp(26.83 - 1.91 * freqlg)
                a = (
                    x
                    * 1.0
                    * np.exp(-166277.546 * hckt)
                    * (bhe1[:, 2] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 2] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 38454.691 and bhe1.shape[1] > 1:  # 2S 3S
                x = np.exp(-390.026 + (21.035 - 0.318 * freqlg) * freqlg)
                a = (
                    x
                    * 3.0
                    * np.exp(-159856.069 * hckt)
                    * (bhe1[:, 1] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 1] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # N=1 level (BHE1 index 1)
            if wno >= 198310.760 and bhe1.shape[1] > 0:  # 1S 1S
                x = np.exp(33.32 - 2.0 * freqlg)
                a = x * 1.0 * 1.0 * (bhe1[:, 0] - bhe2_1 * ehvkt_j)
                h = h + a
                denom = bhe1[:, 0] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # Multiply by populations and normalize (atlas7v.for line 5691-5692)
            if xnfphe.shape[1] > 0:
                xnfphe1 = xnfphe[:, 0]
                h = h * xnfphe1 / rho
                s = s * xnfphe1 / rho

            # Free-free contribution (atlas7v.for line 5694-5697)
            freqlg_arr = np.full(n_layers, freqlg)
            tlog_arr = np.log(np.maximum(temp, 1e-10))
            coulff_arr = np.array(
                [
                    _coulff(j_idx, 1, f, freqlg, temp, tlog_arr)
                    for j_idx in range(n_layers)
                ]
            )

            if xnfhe.shape[1] > 1:
                xnfhe2 = xnfhe[:, 1]
                a_ff = (
                    3.619e8
                    / np.sqrt(temp)
                    * coulff_arr
                    / f
                    * xne
                    / f
                    * xnfhe2
                    / f
                    * stim_j
                    / rho
                )
            else:
                a_ff = (
                    3.619e8
                    / np.sqrt(temp)
                    * coulff_arr
                    / f
                    * xne
                    / f
                    / f
                    * stim_j
                    / rho
                )

            h = h + a_ff
            s = s + a_ff * bnu_j

            ahe1[:, j] = h
            she1[:, j] = np.where(h > 0, s / h, bnu_j)

    # HE2OP: Helium II opacity (atlas7v.for line 5705-5793)
    if atmosphere.xnf_he2 is not None:
        logger.info("Computing HE2OP (Helium II opacity)...")
        if xnfphe_mode11 is not None:
            xnfphe = xnfphe_mode11
        else:
            xnfphe = np.asarray(atmosphere.xnf_he2, dtype=np.float64)
            if xnfphe.ndim == 1:
                xnfphe = np.column_stack([np.zeros(n_layers), xnfphe, np.zeros(n_layers)])
            elif xnfphe.shape[1] == 1:
                xnfphe = np.column_stack([np.zeros(n_layers), xnfphe[:, 0], np.zeros(n_layers)])
            elif xnfphe.shape[1] == 2:
                xnfphe = np.column_stack([np.zeros(n_layers), xnfphe[:, 0], xnfphe[:, 1]])
        xnfphe3 = (
            xnfphe[:, 2]
            if xnfphe.shape[1] > 2
            else np.zeros(n_layers, dtype=np.float64)
        )

        bhe2 = atlas_tables.get("bhe2", np.ones((n_layers, 6), dtype=np.float64))

        for j in range(nfreq):
            f = freq[j]
            wno = waveno[j]
            freq3 = 2.815e29 / (f * f * f)
            bnu_j = bnu_all[:, j]
            ehvkt_j = ehvkt[:, j]
            stim_j = stim[:, j]

            # XNFPRHO = XNFPHE(J,2)/RHO(J) (atlas7v.for line 5719)
            xnfprho = (
                xnfphe[:, 1] if xnfphe.shape[1] > 1 else np.zeros(n_layers)
            ) / rho

            # N=10 to infinity (atlas7v.for line 5720-5723)
            rydberg_he2 = 438889.068
            h = (
                freq3
                * 16.0
                * 2.0
                / 2.0
                / (rydberg_he2 * hckt)
                * (
                    np.exp(-np.maximum(434519.959, 438908.85 - wno) * hckt)
                    - np.exp(-438908.85 * hckt)
                )
                * stim_j
                * xnfprho
            )
            s = h * bnu_j

            # Add bound-free contributions (N=9 down to N=1)
            if wno >= 5418.390:  # N=9
                x = freq3 / 59049.0 * 16.0
                a = x * 162.0 * np.exp(-433490.46 * hckt) * stim_j * xnfprho
                h = h + a
                s = s + a * bnu_j

            if wno >= 6857.660:  # N=8
                x = freq3 * 16.0 / 32768.0
                a = x * 128.0 * np.exp(-432051.19 * hckt) * stim_j * xnfprho
                h = h + a
                s = s + a * bnu_j

            if wno >= 8956.950:  # N=7
                x = freq3 * 16.0 / 16807.0
                a = x * 98.0 * np.exp(-429951.90 * hckt) * stim_j * xnfprho
                h = h + a
                s = s + a * bnu_j

            if wno >= 12191.437:  # N=6
                bhe2_6 = bhe2[:, 5] if bhe2.shape[1] > 5 else np.ones(n_layers)
                x = freq3 * 16.0 / 7776.0 * (1.0986 + (-2.704e13 + 1.229e27 / f) / f)
                a = x * 72.0 * np.exp(-426717.413 * hckt) * (bhe2_6 - ehvkt_j) * xnfprho
                h = h + a
                s = s + a * bnu_j * stim_j / np.maximum(bhe2_6 - ehvkt_j, 1e-40)

            if wno >= 17555.715:  # N=5
                bhe2_5 = bhe2[:, 4] if bhe2.shape[1] > 4 else np.ones(n_layers)
                x = freq3 * 16.0 / 3125.0 * (1.102 + (-3.909e13 + 2.371e27 / f) / f)
                a = x * 50.0 * np.exp(-421353.135 * hckt) * (bhe2_5 - ehvkt_j) * xnfprho
                h = h + a
                s = s + a * bnu_j * stim_j / np.maximum(bhe2_5 - ehvkt_j, 1e-40)

            if wno >= 27430.925:  # N=4
                bhe2_4 = bhe2[:, 3] if bhe2.shape[1] > 3 else np.ones(n_layers)
                x = freq3 * 16.0 / 1024.0 * (1.101 + (-5.765e13 + 4.593e27 / f) / f)
                a = x * 32.0 * np.exp(-411477.925 * hckt) * (bhe2_4 - ehvkt_j) * xnfprho
                h = h + a
                s = s + a * bnu_j * stim_j / np.maximum(bhe2_4 - ehvkt_j, 1e-40)

            if wno >= 48766.491:  # N=3
                bhe2_3 = bhe2[:, 2] if bhe2.shape[1] > 2 else np.ones(n_layers)
                x = freq3 * 16.0 / 243.0 * (1.101 + (-9.863e13 + 1.035e28 / f) / f)
                a = x * 18.0 * np.exp(-390142.359 * hckt) * (bhe2_3 - ehvkt_j) * xnfprho
                h = h + a
                s = s + a * bnu_j * stim_j / np.maximum(bhe2_3 - ehvkt_j, 1e-40)

            if wno >= 109726.529:  # N=2
                bhe2_2 = bhe2[:, 1] if bhe2.shape[1] > 1 else np.ones(n_layers)
                x = freq3 * 16.0 / 32.0 * (1.105 + (-2.375e14 + 4.077e28 / f) / f)
                a = x * 8.0 * np.exp(-329182.321 * hckt) * (bhe2_2 - ehvkt_j) * xnfprho
                h = h + a
                s = s + a * bnu_j * stim_j / np.maximum(bhe2_2 - ehvkt_j, 1e-40)

            if wno >= 438908.850:  # N=1
                bhe2_1 = bhe2[:, 0] if bhe2.shape[1] > 0 else np.ones(n_layers)
                x = freq3 * 16.0 / 1.0 * (0.9916 + (2.719e13 - 2.268e30 / f) / f)
                a = x * 2.0 * 1.0 * (bhe2_1 - ehvkt_j) * xnfprho
                h = h + a
                s = s + a * bnu_j * stim_j / np.maximum(bhe2_1 - ehvkt_j, 1e-40)

            # Free-free contribution (atlas7v.for line 5783-5786)
            freqlg_arr = np.full(n_layers, freqlg)
            tlog_arr = np.log(np.maximum(temp, 1e-10))
            coulff_arr = np.array(
                [
                    _coulff(j_idx, 2, f, freqlg, temp, tlog_arr)
                    for j_idx in range(n_layers)
                ]
            )

            a_ff = (
                3.6919e8
                * 4.0
                / np.sqrt(temp)
                * coulff_arr
                / f
                * xne
                / f
                * xnfphe3
                / f
                * stim_j
                / rho
            )
            h = h + a_ff
            s = s + a_ff * bnu_j

            ahe2[:, j] = h
            she2[:, j] = np.where(h > 0, s / h, bnu_j)

    # HEMIOP: He- opacity (atlas7v.for line 7296-7318)
    # Fortran evidence:
    #   AHEMIN(J)=(A*T(J)+B+C/T(J))/1.D15*XNE(J)/1.D15*XNFPHE(J,1)/1.D15/RHO(J)
    # where A, B, C are frequency-dependent polynomials in 1/FREQ.
    if ifop[6] == 1 and atmosphere.xnf_he1 is not None and atmosphere.electron_density is not None:
        logger.info("Computing HEMIOP (He- opacity)...")
        xnfphe = np.asarray(atmosphere.xnf_he1, dtype=np.float64)
        if xnfphe.ndim == 1:
            xnfphe = xnfphe[:, np.newaxis]
        xnfphe1 = xnfphe[:, 0] if xnfphe.shape[1] > 0 else np.ones(n_layers)
        xne = np.asarray(atmosphere.electron_density, dtype=np.float64)

        for j in range(nfreq):
            f = freq[j]
            a_coeff = 3.397e-01 + (-5.216e14 + 7.039e30 / f) / f
            b_coeff = -4.116e03 + (1.067e19 + 8.135e34 / f) / f
            c_coeff = 5.081e08 + (-8.724e22 - 5.659e37 / f) / f
            ahemin[:, j] = (
                (a_coeff * temp + b_coeff + c_coeff / temp)
                / 1.0e15
                * xne
                / 1.0e15
                * xnfphe1
                / 1.0e15
                / rho
            )

    # TODO: Complete HE1OP implementation (all 29 levels)
    # HMINOP: H- opacity (atlas7v.for line 5212-5316)
    if atmosphere.xnfph is not None and atmosphere.electron_density is not None:
        logger.info("Computing HMINOP (H- opacity)...")
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

        for j in range(nfreq):
            f = freq[j]
            wno = waveno[j]
            bnu_j = bnu_all[:, j]
            ehvkt_j = ehvkt[:, j]
            stim_j = stim[:, j]

            # WAVE = 2.99792458e17 / FREQ (atlas7v.for line 5300)
            wave = 2.99792458e17 / f  # wavelength in nm
            # WAVELOG matches Fortran exactly - log of wavelength in nm
            # The WFFLOG array uses log(91.134/WAVEK) where WAVEK = 91.134/wavelength_nm
            # So WFFLOG = log(91.134 / (91.134/wave_nm)) = log(wave_nm)
            wavelog = np.log(wave)

            # Interpolate FFLOG to get FFTT for each THETA (atlas7v.for line 5302-5304)
            fftheta = np.zeros((n_layers,), dtype=np.float64)
            for layer_idx in range(n_layers):
                # For each THETA, interpolate FFLOG over wavelength
                fftt_for_theta = np.zeros((nthetaff,), dtype=np.float64)
                for it in range(nthetaff):
                    # Interpolate FFLOG over WFFLOG (wavelength dimension)
                    fftt_val = _linter(wfflog, fflog[:, it], np.array([wavelog]))[0]
                    fftt_for_theta[it] = np.exp(fftt_val)

                # Interpolate FFTT over THETA (atlas7v.for line 5308)
                fftheta[layer_idx] = _linter(
                    HMINOP_THETAFF, fftt_for_theta, np.array([theta[layer_idx]])
                )[0]

            # HMINBF from MAP1 (atlas7v.for line 5306)
            hminbf = 0.0
            if f > 1.82365e14:
                hminbf = _map1_simple(HMINOP_WBF, HMINOP_BF, wave)

            # Compute H- opacity (atlas7v.for line 5309-5313)
            # HMINFF = FFTETA * XNFPH(J,1) * 2. * BHYD(J,1) * XNE(J) / RHO(J) * 1e-26
            hminff = _SCALE_HMINFF * fftheta * xnfph1 * 2.0 * bhyd1 * xne / rho * 1e-26

            # H = HMINBF * 1e-18 * (1. - EHVKT(J)/BMIN(J)) * XHMIN(J) / RHO(J)
            h_bf = (
                hminbf * 1e-18 * (1.0 - ehvkt_j / np.maximum(bmin, 1e-40)) * xhmin / rho
            )

            ahmin[:, j] = h_bf + hminff

            # Source function (atlas7v.for line 5313-5314)
            # SHMIN = (H * BNU(J) * STIM(J) / (BMIN(J) - EHVKT(J)) + HMINFF * BNU(J)) / AHMIN(J)
            bmin_expanded = np.broadcast_to(bmin, (n_layers,))
            denom = bmin_expanded - ehvkt_j
            h_bf_src = h_bf * bnu_j * stim_j / np.maximum(denom, 1e-40)
            shmin[:, j] = np.where(
                ahmin[:, j] > 0, (h_bf_src + hminff * bnu_j) / ahmin[:, j], bnu_j
            )

    # Scattering subroutines
    # ELECOP: Electron scattering (atlas7v.for line 7806-7817) - Simple!
    if atmosphere.electron_density is not None:
        logger.info("Computing ELECOP (electron scattering)...")
        xne = np.asarray(atmosphere.electron_density, dtype=np.float64)
        for j in range(nfreq):
            # SIGEL = 0.6653e-24 * XNE / RHO (atlas7v.for line 7815)
            sigel[:, j] = _SCALE_ELECOP * 0.6653e-24 * xne / rho

    # HRAYOP: Hydrogen Rayleigh scattering (atlas7v.for line 5332-5482)
    # CRITICAL FIX: Fortran uses XNFPH(J,1) which is GROUND-STATE hydrogen population,
    # computed by POPS(1.01D0,11,XNFPH). But fort.10 only stores XNFH (total neutral H).
    # We must compute ground-state population from total H using partition function.
    xnfph1 = None
    if atmosphere.xnf_h is not None:
        logger.info("Computing HRAYOP (hydrogen Rayleigh scattering)...")
        xnf_h_total = np.asarray(atmosphere.xnf_h, dtype=np.float64)
        # Compute ground-state hydrogen from total neutral hydrogen
        xnfph1 = compute_ground_state_hydrogen(xnf_h_total, temp)
        logger.info(
            f"  Ground-state H fraction at layer 0: {xnfph1[0]/xnf_h_total[0]:.4f}"
        )
        logger.info(
            f"  XNFH[0] (total): {xnf_h_total[0]:.6e}, XNFPH[0] (ground): {xnfph1[0]:.6e}"
        )
    elif atmosphere.xnfph is not None:
        # Fallback: use xnfph if available (legacy behavior)
        logger.info("Computing HRAYOP using legacy xnfph (may be inaccurate)...")
        xnfph_arr = np.asarray(atmosphere.xnfph, dtype=np.float64)
        xnfph1 = xnfph_arr[:, 0] if xnfph_arr.shape[1] > 0 else np.ones(n_layers)

    if xnfph1 is not None:
        bhyd = atlas_tables.get("bhyd", np.ones((n_layers, 8), dtype=np.float64))
        bhyd1 = bhyd[:, 0] if bhyd.shape[1] > 0 else np.ones(n_layers)

        freq_lyman = 3.288051e15  # Lyman limit frequency
        freq_step = 3.288051e13  # Step size for GAVRILAM

        for j in range(nfreq):
            f = freq[j]
            g = 0.0

            # Compute G from Gavrila tables (atlas7v.for line 5421-5477)
            if f < freq_lyman * 0.01:  # FREQ < 3.288051e13
                # Linear extrapolation below table (atlas7v.for line 5422-5424)
                g = HRAYOP_GAVRILAM[0] * (f / freq_step) ** 2
            elif f <= freq_lyman * 0.74:  # FREQ <= 0.74 * Lyman
                # Interpolate in GAVRILAM (atlas7v.for line 5426-5431)
                # Fortran: I=FREQ/3.288051D13, I=MIN(I+1,74)
                #          G=GAVRILAM(I-1)+(GAVRILAM(I)-GAVRILAM(I-1))/3.288051E13*(FREQ-(I-1)*3.288051D13)
                i = int(f / freq_step)
                i = min(i + 1, 74)
                i = max(1, i)
                if i >= len(HRAYOP_GAVRILAM):
                    i = len(HRAYOP_GAVRILAM) - 1
                if i > 1:
                    # CRITICAL FIX: Fortran 1-based indexing to Python 0-based:
                    # Fortran I=31 uses GAVRILAM(30) and GAVRILAM(31)
                    # Python must use HRAYOP_GAVRILAM[29] and HRAYOP_GAVRILAM[30]
                    # So use [i-2] and [i-1] for GAVRILAM(I-1) and GAVRILAM(I)
                    g = HRAYOP_GAVRILAM[i - 2] + (
                        HRAYOP_GAVRILAM[i - 1] - HRAYOP_GAVRILAM[i - 2]
                    ) / freq_step * (f - (i - 1) * freq_step)
                else:
                    g = HRAYOP_GAVRILAM[0]
            elif f < freq_lyman * 0.755:  # FREQ < 0.755 * Lyman
                g = 15.57  # Constant (atlas7v.for line 5433-5435)
            elif f <= freq_lyman * 0.885:  # FREQ <= 0.885 * Lyman
                # Interpolate in GAVRILAMAB (atlas7v.for line 5437-5444)
                # Fortran: I=(FREQ-.755D0*3.288051D15)/1.644026D13
                #          I=I+1
                #          I=MIN(I+1,27)
                #          G=GAVRILAMAB(I-1)+(GAVRILAMAB(I)-GAVRILAMAB(I-1))/1.644026D13*
                #            (FREQ-(.755D0*3.288051D15+((I-1)-1)*1.664026D13))
                step_ab = 1.644026e13
                i = int((f - freq_lyman * 0.755) / step_ab)
                i = i + 1  # First increment (matches Fortran I=I+1)
                i = min(i + 1, 27)  # Second increment (matches Fortran I=MIN(I+1,27))
                i = max(1, i)
                if i >= len(HRAYOP_GAVRILAMAB):
                    i = len(HRAYOP_GAVRILAMAB) - 1
                if i > 1:
                    # CRITICAL FIX: Fortran 1-based indexing to Python 0-based
                    # Fortran uses GAVRILAMAB(I-1) and GAVRILAMAB(I)
                    # Python uses [i-2] and [i-1]
                    # Note: Fortran uses 1.664026D13 in freq offset (line 5442), might be typo but match exactly
                    freq_base = freq_lyman * 0.755
                    freq_offset_step = (
                        1.664026e13  # From Fortran line 5442 (different from step_ab!)
                    )
                    freq1 = freq_base + ((i - 1) - 1) * freq_offset_step
                    g = HRAYOP_GAVRILAMAB[i - 2] + (
                        HRAYOP_GAVRILAMAB[i - 1] - HRAYOP_GAVRILAMAB[i - 2]
                    ) / step_ab * (f - freq1)
                else:
                    g = HRAYOP_GAVRILAMAB[0]
            elif f < freq_lyman * 0.890:  # FREQ < 0.890 * Lyman
                g = 8.0  # Constant (atlas7v.for line 5446-5448)
            elif f <= freq_lyman * 0.936:  # FREQ <= 0.936 * Lyman
                # Interpolate in GAVRILAMBC (atlas7v.for line 5450-5457)
                # Fortran: I=(FREQ-.890D0*3.28851D15)/0.657610D13
                #          I=I+1
                #          I=MIN(I+1,24)
                #          G=GAVRILAMBC(I-1)+(GAVRILAMBC(I)-GAVRILAMBC(I-1))/0.657610D13*
                #            (FREQ-(.890D0*3.288051D15+((I-1)-1)*0.657610D13))
                step_bc = 0.657610e13
                i = int((f - freq_lyman * 0.890) / step_bc)
                i = i + 1  # First increment (matches Fortran I=I+1)
                i = min(i + 1, 24)  # Second increment (matches Fortran I=MIN(I+1,24))
                i = max(1, i)
                if i >= len(HRAYOP_GAVRILAMBC):
                    i = len(HRAYOP_GAVRILAMBC) - 1
                if i > 1:
                    # CRITICAL FIX: Fortran 1-based indexing to Python 0-based
                    # Fortran uses GAVRILAMBC(I-1) and GAVRILAMBC(I)
                    # Python uses [i-2] and [i-1]
                    freq_base = freq_lyman * 0.890
                    freq1 = freq_base + ((i - 1) - 1) * step_bc
                    g = HRAYOP_GAVRILAMBC[i - 2] + (
                        HRAYOP_GAVRILAMBC[i - 1] - HRAYOP_GAVRILAMBC[i - 2]
                    ) / step_bc * (f - freq1)
                else:
                    g = HRAYOP_GAVRILAMBC[0]
            elif f < freq_lyman * 0.938:  # FREQ < 0.938 * Lyman
                g = 9.0  # Constant (atlas7v.for line 5459-5461)
            elif f <= freq_lyman * 0.959:  # FREQ <= 0.959 * Lyman
                # Interpolate in GAVRILAMCD (atlas7v.for line 5463-5470)
                # Fortran: I=(FREQ-.938D0*3.288051D15)/0.3288051D13
                #          I=I+1
                #          I=MIN(I+1,22)
                #          G=GAVRILAMCD(I-1)+(GAVRILAMCD(I)-GAVRILAMCD(I-1))/0.3288051D13*
                #            (FREQ-(.938D0*3.288051D15+((I-1)-1)*0.3288051D13))
                step_cd = 0.3288051e13
                i = int((f - freq_lyman * 0.938) / step_cd)
                i = i + 1  # First increment (matches Fortran I=I+1)
                i = min(i + 1, 22)  # Second increment (matches Fortran I=MIN(I+1,22))
                i = max(1, i)
                if i >= len(HRAYOP_GAVRILAMCD):
                    i = len(HRAYOP_GAVRILAMCD) - 1
                if i > 1:
                    # CRITICAL FIX: Fortran 1-based indexing to Python 0-based
                    # Fortran uses GAVRILAMCD(I-1) and GAVRILAMCD(I)
                    # Python uses [i-2] and [i-1]
                    freq_base = freq_lyman * 0.938
                    freq1 = freq_base + ((i - 1) - 1) * step_cd
                    g = HRAYOP_GAVRILAMCD[i - 2] + (
                        HRAYOP_GAVRILAMCD[i - 1] - HRAYOP_GAVRILAMCD[i - 2]
                    ) / step_cd * (f - freq1)
                else:
                    g = HRAYOP_GAVRILAMCD[0]
            elif f <= freq_lyman:  # FREQ <= 1.000 * Lyman
                g = HRAYOP_GAVRILALYMANCONT[0]  # Constant (atlas7v.for line 5472-5474)
            else:  # FREQ > Lyman
                # Use MAP1 interpolation in GAVRILALYMANCONT (atlas7v.for line 5476-5477)
                freqlg_normalized = f / freq_lyman
                g = _map1_simple(
                    HRAYOP_FGAVRILALYMANCONT, HRAYOP_GAVRILALYMANCONT, freqlg_normalized
                )

            # XSECT = 6.65e-25 * G^2 (atlas7v.for line 5478)
            xsect = 6.65e-25 * g**2

            # SIGH = XSECT * XNFPH(J,1) * 2. * BHYD(J,1) / RHO(J) (atlas7v.for line 5480)
            sigh[:, j] = _SCALE_HRAYOP * xsect * xnfph1 * 2.0 * bhyd1 / rho

    # HERAOP: Helium Rayleigh scattering (atlas7v.for line 5818-5832)
    # CRITICAL: Fortran only calls HERAOP if IFOP(8) == 1 (atlas7v.for line 4046)
    # Fortran's default is IFOP(8) = 0 (atlas7v.for line 2822: DATA IFOP/...,0,0,.../)
    # This means HERAOP is DISABLED by default in Fortran!
    if (
        ifop[7] == 1 and atmosphere.xnf_he1 is not None
    ):  # IFOP(8) in Fortran = ifop[7] in Python (0-indexed)
        logger.info("Computing HERAOP (helium Rayleigh scattering)...")
        xnfphe = np.asarray(atmosphere.xnf_he1, dtype=np.float64)
        if xnfphe.ndim == 1:
            xnfphe = xnfphe[:, np.newaxis]  # Make it 2D
        bhe1 = atlas_tables.get("bhe1", np.ones((n_layers, 29), dtype=np.float64))

        for j in range(nfreq):
            f = freq[j]
            # WAVE = 2.99792458e18 / min(FREQ, 5.15e15) (atlas7v.for line 5826)
            wave = 2.99792458e18 / min(f, 5.15e15)
            ww = wave**2
            # SIG = 5.484e-14 / WW / WW * (1. + (2.44e5 + 5.94e10 / (WW - 2.90e5)) / WW)^2 (atlas7v.for line 5828)
            sig = (
                5.484e-14
                / (ww * ww)
                * (1.0 + (2.44e5 + 5.94e10 / max(ww - 2.90e5, 1e-10)) / ww) ** 2
            )
            xnfphe1 = xnfphe[:, 0] if xnfphe.shape[1] > 0 else np.ones(n_layers)
            bhe1_1 = bhe1[:, 0] if bhe1.shape[1] > 0 else np.ones(n_layers)
            sighe[:, j] = sig * xnfphe1 / rho * bhe1_1
    else:
        logger.info("Skipping HERAOP (helium Rayleigh scattering) - IFOP(8)=0")

    # H2RAOP: H2 Rayleigh scattering (atlas7v.for line 6823-6853)
    # CRITICAL: Fortran only calls H2RAOP if IFOP(13) == 1
    if ifop[12] == 1 and xnfph1 is not None:
        logger.info("Computing H2RAOP (H2 Rayleigh scattering)...")
        bhyd1 = bhyd[:, 0] if bhyd.shape[1] > 0 else np.ones(n_layers)

        # Compute XNH2 (H2 number density per gram) from XNFPH using equilibrium
        # Formula from atlas7v.for H2RAOP subroutine (lines 9744-9747):
        # XNH2(J) = (XNFPH(J,1)*2.*BHYD(J,1))**2 * EXP(4.478D0/TKEV(J) -
        #   4.64584D1 + poly(T) - 1.5*TLOG(J)) / RHO(J)
        # The /RHO(J) is outside the EXP() on line 9747 continuation line 3.
        tkev_arr = KBOLTZ_EV * temp
        tlog_arr = np.log(temp)

        # Polynomial in T: (1.63660e-3 + (-4.93992e-7 + (1.11822e-10 + (-1.49567e-14 +
        #                  (1.06206e-18 - 3.08720e-23*T)*T)*T)*T)*T)*T
        poly_T = (
            1.63660e-3
            + (
                -4.93992e-7
                + (
                    1.11822e-10
                    + (-1.49567e-14 + (1.06206e-18 - 3.08720e-23 * temp) * temp) * temp
                )
                * temp
            )
            * temp
        ) * temp

        exp_term = 4.478 / tkev_arr - 4.64584e1 + poly_T - 1.5 * tlog_arr

        # Avoid overflow
        exp_term = np.clip(exp_term, -100, 100)

        xnh2 = (xnfph1 * 2.0 * bhyd1) ** 2 * np.exp(exp_term) / rho

        for j in range(nfreq):
            f = freq[j]
            # Wave in Angstrom, capped at frequency 2.922e15 Hz
            wave = 2.99792458e18 / min(f, 2.922e15)
            ww = wave**2

            # Cross-section formula (atlas7v.for line 6847)
            sig = (8.14e-13 + 1.28e-6 / ww + 1.61 / (ww * ww)) / (ww * ww)

            sigh2[:, j] = _SCALE_H2RAOP * sig * xnh2

        logger.info(f"  SIGH2[0] at first freq: {sigh2[0, 0]:.6e}")
    else:
        logger.info("Skipping H2RAOP (H2 Rayleigh scattering) - IFOP(13)=0 or no XNFPH")

    # XSOP: Dummy scattering (atlas7v.for line 8083-8091) - does nothing
    # sigx remains zeros

    # Metal opacities
    # C1OP: Carbon I opacity (atlas7v.for line 5859-6033)
    if hasattr(atmosphere, "xnfpc") and atmosphere.xnfpc is not None:
        logger.info("Computing C1OP (Carbon I opacity)...")
        xnfpc = np.asarray(atmosphere.xnfpc, dtype=np.float64)
        bc1 = atlas_tables.get("bc1", np.ones((n_layers, 14), dtype=np.float64))
        bc2 = atlas_tables.get("bc2", np.ones((n_layers, 6), dtype=np.float64))
        ryd = 109732.298  # Carbon Rydberg constant

        for j in range(nfreq):
            f = freq[j]
            wno = waveno[j]
            bnu_j = bnu_all[:, j]
            ehvkt_j = ehvkt[:, j]
            stim_j = stim[:, j]

            h = 1e-30 * np.ones(n_layers)
            s = np.zeros(n_layers)

            # Bound-free contributions (atlas7v.for line 5873-6025)
            # PP 1S (BC1 index 13)
            if wno >= 16886.790:
                x = 0.0  # Placeholder - would need full cross-section
                bc1_13 = bc1[:, 12] if bc1.shape[1] > 12 else np.ones(n_layers)
                bc2_1 = bc2[:, 0] if bc2.shape[1] > 0 else np.ones(n_layers)
                a = x * 1.0 * np.exp(-73975.91 * hckt) * (bc1_13 - bc2_1 * ehvkt_j)
                h = h + a
                denom = bc1_13 / np.maximum(bc2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # PP 1D (BC1 index 12)
            if wno >= 18251.980:
                x = 0.0
                bc1_12 = bc1[:, 11] if bc1.shape[1] > 11 else np.ones(n_layers)
                bc2_1 = bc2[:, 0] if bc2.shape[1] > 0 else np.ones(n_layers)
                a = x * 5.0 * np.exp(-72610.72 * hckt) * (bc1_12 - bc2_1 * ehvkt_j)
                h = h + a
                denom = bc1_12 / np.maximum(bc2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # PP 3P (BC1 index 11)
            if wno >= 19487.800:
                x = 0.0
                bc1_11 = bc1[:, 10] if bc1.shape[1] > 10 else np.ones(n_layers)
                bc2_1 = bc2[:, 0] if bc2.shape[1] > 0 else np.ones(n_layers)
                a = x * 9.0 * np.exp(-71374.90 * hckt) * (bc1_11 - bc2_1 * ehvkt_j)
                h = h + a
                denom = bc1_11 / np.maximum(bc2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # PP 3S (BC1 index 10)
            if wno >= 20118.750:
                x = 0.0
                bc1_10 = bc1[:, 9] if bc1.shape[1] > 9 else np.ones(n_layers)
                bc2_1 = bc2[:, 0] if bc2.shape[1] > 0 else np.ones(n_layers)
                a = x * 3.0 * np.exp(-70743.95 * hckt) * (bc1_10 - bc2_1 * ehvkt_j)
                h = h + a
                denom = bc1_10 / np.maximum(bc2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # PP 3D (BC1 index 9)
            if wno >= 21140.700:
                x = 0.0
                bc1_9 = bc1[:, 8] if bc1.shape[1] > 8 else np.ones(n_layers)
                bc2_1 = bc2[:, 0] if bc2.shape[1] > 0 else np.ones(n_layers)
                a = x * 15.0 * np.exp(-69722.00 * hckt) * (bc1_9 - bc2_1 * ehvkt_j)
                h = h + a
                denom = bc1_9 / np.maximum(bc2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # PP 1P (BC1 index 8) - has cross-section formula
            if wno >= 22006.370:
                x = 2.1e-18 * (22006.370 / wno) ** 1.5
                bc1_8 = bc1[:, 7] if bc1.shape[1] > 7 else np.ones(n_layers)
                bc2_1 = bc2[:, 0] if bc2.shape[1] > 0 else np.ones(n_layers)
                a = x * 3.0 * np.exp(-68856.33 * hckt) * (bc1_8 - bc2_1 * ehvkt_j)
                h = h + a
                denom = bc1_8 / np.maximum(bc2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # PS 1P (BC1 index 6)
            if wno >= 28880.880:
                x = 1.54e-18 * (28880.880 / wno) ** 1.2
                bc1_6 = bc1[:, 5] if bc1.shape[1] > 5 else np.ones(n_layers)
                bc2_1 = bc2[:, 0] if bc2.shape[1] > 0 else np.ones(n_layers)
                a = x * 3.0 * np.exp(-61981.82 * hckt) * (bc1_6 - bc2_1 * ehvkt_j)
                h = h + a
                denom = bc1_6 / np.maximum(bc2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # PS 3P (BC1 index 5)
            if wno >= 30489.700:
                x = 0.2e-18 * (30489.700 / wno) ** 1.2
                bc1_5 = bc1[:, 4] if bc1.shape[1] > 4 else np.ones(n_layers)
                bc2_1 = bc2[:, 0] if bc2.shape[1] > 0 else np.ones(n_layers)
                a = x * 9.0 * np.exp(-60373.00 * hckt) * (bc1_5 - bc2_1 * ehvkt_j)
                h = h + a
                denom = bc1_5 / np.maximum(bc2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # P2 1S (BC1 index 3) - complex formula with resonances
            if wno >= 69172.400:
                x = 10.0 ** (-16.80 - (wno - 69172.400) / 3.0 / ryd)
                eps = (wno - 97700.0) * 2.0 / 2743.0
                a_val = 68e-18
                b_val = 118e-18
                x = x + (a_val * eps + b_val) / (eps**2 + 1.0)
                x = x / 3.0
                bc1_3 = bc1[:, 2] if bc1.shape[1] > 2 else np.ones(n_layers)
                bc2_1 = bc2[:, 0] if bc2.shape[1] > 0 else np.ones(n_layers)
                a = x * 1.0 * np.exp(-21648.02 * hckt) * (bc1_3 - bc2_1 * ehvkt_j)
                h = h + a
                denom = bc1_3 / np.maximum(bc2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

                # Second contribution (atlas7v.for line 5944-5948)
                if wno >= 69235.820:
                    a = a * 2.0
                    h = h + a
                    s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # P2 1D (BC1 index 2) - complex formula
            if wno >= 80627.760:
                x = 10.0 ** (-16.80 - (wno - 80627.760) / 3.0 / ryd)
                eps1 = (wno - 93917.0) * 2.0 / 9230.0
                a1 = 22e-18
                b1 = 26e-18
                x = x + (a1 * eps1 + b1) / (eps1**2 + 1.0)
                eps2 = (wno - 111130.0) * 2.0 / 2743.0
                a2 = -10.5e-18
                b2 = 46e-18
                x = x + (a2 * eps2 + b2) / (eps2**2 + 1.0)
                x = x / 3.0
                bc1_2 = bc1[:, 1] if bc1.shape[1] > 1 else np.ones(n_layers)
                bc2_1 = bc2[:, 0] if bc2.shape[1] > 0 else np.ones(n_layers)
                a = x * 5.0 * np.exp(-10192.66 * hckt) * (bc1_2 - bc2_1 * ehvkt_j)
                h = h + a
                denom = bc1_2 / np.maximum(bc2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

                if wno >= 80691.180:
                    a = a * 2.0
                    h = h + a
                    s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # P2 3P (BC1 index 1) - complex with multiple contributions
            if wno >= 90777.000:
                x = 10.0 ** (-16.80 - (wno - 90777.000) / 3.0 / ryd)
                x = x / 3.0
                bc1_1 = bc1[:, 0] if bc1.shape[1] > 0 else np.ones(n_layers)
                bc2_1 = bc2[:, 0] if bc2.shape[1] > 0 else np.ones(n_layers)

                if wno >= 90777.000:
                    a = x * 5.0 * np.exp(-43.42 * hckt) * (bc1_1 - bc2_1 * ehvkt_j)
                    h = h + a
                    denom = bc1_1 / np.maximum(bc2_1, 1e-40) - ehvkt_j
                    s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

                if wno >= 90804.000:
                    a = x * 3.0 * np.exp(-16.42 * hckt) * (bc1_1 - bc2_1 * ehvkt_j)
                    h = h + a
                    s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

                if wno >= 90820.420:
                    a = x * 1.0 * 1.0 * (bc1_1 - bc2_1 * ehvkt_j)
                    h = h + a
                    s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

                if wno >= 90840.420:
                    x = x * 2.0
                    a = x * 5.0 * np.exp(-43.42 * hckt) * (bc1_1 - bc2_1 * ehvkt_j)
                    h = h + a
                    s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

                if wno >= 90867.420:
                    a = x * 3.0 * np.exp(-16.42 * hckt) * (bc1_1 - bc2_1 * ehvkt_j)
                    h = h + a
                    s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

                if wno >= 90883.840:
                    a = x * 1.0 * 1.0 * (bc1_1 - bc2_1 * ehvkt_j)
                    h = h + a
                    s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # P3 5S (BC1 index 4)
            if wno >= 100121.000:
                x = 1e-18 * (100121.000 / wno) ** 3
                bc1_4 = bc1[:, 3] if bc1.shape[1] > 3 else np.ones(n_layers)
                bc2_1 = bc2[:, 0] if bc2.shape[1] > 0 else np.ones(n_layers)
                a = x * 5.0 * np.exp(-33735.20 * hckt) * (bc1_4 - bc2_1 * ehvkt_j)
                h = h + a
                denom = bc1_4 / np.maximum(bc2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # Normalize and store (atlas7v.for line 6027-6030)
            xnfpc1 = xnfpc[:, 0] if xnfpc.shape[1] > 0 else np.ones(n_layers)
            h = h * xnfpc1 / rho
            s = s * xnfpc1 / rho

            ac1[:, j] = h
            sc1[:, j] = np.where(h > 0, s / h, bnu_j)

    # MG1OP: Magnesium I opacity (atlas7v.for line 6187-6261)
    if hasattr(atmosphere, "xnfpmg") and atmosphere.xnfpmg is not None:
        logger.info("Computing MG1OP (Magnesium I opacity)...")
        xnfpmg = np.asarray(atmosphere.xnfpmg, dtype=np.float64)
        bmg1 = atlas_tables.get("bmg1", np.ones((n_layers, 11), dtype=np.float64))
        bmg2 = atlas_tables.get("bmg2", np.ones((n_layers, 6), dtype=np.float64))

        for j in range(nfreq):
            f = freq[j]
            wno = waveno[j]
            bnu_j = bnu_all[:, j]
            ehvkt_j = ehvkt[:, j]
            stim_j = stim[:, j]

            h = 1e-30 * np.ones(n_layers)
            s = np.zeros(n_layers)

            # 3D 3D (BMG1 index 8)
            if wno >= 13713.986:
                x = 25e-18 * (13713.986 / wno) ** 2.7
                bmg1_8 = bmg1[:, 7] if bmg1.shape[1] > 7 else np.ones(n_layers)
                bmg2_1 = bmg2[:, 0] if bmg2.shape[1] > 0 else np.ones(n_layers)
                a = x * 15.0 * np.exp(-47957.034 * hckt) * (bmg1_8 - bmg2_1 * ehvkt_j)
                h = h + a
                denom = bmg1_8 / np.maximum(bmg2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # 4P 3P (BMG1 index 7)
            if wno >= 13823.223:
                x = 33.8e-18 * (13823.223 / wno) ** 2.8
                bmg1_7 = bmg1[:, 6] if bmg1.shape[1] > 6 else np.ones(n_layers)
                bmg2_1 = bmg2[:, 0] if bmg2.shape[1] > 0 else np.ones(n_layers)
                a = x * 9.0 * np.exp(-47847.797 * hckt) * (bmg1_7 - bmg2_1 * ehvkt_j)
                h = h + a
                denom = bmg1_7 / np.maximum(bmg2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # 3D 1D (BMG1 index 6)
            if wno >= 15267.955:
                x = 45e-18 * (15267.955 / wno) ** 2.7
                bmg1_6 = bmg1[:, 5] if bmg1.shape[1] > 5 else np.ones(n_layers)
                bmg2_1 = bmg2[:, 0] if bmg2.shape[1] > 0 else np.ones(n_layers)
                a = x * 5.0 * np.exp(-46403.065 * hckt) * (bmg1_6 - bmg2_1 * ehvkt_j)
                h = h + a
                denom = bmg1_6 / np.maximum(bmg2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # 4S 1S (BMG1 index 5)
            if wno >= 18167.687:
                x = 0.43e-18 * (18167.687 / wno) ** 2.6
                bmg1_5 = bmg1[:, 4] if bmg1.shape[1] > 4 else np.ones(n_layers)
                bmg2_1 = bmg2[:, 0] if bmg2.shape[1] > 0 else np.ones(n_layers)
                a = x * 1.0 * np.exp(-43503.333 * hckt) * (bmg1_5 - bmg2_1 * ehvkt_j)
                h = h + a
                denom = bmg1_5 / np.maximum(bmg2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # 4S 3S (BMG1 index 4)
            if wno >= 20473.617:
                x = 2.1e-18 * (20473.617 / wno) ** 2.6
                bmg1_4 = bmg1[:, 3] if bmg1.shape[1] > 3 else np.ones(n_layers)
                bmg2_1 = bmg2[:, 0] if bmg2.shape[1] > 0 else np.ones(n_layers)
                a = x * 3.0 * np.exp(-41197.043 * hckt) * (bmg1_4 - bmg2_1 * ehvkt_j)
                h = h + a
                denom = bmg1_4 / np.maximum(bmg2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # 3P 1P (BMG1 index 3)
            if wno >= 26619.756:
                x = (
                    16e-18 * (26619.756 / wno) ** 2.1
                    - 7.8e-18 * (26619.756 / wno) ** 9.5
                )
                bmg1_3 = bmg1[:, 2] if bmg1.shape[1] > 2 else np.ones(n_layers)
                bmg2_1 = bmg2[:, 0] if bmg2.shape[1] > 0 else np.ones(n_layers)
                a = x * 3.0 * np.exp(-35051.264 * hckt) * (bmg1_3 - bmg2_1 * ehvkt_j)
                h = h + a
                denom = bmg1_3 / np.maximum(bmg2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # 3P 3P (BMG1 index 2) - multiple contributions
            if wno >= 39759.842:
                x = 20e-18 * (39759.842 / wno) ** 2.7
                x = np.maximum(x, 40e-18 * (39759.842 / wno) ** 14)
                bmg1_2 = bmg1[:, 1] if bmg1.shape[1] > 1 else np.ones(n_layers)
                bmg2_1 = bmg2[:, 0] if bmg2.shape[1] > 0 else np.ones(n_layers)

                a = x * 5.0 * np.exp(-21911.178 * hckt) * (bmg1_2 - bmg2_1 * ehvkt_j)
                h = h + a
                denom = bmg1_2 / np.maximum(bmg2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

                if wno >= 39800.556:
                    a = (
                        x
                        * 3.0
                        * np.exp(-21870.464 * hckt)
                        * (bmg1_2 - bmg2_1 * ehvkt_j)
                    )
                    h = h + a
                    s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

                if wno >= 39820.615:
                    a = (
                        x
                        * 1.0
                        * np.exp(-21850.405 * hckt)
                        * (bmg1_2 - bmg2_1 * ehvkt_j)
                    )
                    h = h + a
                    s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # 3S 1S (BMG1 index 1)
            if wno >= 61671.020:
                x = 1.1e-18 * (61671.020 / wno) ** 10
                bmg1_1 = bmg1[:, 0] if bmg1.shape[1] > 0 else np.ones(n_layers)
                bmg2_1 = bmg2[:, 0] if bmg2.shape[1] > 0 else np.ones(n_layers)
                a = x * 1.0 * 1.0 * (bmg1_1 - bmg2_1 * ehvkt_j)
                h = h + a
                denom = bmg1_1 / np.maximum(bmg2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # Normalize and store (atlas7v.for line 6258-6260)
            # Handle both 1D (n_layers,) and 2D (n_layers, n_ions) arrays
            xnfpmg1 = (
                xnfpmg
                if xnfpmg.ndim == 1
                else (xnfpmg[:, 0] if xnfpmg.shape[1] > 0 else np.ones(n_layers))
            )
            h = h * xnfpmg1 / rho
            s = s * xnfpmg1 / rho

            amg1[:, j] = h
            smg1[:, j] = np.where(h > 0, s / h, bnu_j)

    # FE1OP: Iron I opacity (atlas7v.for line 6623-6665) - simpler structure
    if hasattr(atmosphere, "xnfpfe") and atmosphere.xnfpfe is not None:
        logger.info("Computing FE1OP (Iron I opacity)...")
        xnfpfe = np.asarray(atmosphere.xnfpfe, dtype=np.float64)
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

        # FE1OP processes all frequencies - individual transitions are checked inside
        for j in range(nfreq):
            wno = waveno[j]
            if wno < 21000.0:  # Skip if below Fe I first edge
                continue

            bnu_j = bnu_all[:, j]
            ehvkt_j = ehvkt[:, j]
            stim_j = stim[:, j]

            # BFUDGE = BSI1(J,1) (atlas7v.for line 6652)
            bfudge = bsi1[:, 0] if bsi1.shape[1] > 0 else np.ones(n_layers)

            h = np.zeros(n_layers)

            # Sum contributions from all transitions (atlas7v.for line 6655-6660)
            for i in range(len(fe1_wno)):
                if fe1_wno[i] <= wno:
                    xsect = 3e-18 / (
                        1.0 + ((fe1_wno[i] + 3000.0 - wno) / fe1_wno[i] / 0.1) ** 4
                    )
                    h = h + xsect * fe1_g[i] * np.exp(-fe1_e[i] * hckt)

            # Normalize and store (atlas7v.for line 6661-6663)
            # Handle both 1D (n_layers,) and 2D (n_layers, n_ions) arrays
            xnfpfe1 = (
                xnfpfe
                if xnfpfe.ndim == 1
                else (xnfpfe[:, 0] if xnfpfe.shape[1] > 0 else np.ones(n_layers))
            )
            h = h * stim_j * xnfpfe1 / rho

            afe1[:, j] = h
            sfe1[:, j] = bnu_j * stim_j / np.maximum(bfudge - ehvkt_j, 1e-40)

    # AL1OP: Aluminum I opacity (atlas7v.for line 7716-7792)
    if hasattr(atmosphere, "xnfpal") and atmosphere.xnfpal is not None:
        logger.info("Computing AL1OP (Aluminum I opacity)...")
        xnfpal = np.asarray(atmosphere.xnfpal, dtype=np.float64)
        bal1 = atlas_tables.get("bal1", np.ones((n_layers, 9), dtype=np.float64))
        bal2 = atlas_tables.get("bal2")
        # BAL2 is exp(-E_ion * HCKT) where E_ion = 48278.37 cm^-1 for Al I ionization limit
        if bal2 is None:
            # Compute BAL2 in LTE: departure coefficient at ionization limit
            bal2 = np.exp(-48278.37 * hckt)[:, np.newaxis]  # (n_layers, 1)

        for j in range(nfreq):
            wno = waveno[j]
            bnu_j = bnu_all[:, j]
            ehvkt_j = ehvkt[:, j]
            stim_j = stim[:, j]

            h = 1e-30 * np.ones(n_layers)
            s = np.zeros(n_layers)

            # Get BAL2(J,1) for this layer
            bal2_1 = bal2[:, 0] if bal2.shape[1] > 0 else np.ones(n_layers)

            # 4F 2F (BAL1 index 9)
            if wno >= 6958.993:
                x = 0.0  # X=0 in Fortran
                bal1_9 = bal1[:, 8] if bal1.shape[1] > 8 else np.ones(n_layers)
                a = x * 14.0 * np.exp(-41319.377 * hckt) * (bal1_9 - bal2_1 * ehvkt_j)
                h = h + a
                denom = bal1_9 / np.maximum(bal2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # 5P 2P (BAL1 index 8)
            if wno >= 8002.467:
                x = 50e-18 * (8002.467 / wno) ** 3
                bal1_8 = bal1[:, 7] if bal1.shape[1] > 7 else np.ones(n_layers)
                a = x * 6.0 * np.exp(-40275.903 * hckt) * (bal1_8 - bal2_1 * ehvkt_j)
                h = h + a
                denom = bal1_8 / np.maximum(bal2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # 4D 2D (BAL1 index 7)
            if wno >= 9346.231:
                x = 50e-18 * (9346.231 / wno) ** 3
                bal1_7 = bal1[:, 6] if bal1.shape[1] > 6 else np.ones(n_layers)
                a = x * 10.0 * np.exp(-38932.139 * hckt) * (bal1_7 - bal2_1 * ehvkt_j)
                h = h + a
                denom = bal1_7 / np.maximum(bal2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # 5S 2S (BAL1 index 6)
            if wno >= 10588.957:
                x = 56.7e-18 * (10588.957 / wno) ** 1.9
                bal1_6 = bal1[:, 5] if bal1.shape[1] > 5 else np.ones(n_layers)
                a = x * 2.0 * np.exp(-37689.413 * hckt) * (bal1_6 - bal2_1 * ehvkt_j)
                h = h + a
                denom = bal1_6 / np.maximum(bal2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # 4P 2P (BAL1 index 5)
            if wno >= 15318.007:
                x = 14.5e-18 * 15318.007 / wno
                bal1_5 = bal1[:, 4] if bal1.shape[1] > 4 else np.ones(n_layers)
                a = x * 6.0 * np.exp(-32960.363 * hckt) * (bal1_5 - bal2_1 * ehvkt_j)
                h = h + a
                denom = bal1_5 / np.maximum(bal2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # 3D 2D (BAL1 index 4)
            if wno >= 15842.129:
                x = 47e-18 * (15842.129 / wno) ** 1.83
                bal1_4 = bal1[:, 3] if bal1.shape[1] > 3 else np.ones(n_layers)
                a = x * 10.0 * np.exp(-32436.241 * hckt) * (bal1_4 - bal2_1 * ehvkt_j)
                h = h + a
                denom = bal1_4 / np.maximum(bal2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # 4S 2S (BAL1 index 2)
            if wno >= 22930.614:
                x = 10e-18 * (22930.614 / wno) ** 2
                bal1_2 = bal1[:, 1] if bal1.shape[1] > 1 else np.ones(n_layers)
                a = x * 2.0 * np.exp(-25347.756 * hckt) * (bal1_2 - bal2_1 * ehvkt_j)
                h = h + a
                denom = bal1_2 / np.maximum(bal2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # 3P 2P (BAL1 index 1) - ground state edge at 48278.37 cm^-1
            if wno >= 48166.309:
                x = 65e-18 * (48166.309 / wno) ** 5
                bal1_1 = bal1[:, 0] if bal1.shape[1] > 0 else np.ones(n_layers)
                a = x * 4.0 * np.exp(-112.061 * hckt) * (bal1_1 - bal2_1 * ehvkt_j)
                h = h + a
                denom = bal1_1 / np.maximum(bal2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

                if wno >= 48278.370:
                    a = x * 2.0 * 1.0 * (bal1_1 - bal2_1 * ehvkt_j)
                    h = h + a
                    s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # P2 4P (BAL1 index 3)
            if wno >= 55903.260:
                x = 10e-18 * (55903.260 / wno) ** 2
                bal1_3 = bal1[:, 2] if bal1.shape[1] > 2 else np.ones(n_layers)
                a = x * 12.0 * np.exp(-29097.11 * hckt) * (bal1_3 - bal2_1 * ehvkt_j)
                h = h + a
                denom = bal1_3 / np.maximum(bal2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # Normalize and store (atlas7v.for line 7789-7790)
            # Handle both 1D (n_layers,) and 2D (n_layers, n_ions) arrays
            xnfpal1 = (
                xnfpal
                if xnfpal.ndim == 1
                else (xnfpal[:, 0] if xnfpal.shape[1] > 0 else np.ones(n_layers))
            )
            h = h * xnfpal1 / rho
            s = s * xnfpal1 / rho

            aal1[:, j] = h
            sal1[:, j] = np.where(h > 0, s / h, bnu_j)

    # SI1OP: Silicon I opacity (atlas7v.for line 7793-7948)
    if hasattr(atmosphere, "xnfpsi") and atmosphere.xnfpsi is not None:
        logger.info("Computing SI1OP (Silicon I opacity)...")
        xnfpsi = np.asarray(atmosphere.xnfpsi, dtype=np.float64)
        bsi1 = atlas_tables.get("bsi1", np.ones((n_layers, 11), dtype=np.float64))
        bsi2 = atlas_tables.get("bsi2", np.ones((n_layers, 10), dtype=np.float64))

        for j in range(nfreq):
            wno = waveno[j]
            bnu_j = bnu_all[:, j]
            ehvkt_j = ehvkt[:, j]
            stim_j = stim[:, j]

            h = 1e-30 * np.ones(n_layers)
            s = np.zeros(n_layers)

            # Get BSI2(J,1) for this layer
            bsi2_1 = bsi2[:, 0] if bsi2.shape[1] > 0 else np.ones(n_layers)

            # PP 3P (BSI1 index 11)
            if wno >= 16810.969:
                x = 0.0  # X=0 in Fortran
                bsi1_11 = bsi1[:, 10] if bsi1.shape[1] > 10 else np.ones(n_layers)
                a = x * 9.0 * np.exp(-49128.131 * hckt) * (bsi1_11 - bsi2_1 * ehvkt_j)
                h = h + a
                denom = bsi1_11 / np.maximum(bsi2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # PP 3D (BSI1 index 10)
            if wno >= 17777.641:
                x = 18e-18 * (17777.641 / wno) ** 3
                bsi1_10 = bsi1[:, 9] if bsi1.shape[1] > 9 else np.ones(n_layers)
                a = x * 15.0 * np.exp(-48161.459 * hckt) * (bsi1_10 - bsi2_1 * ehvkt_j)
                h = h + a
                denom = bsi1_10 / np.maximum(bsi2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # PD 1D (BSI1 index 9)
            if wno >= 18587.546:
                x = 0.0  # X=0 in Fortran
                bsi1_9 = bsi1[:, 8] if bsi1.shape[1] > 8 else np.ones(n_layers)
                a = x * 5.0 * np.exp(-47351.554 * hckt) * (bsi1_9 - bsi2_1 * ehvkt_j)
                h = h + a
                denom = bsi1_9 / np.maximum(bsi2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # PP 1P (BSI1 index 8)
            if wno >= 18655.039:
                x = 0.0  # X=0 in Fortran
                bsi1_8 = bsi1[:, 7] if bsi1.shape[1] > 7 else np.ones(n_layers)
                a = x * 3.0 * np.exp(-47284.061 * hckt) * (bsi1_8 - bsi2_1 * ehvkt_j)
                h = h + a
                denom = bsi1_8 / np.maximum(bsi2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # PS 1P (BSI1 index 6)
            if wno >= 24947.216:
                x = 4.09e-18 * (24947.216 / wno) ** 2
                bsi1_6 = bsi1[:, 5] if bsi1.shape[1] > 5 else np.ones(n_layers)
                a = x * 3.0 * np.exp(-40991.884 * hckt) * (bsi1_6 - bsi2_1 * ehvkt_j)
                h = h + a
                denom = bsi1_6 / np.maximum(bsi2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # PS 3P (BSI1 index 5)
            if wno >= 26079.180:
                x = 1.25e-18 * (26079.180 / wno) ** 2
                bsi1_5 = bsi1[:, 4] if bsi1.shape[1] > 4 else np.ones(n_layers)
                a = x * 9.0 * np.exp(-39859.920 * hckt) * (bsi1_5 - bsi2_1 * ehvkt_j)
                h = h + a
                denom = bsi1_5 / np.maximum(bsi2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # P2 1S (BSI1 index 3) with resonance
            if wno >= 50353.180:
                eps = (wno - 70000.0) * 2.0 / 6500.0
                reson1 = (97e-18 * eps + 94e-18) / (eps**2 + 1.0)
                x = 37e-18 * (50353.180 / wno) ** 2.40 + reson1
                bsi1_3 = bsi1[:, 2] if bsi1.shape[1] > 2 else np.ones(n_layers)
                bolt = 1.0 * np.exp(-15394.370 * hckt) * (bsi1_3 - bsi2_1 * ehvkt_j)
                a = x * bolt / 3.0
                h = h + a
                denom = bsi1_3 / np.maximum(bsi2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

                # Second limit at 50640.630
                if wno >= 50640.630:
                    x = 37e-18 * (50640.630 / wno) ** 2.40 + reson1
                    a = x * bolt * 2.0 / 3.0
                    h = h + a
                    s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # P2 1D (BSI1 index 2) with resonance
            if wno >= 59448.700:
                eps = (wno - 78600.0) * 2.0 / 13000.0
                reson1 = (-10e-18 * eps + 77e-18) / (eps**2 + 1.0)
                x = 24.5e-18 * (59448.700 / wno) ** 1.85 + reson1
                bsi1_2 = bsi1[:, 1] if bsi1.shape[1] > 1 else np.ones(n_layers)
                bolt = 5.0 * np.exp(-6298.850 * hckt) * (bsi1_2 - bsi2_1 * ehvkt_j)
                a = x * bolt / 3.0
                h = h + a
                denom = bsi1_2 / np.maximum(bsi2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

                # Second limit at 59736.150
                if wno >= 59736.150:
                    x = 24.5e-18 * (59736.150 / wno) ** 1.85 + reson1
                    a = x * bolt * 2.0 / 3.0
                    h = h + a
                    bsi1_1 = bsi1[:, 0] if bsi1.shape[1] > 0 else np.ones(n_layers)
                    denom = bsi1_1 / np.maximum(bsi2_1, 1e-40) - ehvkt_j
                    s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # P3 3D (BSI1 index 7)
            if wno >= 63446.510:
                x = 18e-18 * (63446.510 / wno) ** 3
                bsi1_7 = bsi1[:, 6] if bsi1.shape[1] > 6 else np.ones(n_layers)
                a = x * 15.0 * np.exp(-45303.310 * hckt) * (bsi1_7 - bsi2_1 * ehvkt_j)
                h = h + a
                denom = bsi1_7 / np.maximum(bsi2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # P2 3P (BSI1 index 1) - ground state edge with multiple contributions
            if wno >= 65524.393:
                x = 72e-18 * (65524.393 / wno) ** 1.90
                if wno > 74000.0:
                    x = 93e-18 * (65524.393 / wno) ** 4.00
                x = x / 3.0
                bsi1_1 = bsi1[:, 0] if bsi1.shape[1] > 0 else np.ones(n_layers)
                a = x * 5.0 * np.exp(-223.157 * hckt) * (bsi1_1 - bsi2_1 * ehvkt_j)
                h = h + a
                denom = bsi1_1 / np.maximum(bsi2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

                if wno >= 65670.435:
                    a = x * 3.0 * np.exp(-77.115 * hckt) * (bsi1_1 - bsi2_1 * ehvkt_j)
                    h = h + a
                    s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

                if wno >= 65747.550:
                    a = x * 1.0 * 1.0 * (bsi1_1 - bsi2_1 * ehvkt_j)
                    h = h + a
                    s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

                # Second fine-structure component
                if wno >= 65811.843:
                    x = x * 2.0
                    a = x * 5.0 * np.exp(-223.157 * hckt) * (bsi1_1 - bsi2_1 * ehvkt_j)
                    h = h + a
                    s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

                if wno >= 65957.885:
                    a = x * 3.0 * np.exp(-77.115 * hckt) * (bsi1_1 - bsi2_1 * ehvkt_j)
                    h = h + a
                    s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

                if wno >= 66035.000:
                    a = x * 1.0 * 1.0 * (bsi1_1 - bsi2_1 * ehvkt_j)
                    h = h + a
                    s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # P3 5S (BSI1 index 4)
            if wno >= 75423.767:
                x = 15e-18 * (75423.767 / wno) ** 3
                bsi1_4 = bsi1[:, 3] if bsi1.shape[1] > 3 else np.ones(n_layers)
                a = x * 5.0 * np.exp(-33326.053 * hckt) * (bsi1_4 - bsi2_1 * ehvkt_j)
                h = h + a
                denom = bsi1_4 / np.maximum(bsi2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # Normalize and store (atlas7v.for line 7944-7946)
            # Handle both 1D (n_layers,) and 2D (n_layers, n_ions) arrays
            xnfpsi1 = (
                xnfpsi
                if xnfpsi.ndim == 1
                else (xnfpsi[:, 0] if xnfpsi.shape[1] > 0 else np.ones(n_layers))
            )
            h = h * xnfpsi1 / rho
            s = s * xnfpsi1 / rho

            asi1[:, j] = h
            ssi1[:, j] = np.where(h > 0, s / h, bnu_j)

    # LUKEOP: Lukewarm star opacity (atlas7v.for line 8952-8977)
    # Computes: N1OP, O1OP, MG2OP, SI2OP, CA2OP
    # Only computed if IFOP(10) = 1
    if ifop[9] == 1:  # IFOP(10) in Fortran = ifop[9] in Python (0-indexed)
        logger.info("Computing LUKEOP (N1, O1, Mg2, Si2, Ca2 opacity)...")
        if has_pop_grid and pop.shape[1] > 1 and pop.shape[2] > 19:
            # POPS grid is stored as [layer, ion_stage(0-based), element(0-based)].
            # Map Fortran quantities: XNFPN, XNFPO, XNFPMG(II), XNFPSI(II), XNFPCA(II).
            xnfpn = pop[:, 0, 6]
            xnfpo = pop[:, 0, 7]
            xnfpmg2 = pop[:, 1, 11]
            xnfpsi2 = pop[:, 1, 13]
            xnfpca2 = pop[:, 1, 19]
        else:
            logger.warning(
                "LUKEOP enabled but population_per_ion is unavailable/incomplete; using zero LUKEOP to avoid non-Fortran placeholder opacity."
            )
            xnfpn = np.zeros(n_layers, dtype=np.float64)
            xnfpo = np.zeros(n_layers, dtype=np.float64)
            xnfpmg2 = np.zeros(n_layers, dtype=np.float64)
            xnfpsi2 = np.zeros(n_layers, dtype=np.float64)
            xnfpca2 = np.zeros(n_layers, dtype=np.float64)

        tkev = KBOLTZ_EV * temp  # eV (matches Fortran TKEV(J) = 8.6171D-5 * T(J))

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

            # MG2OP: Magnesium II opacity (atlas7v.for line 9020-9042)
            c1169 = 6.0 * np.exp(-4.43 / tkev)
            x824 = 0.0
            x1169 = 0.0
            if f >= 3.635492e15:  # 824 Å edge
                x824 = _seaton(3.635492e15, 1.40e-19, 4.0, 6.7, f)
            if f >= 2.564306e15:  # 1169 Å edge
                x1169 = 5.11e-19 * (2.564306e15 / f) ** 3
            mg2op = x824 * 2.0 + x1169 * c1169  # (n_layers,)

            # SI2OP: Silicon II opacity (atlas7v.for line 9043-9097)
            # Uses Peach tables with temperature/frequency interpolation
            si2op = _si2op_vectorized(f, freqlg, temp, np.log(temp))

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
            aluke[:, j] = (
                n1op * xnfpn
                + o1op * xnfpo
                + mg2op * xnfpmg2
                + si2op * xnfpsi2
                + ca2op * xnfpca2
            ) * stim_j / rho
    else:
        logger.info("Skipping LUKEOP - IFOP(10)=0")

    # HOTOP: Hot star opacity (atlas7v.for line 9124-9251)
    # Free-free from C, N, O, Ne, Mg, Si, S, Fe ionization stages I-V
    # Only computed if IFOP(11) = 1
    if ifop[10] == 1:  # IFOP(11) in Fortran = ifop[10] in Python (0-indexed)
        logger.info("Computing HOTOP (hot star opacity)...")
        xne = np.asarray(atmosphere.electron_density, dtype=np.float64)
        tlog_arr = np.log(np.maximum(temp, 1e-10))
        tkev = KBOLTZ_EV * temp

        # Build HOTOP population vectors matching Fortran POPS calls:
        # XNFP(1:4)=C I-IV, XNFP(5:9)=N I-V, XNFP(10:15)=O I-VI, XNFP(16:21)=Ne I-VI.
        hotop_xnfp = np.zeros((n_layers, 21), dtype=np.float64)
        xnf_sumqq = np.zeros((n_layers, 5), dtype=np.float64)
        if has_pop_grid and pop.shape[1] > 5 and pop.shape[2] > 25:
            hotop_xnfp[:, 0:4] = pop[:, 0:4, 5]
            hotop_xnfp[:, 4:9] = pop[:, 0:5, 6]
            hotop_xnfp[:, 9:15] = pop[:, 0:6, 7]
            hotop_xnfp[:, 15:21] = pop[:, 0:6, 9]

            # XNFSUMQQ = sum_elements[ IZ^2 * XNF(IZ+1) ], IZ=1..5 (Fortran line 9281)
            for elem_idx in (5, 6, 7, 9, 11, 13, 15, 25):  # C,N,O,Ne,Mg,Si,S,Fe
                for iz in range(1, 6):
                    xnf_sumqq[:, iz - 1] += (iz * iz) * pop[:, iz, elem_idx]
        else:
            logger.warning(
                "HOTOP enabled but population_per_ion is unavailable/incomplete; using zero HOTOP populations."
            )

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
        logger.info("Skipping HOTOP - IFOP(11)=0")

    # Molecular opacities for COOLOP (atlas7v.for lines 7302-7310)
    # ACOOL = AC1 + AMG1 + AAL1 + ASI1 + AFE1 + CHOP*XNFPCH + OHOP*XNFPOH + AH2COLL
    # C1, Mg1, Al1, Si1, Fe1 are already computed above
    # Now compute CHOP, OHOP, H2COLL for cool stars (T < 9000K)
    acool_mol = np.zeros((n_layers, nfreq), dtype=np.float64)

    # Check if we have cool temperatures (molecular opacities only matter for T < 9000K)
    t_min = temp.min()
    if t_min < 9000.0 and ifop[8] == 1:  # IFOP(9) = COOLOP enabled
        logger.info("Computing molecular opacities (CHOP, OHOP, H2COLL) for COOLOP...")

        # Get molecular populations if available
        # For now, use placeholder scaling - full implementation would need XNFPCH, XNFPOH
        # These populations come from molecular equilibrium (NMOLEC)
        xnfpch = getattr(atmosphere, "xnfpch", None)
        xnfpoh = getattr(atmosphere, "xnfpoh", None)

        # Use hydrogen and helium populations for H2 collision-induced
        tkev_arr = KBOLTZ_EV * temp
        tlog_arr = np.log(temp)

        for j in range(nfreq):
            f = freq[j]
            stim_j = stim[:, j]

            # CHOP: CH molecular opacity (scaled by CH population)
            if xnfpch is not None:
                chop_xsect = _chop_opacity(f, temp)
                acool_mol[:, j] += chop_xsect * xnfpch / rho * stim_j

            # OHOP: OH molecular opacity (scaled by OH population)
            if xnfpoh is not None:
                ohop_xsect = _ohop_opacity(f, temp)
                acool_mol[:, j] += ohop_xsect * xnfpoh / rho * stim_j

            # H2COLL: H2 collision-induced absorption
            # This is computed from H2 equilibrium, not a stored population
            if xnfph1 is not None:
                xnfhe1_arr = (
                    np.asarray(atmosphere.xnf_he1, dtype=np.float64)
                    if atmosphere.xnf_he1 is not None
                    else np.zeros(n_layers)
                )
                h2coll = _h2_collision_opacity(
                    f,
                    temp,
                    xnfph1,
                    bhyd[:, 0] if bhyd.shape[1] > 0 else np.ones(n_layers),
                    xnfhe1_arr,
                    rho,
                    tkev_arr,
                    tlog_arr,
                    stim_j,
                )
                acool_mol[:, j] += h2coll

        logger.info(
            f"  Molecular opacity range: [{acool_mol.min():.6e}, {acool_mol.max():.6e}]"
        )

    # Sum ACONT (atlas7v.for line 4571-4573)
    # a_base includes ALUKE, AHOT (if enabled)
    # Fortran KAPP (atlas7v.for line 6280) does NOT include ACOOL in ACONT.
    a_base = ah2p + ahemin + aluke + ahot
    acont = (
        a_base + ahyd + ahmin + axcont + ahe1 + ahe2 + ac1 + amg1 + aal1 + asi1 + afe1
    )

    # Compute SCONT (atlas7v.for line 4575-4579)
    scont = bnu_all.copy()
    mask = acont > 0
    numerator = (
        a_base * bnu_all
        + ahyd * shyd
        + ahmin * shmin
        + axcont * sxcont
        + ahe1 * she1
        + ahe2 * she2
        + ac1 * sc1
        + amg1 * smg1
        + aal1 * sal1
        + asi1 * ssi1
        + afe1 * sfe1
    )
    scont[mask] = numerator[mask] / acont[mask]

    # Sum SIGMAC (atlas7v.for line 4584)
    sigmac = sigh + sighe + sigel + sigh2 + sigx

    logger.info(
        f"KAPP continuum computed: ACONT range [{acont.min():.6e}, {acont.max():.6e}]"
    )

    return acont, sigmac, scont
