#!/usr/bin/env python3
"""Exact implementation of POPS and PFSAHA from Fortran atlas7v.for.

This module implements the exact Fortran logic for computing ion populations
using the Saha equation with partition functions.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Dict, Set
import math
import os
import warnings

from numba import jit, prange

from ._pfground_table import (
    FIRST_RANGE_LABELS,
    SECOND_RANGE_LABELS,
    PFGROUND_EXPRESSIONS,
)



def _compute_xnfpmol(
    temperature: np.ndarray,
    tkev: np.ndarray,
    tk: np.ndarray,
    hkt: np.ndarray,
    hckt: np.ndarray,
    tlog: np.ndarray,
    gas_pressure: np.ndarray,
    electron_density: np.ndarray,
    xnatom: np.ndarray,
    xnz: np.ndarray,
    xnmol: np.ndarray,
    code_mol: np.ndarray,
    equil: np.ndarray,
    locj: np.ndarray,
    kcomps: np.ndarray,
    idequa: np.ndarray,
    nequa: int,
    bhyd: np.ndarray,
    bc1: np.ndarray,
    bo1: np.ndarray,
    bmg1: np.ndarray,
    bal1: np.ndarray,
    bsi1: np.ndarray,
    bca1: np.ndarray,
) -> np.ndarray:
    n_layers = len(temperature)
    nummol = len(code_mol)
    xnfpmol = np.zeros((n_layers, nummol), dtype=np.float64)
    xnz_adj = xnz.copy()

    # Normalize XNZ as in atlas7v.for lines 6022-6038
    for k in range(1, nequa):
        id_val = int(idequa[k])
        if id_val == 100:
            denom = 2.0 * 2.4148e15 * temperature * np.sqrt(temperature)
            xnz_adj[:, k] = xnz_adj[:, k] / denom
            continue
        if id_val <= 0 or id_val > 99:
            continue

        pf = np.zeros((n_layers, 31), dtype=np.float64)
        pfsaha_exact(
            0,
            id_val,
            1,
            3,
            temperature,
            tkev,
            tk,
            hkt,
            hckt,
            tlog,
            gas_pressure,
            electron_density,
            xnatom,
            pf,
        )
        pf_val = pf[:, 0]
        if id_val == 1:
            pf_val = pf_val / bhyd[:, 0]
        elif id_val == 6:
            pf_val = pf_val / bc1[:, 0]
        elif id_val == 8:
            pf_val = pf_val / bo1[:, 0]
        elif id_val == 12:
            pf_val = pf_val / bmg1[:, 0]
        elif id_val == 13:
            pf_val = pf_val / bal1[:, 0]
        elif id_val == 14:
            pf_val = pf_val / bsi1[:, 0]
        elif id_val == 20:
            pf_val = pf_val / bca1[:, 0]

        denom = pf_val * 1.8786e20 * np.sqrt((ATMASS[id_val - 1] * temperature) ** 3)
        xnz_adj[:, k] = xnz_adj[:, k] / denom

    # Compute XNFPMOL for each molecule (atlas7v.for lines 6040-6066)
    for jmol in range(nummol):
        if equil[0, jmol] != 0.0:
            xnf = np.exp(equil[0, jmol] / tkev)
            amass = 0.0
            locj1 = locj[jmol]
            locj2 = locj[jmol + 1] - 1
            for lock in range(locj1, locj2 + 1):
                k = kcomps[lock]
                if k == nequa:
                    xnf = xnf / xnz_adj[:, nequa - 1]
                    continue
                id_val = int(idequa[k])
                if id_val <= 0 or id_val > 99:
                    continue
                if id_val < 100:
                    amass += ATMASS[id_val - 1]
                xnf = xnf * xnz_adj[:, k]
            xnf = xnf * 1.8786e20 * np.sqrt((amass * temperature) ** 3)
            xnfpmol[:, jmol] = xnf
        else:
            id_val = int(code_mol[jmol])
            if id_val <= 0 or id_val > 99:
                continue
            ncomp = locj[jmol + 1] - locj[jmol]
            pf = np.zeros((n_layers, 31), dtype=np.float64)
            pfsaha_exact(
                0,
                id_val,
                ncomp,
                3,
                temperature,
                tkev,
                tk,
                hkt,
                hckt,
                tlog,
                gas_pressure,
                electron_density,
                xnatom,
                pf,
            )
            pf_val = pf[:, 0]
            xnfpmol[:, jmol] = xnmol[:, jmol] / pf_val

    return xnfpmol


_PFGROUND_WARNED: Set[int] = set()

# Constants matching Fortran exactly
C_LIGHT_CMS = 2.99792458e10  # cm/s
H_PLANCK_ERG_S = 6.62607015e-27  # erg * s
K_BOLTZ_ERG_K = 1.380649e-16  # erg / K
# CRITICAL: Match Fortran's TKEV calculation exactly (atlas7v.for line 1954: TKEV(J)=8.6171D-5*T(J))
# Fortran uses 8.6171e-5, so KBOLTZ_EV = 1 / 8.6171e-5 = 11604.83225...
KBOLTZ_EV = 1.0 / 8.6171e-5  # K/eV (matches Fortran: 1/(8.6171e-5))
EV_TO_CM = 8065.479  # Conversion: 1 eV = 8065.479 cm^-1
M_H = 1.660e-24  # g (atomic mass unit in Fortran: 1.660D-24)

_HCK = 6.6256e-27 * 2.99792458e10 / 1.38054e-16

_PFGROUND_CONTEXT = {"math": math, "_HCK": _HCK}
_PFGROUND_FUNCS = {
    label: eval(f"lambda T: {expr}", _PFGROUND_CONTEXT)
    for label, expr in PFGROUND_EXPRESSIONS.items()
}
_FIRST_RANGE_LABELS = tuple(FIRST_RANGE_LABELS)
_SECOND_RANGE_LABELS = tuple(SECOND_RANGE_LABELS)


@jit(nopython=True, cache=True)
def _compute_saha_f_kernel(
    F: np.ndarray,
    PART: np.ndarray,
    IP: np.ndarray,
    POTLO: np.ndarray,
    CF: float,
    TV: float,
    nion2: int,
) -> None:
    """
    Numba-compiled kernel for computing Saha equation ionization fractions F.

    Matches Fortran logic from atlas7v.for lines 3900-3926.
    """
    F[0] = 1.0  # F(1) = 1

    # Compute F(ION) for ION = 2 to NION2
    # F(ION) = CF*PART(ION)/PART(ION-1)*EXP(-(IP(ION-1)-POTLO(ION-1))/TV)
    for ion_idx in range(2, nion2 + 1):
        part_curr = PART[ion_idx - 1]
        part_prev = PART[ion_idx - 2]
        ip_prev = IP[ion_idx - 2]
        potlo_prev = POTLO[ion_idx - 2]

        if part_prev > 0:
            exp_arg = -(ip_prev - potlo_prev) / TV
            exp_val = np.exp(exp_arg)
            F[ion_idx - 1] = CF * part_curr / part_prev * exp_val
        else:
            F[ion_idx - 1] = 0.0

    # Normalize ionization fractions (from atlas7v.for lines 3920-3926)
    L = nion2 + 1
    for ion_idx in range(2, nion2 + 1):
        L = L - 1
        F[0] = 1.0 + F[L - 1] * F[0]
    F[0] = 1.0 / F[0]

    # Final F computation
    for ion_idx in range(2, nion2 + 1):
        F[ion_idx - 1] = F[ion_idx - 2] * F[ion_idx - 1]


def _pfground_lookup(nelion: int, temperature: float) -> float:
    """Evaluate the PFGROUND correction for a given nelion/temperature."""
    if temperature <= 0.0 or nelion <= 0:
        return 1.0

    label: int
    if nelion <= len(_FIRST_RANGE_LABELS):
        label = _FIRST_RANGE_LABELS[nelion - 1]
    elif nelion < 169:
        return 1.0
    elif nelion - 169 < len(_SECOND_RANGE_LABELS):
        label = _SECOND_RANGE_LABELS[nelion - 169]
    else:
        label = 666

    func = _PFGROUND_FUNCS.get(label)
    if func is None:
        return 1.0
    return float(func(temperature))


# LOCZ array: starting indices for each element in partition function tables
LOCZ = np.array(
    [
        1,
        3,
        6,
        10,
        14,
        18,
        22,
        27,
        33,
        39,
        45,
        51,
        57,
        63,
        69,
        75,
        81,
        86,
        91,
        96,
        101,
        106,
        111,
        116,
        121,
        126,
        131,
        136,
        141,
    ],
    dtype=np.int32,
)

# SCALE array for partition function interpolation
SCALE = np.array([0.001, 0.01, 0.1, 1.0], dtype=np.float64)

# Global arrays - loaded from extracted data
POTION: Optional[np.ndarray] = None
NNN: Optional[np.ndarray] = None
ATMASS: Optional[np.ndarray] = None
PFTAB: Optional[np.ndarray] = None  # PFIRON partition function table (7, 56, 10, 9)
POTLO_PFIRON: Optional[np.ndarray] = (
    None  # POTLO values for PFIRON [500, 1000, 2000, 4000, 8000, 16000, 32000]
)
POTLOLOG_PFIRON: Optional[np.ndarray] = None  # log10(POTLO) values

# KAPP continuum opacity tables (for TTAUP subroutine)
TABT_KAPP: Optional[np.ndarray] = None  # Temperature grid (36 values, log10)
TABP_KAPP: Optional[np.ndarray] = None  # Pressure grid (30 values, log10)
TABKAP_KAPP: Optional[np.ndarray] = (
    None  # Opacity table (36, 30) - (temperature, pressure)
)

# Module-level variables to track ITEMP state (matching Fortran COMMON /TEMP/ and /IF/)
# From xnfpelsyn.for line 242: ITEMP=ITEMP+1 (called before first POPS)
# From atlas7v.for line 3050: IF(IFPRES.EQ.1.AND.ITEMP.NE.ITEMP1)CALL NMOLEC
# From atlas7v.for line 3051: ITEMP1=ITEMP
_ITEMP = 0
_ITEMP1 = 0
# xnfpelsyn/spectrv debug log shows: "IFCORR 0  IFPRES 0 ... IFMOL 1"
# For spectrum synthesis we must keep IFPRES=0 (no NELECT/NMOLEC calls).
_IFPRES = 0
# CRITICAL FIX: IFMOL=1 to match Fortran xnfpelsyn/spectrv actual behavior!
# When IFMOL=0, POPS calls NELECT (simpler atomic-only electron iteration)
# When IFMOL=1, POPS calls NMOLEC (complex molecular equilibrium)
# The DATA statement default is 0, but xnfpelsyn sets IFMOL=1 for spectrum synthesis
_IFMOL = 1  # Default: molecules ON (matching Fortran xnfpelsyn behavior)
# Storage for NMOLEC results (molecular equilibrium)
# xnz[j, k] = atomic number density for element with idequa[k] at layer j
# This accounts for atoms locked in molecules (e.g., C in CO)
_NMOLEC_XNZ: Optional[np.ndarray] = None
_NMOLEC_IDEQUA: Optional[np.ndarray] = None
_NMOLEC_NEQUA: int = 0
_NMOLEC_XNMOL: Optional[np.ndarray] = None
_NMOLEC_XNFPMOL: Optional[np.ndarray] = None
_NMOLEC_CODEMOL: Optional[np.ndarray] = None
_NMOLEC_NUMMOL: int = 0
_NMOLEC_LOCJ: Optional[np.ndarray] = None
_NMOLEC_KCOMPS: Optional[np.ndarray] = None
_NMOLEC_EQUIL: Optional[np.ndarray] = None




def nelect_exact(
    temperature: np.ndarray,
    tk: np.ndarray,
    tkev: np.ndarray,
    hkt: np.ndarray,
    hckt: np.ndarray,
    tlog: np.ndarray,
    gas_pressure: np.ndarray,
    electron_density: np.ndarray,  # Modified in-place
    xnatom: np.ndarray,  # Modified in-place
    xabund: np.ndarray,
    departure_tables: Optional[dict] = None,
    max_iter: int = 200,
    tol: float = 0.0005,
) -> None:
    """NELECT: Compute electron density for atomic-only atmosphere (IFMOL=0).

    This is a MUCH simpler algorithm than NMOLEC. It computes XNE by iterating
    over the ionization of 10 key elements only: H, He, C, Na, Mg, Al, Si, K, Ca, Fe.

    Matches atlas7v_1.for SUBROUTINE NELECT (lines 2939-2997) exactly.

    Parameters
    ----------
    temperature : np.ndarray
        Temperature array (K), shape (n_layers,)
    tk : np.ndarray
        Boltzmann constant * T (erg), shape (n_layers,)
    tkev : np.ndarray
        T in eV, shape (n_layers,)
    hkt : np.ndarray
        Planck constant / (k_B * T), shape (n_layers,)
    hckt : np.ndarray
        Planck constant * c / (k_B * T), shape (n_layers,)
    tlog : np.ndarray
        Log(T), shape (n_layers,)
    gas_pressure : np.ndarray
        Gas pressure (dyn/cm²), shape (n_layers,)
    electron_density : np.ndarray
        Electron density, modified in-place, shape (n_layers,)
    xnatom : np.ndarray
        Atomic number density, modified in-place, shape (n_layers,)
    xabund : np.ndarray
        Abundance array for elements 1-99, shape (99,)
    departure_tables : Optional[dict]
        Departure coefficients (optional)
    max_iter : int
        Maximum iterations (default 200, matching Fortran)
    tol : float
        Convergence tolerance (default 0.0005, matching Fortran)
    """
    n_layers = len(temperature)

    # Elements used by NELECT (from atlas7v_1.for line 2948)
    # NELEMZ: H(1), He(2), C(6), Na(11), Mg(12), Al(13), Si(14), K(19), Ca(20), Fe(26)
    # NIONZ: number of ions to consider for each element
    NELEMZ = np.array([1, 2, 6, 11, 12, 13, 14, 19, 20, 26], dtype=np.int32)
    NIONZ = np.array([1, 2, 2, 2, 2, 2, 2, 2, 2, 2], dtype=np.int32)
    NZ = 10

    # Working arrays
    elec = np.zeros(n_layers, dtype=np.float64)
    x_arr = np.zeros(NZ, dtype=np.float64)
    mask = np.ones(NZ, dtype=np.int32)

    # Scratch array for PFSAHA
    answer = np.zeros((n_layers, 31), dtype=np.float64)

    # Initial guess for layer 0: XNE = P/(2*TK) (half of total particle density)
    # From atlas7v_1.for line 2956: XNE(1)=P(1)/TK(1)/2.
    electron_density[0] = gas_pressure[0] / tk[0] / 2.0

    for j in range(n_layers):
        # Extrapolate XNE from previous layer (atlas7v_1.for line 2958)
        if j > 0:
            electron_density[j] = (
                electron_density[j - 1] * gas_pressure[j] / gas_pressure[j - 1]
            )

        # Total particle density (atlas7v_1.for line 2959)
        xntot = gas_pressure[j] / tk[j]

        # Initial XNATOM (atlas7v_1.for line 2960)
        xnatom[j] = xntot - electron_density[j]

        # Reset mask for this layer
        mask[:] = 1

        # Iterate to convergence (atlas7v_1.for lines 2963-2985)
        converged = False
        for iteration in range(max_iter):
            xnenew = 0.0

            # Sum electron contributions from each element
            for i in range(NZ):
                if mask[i] == 0:
                    continue

                iz = NELEMZ[i]
                nion = NIONZ[i]

                # Call PFSAHA with mode=4 to get electron contribution
                # This returns answer[j, 0] = sum over ions of F[ion] * (ion-1)
                # where F[ion] is the fraction in ionization state ion
                answer.fill(0.0)
                pfsaha_exact(
                    j=j,
                    iz=iz,
                    nion=nion,
                    mode=4,  # Mode 4: return number of electrons produced
                    temperature=temperature,
                    tkev=tkev,
                    tk=tk,
                    hkt=hkt,
                    hckt=hckt,
                    tlog=tlog,
                    gas_pressure=gas_pressure,
                    electron_density=electron_density,
                    xnatom=xnatom,
                    answer=answer,
                    departure_tables=departure_tables,
                    nlte_on=0,
                )

                # ELEC(J) from PFSAHA (atlas7v_1.for line 2970)
                elec_j = answer[j, 0]

                # X(I) = electrons from this element (atlas7v_1.for line 2971)
                x_arr[i] = elec_j * xnatom[j] * xabund[iz - 1]

                # Accumulate (atlas7v_1.for line 2972)
                xnenew += x_arr[i]

            # Damped update (atlas7v_1.for line 2974)
            xnenew = (xnenew + electron_density[j]) / 2.0

            # Convergence check (atlas7v_1.for lines 2975-2978)
            if xnenew > 0:
                error = abs((electron_density[j] - xnenew) / xnenew)
            else:
                error = abs(electron_density[j])

            # Update (atlas7v_1.for lines 2976-2977)
            electron_density[j] = xnenew
            xnatom[j] = xntot - electron_density[j]

            if error < tol:
                converged = True
                break

            # Masking optimization: skip small contributors (atlas7v_1.for lines 2979-2984)
            if j > 0:
                x1 = 1e-5 * electron_density[j]
                if error < 0.05:
                    x1 = x1 * 10.0
                for i in range(NZ):
                    if x_arr[i] < x1:
                        mask[i] = 0

        if not converged:
            warnings.warn(
                f"NELECT: XNE did not converge for layer {j} after {max_iter} iterations. "
                f"Final error = {error:.6e}"
            )


def set_ifmol(value: int) -> None:
    """Set the IFMOL flag to control NELECT vs NMOLEC behavior.

    Parameters
    ----------
    value : int
        0 = molecules OFF, use NELECT (simpler, atomic-only)
        1 = molecules ON, use NMOLEC (complex molecular equilibrium)
    """
    global _IFMOL
    _IFMOL = value


def set_ifpres(value: int) -> None:
    """Set the IFPRES flag (0 = skip NELECT/NMOLEC, 1 = run them)."""
    global _IFPRES
    _IFPRES = value


def get_ifmol() -> int:
    """Get the current IFMOL flag value."""
    return _IFMOL


# Special element energy and statistical weight arrays (from DATA statements)
# These are constants, so we can define them directly
EHYD = np.array(
    [0.0, 82259.105, 97492.302, 102823.893, 105291.651, 106632.160], dtype=np.float64
)
GHYD = np.array([2.0, 8.0, 18.0, 32.0, 50.0, 72.0], dtype=np.float64)

EHE1 = np.array(
    [
        0.0,
        159856.069,
        166277.546,
        169087.007,
        171135.000,
        183236.892,
        184864.936,
        185564.694,
        186101.654,
        186105.065,
        186209.471,
        190298.210,
        190940.331,
        191217.14,
        191444.588,
        191446.559,
        191451.80,
        191452.08,
        191492.817,
        193347.089,
        193663.627,
        193800.78,
        193917.245,
        193918.391,
        193921.31,
        193921.37,
        193922.5,
        193922.5,
        193942.57,
    ],
    dtype=np.float64,
)
GHE1 = np.array(
    [
        1,
        3.0,
        1.0,
        9.0,
        3.0,
        3.0,
        1.0,
        9.0,
        15.0,
        5.0,
        3.0,
        3.0,
        1.0,
        9.0,
        15.0,
        5.0,
        21.0,
        7.0,
        3.0,
        3.0,
        1.0,
        9.0,
        15.0,
        5.0,
        21.0,
        7.0,
        27.0,
        9.0,
        3.0,
    ],
    dtype=np.float64,
)

EHE2 = np.array(
    [0.0, 329182.321, 390142.359, 411477.925, 421353.135, 426717.413], dtype=np.float64
)
GHE2 = np.array([2.0, 8.0, 18.0, 32.0, 50.0, 72.0], dtype=np.float64)

EC1 = np.array(
    [
        29.60,
        10192.66,
        21648.02,
        33735.20,
        60373.00,
        61981.82,
        64088.85,
        68856.33,
        69722.00,
        70743.95,
        71374.90,
        72610.72,
        73975.91,
        75254.93,
    ],
    dtype=np.float64,
)
GC1 = np.array(
    [9.0, 5.0, 1.0, 5.0, 9.0, 3.0, 15.0, 3.0, 15.0, 3.0, 9.0, 5.0, 1.0, 9.0],
    dtype=np.float64,
)

EC2 = np.array(
    [42.48, 43035.8, 74931.11, 96493.74, 110652.10, 116537.65], dtype=np.float64
)
GC2 = np.array([6.0, 12.0, 10.0, 2.0, 6.0, 2.0], dtype=np.float64)

EO1 = np.array(
    [
        77.975,
        15867.862,
        33792.583,
        73768.200,
        76794.978,
        86629.089,
        88630.977,
        95476.728,
        96225.049,
        97420.748,
        97488.476,
        99094.065,
        99681.051,
    ],
    dtype=np.float64,
)
GO1 = np.array(
    [9.0, 5.0, 1.0, 5.0, 3.0, 15.0, 9.0, 5.0, 3.0, 25.0, 15.0, 15.0, 9.0],
    dtype=np.float64,
)

EMG1 = np.array(
    [
        0.0,
        21890.854,
        35051.264,
        41197.403,
        43503.333,
        46403.065,
        47847.797,
        47957.034,
        49346.729,
        51872.526,
        52556.206,
    ],
    dtype=np.float64,
)
GMG1 = np.array(
    [1.0, 9.0, 3.0, 3.0, 1.0, 5.0, 9.0, 15.0, 3.0, 3.0, 1.0], dtype=np.float64
)

EMG2 = np.array(
    [0.0, 35730.36, 69804.95, 71490.54, 80639.85, 92790.51], dtype=np.float64
)
GMG2 = np.array([2.0, 6.0, 2.0, 10.0, 6.0, 2.0], dtype=np.float64)

EAL1 = np.array(
    [
        74.707,
        25347.756,
        29097.11,
        32436.241,
        32960.363,
        37689.413,
        38932.139,
        40275.903,
        41319.377,
    ],
    dtype=np.float64,
)
GAL1 = np.array([6.0, 2.0, 12.0, 10.0, 6.0, 2.0, 10.0, 6.0, 14.0], dtype=np.float64)

ESI1 = np.array(
    [
        149.681,
        6298.850,
        15394.370,
        33326.053,
        39859.920,
        40991.884,
        45303.310,
        47284.061,
        47351.554,
        48161.459,
        49128.131,
    ],
    dtype=np.float64,
)
GSI1 = np.array(
    [9.0, 5.0, 1.0, 5.0, 9.0, 3.0, 15.0, 3.0, 5.0, 15.0, 9.0], dtype=np.float64
)

ESI2 = np.array(
    [191.55, 43002.27, 55319.11, 65500.73, 76665.61, 79348.67], dtype=np.float64
)
GSI2 = np.array([6.0, 12.0, 10.0, 2.0, 2.0, 10.0], dtype=np.float64)

ECA1 = np.array(
    [0.0, 15263.089, 20356.265, 21849.634, 23652.304, 31539.495, 33317.264, 35831.203],
    dtype=np.float64,
)
GCA1 = np.array([1.0, 9.0, 15.0, 5.0, 3.0, 3.0, 1.0, 21.0], dtype=np.float64)

ECA2 = np.array([0.0, 13686.60, 25340.10, 52166.93, 56850.78], dtype=np.float64)
GCA2 = np.array([2.0, 10.0, 6.0, 2.0, 10.0], dtype=np.float64)

ENA1 = np.array(
    [0.0, 16956.172, 16973.368, 25739.991, 29172.889, 29172.839, 30266.99, 30272.58],
    dtype=np.float64,
)
GNA1 = np.array([2.0, 2.0, 4.0, 2.0, 6.0, 4.0, 2.0, 4.0], dtype=np.float64)

EB1 = np.array(
    [10.17, 28810.0, 40039.65, 47856.99, 48613.01, 54767.74, 55010.08], dtype=np.float64
)
GB1 = np.array([6.0, 12.0, 2.0, 10.0, 6.0, 10.0, 2.0], dtype=np.float64)

EK1 = np.array(
    [0.0, 12985.170, 13042.876, 21026.551, 21534.680, 21536.988, 24701.382, 24720.139],
    dtype=np.float64,
)
GK1 = np.array([2.0, 2.0, 4.0, 2.0, 6.0, 4.0, 2.0, 4.0], dtype=np.float64)

# B arrays (BHYD, BHE1, etc.) are computed at runtime or read from files
# For now, use default value of 1.0 (NLTEON=-1 case)
# These would need to be computed or loaded if NLTE effects are important


def load_fortran_data(data_path: Optional[Path] = None) -> None:
    """Load POTION, NNN, ATMASS, PFIRON, and KAPP arrays from extracted data files."""
    global POTION, NNN, ATMASS, PFTAB, POTLO_PFIRON, POTLOLOG_PFIRON
    global TABT_KAPP, TABP_KAPP, TABKAP_KAPP

    if data_path is None:
        # Default to data directory (where it should be committed)
        data_path = Path(__file__).parent.parent / "data" / "fortran_data.npz"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Fortran data file not found: {data_path}\n"
            "  This file should be committed to the repository in synthe_py/data/.\n"
            "  If missing, run once: python synthe_py/tools/extract_fortran_data.py\n"
            "  Then commit synthe_py/data/fortran_data.npz to the repository."
        )

    data = np.load(data_path, allow_pickle=False)
    potion_data = data.get("potion", np.zeros(999, dtype=np.float64))
    nnn_data = data.get("nnn", np.zeros((6, 374), dtype=np.int32))
    atmass_data = data.get("atmass", np.zeros(99, dtype=np.float64))

    if potion_data is None or np.count_nonzero(potion_data) == 0:
        raise ValueError("POTION data not found or empty in data file")

    # NNN is stored as a 2D array in fortran_data.npz with the correct
    # Fortran indexing already applied by extract_fortran_data.py.
    # Only reshape if an older 1D array is encountered.
    if nnn_data is not None and nnn_data.size > 0 and nnn_data.ndim == 1:
        nnn_data = nnn_data.reshape((6, 374), order="F")

    # Assign to global variables
    POTION = potion_data
    NNN = nnn_data
    ATMASS = atmass_data


    # Load PFIRON data (separate file)
    pfiron_path = Path(__file__).parent.parent / "data" / "pfiron_data.npz"
    if not pfiron_path.exists():
        raise FileNotFoundError(
            f"PFIRON data file not found: {pfiron_path}\n"
            "  Run python extract_pfiron_data.py to generate it."
        )

    pfiron_data = np.load(pfiron_path, allow_pickle=False)
    PFTAB = pfiron_data.get("pftab")
    POTLO_PFIRON = pfiron_data.get("potlo")
    POTLOLOG_PFIRON = pfiron_data.get("potlolog")

    if PFTAB is None or POTLO_PFIRON is None or POTLOLOG_PFIRON is None:
        raise ValueError(
            "PFIRON data file is missing required arrays (pftab/potlo/potlolog)"
        )


    # Load KAPP tables (separate file)
    kapp_path = Path(__file__).parent.parent / "data" / "kapp_tables.npz"
    if not kapp_path.exists():
        raise FileNotFoundError(
            f"KAPP data file not found: {kapp_path}\n"
            "  Run python extract_kapp_tables.py to generate it."
        )

    kapp_data = np.load(kapp_path, allow_pickle=False)
    TABT_KAPP = kapp_data.get("tabt")
    TABP_KAPP = kapp_data.get("tabp")
    TABKAP_KAPP = kapp_data.get("ktab")

    if TABT_KAPP is None or TABP_KAPP is None or TABKAP_KAPP is None:
        raise ValueError("KAPP data file missing tabt/tabp/ktab arrays")



def pfground(nelion: int, temperature: float) -> float:
    """Exact implementation of PFGROUND function from atlas7v.for lines 16626-17387.

    Computes low-temperature partition function corrections.
    NELION = (IZ-1)*6 + ION (1-based element/ion index)
    """
    return _pfground_lookup(nelion, temperature)


def pfiron(iz: int, ion: int, tlog8: float, potlow8: float) -> float:
    """Exact implementation of PFIRON subroutine for Fe-group elements.

    From atlas7v.for lines 11604-17640. Interpolates partition functions
    from the PFTAB table for Fe-group elements (Ca=20 through Ni=28).

    Args:
        iz: Atomic number (20-28 for Ca through Ni)
        ion: Ionization stage (1-10)
        tlog8: log10(temperature) (base 10)
        potlow8: Lowered ionization potential in cm^-1

    Returns:
        Partition function value
    """
    global PFTAB, POTLO_PFIRON, POTLOLOG_PFIRON

    # Check if PFIRON data is loaded
    if PFTAB is None or POTLO_PFIRON is None or POTLOLOG_PFIRON is None:
        raise RuntimeError("PFIRON tables not loaded; call load_fortran_data() first")

    # Validate inputs
    if iz < 20 or iz > 28:
        raise ValueError(f"PFIRON only valid for elements 20-28, got {iz}")
    if ion < 1 or ion > 10:
        raise ValueError(f"PFIRON ion must be 1-10, got {ion}")

    # Convert to 0-based indices for Python arrays
    # Fortran uses NELEM-19 where NELEM=20-28, so NELEM-19=1-9
    elem_idx = iz - 20  # 0-based: 0-8 for Ca-Ni
    ion_idx = ion - 1  # 0-based: 0-9

    # Temperature interpolation (from Fortran lines 17611-17624)
    tlog = tlog8
    potlow = potlow8

    if tlog > 4.0:
        # High temperature: IT = (TLOG-4.0)/0.05 + 31, clamped to 1-56
        it_float = (tlog - 4.0) / 0.05 + 31.0
        it = max(1, min(56, int(it_float)))
        f = (tlog - (it - 31) * 0.05 - 4.0) / 0.05
    elif tlog < 3.7:
        # Low temperature: IT = (TLOG-3.32)/0.02 + 2, clamped to >= 2
        it_float = (tlog - 3.32) / 0.02 + 2.0
        it = max(2, int(it_float))
        f = (tlog - (it - 2) * 0.02 - 3.32) / 0.02
    else:
        # Medium temperature: IT = (TLOG-3.7)/0.03 + 21
        it_float = (tlog - 3.7) / 0.03 + 21.0
        it = int(it_float)
        f = (tlog - (it - 21) * 0.03 - 3.7) / 0.03

    # Clamp IT to valid range (1-56). Do NOT clamp F; Fortran allows F < 0
    # below the PFIRON table minimum (tlog < 3.32), and PFIRON interpolates
    # with the raw F value.
    it = max(1, min(56, it))
    it_idx = it - 1  # Convert to 0-based

    # POTLOW interpolation (from Fortran lines 17625-17638)
    # Fortran logic:
    #   LOW=1
    #   IF(POTLOW.LT.POTLO(LOW))GO TO 32  (simple interpolation)
    #   DO LOW=2,7
    #     IF(POTLOW.LT.POTLO(LOW))GO TO 35  (bilinear interpolation)
    #   LOW=7
    #   32: simple interpolation
    #   35: bilinear interpolation
    low = 0  # 0-based, Fortran LOW=1 corresponds to index 0
    if potlow < POTLO_PFIRON[0]:
        # POTLOW < POTLO(1): simple interpolation (Fortran label 32)
        low = 0
        pf = (
            f * PFTAB[low, it_idx, ion_idx, elem_idx]
            + (1.0 - f) * PFTAB[low, max(0, it_idx - 1), ion_idx, elem_idx]
        )
    else:
        # Find LOW such that POTLO(LOW-1) <= POTLOW < POTLO(LOW)
        low = len(POTLO_PFIRON) - 1  # Default to highest
        for i in range(1, len(POTLO_PFIRON)):
            if potlow < POTLO_PFIRON[i]:
                low = i
                break

        # Bilinear interpolation in POTLOW and temperature (Fortran label 35)
        # P = (LOG10(POTLOW) - POTLOLOG(LOW-1)) / 0.30103
        p = (np.log10(potlow) - POTLOLOG_PFIRON[low - 1]) / 0.30103
        pf = p * (
            f * PFTAB[low, it_idx, ion_idx, elem_idx]
            + (1.0 - f) * PFTAB[low, max(0, it_idx - 1), ion_idx, elem_idx]
        ) + (1.0 - p) * (
            f * PFTAB[low - 1, it_idx, ion_idx, elem_idx]
            + (1.0 - f) * PFTAB[low - 1, max(0, it_idx - 1), ion_idx, elem_idx]
        )

    return float(pf)


def get_potion_index(iz: int, ion: int) -> int:
    """Get POTION array index for element IZ, ion ION (1-based).

    From atlas7v.for lines 3599-3600:
    - For IZ <= 30: INDEX = IZ*(IZ+1)/2 + ION - 1
    - For IZ > 30: INDEX = IZ*5 + 341 + ION - 1
    """
    if iz <= 30:
        return iz * (iz + 1) // 2 + ion - 1
    else:
        return iz * 5 + 341 + ion - 1


def pfsaha_exact(
    j: int,  # Layer index (0-based, matching Fortran 1-based) - not used, process all layers
    iz: int,  # Element number (1-99)
    nion: int,  # Ion number (1-6)
    mode: int,  # Mode: 1=ionization fraction/partition, 2=ionization fraction, 3=partition, 4=electrons
    temperature: np.ndarray,
    tkev: np.ndarray,
    tk: np.ndarray,
    hkt: np.ndarray,
    hckt: np.ndarray,
    tlog: np.ndarray,
    gas_pressure: np.ndarray,
    electron_density: np.ndarray,
    xnatom: np.ndarray,
    answer: np.ndarray,  # Output array shape (n_layers, 31)
    # Additional arrays needed for special elements (to be added)
    departure_tables: Optional[dict[str, np.ndarray]] = None,
    nlte_on: int = 0,
) -> None:
    global POTION, NNN, ATMASS  # Use global arrays loaded by load_fortran_data()
    """Exact implementation of PFSAHA subroutine from atlas7v.for.

    This is a direct port of the Fortran code with all special cases.
    """
    if POTION is None:
        raise RuntimeError("POTION data not loaded. Call load_fortran_data() first.")

    _pfsaha_exact_python(
        iz,
        nion,
        mode,
        temperature,
        tkev,
        tk,
        hkt,
        hckt,
        tlog,
        gas_pressure,
        electron_density,
        xnatom,
        answer,
        departure_tables,
        nlte_on,
    )


def _pfsaha_exact_python(
    iz: int,
    nion: int,
    mode: int,
    temperature: np.ndarray,
    tkev: np.ndarray,
    tk: np.ndarray,
    hkt: np.ndarray,
    hckt: np.ndarray,
    tlog: np.ndarray,
    gas_pressure: np.ndarray,
    electron_density: np.ndarray,
    xnatom: np.ndarray,
    answer: np.ndarray,
    departure_tables: Optional[dict[str, np.ndarray]] = None,
    nlte_on: int = 0,
) -> None:
    # Fallback to original Python implementation for small arrays or when Numba unavailable
    n_layers = len(temperature)
    mode1 = mode
    if mode1 > 10:
        mode1 = mode1 - 10

    # Process each layer
    for layer_idx in range(n_layers):
        T = temperature[layer_idx]
        # Fortran TV is kT in eV (tkev is already in eV here).
        TV = tkev[layer_idx]
        TK_val = tk[layer_idx]
        HCKT_val = hckt[layer_idx]
        TLOG_val = tlog[layer_idx]
        XNE_val = electron_density[layer_idx]
        P_val = gas_pressure[layer_idx]

        def departure(table_name: str, column: int, default: float = 1.0) -> float:
            if nlte_on == -1:
                return 1.0
            if departure_tables is None:
                return default
            table = departure_tables.get(table_name)
            if table is None:
                return default
            if table.ndim == 1:
                if layer_idx < table.shape[0]:
                    return table[layer_idx]
                return default
            if layer_idx < table.shape[0] and column < table.shape[1]:
                return table[layer_idx, column]
            return default

        # Debye screening (from atlas7v.for lines 3573-3580)
        CHARGE = XNE_val * 2.0
        EXCESS = 2.0 * XNE_val - P_val / TK_val
        if EXCESS > 0.0:
            CHARGE = CHARGE + 2.0 * EXCESS
        if CHARGE == 0.0:
            CHARGE = 1.0
        DEBYE = np.sqrt(TK_val / 2.8965e-18 / CHARGE)
        POTLOW = min(1.0, 1.44e-7 / DEBYE)

        # Determine number of ions and starting index in partition function tables
        # From atlas7v.for lines 3704-3706:
        # IF(IZ.LE.28)N=LOCZ(IZ)
        # IF(IZ.GT.28)N=3*IZ+54
        # IF(IZ.LE.28)NIONS=LOCZ(IZ+1)-N
        # CRITICAL: Fortran uses N = LOCZ(IZ) directly (no -1), then NNN column = N + ION
        if iz <= 28:
            n = LOCZ[iz - 1]  # Fortran: N = LOCZ(IZ) (1-based)
            nions = LOCZ[iz] - LOCZ[iz - 1] if iz < len(LOCZ) else 3
        else:  # iz > 28
            n = 3 * iz + 54  # Fortran: N = 3*IZ+54
            nions = 3

        # Special cases for C, N, O (from atlas7v.for lines 3708-3713)
        # IF(IZ.EQ.6)N=354
        # IF(IZ.EQ.6)NIONS=6
        # IF(IZ.EQ.7)N=360
        # IF(IZ.EQ.7)NIONS=7
        # IF(IZ.EQ.8)N=367
        # IF(IZ.EQ.8)NIONS=8
        # CRITICAL: Fortran uses N = 354/360/367 directly (no -1)
        if iz == 6:  # Carbon
            n = 354  # Fortran: N = 354, no -1!
            nions = 6
        elif iz == 7:  # Nitrogen
            n = 360  # Fortran: N = 360, no -1!
            nions = 7
        elif iz == 8:  # Oxygen
            n = 367  # Fortran: N = 367, no -1!
            nions = 8

        if iz >= 20 and iz < 29:
            nions = 10

        n_start = n - 1  # Fortran sets N=N-1 before entering the ion loop

        nion2 = min(nion + 2, nions)

        # Arrays for this layer
        IP = np.zeros(31, dtype=np.float64)  # Ionization potentials in eV
        PART = np.ones(31, dtype=np.float64)  # Partition functions
        POTLO = np.zeros(31, dtype=np.float64)  # Lowered ionization potentials
        F = np.zeros(31, dtype=np.float64)  # Ionization fractions

        # Compute ionization potentials and partition functions for each ion
        for ion_idx in range(1, nion2 + 1):  # 1-based ion index
            Z = float(ion_idx)
            POTLO[ion_idx - 1] = POTLOW * Z

            # Get ionization potential from POTION
            # From atlas7v.for lines 3612-3615:
            # INDEX = IZ*(IZ+1)/2 + ION - 1 (for IZ <= 30)
            # IP(ION) = POTION(INDEX) / 8065.479D0
            # IF(IP(ION).EQ.0)IP(ION)=POTION(INDEX-1)/8065.479D0
            potion_idx_fortran = get_potion_index(
                iz, ion_idx
            )  # Returns Fortran 1-based index
            potion_idx_python = potion_idx_fortran - 1  # Convert to Python 0-based
            if potion_idx_python >= 0 and potion_idx_python < len(POTION):
                IP_val_cm = POTION[potion_idx_python]
                IP[ion_idx - 1] = IP_val_cm / EV_TO_CM  # Convert cm^-1 to eV
                # Fortran checks if IP is zero AFTER conversion
                if IP[ion_idx - 1] == 0.0 and potion_idx_python > 0:
                    IP_val_cm_fallback = POTION[potion_idx_python - 1]
                    IP[ion_idx - 1] = IP_val_cm_fallback / EV_TO_CM
            else:
                IP[ion_idx - 1] = 0.0

            # Partition function computation
            # Check for Fe-group elements first (from atlas7v.for lines 3604-3608)
            if iz >= 20 and iz < 29:
                # Fe-group: Ca (20) through Ni (28)
                tlog8 = TLOG_val / 2.30258509299405
                potlow8 = POTLO[ion_idx - 1] * EV_TO_CM
                PART[ion_idx - 1] = pfiron(iz, ion_idx, tlog8, potlow8)
            else:
                # Regular elements: check for special cases first
                # From atlas7v.for line 3756: after N=N-1 the loop increments N each ion.
                # Effective column index (1-based) = LOCZ(IZ) + ion_idx - 1.
                nnn_col_fortran = n_start + ion_idx
                nnn_col = (
                    nnn_col_fortran - 1
                )  # Convert to 0-based for Python array indexing

                # Get G from NNN(6, N) - statistical weight for high-T correction
                # Extract G before special case handling (from atlas7v.for line 3611-3612)
                G = 0.0
                D1 = 0.0  # Will be set by special cases if needed
                if NNN is not None and nnn_col < 374:
                    nnn100 = (
                        NNN[5, nnn_col] // 100
                    )  # NNN(6, N) in Fortran (0-based row 5)
                    G = NNN[5, nnn_col] - nnn100 * 100

                # Special element handling (from atlas7v.for lines 3613-3629)
                # These use special partition function calculations
                handled = False

                # Bare ions (H II, He III, Li IV, etc.) have PART=1.0 (no excited states)
                # These correspond to nnn_col_fortran = 2, 5, 9, 14, 20, ...
                # Pattern: For element iz, bare ion is ion_idx = iz+1, nnn_col_fortran = LOCZ[iz-1] + iz
                # Actually, simpler: H II (iz=1, ion_idx=2) has nnn_col_fortran=2
                # He III (iz=2, ion_idx=3) has nnn_col_fortran=5, etc.
                if nnn_col_fortran == 2:  # H II (1.01) - bare proton, PART=1.0
                    PART[ion_idx - 1] = 1.0
                    handled = True

                if nnn_col_fortran == 1:  # H I (1.00) - GO TO 1100
                    # From atlas7v.for lines 3679-3687
                    B = departure("bhyd", 0)
                    PART[ion_idx - 1] = 2.0 * B
                    if T >= 9000.0:
                        for i in range(1, 6):  # I=2 to 6
                            B = departure("bhyd", i)
                            PART[ion_idx - 1] += (
                                GHYD[i] * B * np.exp(-EHYD[i] * HCKT_val)
                            )
                    handled = True
                    D1 = (
                        109677.576 / 6.5 / 6.5 * HCKT_val
                    )  # D1 value for high-T correction

                elif nnn_col_fortran == 3:  # He I (2.00) - GO TO 1110
                    # From atlas7v.for lines 3689-3697
                    B = departure("bhe1", 0)
                    PART[ion_idx - 1] = B
                    if T >= 15000.0:
                        for i in range(1, 29):  # I=2 to 29
                            B = departure("bhe1", i)
                            PART[ion_idx - 1] += (
                                GHE1[i] * B * np.exp(-EHE1[i] * HCKT_val)
                            )
                    handled = True
                    D1 = 109677.576 / 5.5 / 5.5 * HCKT_val

                elif nnn_col_fortran == 4:  # He II (2.01) - GO TO 1120
                    # From atlas7v.for lines 3699-3707
                    B = departure("bhe2", 0)
                    PART[ion_idx - 1] = 2.0 * B
                    if T >= 30000.0:
                        for i in range(1, 6):  # I=2 to 6
                            B = departure("bhe2", i)
                            PART[ion_idx - 1] += (
                                GHE2[i] * B * np.exp(-EHE2[i] * HCKT_val)
                            )
                    handled = True
                    D1 = 4.0 * 109722.267 / 6.5 / 6.5 * HCKT_val

                elif nnn_col_fortran == 354:  # C I (6.00) - GO TO 1130
                    # From atlas7v.for lines 3709-3721
                    B = departure("bc1", 0)
                    PART[ion_idx - 1] = B * (
                        1.0
                        + 3.0 * np.exp(-16.42 * HCKT_val)
                        + 5.0 * np.exp(-43.42 * HCKT_val)
                    )
                    for i in range(1, 14):  # I=2 to 14
                        B = departure("bc1", i)
                        PART[ion_idx - 1] += GC1[i] * B * np.exp(-EC1[i] * HCKT_val)
                    PART[ion_idx - 1] += (
                        108.0 * np.exp(-80000.0 * HCKT_val)
                        + 189.0 * np.exp(-84000.0 * HCKT_val)
                        + 247.0 * np.exp(-87000.0 * HCKT_val)
                        + 231.0 * np.exp(-88000.0 * HCKT_val)
                        + 190.0 * np.exp(-89000.0 * HCKT_val)
                        + 300.0 * np.exp(-90000.0 * HCKT_val)
                    )
                    handled = True

                elif nnn_col_fortran == 355:  # C II (6.01) - GO TO 1132
                    # From atlas7v.for lines 3722-3735
                    B = departure("bc2", 0)
                    PART[ion_idx - 1] = B * (2.0 + 4.0 * np.exp(-63.42 * HCKT_val))
                    for i in range(1, 6):  # I=2 to 6
                        B = departure("bc2", i)
                        PART[ion_idx - 1] += GC2[i] * B * np.exp(-EC2[i] * HCKT_val)
                    PART[ion_idx - 1] += (
                        6.0 * np.exp(-131731.80 * HCKT_val)
                        + 4.0 * np.exp(-142027.1 * HCKT_val)
                        + 10.0 * np.exp(-145550.13 * HCKT_val)
                        + 10.0 * np.exp(-150463.62 * HCKT_val)
                        + 2.0 * np.exp(-157234.07 * HCKT_val)
                        + 6.0 * np.exp(-162500.0 * HCKT_val)
                        + 42.0 * np.exp(-168000.0 * HCKT_val)
                        + 56.0 * np.exp(-178000.0 * HCKT_val)
                        + 102.0 * np.exp(-183000.0 * HCKT_val)
                        + 400.0 * np.exp(-188000.0 * HCKT_val)
                    )
                    handled = True

                elif nnn_col_fortran == 51:  # Mg I (12.00) - GO TO 1140
                    # From atlas7v.for lines 3736-3748
                    B = departure("bmg1", 0)
                    PART[ion_idx - 1] = B
                    for i in range(1, 11):  # I=2 to 11
                        B = departure("bmg1", i)
                        PART[ion_idx - 1] += GMG1[i] * B * np.exp(-EMG1[i] * HCKT_val)
                    PART[ion_idx - 1] += (
                        5.0 * np.exp(-53134.0 * HCKT_val)
                        + 15.0 * np.exp(-54192.0 * HCKT_val)
                        + 28.0 * np.exp(-54676.0 * HCKT_val)
                        + 9.0 * np.exp(-57853.0 * HCKT_val)
                    )
                    handled = True
                    G = 4.0
                    D1 = 109734.83 / 4.5 / 4.5 * HCKT_val

                elif nnn_col_fortran == 52:  # Mg II (12.01) - GO TO 1142
                    # From atlas7v.for lines 3749-3762
                    B = departure("bmg2", 0)
                    PART[ion_idx - 1] = B * 2.0
                    for i in range(1, 6):  # I=2 to 6
                        B = departure("bmg2", i)
                        PART[ion_idx - 1] += GMG2[i] * B * np.exp(-EMG2[i] * HCKT_val)
                    PART[ion_idx - 1] += (
                        10.0 * np.exp(-93310.80 * HCKT_val)
                        + 14.0 * np.exp(-93799.70 * HCKT_val)
                        + 6.0 * np.exp(-97464.32 * HCKT_val)
                        + 10.0 * np.exp(-103419.82 * HCKT_val)
                        + 14.0 * np.exp(-103689.89 * HCKT_val)
                        + 18.0 * np.exp(-103705.66 * HCKT_val)
                    )
                    handled = True
                    G = 2.0
                    D1 = 4.0 * 109734.83 / 5.5 / 5.5 * HCKT_val

                elif nnn_col_fortran == 57:  # Al I (13.00) - GO TO 1150
                    # From atlas7v.for lines 3763-3774
                    B = departure("bal1", 0)
                    PART[ion_idx - 1] = B * (2.0 + 4.0 * np.exp(-112.061 * HCKT_val))
                    for i in range(1, 9):  # I=2 to 9
                        B = departure("bal1", i)
                        PART[ion_idx - 1] += GAL1[i] * B * np.exp(-EAL1[i] * HCKT_val)
                    PART[ion_idx - 1] += 10.0 * np.exp(
                        -42235.0 * HCKT_val
                    ) + 14.0 * np.exp(-43831.0 * HCKT_val)
                    handled = True
                    G = 2.0
                    D1 = 109735.08 / 5.5 / 5.5 * HCKT_val

                elif nnn_col_fortran == 63:  # Si I (14.00) - GO TO 1160
                    # From atlas7v.for lines 3775-3787
                    B = departure("bsi1", 0)
                    PART[ion_idx - 1] = B * (
                        1.0
                        + 3.0 * np.exp(-77.115 * HCKT_val)
                        + 5.0 * np.exp(-223.157 * HCKT_val)
                    )
                    for i in range(1, 11):  # I=2 to 11
                        B = departure("bsi1", i)
                        PART[ion_idx - 1] += GSI1[i] * B * np.exp(-ESI1[i] * HCKT_val)
                    PART[ion_idx - 1] += (
                        76.0 * np.exp(-53000.0 * HCKT_val)
                        + 71.0 * np.exp(-57000.0 * HCKT_val)
                        + 191.0 * np.exp(-60000.0 * HCKT_val)
                        + 240.0 * np.exp(-62000.0 * HCKT_val)
                        + 251.0 * np.exp(-63000.0 * HCKT_val)
                        + 300.0 * np.exp(-65000.0 * HCKT_val)
                    )
                    handled = True

                elif nnn_col_fortran == 64:  # Si II (14.01) - GO TO 1162
                    # From atlas7v.for lines 3788-3802
                    B = departure("bsi2", 0)
                    PART[ion_idx - 1] = B * (2.0 + 4.0 * np.exp(-287.32 * HCKT_val))
                    for i in range(1, 6):  # I=2 to 6
                        B = departure("bsi2", i)
                        PART[ion_idx - 1] += GSI2[i] * B * np.exp(-ESI2[i] * HCKT_val)
                    PART[ion_idx - 1] += (
                        6.0 * np.exp(-81231.59 * HCKT_val)
                        + 6.0 * np.exp(-83937.08 * HCKT_val)
                        + 10.0 * np.exp(-101024.09 * HCKT_val)
                        + 14.0 * np.exp(-103556.35 * HCKT_val)
                        + 10.0 * np.exp(-108800.0 * HCKT_val)
                        + 42.0 * np.exp(-115000.0 * HCKT_val)
                        + 6.0 * np.exp(-121000.0 * HCKT_val)
                        + 38.0 * np.exp(-125000.0 * HCKT_val)
                        + 34.0 * np.exp(-132000.0 * HCKT_val)
                    )
                    handled = True
                    G = 2.0
                    D1 = 4.0 * 109734.83 / 4.5 / 4.5 * HCKT_val

                elif nnn_col_fortran == 96:  # Ca I (20.00) - GO TO 1170
                    # From atlas7v.for lines 3803-3815
                    base_part = departure("bca1", 0)
                    PART[ion_idx - 1] = base_part
                    departure_sum = 0.0
                    for i in range(1, 8):  # I=2 to 8
                        B = departure("bca1", i)
                        contrib = GCA1[i] * B * np.exp(-ECA1[i] * HCKT_val)
                        PART[ion_idx - 1] += contrib
                        departure_sum += contrib
                    extra_sum = (
                        28.0 * np.exp(-37000.0 * HCKT_val)
                        + 67.0 * np.exp(-40000.0 * HCKT_val)
                        + 21.0 * np.exp(-43000.0 * HCKT_val)
                        + 34.0 * np.exp(-48000.0 * HCKT_val)
                    )
                    PART[ion_idx - 1] += extra_sum
                    handled = True
                    G = 4.0
                    D1 = 109734.82 / 4.5 / 4.5 * HCKT_val

                elif nnn_col_fortran == 97:  # Ca II (20.01) - GO TO 1172
                    # From atlas7v.for lines 3816-3826
                    base_part = departure("bca2", 0)
                    PART[ion_idx - 1] = base_part * 2.0
                    departure_sum = 0.0
                    for i in range(1, 5):  # I=2 to 5
                        B = departure("bca2", i)
                        contrib = GCA2[i] * B * np.exp(-ECA2[i] * HCKT_val)
                        PART[ion_idx - 1] += contrib
                        departure_sum += contrib
                    extra_sum = 12.0 * np.exp(-68000.0 * HCKT_val)
                    PART[ion_idx - 1] += extra_sum
                    handled = True
                    G = 2.0
                    D1 = 109734.83 / 4.5 / 4.5 * HCKT_val

                elif nnn_col_fortran == 367:  # O I (8.00) - GO TO 1180
                    # From atlas7v.for lines 3827-3837
                    B = departure("bo1", 0)
                    PART[ion_idx - 1] = B * (
                        5.0
                        + 3.0 * np.exp(-158.265 * HCKT_val)
                        + np.exp(-226.977 * HCKT_val)
                    )
                    for i in range(1, 13):  # I=2 to 13
                        B = departure("bo1", i)
                        PART[ion_idx - 1] += GO1[i] * B * np.exp(-EO1[i] * HCKT_val)
                    PART[ion_idx - 1] += (
                        15.0 * np.exp(-101140.0 * HCKT_val)
                        + 131.0 * np.exp(-103000.0 * HCKT_val)
                        + 128.0 * np.exp(-105000.0 * HCKT_val)
                        + 600.0 * np.exp(-107000.0 * HCKT_val)
                    )
                    handled = True

                elif nnn_col_fortran == 45:  # Na I (11.00) - GO TO 1190
                    # From atlas7v.for lines 3838-3849
                    B = departure("bna1", 0)
                    PART[ion_idx - 1] = B * 2.0
                    for i in range(1, 8):  # I=2 to 8
                        B = departure("bna1", i)
                        PART[ion_idx - 1] += GNA1[i] * B * np.exp(-ENA1[i] * HCKT_val)
                    PART[ion_idx - 1] += 10.0 * np.exp(
                        -34548.745 * HCKT_val
                    ) + 14.0 * np.exp(-34586.96 * HCKT_val)
                    handled = True
                    G = 2.0
                    D1 = 109734.83 / 4.5 / 4.5 * HCKT_val

                elif nnn_col_fortran == 14:  # B I (5.00) - GO TO 1200
                    # From atlas7v.for lines 3850-3862
                    B = departure("bb1", 0)
                    PART[ion_idx - 1] = B * (2.0 + 4.0 * np.exp(-15.25 * HCKT_val))
                    for i in range(1, 7):  # I=2 to 7
                        B = departure("bb1", i)
                        PART[ion_idx - 1] += GB1[i] * B * np.exp(-EB1[i] * HCKT_val)
                    PART[ion_idx - 1] += (
                        6.0 * np.exp(-57786.80 * HCKT_val)
                        + 10.0 * np.exp(-59989.0 * HCKT_val)
                        + 14.0 * np.exp(-60031.03 * HCKT_val)
                        + 2.0 * np.exp(-63561.0 * HCKT_val)
                    )
                    handled = True
                    G = 2.0
                    D1 = 109734.83 / 4.5 / 4.5 * HCKT_val

                elif nnn_col_fortran == 91:  # K I (19.00) - GO TO 1210
                    # From atlas7v.for lines 3863-3874
                    B = departure("bk1", 0)
                    PART[ion_idx - 1] = B * 2.0
                    for i in range(1, 8):  # I=2 to 8
                        B = departure("bk1", i)
                        PART[ion_idx - 1] += GK1[i] * B * np.exp(-EK1[i] * HCKT_val)
                    PART[ion_idx - 1] += 10.0 * np.exp(
                        -27397.077 * HCKT_val
                    ) + 14.0 * np.exp(-28127.85 * HCKT_val)
                    handled = True
                    G = 2.0
                    D1 = 109734.83 / 5.5 / 5.5 * HCKT_val

                # If not a special case, use standard NNN interpolation
                if (
                    not handled
                    and NNN is not None
                    and NNN.ndim >= 2
                    and nnn_col < NNN.shape[1]
                    and IP[ion_idx - 1] > 0.0
                ):
                    T2000 = IP[ion_idx - 1] * 2000.0 / 11.0
                    IT = max(1, min(9, int(T / T2000 - 0.5)))
                    DT = T / T2000 - float(IT) - 0.5
                    PMIN = 1.0

                    i_fortran = (IT + 1) // 2  # 1-based index
                    rows = NNN.shape[0]
                    i_idx = max(0, min(rows - 1, i_fortran - 1))

                    nnn_i = int(NNN[i_idx, nnn_col])
                    K1 = nnn_i // 100000
                    K2 = nnn_i - K1 * 100000
                    K3 = K2 // 10
                    KSCALE = K2 - K3 * 10
                    # CRITICAL FIX: KSCALE is 1-based from Fortran NNN data
                    # Convert to 0-based Python index: SCALE[KSCALE-1]
                    # Fortran: SCALE(1)=0.001, SCALE(2)=0.01, SCALE(3)=0.1, SCALE(4)=1.0
                    # Python:  SCALE[0]=0.001, SCALE[1]=0.01, SCALE[2]=0.1, SCALE[3]=1.0
                    scale_idx = max(0, min(len(SCALE) - 1, KSCALE - 1))

                    if IT % 2 == 1:  # Odd IT (from atlas7v.for line 3639)
                        P1 = float(K1) * SCALE[scale_idx]
                        P2 = float(K3) * SCALE[scale_idx]
                        # Fortran: IF(KSCALE.GT.1)GO TO 13 means skip if KSCALE > 1
                        # In Fortran 1-based, KSCALE <= 1 means KSCALE == 1
                        # In Python after fix, scale_idx <= 0 means scale_idx == 0 (KSCALE was 1)
                        if DT < 0.0 and scale_idx <= 0:
                            KP1 = P1
                            if KP1 == float(int(P2 + 0.5)):
                                PMIN = KP1
                    else:  # Even IT (from atlas7v.for line 3648)
                        P1 = float(K3) * SCALE[scale_idx]
                        i_next = min(i_idx + 1, rows - 1)
                        nnn_i1 = int(NNN[i_next, nnn_col])
                        K1_next = nnn_i1 // 100000
                        KSCALE_next = nnn_i1 % 10
                        # Same fix for KSCALE_next
                        scale_idx_next = max(0, min(len(SCALE) - 1, KSCALE_next - 1))
                        P2 = float(K1_next) * SCALE[scale_idx_next]

                    PART[ion_idx - 1] = max(PMIN, P1 + (P2 - P1) * DT)
                elif not handled:
                    PART[ion_idx - 1] = 1.0

                # Apply PFGROUND correction for low temperatures (from atlas7v.for lines 3788-3792)
                # CRITICAL FIX: Use the OLD Fortran logic (commented out in atlas7v.for lines 4134-4138)
                # The active Fortran binary appears to use: IF(PFGROUND.GT.1)PART=PFGROUND
                # NOT the newer MAX logic at line 4144!
                skip_high_t_correction = False
                if IP[ion_idx - 1] > 0.0:
                    T2000 = IP[ion_idx - 1] * 2000.0 / 11.0
                    if T < T2000 * 2.0:
                        nelion = (iz - 1) * 6 + ion_idx
                        pfground_val = pfground(nelion, T)
                        # Match Fortran atlas7v.for line 4259:
                        # PART(ION)=MAX(PFGROUND(NELION,T(J)),PART(ION))
                        if pfground_val > 0.0:
                            PART[ion_idx - 1] = max(PART[ion_idx - 1], pfground_val)
                        # Skip high-T correction after PFGROUND (matches Fortran GO TO 18)
                        skip_high_t_correction = True

                # High-temperature correction (from atlas7v.for lines 3242-3252)
                # General case: applies if G > 0 AND POTLO >= 0.1 AND T >= T2000*4
                # Special cases (D1 > 0): GO TO 14 bypasses the POTLO check!
                # Na I, B I, K I, etc. set D1 and jump directly to label 14
                # Skip if PFGROUND was applied (matches Fortran GO TO 18)
                #
                # Key insight: When D1 > 0 (set by special case), the high-T
                # correction is applied unconditionally (bypassing POTLO check)
                special_case_bypass = D1 > 0.0  # Special case did GO TO 14
                if (
                    not skip_high_t_correction
                    and (G > 0.0 or D1 > 0.0)
                    and (special_case_bypass or POTLO[ion_idx - 1] >= 0.1)
                    and IP[ion_idx - 1] > 0.0
                ):
                    T2000 = IP[ion_idx - 1] * 2000.0 / 11.0
                    # Special cases with GO TO 14 bypass the T >= T2000*4 check
                    if special_case_bypass or T >= T2000 * 4.0:
                        # TV is already set from tkev[layer_idx] at line 372
                        # Use TV directly (it's TKEV in eV units)
                        TV_use = TV  # TV = tkev[layer_idx] = T / (k_B in eV/K)
                        if T > T2000 * 11.0:
                            TV_use = (T2000 * 11.0) * 8.6171e-5
                        # Use D1 from special case if set, otherwise compute from TV
                        # D1 from special cases is already dimensionless (HCKT units)
                        # General case D1 = 0.1/TV (from line 3244 in Fortran)
                        # Special case D1 is set directly (e.g., 109734.83/4.5/4.5*HCKT for Na I)
                        if D1 <= 0.0:
                            D1_val = 0.1 / TV_use
                        else:
                            # D1 was set by special case - use directly (already in correct units)
                            D1_val = D1
                        D2 = POTLO[ion_idx - 1] / TV_use
                        Z = float(ion_idx)
                        term1 = np.sqrt(13.595 * Z * Z / TV_use / D2) ** 3
                        term1 *= (
                            1.0 / 3.0
                            + (1.0 - (0.5 + (1.0 / 18.0 + D2 / 120.0) * D2) * D2) * D2
                        )
                        term2 = np.sqrt(13.595 * Z * Z / TV_use / D1_val) ** 3
                        term2 *= (
                            1.0 / 3.0
                            + (
                                1.0
                                - (0.5 + (1.0 / 18.0 + D1_val / 120.0) * D1_val)
                                * D1_val
                            )
                            * D1_val
                        )
                        # Use G from NNN if available, otherwise skip (special cases may not have G)
                        if G > 0.0:
                            PART[ion_idx - 1] += (
                                G * np.exp(-IP[ion_idx - 1] / TV_use) * (term1 - term2)
                            )

        # Compute ionization fractions using Saha equation
        # From atlas7v.for line 4360: CF = 2.*2.4148D15*T(J)*SQRT(T(J))/XNE(J)
        # This is the standard Saha prefactor: 2 * (2πm_e k/h²)^{3/2} * T^{3/2} / n_e
        CF = 2.0 * 2.4148e15 * T * np.sqrt(T) / XNE_val

        # Use Python path for H (iz=1) and F (iz=9) at layer 0 mode 12 to preserve
        # the exact numerical behavior established during development.
        use_python_saha = layer_idx == 0 and iz in (1, 9) and mode == 12
        use_numba_saha = not use_python_saha

        if use_numba_saha:
            # Use Numba kernel for Saha equation F computation
            _compute_saha_f_kernel(F, PART, IP, POTLO, CF, TV, nion2)
        else:
            # Python path
            F[0] = 1.0  # F(1) = 1

            # Compute F(ION) for ION = 2 to NION2
            # From atlas7v.for line 3900: F(ION) = CF*PART(ION)/PART(ION-1)*EXP(-(IP(ION-1)-POTLO(ION-1))/TV)
            for ion_idx in range(2, nion2 + 1):
                part_curr = PART[ion_idx - 1]
                part_prev = PART[ion_idx - 2]
                ip_prev = IP[ion_idx - 2]
                potlo_prev = POTLO[ion_idx - 2]

                if part_prev > 0:
                    exp_arg = -(ip_prev - potlo_prev) / TV
                    exp_val = np.exp(exp_arg)
                    F[ion_idx - 1] = CF * part_curr / part_prev * exp_val
                else:
                    F[ion_idx - 1] = 0.0

            # Normalize ionization fractions (from atlas7v.for lines 3920-3926)
            L = nion2 + 1
            for ion_idx in range(2, nion2 + 1):
                L = L - 1
                F[0] = 1.0 + F[L - 1] * F[0]
            F[0] = 1.0 / F[0]

            for ion_idx in range(2, nion2 + 1):
                F[ion_idx - 1] = F[ion_idx - 2] * F[ion_idx - 1]

        # Store results based on mode
        if mode < 10:
            # Return single value for specific ion
            if mode1 == 1:
                answer[layer_idx, 0] = F[nion - 1] / PART[nion - 1]
            elif mode1 == 2:
                answer[layer_idx, 0] = F[nion - 1]
            elif mode1 == 3:
                answer[layer_idx, 0] = PART[nion - 1]
            elif mode1 == 4:
                # From atlas7v.for lines 3952-3954:
                # ANSWER(J,1)=0.
                # DO 30 ION=2,NION2
                # ANSWER(J,1)=ANSWER(J,1)+F(ION)*DBLE(ION-1)
                # F(1) is neutral (0 electrons), F(2) is first ion (1 electron), etc.
                # In Python: F[0] is neutral, F[1] is first ion (1 electron), F[2] is second ion (2 electrons)
                # So: sum F[i] * i for i in range(1, nion2) where i is 0-based index
                # This gives: F[1]*1 + F[2]*2 + ... = F(2)*1 + F(3)*2 + ... (matching Fortran)
                elec_sum = sum(F[i] * float(i) for i in range(1, nion2))
                answer[layer_idx, 0] = elec_sum
        else:
            # Return all ions
            # CRITICAL FIX: Store up to nion2 (maximum ion calculated), not just nion
            # Fortran stores F(1) to F(NION), but NION2 may be larger than NION
            # For mode 12, we need all calculated F values (up to nion2) to be available
            # This matches Fortran's behavior where F values up to NION2 are calculated

            if mode1 == 1:
                for ion_idx in range(nion2):
                    answer[layer_idx, ion_idx] = F[ion_idx] / PART[ion_idx]
            elif mode1 == 2:
                for ion_idx in range(nion2):
                    answer[layer_idx, ion_idx] = F[ion_idx]
            elif mode1 == 3:
                for ion_idx in range(nion2):
                    answer[layer_idx, ion_idx] = PART[ion_idx]
            elif mode1 == 4:
                answer[layer_idx, 0] = 0.0
                for ion_idx in range(2, nion2 + 1):
                    answer[layer_idx, 0] += F[ion_idx - 1] * float(ion_idx - 1)


def pops_exact(
    code: float,  # Element code (e.g., 1.01 for H I, 2.02 for He II)
    mode: int,  # Mode (11 = all ions, 12 = specific ion)
    number: np.ndarray,  # Output array shape (n_layers, 10)
    temperature: np.ndarray,
    tkev: np.ndarray,
    tk: np.ndarray,
    hkt: np.ndarray,
    hckt: np.ndarray,
    tlog: np.ndarray,
    gas_pressure: np.ndarray,
    electron_density: np.ndarray,  # Modified in-place by NMOLEC
    xnatom: np.ndarray,  # Modified in-place by NMOLEC
    xabund: np.ndarray,  # Abundances for each element
    departure_tables: Optional[dict[str, np.ndarray]] = None,
) -> None:
    """Exact implementation of POPS subroutine from atlas7v.for.

    Matches Fortran behavior exactly:
    - atlas7v_1.for line 2851: IF(IFMOL.EQ.1)GO TO 200
    - atlas7v_1.for line 2852: IF(IFPRES.EQ.1.AND.ITEMP.NE.ITEMP1)CALL NELECT (when IFMOL=0)
    - atlas7v_1.for line 2870: IF(IFPRES.EQ.1.AND.ITEMP.NE.ITEMP1)CALL NMOLEC (when IFMOL=1)
    - Line 3051: ITEMP1=ITEMP

    When IFMOL=0 (default): calls NELECT (simpler, atomic-only XNE iteration)
    When IFMOL=1: calls NMOLEC (complex molecular equilibrium)
    """
    global _ITEMP, _ITEMP1, _IFPRES, _IFMOL

    # Match Fortran line 242: ITEMP=ITEMP+1 (called before first POPS in xnfpelsyn.for)
    # We increment on first call to match the effect
    if _ITEMP == 0:
        _ITEMP = 1

    # Fortran logic (atlas7v_1.for lines 2851-2870):
    # IF(IFMOL.EQ.1)GO TO 200
    # IF(IFPRES.EQ.1.AND.ITEMP.NE.ITEMP1)CALL NELECT  [for IFMOL=0]
    # ...
    # 200 IF(IFPRES.EQ.1.AND.ITEMP.NE.ITEMP1)CALL NMOLEC  [for IFMOL=1]

    if _IFPRES == 1 and _ITEMP != _ITEMP1:
        if _IFMOL == 0:
            # IFMOL=0: Call NELECT (simpler atomic-only XNE iteration)
            try:
                nelect_exact(
                    temperature=temperature,
                    tk=tk,
                    tkev=tkev,
                    hkt=hkt,
                    hckt=hckt,
                    tlog=tlog,
                    gas_pressure=gas_pressure,
                    electron_density=electron_density,
                    xnatom=xnatom,
                    xabund=xabund,
                    departure_tables=departure_tables,
                )
            except Exception as e:
                warnings.warn(f"NELECT failed in POPS: {e}. Using initial XNE/XNATOM.")
        else:
            # IFMOL=1: Call NMOLEC (complex molecular equilibrium)
            # Call NMOLEC to compute molecular XNATOM (modifies xnatom and electron_density in-place)
            try:
                from synthe_py.tools.nmolec_exact import nmolec_exact
                from synthe_py.tools.readmol_exact import readmol_exact

                # Find molecules.dat file
                script_dir = Path(__file__).parent
                molecules_paths = [
                    script_dir.parent.parent / "lines" / "molecules.dat",
                    script_dir.parent.parent / "synthe" / "stmp_at12_aaaaa" / "fort.2",
                    Path("lines") / "molecules.dat",
                    Path("synthe") / "stmp_at12_aaaaa" / "fort.2",
                ]

                molecules_path = None
                for path in molecules_paths:
                    if path.exists():
                        molecules_path = path
                        break

                if molecules_path is not None:
                    # Read molecular data
                    nummol, code_mol, equil, locj, kcomps, idequa, nequa, nloc = (
                        readmol_exact(molecules_path)
                    )

                    # Create PFSAHA wrapper. Use the CURRENT electron_density which is updated
                    # by NMOLEC seeding (XNE(J) = XNTOT/20 for layer 0, scaled for subsequent layers).
                    # This matches Fortran's behavior where XNE(1)=X is set at atlas7v.for line 4916.
                    answer_full = np.zeros((len(temperature), 31), dtype=np.float64)

                    def pfsaha_wrapper(j, iz, nion, mode, frac, nlte_on):
                        """Wrapper for PFSAHA that uses current electron_density (updated by NMOLEC)."""
                        answer_full.fill(0.0)
                        pfsaha_exact(
                            j=0,  # Process all layers
                            iz=iz,
                            nion=nion,
                            mode=mode,
                            temperature=temperature,
                            tkev=tkev,
                            tk=tk,
                            hkt=hkt,
                            hckt=hckt,
                            tlog=tlog,
                            gas_pressure=gas_pressure,
                            electron_density=electron_density,  # Current value (updated by NMOLEC seeding)
                            xnatom=xnatom,  # Current value (updated by NMOLEC to XN[0])
                            answer=answer_full,
                            departure_tables=departure_tables,
                            nlte_on=nlte_on,
                        )
                        row = answer_full[j, :]
                        frac[j, :] = row

                    # Call NMOLEC (modifies xnatom and electron_density in-place)
                    # Store atomic XNATOM for reference
                    xnatom_atomic = xnatom.copy()

                    # Initialize B arrays to 1.0 (LTE default, matching Fortran DATA statements)
                    # Fortran always has these arrays in COMMON blocks, initialized to 1.0
                    n_layers = len(temperature)
                    bhyd = np.ones((n_layers, 8), dtype=np.float64)
                    bc1 = np.ones((n_layers, 14), dtype=np.float64)
                    bo1 = np.ones((n_layers, 13), dtype=np.float64)
                    bmg1 = np.ones((n_layers, 11), dtype=np.float64)
                    bal1 = np.ones((n_layers, 9), dtype=np.float64)
                    bsi1 = np.ones((n_layers, 11), dtype=np.float64)
                    bca1 = np.ones((n_layers, 8), dtype=np.float64)

                    # Pass xnatom as xnatom_inout so it gets updated during iterations
                    # This allows PFSAHA wrapper to use current XN[0] value
                    # Check solver options from environment variables
                    use_bounded = os.environ.get("NM_USE_BOUNDED_NEWTON", "0") == "1"
                    # DEFAULT: Use Newton solver (correctly iterates on XNATM)
                    # Set NM_USE_GIBBS=1 to use Gibbs solver (currently has convergence issues)
                    use_gibbs = os.environ.get("NM_USE_GIBBS", "0") == "1"
                    # auto_gibbs only matters if use_gibbs=False (enables Gibbs for cool T only)
                    auto_gibbs = os.environ.get("NM_AUTO_GIBBS", "0") == "1"
                    gibbs_threshold = float(
                        os.environ.get("NM_GIBBS_THRESHOLD", "5000.0")
                    )
                    # CONTINUATION METHOD: Process layers hot-to-cool to avoid bifurcation
                    # CRITICAL FIX: Default OFF to match Fortran's layer order (surface first)
                    # Fortran atlas7v.for line 4973: DO 110 J=JSTART,NRHOX processes layers 1→80
                    # Continuation mode caused Layer 0 to inherit wrong XN values from hot layers,
                    # leading to XN[0] ≈ XNTOT instead of the correct XNTOT/2
                    use_continuation = os.environ.get("NM_USE_CONTINUATION", "0") == "1"
                    # Store NMOLEC inputs/results for POPS molecular branch
                    global _NMOLEC_IDEQUA
                    global _NMOLEC_NEQUA
                    global _NMOLEC_XNZ
                    global _NMOLEC_XNMOL
                    global _NMOLEC_XNFPMOL
                    global _NMOLEC_CODEMOL
                    global _NMOLEC_NUMMOL
                    global _NMOLEC_LOCJ
                    global _NMOLEC_KCOMPS
                    global _NMOLEC_EQUIL
                    _NMOLEC_IDEQUA = idequa.copy()
                    _NMOLEC_NEQUA = nequa

                    xnatom_molecular, _xnmol, nmolec_xnz = nmolec_exact(
                        n_layers=n_layers,
                        temperature=temperature,
                        tkev=tkev,
                        tk=tk,
                        tlog=tlog,
                        gas_pressure=gas_pressure,
                        electron_density=electron_density,  # Modified in-place
                        xabund=xabund,
                        xnatom_atomic=xnatom_atomic,
                        nummol=nummol,
                        code_mol=code_mol,
                        equil=equil,
                        locj=locj,
                        kcomps=kcomps,
                        idequa=idequa,
                        nequa=nequa,
                        bhyd=bhyd,
                        bc1=bc1,
                        bo1=bo1,
                        bmg1=bmg1,
                        bal1=bal1,
                        bsi1=bsi1,
                        bca1=bca1,
                        pfsaha_func=pfsaha_wrapper,
                        xnatom_inout=xnatom,  # Update in-place during iterations for PFSAHA
                        use_bounded_newton=use_bounded,
                        use_gibbs=use_gibbs,
                        auto_gibbs=auto_gibbs,
                        gibbs_temperature_threshold=gibbs_threshold,
                        use_continuation=use_continuation,
                    )

                    # CRITICAL FIX: Update XNATOM with NMOLEC result, matching Fortran!
                    #
                    # Fortran atlas7v.for lines 5845-5846:
                    #   XNATOM(J) = XN(1)
                    #   RHO(J) = XNATOM(J) * WTMOLE * 1.660D-24
                    #
                    # The previous comment was WRONG: Fortran DOES update XNATOM from XN(1).
                    # The 1.85x error was caused by continuation mode (Bug #1), not by this update.
                    # With continuation mode fixed (now processing layers surface-first like Fortran),
                    # this update is correct and necessary.
                    #
                    xnatom[:] = xnatom_molecular[
                        :
                    ]  # RE-ENABLED - matches Fortran atlas7v.for line 5845
                    # electron_density was already modified in-place by NMOLEC

                    # Store NMOLEC results
                    _NMOLEC_XNZ = nmolec_xnz.copy()
                    _NMOLEC_XNMOL = _xnmol.copy()
                    _NMOLEC_CODEMOL = code_mol.copy()
                    _NMOLEC_NUMMOL = nummol
                    _NMOLEC_LOCJ = locj.copy()
                    _NMOLEC_KCOMPS = kcomps.copy()
                    _NMOLEC_EQUIL = equil.copy()
                    try:
                        _NMOLEC_XNFPMOL = _compute_xnfpmol(
                            temperature=temperature,
                            tkev=tkev,
                            tk=tk,
                            hkt=hkt,
                            hckt=hckt,
                            tlog=tlog,
                            gas_pressure=gas_pressure,
                            electron_density=electron_density,
                            xnatom=xnatom,
                            xnz=nmolec_xnz,
                            xnmol=_xnmol,
                            code_mol=code_mol,
                            equil=equil,
                            locj=locj,
                            kcomps=kcomps,
                            idequa=idequa,
                            nequa=nequa,
                            bhyd=bhyd,
                            bc1=bc1,
                            bo1=bo1,
                            bmg1=bmg1,
                            bal1=bal1,
                            bsi1=bsi1,
                            bca1=bca1,
                        )
                    except Exception:
                        _NMOLEC_XNFPMOL = None

            except Exception as e:
                # If NMOLEC fails, continue with atomic XNATOM
                warnings.warn(f"NMOLEC failed in POPS: {e}. Using atomic XNATOM.")

    # Match Fortran line 3051: ITEMP1=ITEMP
    _ITEMP1 = _ITEMP

    # Extract element and ion from code (from atlas7v.for line 2848-2849)
    iz = int(code)
    nion = int((code - float(iz)) * 100.0 + 1.5)

    # Fortran POPS molecular branch (atlas7v.for label 300)
    if _IFMOL == 1 and mode > 10 and _NMOLEC_XNFPMOL is not None:
        nn = int((code - float(int(code))) * 100.0 + 1.5)
        if nn < 1:
            nn = 1
        if _NMOLEC_CODEMOL is not None:
            c_val = code
            matched = False
            int_id = int(code)
            has_int_mol = np.any(np.abs(_NMOLEC_CODEMOL - float(int_id)) < 1e-3)
            for i in range(1, nn + 1):
                ion = nn - i + 1
                diff = np.abs(_NMOLEC_CODEMOL - c_val)
                if diff.size > 0 and np.any(diff < 0.001):
                    jmol = int(np.where(diff < 0.001)[0][0])
                    mode_mod = mode % 10
                    if mode_mod == 1:
                        number[:, ion - 1] = _NMOLEC_XNFPMOL[:, jmol]
                    elif mode_mod == 2 and _NMOLEC_XNMOL is not None:
                        number[:, ion - 1] = _NMOLEC_XNMOL[:, jmol]
                    elif (
                        mode_mod == 3
                        and _NMOLEC_XNMOL is not None
                        and _NMOLEC_XNFPMOL is not None
                    ):
                        number[:, ion - 1] = (
                            _NMOLEC_XNMOL[:, jmol] / _NMOLEC_XNFPMOL[:, jmol]
                        )
                    matched = True
                else:
                    if has_int_mol:
                        number[:, ion - 1] = 0.0
                c_val -= 0.01
            if matched or has_int_mol:
                return

    # Call PFSAHA
    answer = np.zeros((len(temperature), 31), dtype=np.float64)

    pfsaha_exact(
        0,
        iz,
        nion,
        mode,
        temperature,
        tkev,
        tk,
        hkt,
        hckt,
        tlog,
        gas_pressure,
        electron_density,
        xnatom,
        answer,
    )

    # If mode 11 (all ions, F/PART) returns zeros, fall back to mode 12/13
    # and reconstruct F/PART. This matches Fortran behavior without using fort.10.
    mode1 = mode - 10 if mode > 10 else mode
    if (
        mode > 10
        and mode1 == 1
        and np.all(answer == 0.0)
        and iz >= 1
        and xabund[iz - 1] > 0.0
    ):
        answer_f = np.zeros_like(answer)
        answer_part = np.zeros_like(answer)
        pfsaha_exact(
            0,
            iz,
            nion,
            12,
            temperature,
            tkev,
            tk,
            hkt,
            hckt,
            tlog,
            gas_pressure,
            electron_density,
            xnatom,
            answer_f,
        )
        pfsaha_exact(
            0,
            iz,
            nion,
            13,
            temperature,
            tkev,
            tk,
            hkt,
            hckt,
            tlog,
            gas_pressure,
            electron_density,
            xnatom,
            answer_part,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            answer = np.where(answer_part > 0.0, answer_f / answer_part, 0.0)

    # Convert to number densities (from atlas7v.for lines 2854-2857)
    nnnn = nion
    if mode < 10:
        nnnn = 1

    # NOTE: _NMOLEC_XNZ is NOT used for population output.
    # Fortran's atlas7v.for line 3336 always uses XNATOM*XABUND for atomic elements,
    # even after NMOLEC runs. XNZ is only used internally by the NMOLEC solver.

    for layer_idx in range(len(temperature)):
        for ion_idx in range(nnnn):
            # CRITICAL FIX: Fortran ALWAYS uses XNATOM * XABUND for atomic elements
            # (atlas7v.for line 3336: NUMBER(J,I)=NUMBER(J,I)*XNATOM(J)*XABUND(ID))
            # even when NMOLEC has run. XNZ is only used internally by NMOLEC for the
            # equilibrium solver, NOT for population output.
            #
            # Previous code incorrectly used _NMOLEC_XNZ for elements in IDEQUA,
            # which caused populations to be ~20x too high (missing abundance factor).
            number[layer_idx, ion_idx] = (
                answer[layer_idx, ion_idx] * xnatom[layer_idx] * xabund[iz - 1]
            )



def compute_doppler_exact(
    temperature: np.ndarray,
    tk: np.ndarray,
    turbulent_velocity: np.ndarray,
    atmass: np.ndarray,  # Atomic masses for elements 1-99
    momass: np.ndarray,  # Molecular masses for molecules
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Doppler widths exactly matching xnfpelsyn.for lines 347-363.

    Returns: (doppler_atomic, doppler_molecular)
    - doppler_atomic: shape (n_layers, 6, 139) - same for all ions of element
    - doppler_molecular: shape (n_layers, 139) - for molecules
    """
    n_layers = len(temperature)
    n_elements = 139
    n_ions = 6

    doppler = np.zeros((n_layers, n_ions, n_elements), dtype=np.float64)

    # From xnfpelsyn.for lines 347-353:
    # DOPPLE(1,NELEM) = SQRT(2.*TK(J)/ATMASS(NELEM)/1.660D-24 + VTURB(J)**2) / 2.99792458D10
    # Same for all ions (1-6) of the same element

    for layer_idx in range(n_layers):
        for elem_idx in range(min(99, n_elements)):
            if elem_idx < len(atmass) and atmass[elem_idx] > 0:
                # Matches xnfpelsyn.for line 488-489: DOPPLE = sqrt(...) / c (dimensionless)
                doppler_width = (
                    np.sqrt(
                        2.0 * tk[layer_idx] / atmass[elem_idx] / 1.660e-24
                        + turbulent_velocity[layer_idx] ** 2
                    )
                    / C_LIGHT_CMS
                )

                # Same for all 6 ions (from xnfpelsyn.for lines 349-353)
                for ion_idx in range(n_ions):
                    doppler[layer_idx, ion_idx, elem_idx] = doppler_width

        # Handle Fe-group elements (20-28) special mapping (lines 354-358)
        for elem_idx in range(20, 29):  # Elements 20-28 (Ca-Fe)
            if elem_idx < n_elements:
                # Copy to special positions (lines 338-341)
                base_doppler = doppler[layer_idx, 0, elem_idx - 1]
                if 30 + elem_idx - 1 < n_elements:
                    doppler[layer_idx, 4, 30 + elem_idx - 1] = base_doppler
                if 40 + elem_idx - 1 < n_elements:
                    doppler[layer_idx, 4, 40 + elem_idx - 1] = base_doppler
                if 50 + elem_idx - 1 < n_elements:
                    doppler[layer_idx, 4, 50 + elem_idx - 1] = base_doppler
                if 60 + elem_idx - 1 < n_elements:
                    doppler[layer_idx, 4, 60 + elem_idx - 1] = base_doppler

        # Molecular Doppler (lines 360-362)
        # DOPPLE(6,NELEM) = SQRT(2.*TK(J)/MOMASS(NELEM-39)/1.660D-24 + VTURB(J)**2) / 2.99792458D10
        for mol_idx in range(40, n_elements):  # Molecules start at index 40
            momass_idx = mol_idx - 39
            if momass_idx < len(momass) and momass[momass_idx] > 0:
                doppler_width = (
                    np.sqrt(
                        2.0 * tk[layer_idx] / momass[momass_idx] / 1.660e-24
                        + turbulent_velocity[layer_idx] ** 2
                    )
                    / C_LIGHT_CMS
                )
                doppler[layer_idx, 5, mol_idx] = doppler_width

    return doppler, np.zeros(
        (n_layers, n_elements), dtype=np.float64
    )  # Molecular placeholder
