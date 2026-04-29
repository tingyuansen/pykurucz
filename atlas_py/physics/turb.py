"""Microturbulence update subroutines ported from ``atlas12.for``.

Fortran reference:
- ``TURB`` (atlas12.for lines 5199-5214): power-law microturbulence update.
- ``VTURBSTANDARD`` (atlas12.for lines 5108-5198): standard table-based profile.
"""

from __future__ import annotations

import math

import numpy as np

from .josh_math import _map1

# ---------------------------------------------------------------------------
# VTURBSTANDARD data tables (atlas12.for lines 5131-5168)
# ---------------------------------------------------------------------------

_VSTANDARD = np.array(
    [
        0.50e5, 0.50e5, 0.50e5, 0.51e5, 0.52e5, 0.55e5,
        0.63e5, 0.80e5, 0.90e5, 1.00e5, 1.10e5, 1.20e5, 1.30e5, 1.40e5,
        1.46e5, 1.52e5, 1.56e5, 1.60e5, 1.64e5, 1.68e5, 1.71e5, 1.74e5,
        1.76e5, 1.78e5, 1.80e5, 1.81e5, 1.82e5, 1.83e5, 1.83e5, 1.83e5,
    ],
    dtype=np.float64,
)

_TAUSTANDARD = np.array(
    [
        -20.0, -3.0, -2.67313, -2.49296, -2.31296, -1.95636,
        -1.60768, -1.26699, -1.10007, -0.93587, -0.77416, -0.61500,
        -0.45564, -0.29176, -0.18673, -0.07193,  0.01186,  0.10342,
         0.20400,  0.31605,  0.44498,  0.58875,  0.74365,  0.90604,
         1.07181,  1.23841,  1.39979,  1.55300,  2.00000, 10.00000,
    ],
    dtype=np.float64,
)

# VMAXSTANDARD(13, 25) -- rows: log g bins (-1.0..5.0 step 0.5, 13 values)
# columns: Teff bins (3000..9000 step 250, 25 values)
# (Fortran order: column-major, 13 rows per column; here stored row-major)
# Row index IG (1-based): log g = -1.0 + (IG-1)*0.5  -> IG 1..13
# Col index IT (1-based): Teff = 3000 + (IT-1)*250   -> IT 1..25
_VMAXSTANDARD = np.array(
    [
        # IT=1(3000) to IT=25(9000), one row per log g value
        # IG=1: logg=-1.0
        [3.3, 4.1, 5.2, 6.3, 7.3, 8.0, 8.0, 8.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # IG=2: logg=-0.5
        [3.0, 3.7, 4.6, 5.5, 6.4, 7.7, 8.0, 8.0, 8.0, 8.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # IG=3: logg=0.0
        [2.7, 3.3, 4.0, 4.7, 5.5, 6.4, 7.1, 7.9, 8.0, 8.0, 8.0, 8.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # IG=4: logg=0.5
        [2.4, 2.9, 3.4, 3.9, 4.6, 5.1, 5.7, 6.3, 6.9, 7.5, 8.0, 8.0, 8.0, 4.6, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # IG=5: logg=1.0
        [2.1, 2.5, 2.9, 3.3, 3.7, 4.2, 4.7, 5.2, 5.6, 6.1, 6.6, 7.1, 7.6, 8.0, 4.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # IG=6: logg=1.5
        [1.8, 2.1, 2.4, 2.7, 3.1, 3.5, 3.9, 4.3, 4.7, 5.1, 5.5, 5.9, 6.2, 6.6, 7.0, 4.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # IG=7: logg=2.0
        [1.3, 1.6, 1.9, 2.2, 2.6, 2.9, 3.2, 3.6, 4.0, 4.4, 4.7, 5.0, 5.4, 5.7, 6.1, 6.4, 3.9, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # IG=8: logg=2.5
        [0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0, 3.4, 3.7, 4.0, 4.3, 4.6, 4.9, 5.3, 5.6, 5.9, 3.7, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # IG=9: logg=3.0
        [0.6, 0.9, 1.2, 1.5, 1.8, 2.0, 2.3, 2.5, 2.8, 3.1, 3.4, 3.6, 3.9, 4.2, 4.4, 4.7, 5.0, 5.2, 3.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        # IG=10: logg=3.5
        [0.3, 0.6, 0.9, 1.2, 1.4, 1.6, 1.9, 2.1, 2.3, 2.6, 2.8, 3.0, 3.3, 3.5, 3.7, 4.0, 4.2, 4.4, 4.7, 3.4, 0.7, 0.0, 0.0, 0.0, 0.0],
        # IG=11: logg=4.0
        [0.2, 0.3, 0.6, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1, 3.3, 3.5, 3.7, 3.9, 4.1, 3.6, 1.1, 0.0, 0.0, 0.0],
        # IG=12: logg=4.5
        [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.5, 1.7, 1.9, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 3.5, 1.6, 0.0],
        # IG=13: logg=5.0
        [0.1, 0.1, 0.2, 0.4, 0.6, 0.7, 0.9, 1.1, 1.2, 1.3, 1.5, 1.7, 1.9, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.1, 3.3, 3.5, 3.6, 3.6, 2.3],
    ],
    dtype=np.float64,
)  # shape (13, 25)


def turb(
    *,
    rho: np.ndarray,
    velsnd: np.ndarray,
    trbfdg: float,
    trbcon: float,
    trbpow: float,
    trbsnd: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Port of Fortran ``TURB`` (atlas12.for lines 5199-5214).

    Computes microturbulence velocity and pressure from a power-law formula.

    Parameters
    ----------
    rho:
        Mass density at each layer (g/cm³), shape ``(nrhox,)``.
    velsnd:
        Sound speed at each layer (cm/s), shape ``(nrhox,)``.
    trbfdg:
        Turbulence density coefficient (from TURBPR COMMON).
    trbcon:
        Turbulence constant term (km/s, converted to cm/s by ×1e5).
    trbpow:
        Turbulence density power-law exponent.
    trbsnd:
        Turbulence sound-speed fraction.

    Returns
    -------
    vturb:
        Microturbulence velocity (cm/s), shape ``(nrhox,)``.
    pturb:
        Turbulence pressure (dyne/cm²), shape ``(nrhox,)``.
    """
    vturb = (trbfdg * rho ** trbpow + trbsnd * velsnd / 1.0e5 + trbcon) * 1.0e5
    pturb = rho * vturb ** 2 * 0.5
    return vturb, pturb


def vturbstandard(
    *,
    teff: float,
    glog: float,
    taustd: np.ndarray,
    vnew: float = -99.0e5,
) -> np.ndarray:
    """Port of Fortran ``VTURBSTANDARD`` (atlas12.for lines 5108-5198).

    Computes a depth-dependent microturbulence velocity profile either from
    a standard solar calibration table (when ``vnew == -99e5``) or from a
    user-supplied amplitude ``vnew``.

    Parameters
    ----------
    teff:
        Effective temperature (K).
    glog:
        log10(surface gravity in cm/s²).
    taustd:
        Standard optical depth grid at each layer, shape ``(nrhox,)``.
        Computed as ``10^(tau1lg + j * steplg)`` for j = 0..nrhox-1
        with tau1lg=-6.875, steplg=0.125.
    vnew:
        If ``-99e5``: use bilinear interpolation from VMAXSTANDARD table.
        Otherwise: use ``abs(vnew)`` as the maximum velocity (cm/s).

    Returns
    -------
    vturb:
        Microturbulence velocity at each layer (cm/s), shape ``(nrhox,)``.
    """
    if vnew == -99.0e5:
        ig = int((glog + 1.0) / 0.5) + 1
        ig = max(1, min(ig, 12))
        it = int((teff - 3000.0) / 250.0) + 1
        it = max(1, min(it, 24))

        delg = (glog - ((ig - 1) * 0.5 - 1.0)) / 0.5
        delt = (teff - ((it - 1) * 250.0 + 3000.0)) / 250.0

        # Bilinear interpolation over (IG, IT) → IG-1 and IT-1 are 0-based.
        ig0 = ig - 1
        ig1 = ig  # IG+1 in Fortran (IG+1, 0-based is ig)
        it0 = it - 1
        it1 = it  # IT+1 in Fortran (IT+1, 0-based is it)
        ig1 = min(ig1, 12)  # clamp to table bounds (13 rows, indices 0-12)
        it1 = min(it1, 24)  # clamp to table bounds (25 cols, indices 0-24)

        vmax = (
            _VMAXSTANDARD[ig0, it0] * (1.0 - delg) * (1.0 - delt)
            + _VMAXSTANDARD[ig1, it0] * delg * (1.0 - delt)
            + _VMAXSTANDARD[ig0, it1] * (1.0 - delg) * delt
            + _VMAXSTANDARD[ig1, it1] * delg * delt
        )
        vmax *= 1.0e5  # km/s → cm/s
    else:
        vmax = abs(vnew)

    # Build log10(TAUSTD) for MAP1 interpolation.
    taulog = np.log10(np.maximum(taustd, 1.0e-300))

    # Interpolate VSTANDARD onto TAULOG grid.
    vturb_out, _ = _map1(_TAUSTANDARD, _VSTANDARD, taulog)

    # Scale to VMAX (reference max is 1.83e5 cm/s from the DATA table).
    vturb_out = vturb_out * vmax / 1.83e5
    return vturb_out
