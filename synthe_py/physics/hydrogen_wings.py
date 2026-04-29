"""Hydrogen wing opacity following the Kurucz HLINOP routine."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from numba import jit, prange

from ..io.atmosphere import AtmosphereModel
from .karsas_tables import (
    xkarsas,
    _xkarsas_jit,
    FREQ_LOG,
    XN_LOG,
    XL_LOG_ARRAY,
    EKARSAS,
    LN10,
)

RYD = 3.28805e15
C_LIGHT_CM = 2.99792458e10
RYDBERG_CM = 109677.576
LYMAN_LIMIT = 109678.764
BASE_LIMIT = 109250.336
CMINV_TO_EV = 1.2398419843320026e-4

# Tables from the original Fortran implementation
KNMTAB = np.array(
    [
        # rows correspond to M-N = 1..5, columns to N=1..4
        [3.56e-4, 5.23e-4, 1.09e-3, 1.49e-3],
        [2.25e-3, 1.25e-2, 1.77e-2, 2.80e-2],
        [3.48e-1, 4.93e-2, 1.24e-1, 1.71e-1],
        [2.23e-1, 2.61e-1, 3.42e-1, 6.83e-1],
        [8.66e-1, 1.02e0, 1.19e0, 1.46e0],
    ],
    dtype=np.float64,
)

FSTARK = np.array(
    [
        [0.1387, 0.07910, 0.02126, 0.01394],
        [0.006462, 0.004814, 0.002779, 0.002216],
        [0.001443, 0.001201, 0.3921, 0.1193],
        [0.03766, 0.02209, 0.01139, 0.008036],
        [0.005007, 0.003850, 0.002658, 0.002151],
        [0.6103, 0.1506, 0.04931, 0.02768],
        [0.01485, 0.01023, 0.006588, 0.004996],
        [0.003542, 0.8163, 0.1788, 0.05985],
        [0.03189, 0.01762, 0.01196, 0.007825],
        [0.005882, 0.004233, 0.003375, 0.0],  # last column not used beyond 10
    ],
    dtype=np.float64,
)


@jit(nopython=True)
def _exint(x: float) -> float:
    """EXINT approximation from Fortran - JIT-compiled."""
    return (
        -np.log(x)
        - 0.57516
        + x
        * (
            0.97996
            + x * (-0.21654 + x * (0.033572 + x * (-0.0029222 + x * 1.05439e-4)))
        )
    )


@jit(nopython=True)
def _stark(
    n: int,
    m: int,
    freq: float,
    hkt: float,
    xne: float,
    knmtab: np.ndarray,
    fstark: np.ndarray,
    ryd: float,
) -> float:
    """Port of the STARK helper from Fortran - JIT-compiled."""

    mminn = m - n
    if mminn <= 0:
        return 0.0

    nn = float(n * n)
    mm = float(m * m)
    xx = (float(n) / float(m)) ** 2

    if mminn <= 5:
        knm = knmtab[mminn - 1, n - 1]
    else:
        knm = 5.5e-5 * (nn * mm) ** 2 / (mm - nn)

    if mminn <= 10:
        fnm = fstark[mminn - 1, n - 1]
    else:
        fnm = (
            fstark[9, n - 1]
            * ((20.0 * float(n) + 100.0) / (float(n) + 10.0) / float(m) / (1.0 - xx))
            ** 3
        )

    freqnm = ryd * (1.0 / nn - 1.0 / mm)
    delt = abs(freq - freqnm)

    f0 = 1.25e-9 * max(xne, 1e-20) ** 0.6666667
    dbeta = 2.99792458e18 / (freqnm * freqnm) / f0 / knm
    beta = dbeta * delt

    y1 = mm * delt * hkt / 2.0
    y2 = (
        (np.pi * np.pi / (2.0 * 0.0265384e0 * 2.99792458e10))
        * delt
        * delt
        / max(xne, 1e-30)
    )

    qstat = 1.5 + 0.5 * (y1 * y1 - 1.3840) / (y1 * y1 + 1.3840)
    impact = 0.0
    if y1 <= 8.0 and y1 < y2:
        exy2 = _exint(y2) if y2 <= 8.0 else 0.0
        impact = (
            1.438
            * np.sqrt(y1 * (1.0 - xx))
            * (0.4 * np.exp(-y1) + _exint(y1) - 0.5 * exy2)
        )

    if beta > 20.0:
        prof = 1.5 / (beta * beta * np.sqrt(beta))
        dioi = (
            6.28e0
            * 1.48e-25
            * (2.0 * mm * ryd / max(delt, 1e-30))
            * xne
            * (
                np.sqrt(2.0 * mm * ryd / max(delt, 1e-30))
                * (1.3 * qstat + 0.30 * impact)
                - 3.9 * ryd * hkt
            )
        )
        ratio = qstat * min(1.0 + dioi, 1.25) + impact
    else:
        prof = 8.0 / (80.0 + beta**3)
        ratio = qstat + impact

    return 0.0265384 * fnm * prof * dbeta * ratio


@jit(nopython=True)
def _coulx(n: int, freq: float, z: int, ryd: float) -> float:
    """Port of the COULX helper - JIT-compiled."""

    if freq < z * z * ryd / (n * n):
        return 0.0
    coeff = 2.815e29 / freq**3 / (n**5) * z**4
    if n > 6:
        return coeff
    a = np.array([0.9916, 1.105, 1.101, 1.101, 1.102, 1.0986], dtype=np.float64)
    b = np.array(
        [2.719e13, -2.375e14, -9.863e13, -5.765e13, -3.909e13, -2.704e13],
        dtype=np.float64,
    )
    c = np.array(
        [-2.268e30, 4.077e28, 1.035e28, 4.593e27, 2.371e27, 1.229e27], dtype=np.float64
    )
    zzf = (z * z) / freq
    return coeff * (a[n - 1] + (b[n - 1] + c[n - 1] * zzf) * zzf)


@jit(nopython=True, parallel=True)
def _compute_hydrogen_wings_kernel(
    freq: np.ndarray,
    xne: np.ndarray,
    bhyd: np.ndarray,
    bolt: np.ndarray,
    mlast: np.ndarray,
    hkt: np.ndarray,
    ehvkt: np.ndarray,
    stim: np.ndarray,
    bnu: np.ndarray,
    knmtab: np.ndarray,
    fstark: np.ndarray,
    ryd: float,
    freq_log_table: np.ndarray,
    xn_log_table: np.ndarray,
    xl_log_array: np.ndarray,
    ekarsas_table: np.ndarray,
    ln10: float,
):
    """JIT-compiled kernel for hydrogen wings computation."""
    layers = xne.shape[0]
    nfreq = freq.size
    ahline = np.zeros((layers, nfreq), dtype=np.float64)
    shline = np.zeros((layers, nfreq), dtype=np.float64)

    for k in prange(nfreq):
        f = freq[k]
        if f <= 0.0:
            continue

        n = int(np.sqrt(ryd / f))
        if n <= 0 or n > 4:
            continue
        if n == 1 and f < 2.0e15:
            continue
        if n == 2 and f < 4.44e14:
            continue

        denom = (ryd / (n * n)) - f
        if denom <= 0.0:
            continue
        mfreq = np.sqrt(ryd / denom)

        eh_col = ehvkt[:, k]
        stim_col = stim[:, k]
        bnu_col = bnu[:, k]

        base_m1 = int(mfreq)

        for j in range(layers):
            eh = eh_col[j]
            stim_j = stim_col[j]
            bnu_j = bnu_col[j]
            hkt_j = hkt[j]

            m1 = max(base_m1, n + 1)
            m2 = m1 + 1
            h_val = 0.0
            s_val = 0.0

            if m1 > 6 and m1 <= mlast[j]:
                m1 -= 1
                m2 += 3
                if n >= 4 and m1 <= 8:
                    h_val = (
                        _stark(3, 4, f, hkt_j, xne[j], knmtab, fstark, ryd)
                        * (1.0 - eh * bhyd[j, 3] / bhyd[j, 2])
                        * bolt[j, 2]
                    )
                    denom_sh = bhyd[j, 2] / bhyd[j, 3] - eh
                    if abs(denom_sh) > 1e-12:
                        s_val = h_val * bnu_j * stim_j / denom_sh

            for m in range(m1, m2 + 1):
                bhydjm = 1.0
                if m <= 6:
                    bhydjm = bhyd[j, m - 1]
                a = (
                    _stark(n, m, f, hkt_j, xne[j], knmtab, fstark, ryd)
                    * (1.0 - eh * bhydjm / bhyd[j, n - 1])
                    * bolt[j, n - 1]
                )
                h_val += a
                denom_sh = bhyd[j, n - 1] / bhydjm - eh
                if abs(denom_sh) > 1e-12:
                    s_val += a * bnu_j * stim_j / denom_sh

            if h_val <= 0.0:
                sigma = _xkarsas_jit(
                    f,
                    1.0,
                    n,
                    n,
                    freq_log_table,
                    xn_log_table,
                    xl_log_array,
                    ekarsas_table,
                    ln10,
                )
                if sigma > 0.0 and n <= 6:
                    denom_sh = bhyd[j, n - 1] - eh
                    a = sigma * bolt[j, n - 1] * max(denom_sh, 0.0)
                    h_val = max(a, 0.0)
                    if abs(denom_sh) > 1e-12:
                        s_val = h_val * bnu_j * stim_j / denom_sh
                    else:
                        s_val = h_val * bnu_j
                else:
                    coeff = (
                        _coulx(n, f, 1, ryd)
                        * (1.0 - eh / bhyd[j, n - 1])
                        * bolt[j, n - 1]
                    )
                    h_val = max(coeff, 0.0)
                    denom_sh = bhyd[j, n - 1] - eh
                    if abs(denom_sh) > 1e-12:
                        s_val = h_val * bnu_j * stim_j / denom_sh
                    else:
                        s_val = h_val * bnu_j

            ahline[j, k] = max(h_val, 0.0)
            if h_val > 0.0:
                shline[j, k] = s_val / h_val

    return ahline, shline


def compute_hydrogen_wings(
    atmosphere: AtmosphereModel,
    freq: np.ndarray,
    bnu: np.ndarray,
    ehvkt: np.ndarray,
    stim: np.ndarray,
    hkt: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute AHLINE and SHLINE arrays for each depth/frequency."""

    layers = atmosphere.layers
    nfreq = freq.size

    xne = np.asarray(atmosphere.electron_density, dtype=np.float64)
    rho = np.asarray(atmosphere.mass_density, dtype=np.float64)
    temperature = np.asarray(atmosphere.temperature, dtype=np.float64)
    if atmosphere.xnfph is None:
        return np.zeros((layers, nfreq)), np.zeros((layers, nfreq))
    xnfph = np.asarray(atmosphere.xnfph, dtype=np.float64)

    # Avoid zeros
    xne = np.maximum(xne, 1e-30)
    rho = np.maximum(rho, 1e-30)

    # T in eV
    tkev = temperature * 8.617333262e-5

    if atmosphere.bhyd is not None:
        bhyd = np.asarray(atmosphere.bhyd, dtype=np.float64)
        if bhyd.shape[1] < 8:
            padding = np.ones((layers, 8 - bhyd.shape[1]), dtype=np.float64)
            bhyd = np.hstack((bhyd, padding))
    else:
        bhyd = np.ones((layers, 8), dtype=np.float64)

    atlas_tables = atmosphere.atlas_tables or {}
    ehyd = atlas_tables.get("EHYD")
    ghyd = atlas_tables.get("GHYD")

    if ehyd is not None and ehyd.size >= 4:
        level_energies_ev = np.asarray(ehyd[:4], dtype=np.float64) * CMINV_TO_EV
    else:
        level_energies_ev = np.array(
            [0.0] + [13.595 * (1.0 - 1.0 / (n * n)) for n in range(2, 5)],
            dtype=np.float64,
        )

    if ghyd is not None and ghyd.size >= 4:
        degeneracy = np.asarray(ghyd[:4], dtype=np.float64)
    else:
        degeneracy = np.array([2.0 * (n * n) for n in range(1, 5)], dtype=np.float64)

    bolt = np.zeros((layers, 4), dtype=np.float64)
    for idx in range(4):
        n_level = idx + 1
        energy_ev = level_energies_ev[idx] if idx < level_energies_ev.size else 13.595
        bolt[:, idx] = (
            np.exp(-energy_ev / np.maximum(tkev, 1e-6))
            * degeneracy[idx]
            * bhyd[:, idx]
            * xnfph[:, 0]
            / rho
        )

    mlast = 1100.0 / np.power(xne, 0.133333333)

    ahline, shline = _compute_hydrogen_wings_kernel(
        freq,
        xne,
        bhyd,
        bolt,
        mlast,
        hkt,
        ehvkt,
        stim,
        bnu,
        KNMTAB,
        FSTARK,
        RYD,
        FREQ_LOG,
        XN_LOG,
        XL_LOG_ARRAY,
        EKARSAS,
        LN10,
    )
    return ahline, shline


HIGH_LEVEL_TERMS = (
    (15, 487.456, 450.0, 109191.313),
    (14, 559.579, 392.0, 109119.188),
    (13, 648.980, 338.0, 109029.789),
    (12, 761.649, 288.0, 108917.117),
    (11, 906.426, 242.0, 108772.336),
    (10, 1096.776, 200.0, 108581.992),
    (9, 1354.044, 162.0, 108324.719),
    (8, 1713.713, 128.0, 107965.051),
    (7, 2238.320, 98.0, 107440.444),
)

LOW_LEVEL_TERMS = (
    (6, 3046.604, 72.0, 106632.160),
    (5, 4387.113, 50.0, 105291.651),
    (4, 6854.871, 32.0, 102823.893),
    (3, 12186.462, 18.0, 97492.302),
    (2, 27419.659, 8.0, 82259.105),
)

# Convert to numpy arrays for JIT compatibility
# Shape: (n_terms, 4) where columns are: n, threshold, coeff, exponent
HIGH_LEVEL_TERMS_ARRAY = np.array(HIGH_LEVEL_TERMS, dtype=np.float64)
LOW_LEVEL_TERMS_ARRAY = np.array(LOW_LEVEL_TERMS, dtype=np.float64)


@jit(nopython=True, parallel=True)
def _compute_hydrogen_continuum_kernel(
    freq: np.ndarray,
    waveno: np.ndarray,
    freq3: np.ndarray,
    xnfph: np.ndarray,
    rho: np.ndarray,
    bhyd: np.ndarray,
    hkt: np.ndarray,
    ehvkt: np.ndarray,
    stim: np.ndarray,
    bnu: np.ndarray,
    high_level_terms: np.ndarray,
    low_level_terms: np.ndarray,
    c_light_cm: float,
    rydberg_cm: float,
    lyman_limit: float,
    base_limit: float,
    freq_log_table: np.ndarray,
    xn_log_table: np.ndarray,
    xl_log_array: np.ndarray,
    ekarsas_table: np.ndarray,
    ln10: float,
):
    """JIT-compiled kernel for hydrogen continuum computation."""
    layers = xnfph.shape[0]
    nfreq = freq.size
    ahyd = np.zeros((layers, nfreq), dtype=np.float64)
    shyd = np.zeros((layers, nfreq), dtype=np.float64)

    n_high = high_level_terms.shape[0]
    n_low = low_level_terms.shape[0]

    for k in prange(nfreq):
        f = freq[k]
        if f <= 0.0:
            continue

        wn = waveno[k]
        f3 = freq3[k]

        for j in range(layers):
            stim_j = stim[j, k]
            if stim_j <= 0.0:
                continue

            hkt_j = hkt[j]
            hckt_j = hkt_j * c_light_cm
            bnu_j = bnu[j, k]
            eh = ehvkt[j, k]

            # Base continuum (N=16 to infinity)
            prefactor = f3 / max(rydberg_cm * hckt_j, 1e-30)
            exp_hi = np.exp(-lyman_limit * hckt_j)
            exp_lo = np.exp(-max(base_limit, lyman_limit - wn) * hckt_j)
            h_val = prefactor * (exp_lo - exp_hi) * stim_j
            s_val = h_val * bnu_j

            # High-level terms (N=7-15)
            for i in range(n_high):
                n = int(high_level_terms[i, 0])
                threshold = high_level_terms[i, 1]
                coeff = high_level_terms[i, 2]
                exponent = high_level_terms[i, 3]

                if wn < threshold:
                    break

                sigma = _xkarsas_jit(
                    f,
                    1.0,
                    n,
                    n,
                    freq_log_table,
                    xn_log_table,
                    xl_log_array,
                    ekarsas_table,
                    ln10,
                )
                if sigma <= 0.0:
                    continue

                contrib = sigma * coeff * np.exp(-exponent * hckt_j) * stim_j
                h_val += contrib
                s_val += contrib * bnu_j

            # Low-level terms (N=2-6)
            for i in range(n_low):
                n = int(low_level_terms[i, 0])
                threshold = low_level_terms[i, 1]
                coeff = low_level_terms[i, 2]
                exponent = low_level_terms[i, 3]

                if wn < threshold:
                    break

                sigma = _xkarsas_jit(
                    f,
                    1.0,
                    n,
                    n,
                    freq_log_table,
                    xn_log_table,
                    xl_log_array,
                    ekarsas_table,
                    ln10,
                )
                if sigma <= 0.0:
                    continue

                bh = bhyd[j, n - 1]
                delta = bh - eh
                contrib = sigma * coeff * np.exp(-exponent * hckt_j) * delta
                h_val += contrib
                if abs(delta) > 1e-16:
                    s_val += contrib * bnu_j * stim_j / delta
                else:
                    s_val += contrib * bnu_j

            # N=1 (Lyman limit)
            if wn >= lyman_limit:
                sigma = _xkarsas_jit(
                    f,
                    1.0,
                    1,
                    1,
                    freq_log_table,
                    xn_log_table,
                    xl_log_array,
                    ekarsas_table,
                    ln10,
                )
                if sigma > 0.0:
                    delta = bhyd[j, 0] - eh
                    contrib = sigma * 2.0 * delta
                    h_val += contrib
                    if abs(delta) > 1e-16:
                        s_val += contrib * bnu_j * stim_j / delta
                    else:
                        s_val += contrib * bnu_j

            # Scale by xnfph/rho (matching Fortran HOP line 4140)
            scale = xnfph[j, 0] / rho[j]
            h_scaled = h_val * scale
            if h_scaled <= 0.0:
                continue

            s_scaled = s_val * scale
            ahyd[j, k] = h_scaled
            shyd[j, k] = s_scaled / max(h_scaled, 1e-40)

    return ahyd, shyd


def compute_hydrogen_continuum(
    atmosphere: AtmosphereModel,
    freq: np.ndarray,
    bnu: np.ndarray,
    ehvkt: np.ndarray,
    stim: np.ndarray,
    hkt: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Hydrogen bound-free continuum (HOP) using the Karsas tables."""

    import logging

    logger = logging.getLogger(__name__)

    layers = atmosphere.layers
    nfreq = freq.size
    logger.info(f"Computing hydrogen continuum: {layers} layers × {nfreq} frequencies")

    if atmosphere.xnfph is None:
        logger.info("No hydrogen population data (xnfph), returning zeros")
        zero = np.zeros((layers, nfreq), dtype=np.float64)
        return zero, zero

    xnfph = np.asarray(atmosphere.xnfph, dtype=np.float64)
    rho = np.maximum(np.asarray(atmosphere.mass_density, dtype=np.float64), 1e-30)
    ehvkt = np.asarray(ehvkt, dtype=np.float64)
    stim = np.asarray(stim, dtype=np.float64)
    bnu = np.asarray(bnu, dtype=np.float64)

    if atmosphere.bhyd is not None:
        bhyd = np.asarray(atmosphere.bhyd, dtype=np.float64)
        if bhyd.shape[1] < 8:
            padding = np.ones((layers, 8 - bhyd.shape[1]), dtype=np.float64)
            bhyd = np.hstack((bhyd, padding))
    else:
        bhyd = np.ones((layers, 8), dtype=np.float64)

    ahyd = np.zeros((layers, nfreq), dtype=np.float64)
    shyd = np.zeros_like(ahyd)

    # Constant from Fortran (hydrogen cross-section coefficient)
    # This is a physical constant, not a configuration parameter
    HYDROGEN_CROSS_SECTION_COEFF = 2.815e29

    freq3 = HYDROGEN_CROSS_SECTION_COEFF / np.maximum(freq, 1e-30) ** 3
    waveno = freq / C_LIGHT_CM

    ahyd, shyd = _compute_hydrogen_continuum_kernel(
        freq,
        waveno,
        freq3,
        xnfph,
        rho,
        bhyd,
        hkt,
        ehvkt,
        stim,
        bnu,
        HIGH_LEVEL_TERMS_ARRAY,
        LOW_LEVEL_TERMS_ARRAY,
        C_LIGHT_CM,
        RYDBERG_CM,
        LYMAN_LIMIT,
        BASE_LIMIT,
        FREQ_LOG,
        XN_LOG,
        XL_LOG_ARRAY,
        EKARSAS,
        LN10,
    )
    logger.info("Hydrogen continuum computation complete")
    return ahyd, shyd
