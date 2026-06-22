#!/usr/bin/env python3
"""Numba-JIT replica of ``pops_exact._pfsaha_exact_python``.

This module is a line-for-line ``@njit`` port of the pure-Python PFSAHA
implementation in ``pops_exact.py`` (itself a faithful port of the Fortran
``PFSAHA`` subroutine in ``atlas7v.for``). The pure-Python function remains the
source of truth / reference; ``pops_exact.pfsaha_exact`` dispatches here when
numba is available and falls back to ``_pfsaha_exact_python`` otherwise.

Bit-identity notes (verified empirically on the target interpreter,
numba 0.62.1 / numpy 2.3.4):
- numba ``np.exp`` / ``math.exp`` / ``np.log10`` / ``np.sqrt`` match CPython
  numpy/math for float64 scalars (max rel diff 0.0).
- The high-temperature occupation correction computes ``np.sqrt(...) ** 3`` in
  the reference. CPython evaluates ``float64 ** int`` through libm ``pow``; numba
  ``** 3`` (int exponent) or ``x*x*x`` does NOT match (1 ULP). Writing the
  exponent as a float (``** 3.0``) routes numba through libm ``pow`` and matches
  CPython exactly. This mirrors the warning in the atlas_py port / gap_v2 §9b.
- Constants (LOCZ, SCALE, the E*/G* level arrays) are copied verbatim from
  ``pops_exact.py``; ``tests``/validation assert equality at runtime.
"""

from __future__ import annotations

from typing import Optional, Dict

import numpy as np

from ._pfground_table import (
    FIRST_RANGE_LABELS,
    SECOND_RANGE_LABELS,
    PFGROUND_EXPRESSIONS,
)

try:
    import numba
    from numba import njit

    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - numba is a hard dependency in practice
    numba = None  # type: ignore[assignment]
    _NUMBA_AVAILABLE = False

import math

# ---------------------------------------------------------------------------
# Constants (verbatim copies of the values in pops_exact.py). The validation
# harness asserts these match pops_exact at runtime to catch any copy drift.
# ---------------------------------------------------------------------------
EV_TO_CM = 8065.479
KBOLTZ_FACTOR = 8.6171e-5  # Fortran TKEV factor (atlas7v.for)
_HCK = 6.6256e-27 * 2.99792458e10 / 1.38054e-16

LOCZ = np.array(
    [
        1, 3, 6, 10, 14, 18, 22, 27, 33, 39, 45, 51, 57, 63, 69, 75, 81, 86,
        91, 96, 101, 106, 111, 116, 121, 126, 131, 136, 141,
    ],
    dtype=np.int32,
)

SCALE = np.array([0.001, 0.01, 0.1, 1.0], dtype=np.float64)

EHYD = np.array(
    [0.0, 82259.105, 97492.302, 102823.893, 105291.651, 106632.160], dtype=np.float64
)
GHYD = np.array([2.0, 8.0, 18.0, 32.0, 50.0, 72.0], dtype=np.float64)

EHE1 = np.array(
    [
        0.0, 159856.069, 166277.546, 169087.007, 171135.000, 183236.892,
        184864.936, 185564.694, 186101.654, 186105.065, 186209.471, 190298.210,
        190940.331, 191217.14, 191444.588, 191446.559, 191451.80, 191452.08,
        191492.817, 193347.089, 193663.627, 193800.78, 193917.245, 193918.391,
        193921.31, 193921.37, 193922.5, 193922.5, 193942.57,
    ],
    dtype=np.float64,
)
GHE1 = np.array(
    [
        1, 3.0, 1.0, 9.0, 3.0, 3.0, 1.0, 9.0, 15.0, 5.0, 3.0, 3.0, 1.0, 9.0,
        15.0, 5.0, 21.0, 7.0, 3.0, 3.0, 1.0, 9.0, 15.0, 5.0, 21.0, 7.0, 27.0,
        9.0, 3.0,
    ],
    dtype=np.float64,
)

EHE2 = np.array(
    [0.0, 329182.321, 390142.359, 411477.925, 421353.135, 426717.413], dtype=np.float64
)
GHE2 = np.array([2.0, 8.0, 18.0, 32.0, 50.0, 72.0], dtype=np.float64)

EC1 = np.array(
    [
        29.60, 10192.66, 21648.02, 33735.20, 60373.00, 61981.82, 64088.85,
        68856.33, 69722.00, 70743.95, 71374.90, 72610.72, 73975.91, 75254.93,
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
        77.975, 15867.862, 33792.583, 73768.200, 76794.978, 86629.089,
        88630.977, 95476.728, 96225.049, 97420.748, 97488.476, 99094.065,
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
        0.0, 21890.854, 35051.264, 41197.403, 43503.333, 46403.065, 47847.797,
        47957.034, 49346.729, 51872.526, 52556.206,
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
        74.707, 25347.756, 29097.11, 32436.241, 32960.363, 37689.413,
        38932.139, 40275.903, 41319.377,
    ],
    dtype=np.float64,
)
GAL1 = np.array([6.0, 2.0, 12.0, 10.0, 6.0, 2.0, 10.0, 6.0, 14.0], dtype=np.float64)

ESI1 = np.array(
    [
        149.681, 6298.850, 15394.370, 33326.053, 39859.920, 40991.884,
        45303.310, 47284.061, 47351.554, 48161.459, 49128.131,
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


# ---------------------------------------------------------------------------
# PFGROUND lookup tables, derived from _pfground_table.PFGROUND_EXPRESSIONS.
#
# Every PFGROUND expression has the form:
#     base + c1*exp(-_HCK/T*E1) + c2*exp(-_HCK/T*E2) + ...
# where ``base`` is the leading bare constant. We parse each expression into
# (base, [(coeff, E), ...]) so the njit kernel can evaluate it with the
# identical left-to-right accumulation as the original lambda (bit-for-bit).
# ---------------------------------------------------------------------------
_PFG_INNER_PREFIX = "(-_HCK/T*"


def _parse_pfground_expr(expr: str):
    base = 0.0
    terms = []
    for idx, part in enumerate(expr.split("+")):
        part = part.strip()
        if "math.exp" in part:
            before, after = part.split("math.exp")
            before = before.strip()
            if before == "":
                coeff = 1.0
            elif before.endswith("*"):
                coeff = float(before[:-1])
            else:
                raise ValueError(f"Unexpected PFGROUND term: {part!r}")
            after = after.strip()
            if not (after.startswith(_PFG_INNER_PREFIX) and after.endswith(")")):
                raise ValueError(f"Unexpected PFGROUND exp form: {part!r}")
            inner = after[len(_PFG_INNER_PREFIX) : -1]
            e_val = float(inner)
            terms.append((coeff, e_val))
        else:
            if idx != 0:
                raise ValueError(f"Constant not first in PFGROUND expr: {expr!r}")
            base = float(part)
    return base, terms


def _build_pfground_tables():
    max_label = max(PFGROUND_EXPRESSIONS.keys())
    size = max_label + 1
    pfg_base = np.ones(size, dtype=np.float64)  # default 1.0 (== func None -> 1.0)
    pfg_nterm = np.zeros(size, dtype=np.int64)
    pfg_start = np.zeros(size, dtype=np.int64)
    coeffs: list[float] = []
    evals: list[float] = []
    for label, expr in PFGROUND_EXPRESSIONS.items():
        base, terms = _parse_pfground_expr(expr)
        pfg_base[label] = base
        pfg_start[label] = len(coeffs)
        pfg_nterm[label] = len(terms)
        for c, e in terms:
            coeffs.append(c)
            evals.append(e)
    pfg_tc = np.array(coeffs, dtype=np.float64)
    pfg_te = np.array(evals, dtype=np.float64)
    return pfg_base, pfg_start, pfg_nterm, pfg_tc, pfg_te


PFG_FIRST = np.array(FIRST_RANGE_LABELS, dtype=np.int64)
PFG_SECOND = np.array(SECOND_RANGE_LABELS, dtype=np.int64)
(PFG_BASE, PFG_START, PFG_NTERM, PFG_TC, PFG_TE) = _build_pfground_tables()


def _pfground_lookup_py(nelion: int, temperature: float) -> float:
    """Pure-Python data-driven PFGROUND (mirror of pops_exact._pfground_lookup).

    Used only for offline validation of the data tables; the production path
    uses the njit kernel below.
    """
    if temperature <= 0.0 or nelion <= 0:
        return 1.0
    if nelion <= len(PFG_FIRST):
        label = int(PFG_FIRST[nelion - 1])
    elif nelion < 169:
        return 1.0
    elif nelion - 169 < len(PFG_SECOND):
        label = int(PFG_SECOND[nelion - 169])
    else:
        label = 666
    val = float(PFG_BASE[label])
    start = int(PFG_START[label])
    nt = int(PFG_NTERM[label])
    for k in range(nt):
        val += PFG_TC[start + k] * math.exp(-_HCK / temperature * PFG_TE[start + k])
    return val


if _NUMBA_AVAILABLE:

    @njit(cache=True)
    def _pfground_nb(nelion, temperature):  # noqa: D401 - mirror of _pfground_lookup
        if temperature <= 0.0 or nelion <= 0:
            return 1.0
        if nelion <= len(PFG_FIRST):
            label = PFG_FIRST[nelion - 1]
        elif nelion < 169:
            return 1.0
        elif nelion - 169 < len(PFG_SECOND):
            label = PFG_SECOND[nelion - 169]
        else:
            label = 666
        val = PFG_BASE[label]
        start = PFG_START[label]
        nt = PFG_NTERM[label]
        for k in range(nt):
            val += PFG_TC[start + k] * math.exp(
                -_HCK / temperature * PFG_TE[start + k]
            )
        return val

    @njit(cache=True)
    def _dep(table, dep_on, layer_idx, column):
        if dep_on == 0:
            return 1.0
        if layer_idx < table.shape[0] and column < table.shape[1]:
            return table[layer_idx, column]
        return 1.0

    @njit(cache=True)
    def _pfiron_nb(iz, ion, tlog8, potlow8, pftab, potlo_arr, potlolog_arr):
        elem_idx = iz - 20
        ion_idx = ion - 1
        tlog = tlog8
        potlow = potlow8
        if tlog > 4.0:
            it_float = (tlog - 4.0) / 0.05 + 31.0
            it = int(it_float)
            if it < 1:
                it = 1
            if it > 56:
                it = 56
            f = (tlog - (it - 31) * 0.05 - 4.0) / 0.05
        elif tlog < 3.7:
            it_float = (tlog - 3.32) / 0.02 + 2.0
            it = int(it_float)
            if it < 2:
                it = 2
            f = (tlog - (it - 2) * 0.02 - 3.32) / 0.02
        else:
            it_float = (tlog - 3.7) / 0.03 + 21.0
            it = int(it_float)
            f = (tlog - (it - 21) * 0.03 - 3.7) / 0.03

        if it < 1:
            it = 1
        if it > 56:
            it = 56
        it_idx = it - 1
        it_idx_m1 = it_idx - 1
        if it_idx_m1 < 0:
            it_idx_m1 = 0

        n_potlo = potlo_arr.shape[0]
        if potlow < potlo_arr[0]:
            low = 0
            pf = (
                f * pftab[low, it_idx, ion_idx, elem_idx]
                + (1.0 - f) * pftab[low, it_idx_m1, ion_idx, elem_idx]
            )
        else:
            low = n_potlo - 1
            for i in range(1, n_potlo):
                if potlow < potlo_arr[i]:
                    low = i
                    break
            p = (np.log10(potlow) - potlolog_arr[low - 1]) / 0.30103
            pf = p * (
                f * pftab[low, it_idx, ion_idx, elem_idx]
                + (1.0 - f) * pftab[low, it_idx_m1, ion_idx, elem_idx]
            ) + (1.0 - p) * (
                f * pftab[low - 1, it_idx, ion_idx, elem_idx]
                + (1.0 - f) * pftab[low - 1, it_idx_m1, ion_idx, elem_idx]
            )
        return pf

    @njit(cache=True)
    def _pfsaha_core_nb(
        iz,
        nion,
        mode,
        nlte_on,
        dep_on,
        temperature,
        tkev,
        tk,
        hckt,
        tlog,
        gas_pressure,
        electron_density,
        answer,
        potion,
        nnn,
        pftab,
        potlo_pfiron,
        potlolog_pfiron,
        bhyd,
        bhe1,
        bhe2,
        bc1,
        bc2,
        bmg1,
        bmg2,
        bal1,
        bsi1,
        bsi2,
        bca1,
        bca2,
        bo1,
        bna1,
        bb1,
        bk1,
    ):
        n_layers = temperature.shape[0]
        mode1 = mode
        if mode1 > 10:
            mode1 = mode1 - 10

        nnn_cols = nnn.shape[1]
        nnn_rows = nnn.shape[0]
        n_scale = SCALE.shape[0]

        for layer_idx in range(n_layers):
            T = temperature[layer_idx]
            TV = tkev[layer_idx]
            TK_val = tk[layer_idx]
            HCKT_val = hckt[layer_idx]
            TLOG_val = tlog[layer_idx]
            XNE_val = electron_density[layer_idx]
            P_val = gas_pressure[layer_idx]

            # Debye screening
            CHARGE = XNE_val * 2.0
            EXCESS = 2.0 * XNE_val - P_val / TK_val
            if EXCESS > 0.0:
                CHARGE = CHARGE + 2.0 * EXCESS
            if CHARGE == 0.0:
                CHARGE = 1.0
            DEBYE = np.sqrt(TK_val / 2.8965e-18 / CHARGE)
            POTLOW = 1.44e-7 / DEBYE
            if POTLOW > 1.0:
                POTLOW = 1.0

            # Number of ions / starting index
            if iz <= 28:
                n = LOCZ[iz - 1]
                if iz < LOCZ.shape[0]:
                    nions = LOCZ[iz] - LOCZ[iz - 1]
                else:
                    nions = 3
            else:
                n = 3 * iz + 54
                nions = 3

            if iz == 6:
                n = 354
                nions = 6
            elif iz == 7:
                n = 360
                nions = 7
            elif iz == 8:
                n = 367
                nions = 8

            if iz >= 20 and iz < 29:
                nions = 10

            n_start = n - 1
            nion2 = nion + 2
            if nion2 > nions:
                nion2 = nions

            IP = np.zeros(31, dtype=np.float64)
            PART = np.ones(31, dtype=np.float64)
            POTLO = np.zeros(31, dtype=np.float64)
            F = np.zeros(31, dtype=np.float64)

            for ion_idx in range(1, nion2 + 1):
                Z = float(ion_idx)
                POTLO[ion_idx - 1] = POTLOW * Z

                # Ionization potential from POTION
                if iz <= 30:
                    potion_idx_fortran = iz * (iz + 1) // 2 + ion_idx - 1
                else:
                    potion_idx_fortran = iz * 5 + 341 + ion_idx - 1
                potion_idx_python = potion_idx_fortran - 1
                if 0 <= potion_idx_python < potion.shape[0]:
                    IP_val_cm = potion[potion_idx_python]
                    IP[ion_idx - 1] = IP_val_cm / EV_TO_CM
                    if IP[ion_idx - 1] == 0.0 and potion_idx_python > 0:
                        IP_val_cm_fb = potion[potion_idx_python - 1]
                        IP[ion_idx - 1] = IP_val_cm_fb / EV_TO_CM
                else:
                    IP[ion_idx - 1] = 0.0

                ip_i = IP[ion_idx - 1]

                if iz >= 20 and iz < 29:
                    tlog8 = TLOG_val / 2.30258509299405
                    potlow8 = POTLO[ion_idx - 1] * EV_TO_CM
                    PART[ion_idx - 1] = _pfiron_nb(
                        iz, ion_idx, tlog8, potlow8, pftab, potlo_pfiron, potlolog_pfiron
                    )
                    continue

                nnn_col_fortran = n_start + ion_idx
                nnn_col = nnn_col_fortran - 1

                G = 0.0
                D1 = 0.0
                if nnn_col < 374:
                    nnn6 = nnn[5, nnn_col]
                    nnn100 = nnn6 // 100
                    G = float(nnn6 - nnn100 * 100)

                handled = False

                if nnn_col_fortran == 2:  # H II bare
                    PART[ion_idx - 1] = 1.0
                    handled = True

                if nnn_col_fortran == 1:  # H I
                    B = _dep(bhyd, dep_on, layer_idx, 0)
                    PART[ion_idx - 1] = 2.0 * B
                    if T >= 9000.0:
                        for i in range(1, 6):
                            B = _dep(bhyd, dep_on, layer_idx, i)
                            PART[ion_idx - 1] += GHYD[i] * B * np.exp(
                                -EHYD[i] * HCKT_val
                            )
                    handled = True
                    D1 = 109677.576 / 6.5 / 6.5 * HCKT_val
                elif nnn_col_fortran == 3:  # He I
                    B = _dep(bhe1, dep_on, layer_idx, 0)
                    PART[ion_idx - 1] = B
                    if T >= 15000.0:
                        for i in range(1, 29):
                            B = _dep(bhe1, dep_on, layer_idx, i)
                            PART[ion_idx - 1] += GHE1[i] * B * np.exp(
                                -EHE1[i] * HCKT_val
                            )
                    handled = True
                    D1 = 109677.576 / 5.5 / 5.5 * HCKT_val
                elif nnn_col_fortran == 4:  # He II
                    B = _dep(bhe2, dep_on, layer_idx, 0)
                    PART[ion_idx - 1] = 2.0 * B
                    if T >= 30000.0:
                        for i in range(1, 6):
                            B = _dep(bhe2, dep_on, layer_idx, i)
                            PART[ion_idx - 1] += GHE2[i] * B * np.exp(
                                -EHE2[i] * HCKT_val
                            )
                    handled = True
                    D1 = 4.0 * 109722.267 / 6.5 / 6.5 * HCKT_val
                elif nnn_col_fortran == 354:  # C I
                    B = _dep(bc1, dep_on, layer_idx, 0)
                    PART[ion_idx - 1] = B * (
                        1.0
                        + 3.0 * np.exp(-16.42 * HCKT_val)
                        + 5.0 * np.exp(-43.42 * HCKT_val)
                    )
                    for i in range(1, 14):
                        B = _dep(bc1, dep_on, layer_idx, i)
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
                elif nnn_col_fortran == 355:  # C II
                    B = _dep(bc2, dep_on, layer_idx, 0)
                    PART[ion_idx - 1] = B * (2.0 + 4.0 * np.exp(-63.42 * HCKT_val))
                    for i in range(1, 6):
                        B = _dep(bc2, dep_on, layer_idx, i)
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
                elif nnn_col_fortran == 51:  # Mg I
                    B = _dep(bmg1, dep_on, layer_idx, 0)
                    PART[ion_idx - 1] = B
                    for i in range(1, 11):
                        B = _dep(bmg1, dep_on, layer_idx, i)
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
                elif nnn_col_fortran == 52:  # Mg II
                    B = _dep(bmg2, dep_on, layer_idx, 0)
                    PART[ion_idx - 1] = B * 2.0
                    for i in range(1, 6):
                        B = _dep(bmg2, dep_on, layer_idx, i)
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
                elif nnn_col_fortran == 57:  # Al I
                    B = _dep(bal1, dep_on, layer_idx, 0)
                    PART[ion_idx - 1] = B * (2.0 + 4.0 * np.exp(-112.061 * HCKT_val))
                    for i in range(1, 9):
                        B = _dep(bal1, dep_on, layer_idx, i)
                        PART[ion_idx - 1] += GAL1[i] * B * np.exp(-EAL1[i] * HCKT_val)
                    PART[ion_idx - 1] += 10.0 * np.exp(
                        -42235.0 * HCKT_val
                    ) + 14.0 * np.exp(-43831.0 * HCKT_val)
                    handled = True
                    G = 2.0
                    D1 = 109735.08 / 5.5 / 5.5 * HCKT_val
                elif nnn_col_fortran == 63:  # Si I
                    B = _dep(bsi1, dep_on, layer_idx, 0)
                    PART[ion_idx - 1] = B * (
                        1.0
                        + 3.0 * np.exp(-77.115 * HCKT_val)
                        + 5.0 * np.exp(-223.157 * HCKT_val)
                    )
                    for i in range(1, 11):
                        B = _dep(bsi1, dep_on, layer_idx, i)
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
                elif nnn_col_fortran == 64:  # Si II
                    B = _dep(bsi2, dep_on, layer_idx, 0)
                    PART[ion_idx - 1] = B * (2.0 + 4.0 * np.exp(-287.32 * HCKT_val))
                    for i in range(1, 6):
                        B = _dep(bsi2, dep_on, layer_idx, i)
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
                elif nnn_col_fortran == 96:  # Ca I (dead path; iz=20 uses pfiron)
                    base_part = _dep(bca1, dep_on, layer_idx, 0)
                    PART[ion_idx - 1] = base_part
                    for i in range(1, 8):
                        B = _dep(bca1, dep_on, layer_idx, i)
                        PART[ion_idx - 1] += GCA1[i] * B * np.exp(-ECA1[i] * HCKT_val)
                    PART[ion_idx - 1] += (
                        28.0 * np.exp(-37000.0 * HCKT_val)
                        + 67.0 * np.exp(-40000.0 * HCKT_val)
                        + 21.0 * np.exp(-43000.0 * HCKT_val)
                        + 34.0 * np.exp(-48000.0 * HCKT_val)
                    )
                    handled = True
                    G = 4.0
                    D1 = 109734.82 / 4.5 / 4.5 * HCKT_val
                elif nnn_col_fortran == 97:  # Ca II (dead path)
                    base_part = _dep(bca2, dep_on, layer_idx, 0)
                    PART[ion_idx - 1] = base_part * 2.0
                    for i in range(1, 5):
                        B = _dep(bca2, dep_on, layer_idx, i)
                        PART[ion_idx - 1] += GCA2[i] * B * np.exp(-ECA2[i] * HCKT_val)
                    PART[ion_idx - 1] += 12.0 * np.exp(-68000.0 * HCKT_val)
                    handled = True
                    G = 2.0
                    D1 = 109734.83 / 4.5 / 4.5 * HCKT_val
                elif nnn_col_fortran == 367:  # O I
                    B = _dep(bo1, dep_on, layer_idx, 0)
                    PART[ion_idx - 1] = B * (
                        5.0
                        + 3.0 * np.exp(-158.265 * HCKT_val)
                        + np.exp(-226.977 * HCKT_val)
                    )
                    for i in range(1, 13):
                        B = _dep(bo1, dep_on, layer_idx, i)
                        PART[ion_idx - 1] += GO1[i] * B * np.exp(-EO1[i] * HCKT_val)
                    PART[ion_idx - 1] += (
                        15.0 * np.exp(-101140.0 * HCKT_val)
                        + 131.0 * np.exp(-103000.0 * HCKT_val)
                        + 128.0 * np.exp(-105000.0 * HCKT_val)
                        + 600.0 * np.exp(-107000.0 * HCKT_val)
                    )
                    handled = True
                elif nnn_col_fortran == 45:  # Na I
                    B = _dep(bna1, dep_on, layer_idx, 0)
                    PART[ion_idx - 1] = B * 2.0
                    for i in range(1, 8):
                        B = _dep(bna1, dep_on, layer_idx, i)
                        PART[ion_idx - 1] += GNA1[i] * B * np.exp(-ENA1[i] * HCKT_val)
                    PART[ion_idx - 1] += 10.0 * np.exp(
                        -34548.745 * HCKT_val
                    ) + 14.0 * np.exp(-34586.96 * HCKT_val)
                    handled = True
                    G = 2.0
                    D1 = 109734.83 / 4.5 / 4.5 * HCKT_val
                elif nnn_col_fortran == 14:  # B I
                    B = _dep(bb1, dep_on, layer_idx, 0)
                    PART[ion_idx - 1] = B * (2.0 + 4.0 * np.exp(-15.25 * HCKT_val))
                    for i in range(1, 7):
                        B = _dep(bb1, dep_on, layer_idx, i)
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
                elif nnn_col_fortran == 91:  # K I
                    B = _dep(bk1, dep_on, layer_idx, 0)
                    PART[ion_idx - 1] = B * 2.0
                    for i in range(1, 8):
                        B = _dep(bk1, dep_on, layer_idx, i)
                        PART[ion_idx - 1] += GK1[i] * B * np.exp(-EK1[i] * HCKT_val)
                    PART[ion_idx - 1] += 10.0 * np.exp(
                        -27397.077 * HCKT_val
                    ) + 14.0 * np.exp(-28127.85 * HCKT_val)
                    handled = True
                    G = 2.0
                    D1 = 109734.83 / 5.5 / 5.5 * HCKT_val

                # Standard NNN interpolation
                if (not handled) and (nnn_col < nnn_cols) and (ip_i > 0.0):
                    T2000 = ip_i * 2000.0 / 11.0
                    IT = int(T / T2000 - 0.5)
                    if IT < 1:
                        IT = 1
                    if IT > 9:
                        IT = 9
                    DT = T / T2000 - float(IT) - 0.5
                    PMIN = 1.0

                    i_fortran = (IT + 1) // 2
                    i_idx = i_fortran - 1
                    if i_idx < 0:
                        i_idx = 0
                    if i_idx > nnn_rows - 1:
                        i_idx = nnn_rows - 1

                    nnn_i = nnn[i_idx, nnn_col]
                    K1 = nnn_i // 100000
                    K2 = nnn_i - K1 * 100000
                    K3 = K2 // 10
                    KSCALE = K2 - K3 * 10
                    scale_idx = KSCALE - 1
                    if scale_idx < 0:
                        scale_idx = 0
                    if scale_idx > n_scale - 1:
                        scale_idx = n_scale - 1

                    if IT % 2 == 1:
                        P1 = float(K1) * SCALE[scale_idx]
                        P2 = float(K3) * SCALE[scale_idx]
                        if DT < 0.0 and scale_idx <= 0:
                            KP1 = P1
                            if KP1 == float(int(P2 + 0.5)):
                                PMIN = KP1
                    else:
                        P1 = float(K3) * SCALE[scale_idx]
                        i_next = i_idx + 1
                        if i_next > nnn_rows - 1:
                            i_next = nnn_rows - 1
                        nnn_i1 = nnn[i_next, nnn_col]
                        K1_next = nnn_i1 // 100000
                        KSCALE_next = nnn_i1 % 10
                        scale_idx_next = KSCALE_next - 1
                        if scale_idx_next < 0:
                            scale_idx_next = 0
                        if scale_idx_next > n_scale - 1:
                            scale_idx_next = n_scale - 1
                        P2 = float(K1_next) * SCALE[scale_idx_next]

                    val = P1 + (P2 - P1) * DT
                    if val < PMIN:
                        val = PMIN
                    PART[ion_idx - 1] = val
                elif not handled:
                    PART[ion_idx - 1] = 1.0

                # PFGROUND correction for low temperatures
                skip_high_t_correction = False
                if ip_i > 0.0:
                    T2000 = ip_i * 2000.0 / 11.0
                    if T < T2000 * 2.0:
                        nelion = (iz - 1) * 6 + ion_idx
                        pfground_val = _pfground_nb(nelion, T)
                        if pfground_val > 0.0:
                            if pfground_val > PART[ion_idx - 1]:
                                PART[ion_idx - 1] = pfground_val
                        skip_high_t_correction = True

                special_case_bypass = D1 > 0.0
                if (
                    (not skip_high_t_correction)
                    and (G > 0.0 or D1 > 0.0)
                    and (special_case_bypass or POTLO[ion_idx - 1] >= 0.1)
                    and (ip_i > 0.0)
                ):
                    T2000 = ip_i * 2000.0 / 11.0
                    if special_case_bypass or T >= T2000 * 4.0:
                        TV_use = TV
                        if T > T2000 * 11.0:
                            TV_use = (T2000 * 11.0) * 8.6171e-5
                        if D1 <= 0.0:
                            D1_val = 0.1 / TV_use
                        else:
                            D1_val = D1
                        D2 = POTLO[ion_idx - 1] / TV_use
                        Zc = float(ion_idx)
                        term1 = np.sqrt(13.595 * Zc * Zc / TV_use / D2) ** 3.0
                        term1 *= (
                            1.0 / 3.0
                            + (1.0 - (0.5 + (1.0 / 18.0 + D2 / 120.0) * D2) * D2) * D2
                        )
                        term2 = np.sqrt(13.595 * Zc * Zc / TV_use / D1_val) ** 3.0
                        term2 *= (
                            1.0 / 3.0
                            + (
                                1.0
                                - (0.5 + (1.0 / 18.0 + D1_val / 120.0) * D1_val)
                                * D1_val
                            )
                            * D1_val
                        )
                        if G > 0.0:
                            PART[ion_idx - 1] += (
                                G * np.exp(-ip_i / TV_use) * (term1 - term2)
                            )

            # Saha equation F computation
            CF = 2.0 * 2.4148e15 * T * np.sqrt(T) / XNE_val
            F[0] = 1.0
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

            L = nion2 + 1
            for ion_idx in range(2, nion2 + 1):
                L = L - 1
                F[0] = 1.0 + F[L - 1] * F[0]
            F[0] = 1.0 / F[0]
            for ion_idx in range(2, nion2 + 1):
                F[ion_idx - 1] = F[ion_idx - 2] * F[ion_idx - 1]

            # Store results based on mode
            if mode < 10:
                if mode1 == 1:
                    answer[layer_idx, 0] = F[nion - 1] / PART[nion - 1]
                elif mode1 == 2:
                    answer[layer_idx, 0] = F[nion - 1]
                elif mode1 == 3:
                    answer[layer_idx, 0] = PART[nion - 1]
                elif mode1 == 4:
                    elec_sum = 0.0
                    for i in range(1, nion2):
                        elec_sum += F[i] * float(i)
                    answer[layer_idx, 0] = elec_sum
            else:
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


_DUMMY_DEP = np.ones((1, 1), dtype=np.float64)

_DEP_NAMES = (
    "bhyd", "bhe1", "bhe2", "bc1", "bc2", "bmg1", "bmg2", "bal1", "bsi1",
    "bsi2", "bca1", "bca2", "bo1", "bna1", "bb1", "bk1",
)


def _dep_table(departure_tables: Optional[Dict[str, np.ndarray]], name: str) -> np.ndarray:
    if departure_tables is None:
        return _DUMMY_DEP
    tbl = departure_tables.get(name)
    if tbl is None:
        return _DUMMY_DEP
    return np.ascontiguousarray(tbl, dtype=np.float64)


def pfsaha_core_numba(
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
    answer: np.ndarray,
    potion: np.ndarray,
    nnn: np.ndarray,
    pftab: np.ndarray,
    potlo_pfiron: np.ndarray,
    potlolog_pfiron: np.ndarray,
    departure_tables: Optional[Dict[str, np.ndarray]] = None,
    nlte_on: int = 0,
) -> None:
    """Numba-accelerated PFSAHA; identical math to ``_pfsaha_exact_python``."""
    dep_on = 1 if (departure_tables is not None and nlte_on != -1) else 0
    tbls = [_dep_table(departure_tables, name) for name in _DEP_NAMES]
    _pfsaha_core_nb(
        int(iz),
        int(nion),
        int(mode),
        int(nlte_on),
        int(dep_on),
        np.ascontiguousarray(temperature, dtype=np.float64),
        np.ascontiguousarray(tkev, dtype=np.float64),
        np.ascontiguousarray(tk, dtype=np.float64),
        np.ascontiguousarray(hckt, dtype=np.float64),
        np.ascontiguousarray(tlog, dtype=np.float64),
        np.ascontiguousarray(gas_pressure, dtype=np.float64),
        np.ascontiguousarray(electron_density, dtype=np.float64),
        answer,
        np.ascontiguousarray(potion, dtype=np.float64),
        nnn,
        pftab,
        potlo_pfiron,
        potlolog_pfiron,
        *tbls,
    )
