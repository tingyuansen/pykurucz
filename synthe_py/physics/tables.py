"""Low-level numerical tables mirroring legacy SYNTHE behaviour."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

import numpy as np


@dataclass
class ExpTables:
    """Tabulated exponentials and exponential integrals for FASTEX/EXPI."""

    extab: np.ndarray
    extabf: np.ndarray

    @classmethod
    def build(cls, size: int = 1001) -> "ExpTables":
        """Reproduce the EXTAB/EXTABF tables from the Fortran code."""

        indices = np.arange(size, dtype=np.float64)
        extab = np.exp(-(indices))
        extabf = np.exp(-(indices) * 0.001)
        return cls(extab=extab, extabf=extabf)

    def fast_ex(self, x: float) -> float:
        """Port of the FASTEX helper: exp(-x) using lookup tables."""

        if x <= 0.0:
            return 1.0 if x == 0.0 else math.exp(-x)

        i = int(math.floor(x))
        if i >= self.extab.size:
            return math.exp(-x)

        frac = x - i
        j = int(frac * 1000.0 + 0.5)
        j = max(0, min(j, self.extabf.size - 1))
        return float(self.extab[i] * self.extabf[j])

    def fast_ex_array(self, x: np.ndarray) -> np.ndarray:
        """Vectorized FASTEX lookup with the same table rounding as fast_ex."""

        values = np.asarray(x, dtype=np.float64)
        out = np.empty_like(values, dtype=np.float64)

        zero_mask = values == 0.0
        neg_mask = values < 0.0
        pos_mask = values > 0.0

        out[zero_mask] = 1.0
        out[neg_mask] = np.exp(-values[neg_mask])

        if np.any(pos_mask):
            pos = values[pos_mask]
            i = np.floor(pos).astype(np.int64)
            table_mask = i < self.extab.size
            pos_out = np.empty_like(pos, dtype=np.float64)

            if np.any(table_mask):
                pos_table = pos[table_mask]
                i_table = i[table_mask]
                frac = pos_table - i_table
                j = np.floor(frac * 1000.0 + 0.5).astype(np.int64)
                j = np.clip(j, 0, self.extabf.size - 1)
                pos_out[table_mask] = self.extab[i_table] * self.extabf[j]

            if np.any(~table_mask):
                pos_out[~table_mask] = np.exp(-pos[~table_mask])

            out[pos_mask] = pos_out

        return out


# Lazily constructed singleton tables
_exp_tables: ExpTables | None = None


def _get_exp_tables() -> ExpTables:
    global _exp_tables
    if _exp_tables is None:
        _exp_tables = ExpTables.build()
    return _exp_tables


def fast_ex(x: float) -> float:
    """Public FASTEX wrapper."""

    return _get_exp_tables().fast_ex(x)


def fast_ex_array(x: np.ndarray) -> np.ndarray:
    """Public vectorized FASTEX wrapper."""

    return _get_exp_tables().fast_ex_array(x)


# Coefficients from the legacy EXPI implementation
_A0 = -4.43668255
_A1 = 4.42054938
_A2 = 3.16274620
_B0 = 7.68641124
_B1 = 5.65655216
_C0 = 0.0012102205
_C1 = 0.98147989
_C2 = 0.75339742
_D1 = 1.6198645
_D2 = 0.29135151
_E0 = -0.9969698
_E1 = -0.4257849
_F1 = 2.318261


def exp_integral(n: int, x: float) -> float:
    """Port of the Fortran EXPI function (orders N >= 1)."""

    if x < 0.0:
        raise ValueError("exp_integral defined for non-negative x")

    ex = math.exp(-x) if x < 1e2 else 0.0

    if x > 4.0:
        ex1 = (ex + ex * (_E0 + _E1 / x) / (x + _F1)) / x if x != 0.0 else float("inf")
    elif x > 1.0:
        numerator = ex * (_C2 + (_C1 + _C0 * x) * x)
        denominator = _D2 + (_D1 + x) * x
        ex1 = numerator / denominator
    elif x > 0.0:
        numerator = _A0 + (_A1 + _A2 * x) * x
        denominator = _B0 + (_B1 + x) * x
        ex1 = numerator / denominator - math.log(x)
    else:  # x == 0
        ex1 = 0.0

    result = ex1
    if n == 1:
        return result

    for i in range(1, n):
        denom = float(i)
        result = (ex - x * result) / denom
    return result




@dataclass
class ContinuumTables:
    """Tabulated continuum coefficients corresponding to continuum edges."""

    half_edge: np.ndarray
    delta_edge: np.ndarray
    wledge: np.ndarray
    ablog: np.ndarray


@lru_cache(maxsize=None)
def build_continuum_tables(wledge: Tuple[float, ...], ablog_flat: Tuple[float, ...]) -> ContinuumTables:
    """Construct structured tables from raw edge data."""

    wledge_arr = np.abs(np.asarray(wledge, dtype=np.float64))
    nedge = wledge_arr.size
    half_edge = np.zeros(nedge - 1, dtype=np.float64)
    delta_edge = np.zeros(nedge - 1, dtype=np.float64)

    for idx in range(1, nedge):
        half_edge[idx - 1] = 0.5 * (wledge_arr[idx - 1] + wledge_arr[idx])
        delta = wledge_arr[idx] - wledge_arr[idx - 1]
        delta_edge[idx - 1] = 0.5 * delta * delta

    ablog = np.array(ablog_flat, dtype=np.float64).reshape(3, nedge - 1)
    return ContinuumTables(
        half_edge=half_edge,
        delta_edge=delta_edge,
        wledge=wledge_arr,
        ablog=ablog,
    )


@dataclass(frozen=True)
class MetalWingTables:
    """Static continuum-merge tables used by XLINOP for metal wings."""

    conth: np.ndarray
    contx: np.ndarray


_METAL_CONTH = np.array(
    [
        109678.7640,
        27419.6590,
        12186.4620,
        6854.8710,
        4387.1130,
        3046.6040,
        2238.3200,
        1713.7110,
        1354.0440,
        1096.7760,
        906.4260,
        761.6500,
        648.9800,
        559.5790,
        487.4560,
    ],
    dtype=np.float64,
)

_METAL_CONTX = np.array(
    [
        (109678.764000, 198310.760000, 438908.850000, 90883.840000, 0.000000, 61671.020000, 0.000000, 48278.370000, 0.000000, 66035.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000),
        (27419.659000, 38454.691000, 109726.529000, 90867.420000, 0.000000, 39820.615000, 0.000000, 48166.309000, 0.000000, 65957.885000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000),
        (12186.462000, 32033.214000, 48766.491000, 90840.420000, 0.000000, 39800.556000, 0.000000, 0.000000, 0.000000, 65811.843000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000),
        (6854.871000, 29223.753000, 27430.925000, 90820.420000, 0.000000, 39759.842000, 0.000000, 0.000000, 0.000000, 65747.550000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000),
        (4387.113000, 27175.760000, 17555.715000, 90804.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 65670.435000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000),
        (3046.604000, 15073.868000, 12191.437000, 90777.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 65524.393000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000),
        (2238.320000, 0.000000, 0.000000, 80691.180000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 59736.150000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000),
        (1713.711000, 0.000000, 0.000000, 80627.760000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 59448.700000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000),
        (1354.044000, 0.000000, 0.000000, 69235.820000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 50640.630000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000),
        (1096.776000, 0.000000, 0.000000, 69172.400000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 50553.180000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000),
        (0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000),
        (0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000),
        (0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000),
        (0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000),
        (0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000),
        (0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000),
        (0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000),
        (0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000),
        (0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000),
        (0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000),
        (0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000),
        (0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000),
        (0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000),
        (0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000),
        (0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000),
        (0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000),
    ],
    dtype=np.float64,
)


@lru_cache(maxsize=1)
def metal_wing_tables() -> MetalWingTables:
    """Return immutable metal-wing lookup tables."""

    return MetalWingTables(
        conth=_METAL_CONTH.copy(),
        contx=_METAL_CONTX.copy(),
    )
@dataclass
class VoigtTables:
    """Tables used by the Voigt approximation (matches Fortran /H1TAB/)."""

    vsteps: float
    h0tab: np.ndarray
    h1tab: np.ndarray
    h2tab: np.ndarray

    @classmethod
    def build(cls, vsteps: float = 200.0, size: int = 2001) -> "VoigtTables":
        h0tab = np.zeros(size, dtype=np.float64)
        h1tab = np.zeros(size, dtype=np.float64)
        h2tab = np.zeros(size, dtype=np.float64)

        # Initial grid for H0TAB (frequency grid)
        for i in range(size):
            h0tab[i] = float(i) / vsteps

        tabvi, tabh1 = _voigt_reference_tables()
        h1tab[:] = _map1_interpolate(tabvi, tabh1, h0tab)

        for i in range(size):
            vv = (float(i) / vsteps) ** 2
            h0tab[i] = math.exp(-vv)
            h2tab[i] = h0tab[i] - (vv + vv) * h0tab[i]

        return cls(vsteps=vsteps, h0tab=h0tab, h1tab=h1tab, h2tab=h2tab)


@lru_cache(maxsize=1)
def voigt_tables() -> VoigtTables:
    return VoigtTables.build()


def _voigt_reference_tables() -> Tuple[np.ndarray, np.ndarray]:
    tabvi = np.array([
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
        1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2,
        3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6,
        5.8, 6.0, 6.2, 6.4, 6.6, 6.8, 7.0, 7.2, 7.4, 7.6, 7.8, 8.0, 8.2, 8.4, 8.6, 8.8,
        9.0, 9.2, 9.4, 9.6, 9.8, 10.0, 10.2, 10.4, 10.6, 10.8, 11.0, 11.2, 11.4, 11.6,
        11.8, 12.0
    ], dtype=np.float64)

    tabh1 = np.array([
        -1.12838, -1.10596, -1.04048, -0.93703, -0.80346, -0.64945,
        -0.48552, -0.32192, -0.16772, -0.03012, 0.08594, 0.17789, 0.24537, 0.28981,
        0.31394, 0.32130, 0.31573, 0.30094, 0.28027, 0.25648, 0.231726, 0.207528, 0.184882,
        0.164341, 0.146128, 0.130236, 0.116515, 0.104739, 0.094653, 0.086005, 0.078565,
        0.072129, 0.066526, 0.061615, 0.057281, 0.053430, 0.049988, 0.046894, 0.044098,
        0.041561, 0.039250, 0.035195, 0.031762, 0.028824, 0.026288, 0.024081, 0.022146,
        0.020441, 0.018929, 0.017582, 0.016375, 0.015291, 0.014312, 0.013426, 0.012620,
        0.0118860, 0.0112145, 0.0105990, 0.0100332, 0.0095119, 0.0090306, 0.0085852,
        0.0081722, 0.0077885, 0.0074314, 0.0070985, 0.0067875, 0.0064967, 0.0062243,
        0.0059688, 0.0057287, 0.0055030, 0.0052903, 0.0050898, 0.0049006, 0.0047217,
        0.0045526, 0.0043924, 0.0042405, 0.0040964, 0.0039595
    ], dtype=np.float64)
    return tabvi, tabh1


def _map1_interpolate(xold: np.ndarray, fold: np.ndarray, xnew: np.ndarray) -> np.ndarray:
    """Exact port of the Fortran MAP1 routine used by TABVOIGT."""

    nold = xold.size
    nnew = xnew.size
    fnew = np.zeros_like(xnew, dtype=np.float64)

    # 1-based working arrays to mirror Fortran indexing.
    xold1 = np.zeros(nold + 1, dtype=np.float64)
    fold1 = np.zeros(nold + 1, dtype=np.float64)
    xold1[1:] = xold
    fold1[1:] = fold

    l = 2
    ll = 0
    a = b = c = 0.0
    abac = bbac = cbac = 0.0
    afor = bfor = cfor = 0.0

    for k in range(1, nnew + 1):
        x = float(xnew[k - 1])

        # Label 10 loop
        while True:
            if x < xold1[l]:
                break
            l += 1
            if l > nold:
                break

        # Label 30: boundary/extrapolation
        if l > nold:
            if l != ll:
                l = min(nold, l)
                c = 0.0
                b = (fold1[l] - fold1[l - 1]) / (xold1[l] - xold1[l - 1])
                a = fold1[l] - xold1[l] * b
                ll = l
            fnew[k - 1] = a + (b + c * x) * x
            continue

        # Label 20
        if l == ll:
            fnew[k - 1] = a + (b + c * x) * x
            continue

        if l == 2:
            if l != ll:
                c = 0.0
                b = (fold1[l] - fold1[l - 1]) / (xold1[l] - xold1[l - 1])
                a = fold1[l] - xold1[l] * b
                ll = l
            fnew[k - 1] = a + (b + c * x) * x
            continue

        l1 = l - 1

        if l > ll + 1 or l == 3:
            # Label 21
            l2 = l - 2
            d = (fold1[l1] - fold1[l2]) / (xold1[l1] - xold1[l2])
            cbac = (
                fold1[l]
                / ((xold1[l] - xold1[l1]) * (xold1[l] - xold1[l2]))
                + (
                    fold1[l2] / (xold1[l] - xold1[l2])
                    - fold1[l1] / (xold1[l] - xold1[l1])
                )
                / (xold1[l1] - xold1[l2])
            )
            bbac = d - (xold1[l1] + xold1[l2]) * cbac
            abac = fold1[l2] - xold1[l2] * d + xold1[l1] * xold1[l2] * cbac
            if l == nold:
                # Label 22
                c = cbac
                b = bbac
                a = abac
                ll = l
                fnew[k - 1] = a + (b + c * x) * x
                continue
        else:
            # Reuse backward coefficients
            cbac = cfor
            bbac = bfor
            abac = afor
            if l == nold:
                c = cbac
                b = bbac
                a = abac
                ll = l
                fnew[k - 1] = a + (b + c * x) * x
                continue

        # Label 25: forward coefficients
        d = (fold1[l] - fold1[l1]) / (xold1[l] - xold1[l1])
        cfor = (
            fold1[l + 1]
            / ((xold1[l + 1] - xold1[l]) * (xold1[l + 1] - xold1[l1]))
            + (
                fold1[l1] / (xold1[l + 1] - xold1[l1])
                - fold1[l] / (xold1[l + 1] - xold1[l])
            )
            / (xold1[l] - xold1[l1])
        )
        bfor = d - (xold1[l] + xold1[l1]) * cfor
        afor = fold1[l1] - xold1[l1] * d + xold1[l] * xold1[l1] * cfor

        wt = 0.0
        if abs(cfor) != 0.0:
            wt = abs(cfor) / (abs(cfor) + abs(cbac))
        a = afor + wt * (abac - afor)
        b = bfor + wt * (bbac - bfor)
        c = cfor + wt * (cbac - cfor)
        ll = l
        fnew[k - 1] = a + (b + c * x) * x

    return fnew
