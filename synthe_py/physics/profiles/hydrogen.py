"""Hydrogen line profiles – partial port of the Kurucz HPROF4 stack."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np

from ..populations import DepthState, HydrogenDepthState
from ..tables import fast_ex
from .hydrogen_tables_data import load_hydrogen_profile_tables

RYDH = 3.2880515e15
C_LIGHT = 2.99792458e18
C_LIGHT_CM = 2.99792458e10
LYMAN_ALPHA_CENTER_WN = 82259.10
SQRT_PI = math.sqrt(math.pi)


@dataclass
class HydrogenTables:
    propbm: np.ndarray
    c: np.ndarray
    d: np.ndarray
    pp: np.ndarray
    beta: np.ndarray
    stalph: np.ndarray
    stwtal: np.ndarray
    istal: np.ndarray
    lnghal: np.ndarray
    stcomp: np.ndarray
    stcpwt: np.ndarray
    lncomp: np.ndarray
    cutoff_h2_plus: np.ndarray
    cutoff_h2: np.ndarray
    asum_lyman: np.ndarray
    asum: np.ndarray
    y1wtm: np.ndarray
    xknmtb: np.ndarray
    tabvi: np.ndarray
    tabh1: np.ndarray


@lru_cache(maxsize=1)
def hydrogen_tables() -> HydrogenTables:
    data = load_hydrogen_profile_tables()
    return HydrogenTables(
        propbm=data["propbm"],
        c=data["c"],
        d=data["d"],
        pp=data["pp"],
        beta=data["beta"],
        stalph=data["stalph"],
        stwtal=data["stwtal"],
        istal=data["istal"],
        lnghal=data["lnghal"],
        stcomp=data["stcomp"],
        stcpwt=data["stcpwt"],
        lncomp=data["lncomp"],
        cutoff_h2_plus=data["cutoff_h2_plus"],
        cutoff_h2=data["cutoff_h2"],
        asum_lyman=data["asum_lyman"],
        asum=data["asum"],
        y1wtm=data["y1wtm"],
        xknmtb=data["xknmtb"],
        tabvi=data["tabvi"],
        tabh1=data["tabh1"],
    )


def vcse1f(x: float) -> float:
    if x <= 0:
        return 0.0
    if x <= 0.01:
        return -math.log(x) - 0.577215 + x
    if x <= 1.0:
        return (
            -math.log(x)
            - 0.57721566
            + x
            * (
                0.99999193
                + x
                * (-0.24991055 + x * (0.05519968 + x * (-0.00976004 + x * 0.00107857)))
            )
        )
    if x > 30.0:
        return 0.0
    numerator = x * (x + 2.334733) + 0.25062
    denominator = (x * (x + 3.330657) + 1.681534) * x
    return numerator / denominator * fast_ex(x)


@lru_cache(maxsize=1)
def _e1_table() -> np.ndarray:
    values = np.zeros(2000, dtype=np.float64)
    for idx in range(1, 2000 + 1):
        x = idx * 0.01
        values[idx - 1] = math.exp(-x) / x
    return values


def faste1(x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x < 0.5:
        return (1.0 - 0.22464 * x) * x - math.log(x) - 0.57721
    if x > 20.0:
        return 0.0
    idx = min(int(x * 100.0 + 0.5), 1999)
    return _e1_table()[idx]


def sofbeta(beta: float, p: float, n: int, m: int) -> float:
    tables = hydrogen_tables()
    corr = 1.0
    b2 = beta * beta
    sb = math.sqrt(beta)
    if beta <= 500.0:
        indx = 7
        mmn = m - n
        if n <= 3 and mmn <= 2:
            indx = 2 * (n - 1) + mmn
        indx = max(1, min(indx, 7))
        im = min(int(5.0 * p) + 1, 4)
        im = max(im, 1)
        ip = im + 1
        wtp = 5.0 * (p - tables.pp[im - 1])
        wtp = max(0.0, min(1.0, wtp))
        wtm = 1.0 - wtp
        if beta <= 25.12:
            betagrid = tables.beta
            j = int(np.searchsorted(betagrid, beta, side="right"))
            j = max(1, min(j, betagrid.size - 1))
            jm = j - 1
            jp = j
            denom = betagrid[jp] - betagrid[jm]
            if denom <= 0:
                wtb = 0.0
            else:
                wtb = (beta - betagrid[jm]) / denom
            wtbm = 1.0 - wtb
            prop = tables.propbm[indx - 1]
            cbp = prop[ip - 1, jp] * wtp + prop[im - 1, jp] * wtm
            cbm = prop[ip - 1, jm] * wtp + prop[im - 1, jm] * wtm
            corr = 1.0 + cbp * wtb + cbm * wtbm
            pr1 = 0.0
            pr2 = 0.0
            wt = max(min(0.5 * (10.0 - beta), 1.0), 0.0)
            if beta <= 10.0:
                pr1 = 8.0 / (83.0 + (2.0 + 0.95 * b2) * beta)
            if beta >= 8.0:
                pr2 = (1.5 / sb + 27.0 / b2) / b2
            return (pr1 * wt + pr2 * (1.0 - wt)) * corr
        cc = tables.c[im - 1, indx - 1] * wtp + tables.c[ip - 1, indx - 1] * wtm
        dd = tables.d[im - 1, indx - 1] * wtp + tables.d[ip - 1, indx - 1] * wtm
        corr = 1.0 + dd / (cc + beta * sb)
    return (1.5 / sb + 27.0 / b2) / b2 * corr


@lru_cache(maxsize=256)
def _hf_nm_cached(n: int, m: int) -> float:
    if m <= n:
        return 0.0
    xn = float(n)
    ginf = 0.2027 / xn**0.71
    gca = 0.124 / xn
    fkn = xn * 1.9603
    wtc = 0.45 - 2.4 / xn**3 * (xn - 1.0)
    xm = float(m)
    xmn = xm - xn
    fk = fkn * (xm / (xmn * (xm + xn))) ** 3
    xmn12 = xmn**1.2
    wt = (xmn12 - 1.0) / (xmn12 + wtc)
    fnm = fk * (1.0 - wt * ginf - (0.222 + gca / xm) * (1.0 - wt))
    return fnm


def _fine_structure(
    n: int, m: int, tables: HydrogenTables
) -> Tuple[np.ndarray, np.ndarray]:
    mmn = m - n
    xn = float(n)
    xn2 = xn * xn
    if n > 4 or m > 10:
        return np.array([0.0]), np.array([1.0])
    if mmn != 1:
        ifins = tables.lncomp[n - 1]
        offsets = tables.stcomp[:ifins, n - 1] * 1.0e7
        weights = tables.stcpwt[:ifins, n - 1] / xn2
        return offsets, weights
    ifins = tables.lnghal[n - 1]
    ipos = tables.istal[n - 1]
    offsets = tables.stalph[ipos : ipos + ifins] * 1.0e7
    weights = tables.stwtal[ipos : ipos + ifins] / xn2 / 3.0
    return offsets, weights


@lru_cache(maxsize=64)
def _fine_structure_cached(n: int, m: int) -> Tuple[np.ndarray, np.ndarray]:
    return _fine_structure(n, m, hydrogen_tables())


def hydrogen_line_profile(
    n: int, m: int, depth_state: DepthState, delta_lambda_nm: float
) -> float:
    hyd = depth_state.hydrogen
    if hyd is None:
        return 0.0
    tables = hydrogen_tables()
    mmn = m - n
    xn = float(n)
    xm = float(m)
    xn2 = xn * xn
    xm2 = xm * xm
    xm2mn2 = xm2 - xn2
    xmn2 = xm2 * xn2
    gnm = xm2mn2 / xmn2
    if mmn <= 0:
        return 0.0
    if mmn <= 3 and n <= 4:
        xknm = tables.xknmtb[n - 1, mmn - 1]
    else:
        xknm = 5.5e-5 / gnm * xmn2 / (1.0 + 0.13 / mmn)
    freqnm = RYDH * gnm
    wavenm = C_LIGHT / freqnm
    dbeta = C_LIGHT / (freqnm * freqnm * xknm)
    c1con = xknm / wavenm * gnm * xm2mn2
    c2con = (xknm / wavenm) ** 2
    # ASUM/ASUMLYMAN are Fortran 1-based arrays; convert to Python 0-based indexing.
    radamp = tables.asum[n - 1] + tables.asum[m - 1]
    if n == 1:
        radamp = tables.asum_lyman[m - 1]
    radamp /= 12.5664
    radamp /= freqnm
    resont = _hf_nm_cached(1, m) / xm / (1.0 - 1.0 / xm2)
    if n != 1:
        resont += _hf_nm_cached(1, n) / xn / (1.0 - 1.0 / xn2)
    resont *= 3.579e-24 / gnm
    vdw = 4.45e-26 / gnm * (xm2 * (7.0 * xm2 + 5.0)) ** 0.4
    hwvdw = vdw * hyd.t3nhe + 2.0 * vdw * hyd.t3nh2
    hwrad = radamp
    stark = 1.6678e-18 * freqnm * xknm
    hwres = resont * hyd.xnfph[0] * 2.0 if hyd.xnfph.size > 0 else 0.0
    hwstk = stark * hyd.fo
    hwlor = hwres + hwvdw + hwrad
    finest, finswt = _fine_structure_cached(n, m)
    wl0 = wavenm
    wl = wl0 + delta_lambda_nm * 10.0
    freq = C_LIGHT / wl
    del_freq = abs(freq - freqnm)
    dopph = max(hyd.dopph, 1e-40)
    dop = freqnm * dopph
    hfwidth = freqnm * max(dopph, hwlor, hwstk)
    ifcore = del_freq <= hfwidth

    # Match Fortran NWID selection for core handling.
    nwid = 1
    if not (dopph >= hwstk and dopph >= hwlor):
        nwid = 2
        if hwlor < hwstk:
            nwid = 3

    # Doppler core (same normalization as VOIGT in Fortran: FASTEX only).
    core = 0.0
    for offset, weight in zip(finest, finswt):
        component_freq = freqnm + offset
        d = abs(freq - component_freq) / max(dop, 1e-30)
        if d <= 7.0:
            core += fast_ex(d * d) * weight

    # Lorentz component (including Lyman-alpha special case).
    lorentz = 0.0
    hhw = freqnm * hwlor
    if n == 1 and m == 2:
        lorentz = _lyman_alpha_lorentz(
            freq=freq,
            freqnm=freqnm,
            del_freq=del_freq,
            dop=dop,
            hwres=hwres,
            hwvdw=hwvdw,
            hwrad=hwrad,
            tables=tables,
            hyd=hyd,
        )
    else:
        top = hhw
        if n == 1 and m in {3, 4, 5}:
            freq_ratio = freq / RYDH
            if m == 3 and 0.885 <= freq_ratio <= 0.890:
                top = max(hhw - freqnm * hwrad, 0.0)
            elif m == 4 and 0.936 <= freq_ratio <= 0.938:
                top = max(hhw - freqnm * hwrad, 0.0)
            elif m == 5 and 0.959 <= freq_ratio <= 0.961:
                top = max(hhw - freqnm * hwrad, 0.0)
        if hhw > 0.0:
            lorentz = top / math.pi / (del_freq * del_freq + hhw * hhw) * 1.77245 * dop

    y1num = 320.0
    if m == 2:
        y1num = 550.0
    elif m == 3:
        y1num = 380.0

    y1wht = 1.0e13
    if mmn <= 3:
        y1wht = 1.0e14
    if (
        mmn <= 2
        and n <= 2
        and tables.y1wtm.shape[0] >= n
        and tables.y1wtm.shape[1] >= mmn
    ):
        y1wht = tables.y1wtm[n - 1, mmn - 1]

    wty1 = 1.0 / (1.0 + max(depth_state.electron_density, 0.0) / max(y1wht, 1e-30))
    y1_scal = y1num * hyd.y1s * wty1 + hyd.y1b * (1.0 - wty1)
    c1 = hyd.c1d * c1con * y1_scal
    c2 = hyd.c2d * c2con

    beta = del_freq / max(hyd.fo, 1e-30) * dbeta
    y1 = c1 * beta
    y2 = c2 * beta * beta
    g1 = 6.77 * math.sqrt(max(c1, 1e-30))
    ratio = 0.0
    if c1 > 0.0 and c2 > 0.0:
        ratio = math.sqrt(c2) / max(c1, 1e-30)
    log_term = 0.0
    if ratio > 0.0:
        log_term = math.log(max(ratio, 1e-30))
    gnot = g1 * max(0.0, 0.2114 + log_term) * (1.0 - hyd.gcon1 - hyd.gcon2)
    gamma = gnot
    if y2 > 1e-4 and y1 > 1e-5:
        gamma = (
            g1
            * (0.5 * fast_ex(min(80.0, y1)) + vcse1f(y1) - 0.5 * vcse1f(y2))
            * (
                1.0
                - hyd.gcon1 / (1.0 + (90.0 * y1) ** 3)
                - hyd.gcon2 / (1.0 + 2000.0 * y1)
            )
        )
    f = 0.0
    if gamma > 0:
        f = gamma / math.pi / (gamma * gamma + beta * beta)
    prqs = sofbeta(beta, hyd.pp, n, m)
    stark_extra = 0.0
    if m <= 2:
        prqs *= 0.5
        stark_extra = _lyman_quasistatic_cutoff(
            freq=freq,
            prqs=prqs,
            hyd=hyd,
            dbeta=dbeta,
            dop=dop,
            n=n,
            m=m,
        )
    p1 = (0.9 * y1) ** 2
    fns = (p1 + 0.03 * math.sqrt(max(y1, 0.0))) / (p1 + 1.0)
    stark_core = (prqs * (1.0 + fns) + f) / max(hyd.fo, 1e-30) * dbeta * 1.77245 * dop

    # Fortran core branch uses only the dominant width component.
    if ifcore:
        if nwid == 1:
            return max(core, 0.0)
        if nwid == 2:
            return max(lorentz, 0.0)
        return max(stark_core + stark_extra, 0.0)

    return max(core + lorentz + stark_core + stark_extra, 0.0)


def _interpolate_cutoff(
    delta_wavenumber: float, table: np.ndarray, start: float, step: float
) -> float | None:
    max_delta = start + step * (table.size - 1)
    if delta_wavenumber > max_delta:
        return None
    if delta_wavenumber <= start:
        if table.size < 2:
            return table[0] if table.size else None
        frac = (delta_wavenumber - start) / step
        return table[0] + (table[1] - table[0]) * frac
    position = (delta_wavenumber - start) / step
    index = int(math.floor(position))
    frac = position - index
    if index >= table.size - 1:
        return float(table[-1])
    return float(table[index] + (table[index + 1] - table[index]) * frac)


def _lyman_alpha_lorentz(
    freq: float,
    freqnm: float,
    del_freq: float,
    dop: float,
    hwres: float,
    hwvdw: float,
    hwrad: float,
    tables: HydrogenTables,
    hyd: HydrogenDepthState,
) -> float:
    if dop <= 0.0:
        return 0.0

    hwres_near = hwres * 4.0
    hwlor_near = hwres_near + hwvdw + hwrad
    hhw_near = freqnm * max(hwlor_near, 0.0)
    freq_threshold = (LYMAN_ALPHA_CENTER_WN - 4000.0) * C_LIGHT_CM
    wavenumber = freq / C_LIGHT_CM
    delta_wn = wavenumber - LYMAN_ALPHA_CENTER_WN

    if freq > freq_threshold and hhw_near > 0.0:
        hres_term = (
            hwres_near
            * freqnm
            / math.pi
            / (del_freq * del_freq + hhw_near * hhw_near)
            * 1.77245
            * dop
        )
        hhw_use = hhw_near
    else:
        cutoff_val = 0.0
        cutoff_log = _interpolate_cutoff(delta_wn, tables.cutoff_h2, -22000.0, 200.0)
        if cutoff_log is not None and hyd.xnfph.size > 0:
            cutoff_val = (10.0 ** (cutoff_log - 14.0)) * hyd.xnfph[0] * 2.0 / C_LIGHT_CM
        hres_term = cutoff_val * 1.77245 * dop
        hwlor = hwres + hwvdw + hwrad
        hhw_use = freqnm * max(hwlor, 0.0)

    hrad_term = 0.0
    if hwrad > 0.0 and hhw_use > 0.0:
        freq_low = 2.4190611e15
        freq_high = 0.77 * RYDH
        if freq > freq_low and freq < freq_high:
            hrad_term = (
                hwrad
                * freqnm
                / math.pi
                / (del_freq * del_freq + hhw_use * hhw_use)
                * 1.77245
                * dop
            )

    hvdw_term = 0.0
    if hwvdw > 0.0 and hhw_use > 0.0:
        if freq >= 1.8e15:
            hvdw_term = (
                hwvdw
                * freqnm
                / math.pi
                / (del_freq * del_freq + hhw_use * hhw_use)
                * 1.77245
                * dop
            )

    return hres_term + hrad_term + hvdw_term


def _lyman_quasistatic_cutoff(
    freq: float,
    prqs: float,
    hyd: HydrogenDepthState,
    dbeta: float,
    dop: float,
    n: int,
    m: int,
) -> float:
    if hyd.xnfph.size < 2 or hyd.fo <= 0.0:
        return 0.0

    wavenumber_center = LYMAN_ALPHA_CENTER_WN
    wavenumber = freq / C_LIGHT_CM
    delta_wn = wavenumber - wavenumber_center

    if delta_wn < -20000.0:
        return 0.0

    extra = 0.0
    if delta_wn <= -4000.0:
        cutoff_log = _interpolate_cutoff(
            delta_wn, hydrogen_tables().cutoff_h2_plus, -15000.0, 100.0
        )
        if cutoff_log is not None:
            cutoff_val = (10.0 ** (cutoff_log - 14.0)) * hyd.xnfph[1] / C_LIGHT_CM
            extra += cutoff_val * 1.77245 * dop
    else:
        beta4000 = 4000.0 * C_LIGHT_CM / max(hyd.fo, 1e-30) * dbeta
        prqs4000 = sofbeta(beta4000, hyd.pp, n, m) * 0.5
        normalization = prqs4000 / max(hyd.fo, 1e-30) * dbeta
        cutoff4000 = (10.0 ** (-11.07 - 14.0)) * hyd.xnfph[1] / C_LIGHT_CM
        if normalization > 0.0:
            extra += (
                cutoff4000
                / normalization
                * (prqs / max(hyd.fo, 1e-30) * dbeta)
                * 1.77245
                * dop
            )

    return extra
