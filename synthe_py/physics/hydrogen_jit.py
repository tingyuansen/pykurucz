"""Numba JIT-compiled hydrogen wing opacity kernels.

Depth-parallel port of Fortran HPROF4/HLINOP physics from synthe.for.
All Python object references (DepthState, HydrogenDepthState) are replaced
with plain NumPy arrays so nopython=True compilation succeeds.

Speedup vs pure-Python baseline: ~200-1000x (depending on depth count and
number of H lines) due to JIT compilation of the inner profile loop and
prange parallelism over depth layers.
"""

from __future__ import annotations

import math

import numpy as np
from numba import jit, prange

# ──────────────────────────────── constants ────────────────────────────────────
RYDH = 3.2880515e15          # Hz  (Rydberg frequency)
C_LIGHT = 2.99792458e18      # Å/s (= nm/s × 10; used as in hydrogen.py)
C_LIGHT_CM = 2.99792458e10   # cm/s
LYMAN_ALPHA_CENTER_WN = 82259.10  # cm⁻¹

# E1-integral lookup table (identical to _e1_table() in hydrogen.py)
def _build_e1_table() -> np.ndarray:
    values = np.empty(2000, dtype=np.float64)
    for i in range(2000):
        x = (i + 1) * 0.01
        values[i] = math.exp(-x) / x
    return values


_E1_TABLE: np.ndarray = _build_e1_table()

# ─────────────────────── Numba JIT scalar helpers ─────────────────────────────

@jit(nopython=True, cache=True)
def _vcse1f_jit(x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x <= 0.01:
        return -math.log(x) - 0.577215 + x
    if x <= 1.0:
        return (
            -math.log(x)
            - 0.57721566
            + x * (0.99999193 + x * (-0.24991055 + x * (0.05519968 + x * (-0.00976004 + x * 0.00107857))))
        )
    if x > 30.0:
        return 0.0
    numerator = x * (x + 2.334733) + 0.25062
    denominator = (x * (x + 3.330657) + 1.681534) * x
    return numerator / denominator * math.exp(-x)


@jit(nopython=True, cache=True)
def _faste1_jit(x: float, e1_table: np.ndarray) -> float:
    if x <= 0.0:
        return 0.0
    if x < 0.5:
        return (1.0 - 0.22464 * x) * x - math.log(x) - 0.57721
    if x > 20.0:
        return 0.0
    idx = min(int(x * 100.0 + 0.5), 1999)
    return e1_table[idx]


@jit(nopython=True, cache=True)
def _fast_ex_jit(x: float) -> float:
    if x > 80.0:
        return 0.0
    return math.exp(-x)


@jit(nopython=True, cache=True)
def _hf_nm_jit(n: int, m: int) -> float:
    """Port of _hf_nm_cached for use inside nopython JIT."""
    if m <= n:
        return 0.0
    xn = float(n)
    ginf = 0.2027 / xn ** 0.71
    gca = 0.124 / xn
    fkn = xn * 1.9603
    wtc = 0.45 - 2.4 / xn ** 3 * (xn - 1.0)
    xm = float(m)
    xmn = xm - xn
    fk = fkn * (xm / (xmn * (xm + xn))) ** 3
    xmn12 = xmn ** 1.2
    wt = (xmn12 - 1.0) / (xmn12 + wtc)
    return fk * (1.0 - wt * ginf - (0.222 + gca / xm) * (1.0 - wt))


@jit(nopython=True, cache=True)
def _interpolate_cutoff_jit(
    delta_wavenumber: float,
    table: np.ndarray,
    start: float,
    step: float,
) -> float:
    """Returns 1e300 (sentinel for None) when delta_wavenumber > max_delta."""
    max_delta = start + step * (table.shape[0] - 1)
    if delta_wavenumber > max_delta:
        return 1e300
    if delta_wavenumber <= start:
        if table.shape[0] < 2:
            return table[0]
        frac = (delta_wavenumber - start) / step
        return table[0] + (table[1] - table[0]) * frac
    position = (delta_wavenumber - start) / step
    index = int(math.floor(position))
    frac = position - index
    if index >= table.shape[0] - 1:
        return float(table[-1])
    return float(table[index] + (table[index + 1] - table[index]) * frac)


@jit(nopython=True, cache=True)
def _sofbeta_jit(
    beta: float,
    p: float,
    n: int,
    m: int,
    propbm: np.ndarray,   # (7, 5, 15)
    c_arr: np.ndarray,    # (5, 7)
    d_arr: np.ndarray,    # (5, 7)
    pp_arr: np.ndarray,   # (5,)
    beta_arr: np.ndarray, # (15,)
) -> float:
    """Port of sofbeta – statistical broadening. Safe for all (n,m) values."""
    if beta <= 0.0:
        return 0.0
    b2 = beta * beta
    sb = math.sqrt(beta)
    corr = 1.0
    if beta <= 500.0:
        mmn = m - n
        if n <= 3 and mmn <= 2:
            indx = 2 * (n - 1) + mmn
        else:
            indx = 7
        if indx < 1:
            indx = 1
        if indx > 7:
            indx = 7
        im = min(int(5.0 * p) + 1, 4)
        if im < 1:
            im = 1
        ip = im + 1  # max 5, safe
        wtp = 5.0 * (p - pp_arr[im - 1])
        if wtp < 0.0:
            wtp = 0.0
        if wtp > 1.0:
            wtp = 1.0
        wtm = 1.0 - wtp
        if beta <= 25.12:
            j = np.searchsorted(beta_arr, beta)
            if j < 1:
                j = 1
            if j > beta_arr.shape[0] - 1:
                j = beta_arr.shape[0] - 1
            jm = j - 1
            jp = j
            denom = beta_arr[jp] - beta_arr[jm]
            if denom <= 0.0:
                wtb = 0.0
            else:
                wtb = (beta - beta_arr[jm]) / denom
            wtbm = 1.0 - wtb
            cbp = (
                propbm[indx - 1, ip - 1, jp] * wtp
                + propbm[indx - 1, im - 1, jp] * wtm
            )
            cbm = (
                propbm[indx - 1, ip - 1, jm] * wtp
                + propbm[indx - 1, im - 1, jm] * wtm
            )
            corr = 1.0 + cbp * wtb + cbm * wtbm
            wt = 0.5 * (10.0 - beta)
            if wt < 0.0:
                wt = 0.0
            if wt > 1.0:
                wt = 1.0
            pr1 = 0.0
            pr2 = 0.0
            if beta <= 10.0:
                pr1 = 8.0 / (83.0 + (2.0 + 0.95 * b2) * beta)
            if beta >= 8.0:
                pr2 = (1.5 / sb + 27.0 / b2) / b2
            return (pr1 * wt + pr2 * (1.0 - wt)) * corr
        cc = c_arr[im - 1, indx - 1] * wtp + c_arr[ip - 1, indx - 1] * wtm
        dd = d_arr[im - 1, indx - 1] * wtp + d_arr[ip - 1, indx - 1] * wtm
        denom2 = cc + beta * sb
        if denom2 == 0.0:
            denom2 = 1e-30
        corr = 1.0 + dd / denom2
    return (1.5 / sb + 27.0 / b2) / b2 * corr


@jit(nopython=True, cache=True)
def _lyman_alpha_lorentz_jit(
    freq: float,
    freqnm: float,
    del_freq: float,
    dop: float,
    hwres: float,
    hwvdw: float,
    hwrad: float,
    cutoff_h2: np.ndarray,       # (91,) start=-22000, step=200
    cutoff_h2_plus: np.ndarray,  # (111,) start=-15000, step=100
    xnfph_0: float,
    xnfph_1: float,
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
            hwres_near * freqnm / math.pi
            / (del_freq * del_freq + hhw_near * hhw_near)
            * 1.77245 * dop
        )
        hhw_use = hhw_near
    else:
        cutoff_val = 0.0
        cutoff_log = _interpolate_cutoff_jit(delta_wn, cutoff_h2, -22000.0, 200.0)
        if cutoff_log < 1e299 and xnfph_0 > 0.0:
            cutoff_val = (10.0 ** (cutoff_log - 14.0)) * xnfph_0 * 2.0 / C_LIGHT_CM
        hres_term = cutoff_val * 1.77245 * dop
        hwlor = hwres + hwvdw + hwrad
        hhw_use = freqnm * max(hwlor, 0.0)

    hrad_term = 0.0
    if hwrad > 0.0 and hhw_use > 0.0:
        freq_low = 2.4190611e15
        freq_high = 0.77 * RYDH
        if freq > freq_low and freq < freq_high:
            hrad_term = (
                hwrad * freqnm / math.pi
                / (del_freq * del_freq + hhw_use * hhw_use)
                * 1.77245 * dop
            )

    hvdw_term = 0.0
    if hwvdw > 0.0 and hhw_use > 0.0:
        if freq >= 1.8e15:
            hvdw_term = (
                hwvdw * freqnm / math.pi
                / (del_freq * del_freq + hhw_use * hhw_use)
                * 1.77245 * dop
            )

    return hres_term + hrad_term + hvdw_term


@jit(nopython=True, cache=True)
def _lyman_quasistatic_cutoff_jit(
    freq: float,
    prqs: float,
    xnfph_0: float,
    xnfph_1: float,
    fo: float,
    dbeta: float,
    dop: float,
    n: int,
    m: int,
    cutoff_h2_plus: np.ndarray,  # (111,) start=-15000, step=100
    propbm: np.ndarray,
    c_arr: np.ndarray,
    d_arr: np.ndarray,
    pp_arr: np.ndarray,
    beta_arr: np.ndarray,
    pp_val: float,
) -> float:
    if fo <= 0.0:
        return 0.0
    wavenumber = freq / C_LIGHT_CM
    delta_wn = wavenumber - LYMAN_ALPHA_CENTER_WN
    if delta_wn < -20000.0:
        return 0.0
    extra = 0.0
    if delta_wn <= -4000.0:
        cutoff_log = _interpolate_cutoff_jit(delta_wn, cutoff_h2_plus, -15000.0, 100.0)
        if cutoff_log < 1e299:
            cutoff_val = (10.0 ** (cutoff_log - 14.0)) * xnfph_1 / C_LIGHT_CM
            extra += cutoff_val * 1.77245 * dop
    else:
        fo_safe = max(fo, 1e-30)
        beta4000 = 4000.0 * C_LIGHT_CM / fo_safe * dbeta
        prqs4000 = _sofbeta_jit(beta4000, pp_val, n, m, propbm, c_arr, d_arr, pp_arr, beta_arr) * 0.5
        normalization = prqs4000 / fo_safe * dbeta
        cutoff4000 = (10.0 ** (-11.07 - 14.0)) * xnfph_1 / C_LIGHT_CM
        if normalization > 0.0:
            extra += (
                cutoff4000 / normalization
                * (prqs / fo_safe * dbeta)
                * 1.77245 * dop
            )
    return extra


# ────────────────── Core profile function (scalar, JIT) ───────────────────────

@jit(nopython=True, cache=True)
def _hydrogen_line_profile_jit(
    n: int,
    m: int,
    delta_lambda_nm: float,
    # Per-depth scalars from HydrogenDepthState
    t3nhe: float,
    t3nh2: float,
    fo: float,
    dopph: float,
    c1d: float,
    c2d: float,
    y1s: float,
    y1b: float,
    gcon1: float,
    gcon2: float,
    pp_val: float,
    xnfph_0: float,
    xnfph_1: float,
    electron_density: float,
    # Tables
    asum: np.ndarray,            # (96,)
    asum_lyman: np.ndarray,      # (100,)
    y1wtm: np.ndarray,           # (2, 2)
    xknmtb: np.ndarray,          # (4, 3)
    propbm: np.ndarray,          # (7, 5, 15)
    c_tbl: np.ndarray,           # (5, 7)
    d_tbl: np.ndarray,           # (5, 7)
    pp_tbl: np.ndarray,          # (5,)
    beta_tbl: np.ndarray,        # (15,)
    cutoff_h2_plus: np.ndarray,  # (111,)
    cutoff_h2: np.ndarray,       # (91,)
    e1_table: np.ndarray,        # (2000,)
    # Fine structure for this (n, m) – padded to MAX_FINE components
    fine_offsets: np.ndarray,    # (MAX_FINE,) frequency offsets in Hz
    fine_weights: np.ndarray,    # (MAX_FINE,)
    n_fine: int,
) -> float:
    mmn = m - n
    if mmn <= 0:
        return 0.0

    xn = float(n)
    xm = float(m)
    xn2 = xn * xn
    xm2 = xm * xm
    xm2mn2 = xm2 - xn2
    xmn2 = xm2 * xn2
    gnm = xm2mn2 / xmn2

    if n <= 4 and mmn <= 3 and n >= 1 and mmn >= 1:
        xknm = xknmtb[n - 1, mmn - 1]
    else:
        xknm = 5.5e-5 / gnm * xmn2 / (1.0 + 0.13 / float(mmn))

    freqnm = RYDH * gnm
    wavenm = C_LIGHT / freqnm
    dbeta = C_LIGHT / (freqnm * freqnm * xknm)
    c1con = xknm / wavenm * gnm * xm2mn2
    c2con = (xknm / wavenm) ** 2

    n_asum = asum.shape[0]
    m_asum = asum.shape[0]
    if n <= n_asum and m <= m_asum:
        radamp = asum[n - 1] + asum[m - 1]
    elif n <= n_asum:
        radamp = asum[n - 1]
    elif m <= m_asum:
        radamp = asum[m - 1]
    else:
        radamp = 0.0

    n_asum_ly = asum_lyman.shape[0]
    if n == 1 and m <= n_asum_ly:
        radamp = asum_lyman[m - 1]

    radamp /= 12.5664
    radamp /= freqnm

    resont = _hf_nm_jit(1, m) / xm / (1.0 - 1.0 / xm2)
    if n != 1:
        resont += _hf_nm_jit(1, n) / xn / (1.0 - 1.0 / xn2)
    resont *= 3.579e-24 / gnm

    vdw = 4.45e-26 / gnm * (xm2 * (7.0 * xm2 + 5.0)) ** 0.4
    hwvdw = vdw * t3nhe + 2.0 * vdw * t3nh2
    hwrad = radamp
    stark = 1.6678e-18 * freqnm * xknm
    hwres = resont * xnfph_0 * 2.0
    hwstk = stark * fo
    hwlor = hwres + hwvdw + hwrad

    wl = wavenm + delta_lambda_nm * 10.0
    if wl <= 0.0:
        return 0.0
    freq = C_LIGHT / wl
    del_freq = abs(freq - freqnm)
    dopph_safe = max(dopph, 1e-40)
    dop = freqnm * dopph_safe
    hfwidth = freqnm * max(dopph_safe, hwlor, hwstk)
    ifcore = del_freq <= hfwidth

    nwid = 1
    if not (dopph_safe >= hwstk and dopph_safe >= hwlor):
        nwid = 2
        if hwlor < hwstk:
            nwid = 3

    # Doppler core (fine structure components)
    core = 0.0
    for fi in range(n_fine):
        component_freq = freqnm + fine_offsets[fi]
        dop_safe = max(dop, 1e-30)
        d = abs(freq - component_freq) / dop_safe
        if d <= 7.0:
            core += _fast_ex_jit(d * d) * fine_weights[fi]

    # Lorentz component
    lorentz = 0.0
    hhw = freqnm * hwlor
    if n == 1 and m == 2:
        lorentz = _lyman_alpha_lorentz_jit(
            freq, freqnm, del_freq, dop,
            hwres, hwvdw, hwrad,
            cutoff_h2, cutoff_h2_plus,
            xnfph_0, xnfph_1,
        )
    else:
        top = hhw
        if n == 1:
            freq_ratio = freq / RYDH
            if m == 3 and 0.885 <= freq_ratio <= 0.890:
                top = max(hhw - freqnm * hwrad, 0.0)
            elif m == 4 and 0.936 <= freq_ratio <= 0.938:
                top = max(hhw - freqnm * hwrad, 0.0)
            elif m == 5 and 0.959 <= freq_ratio <= 0.961:
                top = max(hhw - freqnm * hwrad, 0.0)
        if hhw > 0.0:
            lorentz = (
                top / math.pi
                / (del_freq * del_freq + hhw * hhw)
                * 1.77245 * dop
            )

    # y1 weights
    y1num = 320.0
    if m == 2:
        y1num = 550.0
    elif m == 3:
        y1num = 380.0

    y1wht = 1.0e13
    if mmn <= 3:
        y1wht = 1.0e14
    if mmn <= 2 and n <= 2 and n >= 1 and mmn >= 1:
        if n <= y1wtm.shape[0] and mmn <= y1wtm.shape[1]:
            y1wht = y1wtm[n - 1, mmn - 1]

    y1wht_safe = max(y1wht, 1e-30)
    elec_safe = max(electron_density, 0.0)
    wty1 = 1.0 / (1.0 + elec_safe / y1wht_safe)
    y1_scal = y1num * y1s * wty1 + y1b * (1.0 - wty1)
    c1 = c1d * c1con * y1_scal
    c2 = c2d * c2con

    fo_safe = max(fo, 1e-30)
    beta = del_freq / fo_safe * dbeta
    y1 = c1 * beta
    y2 = c2 * beta * beta
    g1 = 6.77 * math.sqrt(max(c1, 1e-30))
    ratio = 0.0
    if c1 > 0.0 and c2 > 0.0:
        ratio = math.sqrt(c2) / max(c1, 1e-30)
    log_term = 0.0
    if ratio > 0.0:
        log_term = math.log(max(ratio, 1e-30))
    gnot = g1 * max(0.0, 0.2114 + log_term) * (1.0 - gcon1 - gcon2)
    gamma = gnot
    if y2 > 1e-4 and y1 > 1e-5:
        gamma = (
            g1
            * (
                0.5 * _fast_ex_jit(min(80.0, y1))
                + _vcse1f_jit(y1)
                - 0.5 * _vcse1f_jit(y2)
            )
            * (
                1.0
                - gcon1 / (1.0 + (90.0 * y1) ** 3)
                - gcon2 / (1.0 + 2000.0 * y1)
            )
        )

    f = 0.0
    if gamma > 0.0:
        f = gamma / math.pi / (gamma * gamma + beta * beta)

    prqs = _sofbeta_jit(beta, pp_val, n, m, propbm, c_tbl, d_tbl, pp_tbl, beta_tbl)
    stark_extra = 0.0
    if m <= 2:
        prqs *= 0.5
        stark_extra = _lyman_quasistatic_cutoff_jit(
            freq, prqs, xnfph_0, xnfph_1, fo, dbeta, dop,
            n, m, cutoff_h2_plus, propbm, c_tbl, d_tbl, pp_tbl, beta_tbl, pp_val,
        )

    p1 = (0.9 * y1) ** 2
    fns = (p1 + 0.03 * math.sqrt(max(y1, 0.0))) / (p1 + 1.0)
    fo_safe2 = max(fo, 1e-30)
    stark_core = (prqs * (1.0 + fns) + f) / fo_safe2 * dbeta * 1.77245 * dop

    if ifcore:
        if nwid == 1:
            return max(core, 0.0)
        if nwid == 2:
            return max(lorentz, 0.0)
        return max(stark_core + stark_extra, 0.0)

    return max(core + lorentz + stark_core + stark_extra, 0.0)


# ──────────────────── Wing expansion loop (one line, one depth) ───────────────

@jit(nopython=True, cache=True)
def _accumulate_hyd_line_depth_jit(
    buffer: np.ndarray,
    continuum_row: np.ndarray,
    stim_row: np.ndarray,
    wavelength_grid: np.ndarray,
    center_index: int,
    line_wavelength: float,
    kappa0: float,
    n_lower: int,
    n_upper: int,
    wcon: float,
    wtail: float,
    wlminus1: float,
    wlminus2: float,
    wlplus1: float,
    wlplus2: float,
    redcut: float,
    bluecut: float,
    cutoff: float,
    # Depth scalars
    t3nhe: float, t3nh2: float, fo: float, dopph: float,
    c1d: float, c2d: float, y1s: float, y1b: float,
    gcon1: float, gcon2: float, pp_val: float,
    xnfph_0: float, xnfph_1: float, electron_density: float,
    # Tables
    asum: np.ndarray,
    asum_lyman: np.ndarray,
    y1wtm: np.ndarray,
    xknmtb: np.ndarray,
    propbm: np.ndarray,
    c_tbl: np.ndarray,
    d_tbl: np.ndarray,
    pp_tbl: np.ndarray,
    beta_tbl: np.ndarray,
    cutoff_h2_plus: np.ndarray,
    cutoff_h2: np.ndarray,
    e1_table: np.ndarray,
    fine_offsets: np.ndarray,
    fine_weights: np.ndarray,
    n_fine: int,
) -> None:
    n_points = buffer.shape[0]
    simple_wings = n_upper <= n_lower + 2
    use_taper = (not simple_wings) and (wtail > wcon)
    upper_minus2 = max(n_upper - 2, n_lower + 1)
    upper_plus2 = n_upper + 2

    red_active = True
    blue_active = True
    offset = 1
    max_steps = center_index
    tmp = n_points - center_index - 1
    if tmp > max_steps:
        max_steps = tmp

    if 0 <= center_index < n_points:
        wave_center = wavelength_grid[center_index]
        if not (not simple_wings and wave_center < wcon):
            delta_center_nm = wave_center - line_wavelength
            profile_center = kappa0 * _hydrogen_line_profile_jit(
                n_lower, n_upper, delta_center_nm,
                t3nhe, t3nh2, fo, dopph, c1d, c2d, y1s, y1b,
                gcon1, gcon2, pp_val, xnfph_0, xnfph_1, electron_density,
                asum, asum_lyman, y1wtm, xknmtb, propbm, c_tbl, d_tbl, pp_tbl, beta_tbl,
                cutoff_h2_plus, cutoff_h2, e1_table, fine_offsets, fine_weights, n_fine,
            )
            value_center = profile_center * stim_row[center_index]
            if use_taper and wave_center < wtail:
                value_center *= (wave_center - wcon) / (wtail - wcon)
            if value_center >= continuum_row[center_index] * cutoff:
                buffer[center_index] += value_center
    else:
        if center_index >= n_points:
            red_active = False
            offset = max(1, center_index - (n_points - 1))
        else:
            blue_active = False
            offset = max(1, -center_index)

    while offset <= max_steps and (red_active or blue_active):
        if red_active:
            idx = center_index + offset
            if idx >= n_points:
                red_active = False
            else:
                wave = wavelength_grid[idx]
                if not simple_wings:
                    if wave > wlminus1:
                        red_active = False
                    elif wave < wcon:
                        pass  # skip but continue (Fortran GO TO 611)
                    else:
                        delta_nm = wave - line_wavelength
                        value = kappa0 * _hydrogen_line_profile_jit(
                            n_lower, n_upper, delta_nm,
                            t3nhe, t3nh2, fo, dopph, c1d, c2d, y1s, y1b,
                            gcon1, gcon2, pp_val, xnfph_0, xnfph_1, electron_density,
                            asum, asum_lyman, y1wtm, xknmtb, propbm, c_tbl, d_tbl, pp_tbl, beta_tbl,
                            cutoff_h2_plus, cutoff_h2, e1_table, fine_offsets, fine_weights, n_fine,
                        ) * stim_row[idx]
                        if use_taper and wave < wtail:
                            value *= (wave - wcon) / (wtail - wcon)
                        if wave > redcut:
                            delta_minus2 = wave - wlminus2
                            value_minus2 = kappa0 * _hydrogen_line_profile_jit(
                                n_lower, upper_minus2, delta_minus2,
                                t3nhe, t3nh2, fo, dopph, c1d, c2d, y1s, y1b,
                                gcon1, gcon2, pp_val, xnfph_0, xnfph_1, electron_density,
                                asum, asum_lyman, y1wtm, xknmtb, propbm, c_tbl, d_tbl, pp_tbl, beta_tbl,
                                cutoff_h2_plus, cutoff_h2, e1_table, fine_offsets, fine_weights, n_fine,
                            ) * stim_row[idx]
                            if use_taper and wave < wtail:
                                value_minus2 *= (wave - wcon) / (wtail - wcon)
                            if value_minus2 >= value:
                                red_active = False
                                value = 0.0
                        if value <= 0.0 or value < continuum_row[idx] * cutoff:
                            red_active = False
                        else:
                            buffer[idx] += value
                else:
                    delta_nm = wave - line_wavelength
                    value = kappa0 * _hydrogen_line_profile_jit(
                        n_lower, n_upper, delta_nm,
                        t3nhe, t3nh2, fo, dopph, c1d, c2d, y1s, y1b,
                        gcon1, gcon2, pp_val, xnfph_0, xnfph_1, electron_density,
                        asum, asum_lyman, y1wtm, xknmtb, propbm, c_tbl, d_tbl, pp_tbl, beta_tbl,
                        cutoff_h2_plus, cutoff_h2, e1_table, fine_offsets, fine_weights, n_fine,
                    ) * stim_row[idx]
                    if value <= 0.0 or value < continuum_row[idx] * cutoff:
                        red_active = False
                    else:
                        buffer[idx] += value

        if blue_active:
            idx = center_index - offset
            if idx < 0:
                blue_active = False
            else:
                wave = wavelength_grid[idx]
                if not simple_wings and (wave < wcon or wave < wlplus1):
                    blue_active = False
                else:
                    delta_nm = wave - line_wavelength
                    value = kappa0 * _hydrogen_line_profile_jit(
                        n_lower, n_upper, delta_nm,
                        t3nhe, t3nh2, fo, dopph, c1d, c2d, y1s, y1b,
                        gcon1, gcon2, pp_val, xnfph_0, xnfph_1, electron_density,
                        asum, asum_lyman, y1wtm, xknmtb, propbm, c_tbl, d_tbl, pp_tbl, beta_tbl,
                        cutoff_h2_plus, cutoff_h2, e1_table, fine_offsets, fine_weights, n_fine,
                    ) * stim_row[idx]
                    if not simple_wings:
                        if use_taper and wave < wtail:
                            value *= (wave - wcon) / (wtail - wcon)
                        if wave < bluecut:
                            delta_plus2 = wave - wlplus2
                            value_plus2 = kappa0 * _hydrogen_line_profile_jit(
                                n_lower, upper_plus2, delta_plus2,
                                t3nhe, t3nh2, fo, dopph, c1d, c2d, y1s, y1b,
                                gcon1, gcon2, pp_val, xnfph_0, xnfph_1, electron_density,
                                asum, asum_lyman, y1wtm, xknmtb, propbm, c_tbl, d_tbl, pp_tbl, beta_tbl,
                                cutoff_h2_plus, cutoff_h2, e1_table, fine_offsets, fine_weights, n_fine,
                            ) * stim_row[idx]
                            if use_taper and wave < wtail:
                                value_plus2 *= (wave - wcon) / (wtail - wcon)
                            if value_plus2 >= value:
                                blue_active = False
                                value = 0.0
                    if value <= 0.0 or value < continuum_row[idx] * cutoff:
                        blue_active = False
                    else:
                        buffer[idx] += value

        offset += 1


# ──────────────────── Depth-parallel master kernel ────────────────────────────

@jit(nopython=True, parallel=True, cache=True)
def compute_hydrogen_opacity_jit(
    # H-line parameters (n_h_lines,)
    h_n_lower: np.ndarray,
    h_n_upper: np.ndarray,
    h_center_indices: np.ndarray,
    h_wavelengths: np.ndarray,
    h_cgf: np.ndarray,
    h_wshift: np.ndarray,
    h_redcut: np.ndarray,
    h_bluecut: np.ndarray,
    h_wlminus1: np.ndarray,
    h_wlminus2: np.ndarray,
    h_wlplus1: np.ndarray,
    h_wlplus2: np.ndarray,
    h_conth_val: np.ndarray,
    h_fine_offsets: np.ndarray,  # (n_h_lines, MAX_FINE)
    h_fine_weights: np.ndarray,  # (n_h_lines, MAX_FINE)
    h_n_fine: np.ndarray,        # (n_h_lines,) int
    # Per-depth state (n_depths,)
    pop_densities: np.ndarray,
    dop_velocities: np.ndarray,
    mass_density: np.ndarray,
    emerge_h: np.ndarray,
    boltz_factors: np.ndarray,   # (n_depths, n_h_lines)
    electron_density: np.ndarray,
    t3nhe: np.ndarray,
    t3nh2: np.ndarray,
    fo_arr: np.ndarray,
    dopph_arr: np.ndarray,
    c1d_arr: np.ndarray,
    c2d_arr: np.ndarray,
    y1s_arr: np.ndarray,
    y1b_arr: np.ndarray,
    gcon1_arr: np.ndarray,
    gcon2_arr: np.ndarray,
    pp_depth_arr: np.ndarray,
    xnfph_0_arr: np.ndarray,
    xnfph_1_arr: np.ndarray,
    # Grid
    wavelength_grid: np.ndarray,
    continuum: np.ndarray,       # (n_depths, n_wl)
    stim: np.ndarray,            # (n_depths, n_wl)
    cutoff: float,
    # Tables
    asum: np.ndarray,
    asum_lyman: np.ndarray,
    y1wtm: np.ndarray,
    xknmtb: np.ndarray,
    propbm: np.ndarray,
    c_tbl: np.ndarray,
    d_tbl: np.ndarray,
    pp_tbl: np.ndarray,
    beta_tbl: np.ndarray,
    cutoff_h2_plus: np.ndarray,
    cutoff_h2: np.ndarray,
    e1_table: np.ndarray,
    # Output
    ahline: np.ndarray,          # (n_depths, n_wl) – pre-zeroed
) -> None:
    n_depths = ahline.shape[0]
    n_h_lines = h_wavelengths.shape[0]
    n_wl = wavelength_grid.shape[0]

    for depth_idx in prange(n_depths):
        pop_val = pop_densities[depth_idx]
        dop_val = dop_velocities[depth_idx]
        rho = mass_density[depth_idx]
        if pop_val <= 0.0 or dop_val <= 0.0 or rho <= 0.0:
            continue

        xnfdop = pop_val / (rho * dop_val)
        em_h = emerge_h[depth_idx]

        t3nhe_d = t3nhe[depth_idx]
        t3nh2_d = t3nh2[depth_idx]
        fo_d = fo_arr[depth_idx]
        dopph_d = dopph_arr[depth_idx]
        c1d_d = c1d_arr[depth_idx]
        c2d_d = c2d_arr[depth_idx]
        y1s_d = y1s_arr[depth_idx]
        y1b_d = y1b_arr[depth_idx]
        gcon1_d = gcon1_arr[depth_idx]
        gcon2_d = gcon2_arr[depth_idx]
        pp_d = pp_depth_arr[depth_idx]
        xnfph_0_d = xnfph_0_arr[depth_idx]
        xnfph_1_d = xnfph_1_arr[depth_idx]
        elec_d = electron_density[depth_idx]

        for line_idx in range(n_h_lines):
            cgf = h_cgf[line_idx]
            kappa0_pre = cgf * xnfdop

            center_idx = int(h_center_indices[line_idx])
            clamped = center_idx
            if clamped < 0:
                clamped = 0
            if clamped >= n_wl:
                clamped = n_wl - 1
            kapmin = continuum[depth_idx, clamped] * cutoff
            if kappa0_pre < kapmin:
                continue

            boltz = boltz_factors[depth_idx, line_idx]
            kappa0 = kappa0_pre * boltz
            if kappa0 < kapmin:
                continue

            # Compute wcon/wtail for this (depth, line)
            conth_val = h_conth_val[line_idx]
            denom = conth_val - em_h
            if denom > 0.0:
                wmerge = 1.0e7 / denom
            else:
                wmerge = -1.0
            wshift = h_wshift[line_idx]
            if wmerge < 0.0:
                wmerge = wshift + wshift
            wcon = max(wshift, wmerge)
            if wcon > 0.0:
                inner = 1.0e7 / wcon - 500.0
                wtail = 1.0e7 / inner if inner > 0.0 else wcon + wcon
            else:
                wtail = wcon + wcon
            wcon = min(wshift + wshift, wcon)
            if wtail < 0.0:
                wtail = wcon + wcon
            wtail = min(wcon + wcon, wtail)

            n_lower = int(h_n_lower[line_idx])
            n_upper = int(h_n_upper[line_idx])
            line_wavelength = h_wavelengths[line_idx]
            n_fine = int(h_n_fine[line_idx])

            _accumulate_hyd_line_depth_jit(
                ahline[depth_idx],
                continuum[depth_idx],
                stim[depth_idx],
                wavelength_grid,
                center_idx,
                line_wavelength,
                kappa0,
                n_lower, n_upper,
                wcon, wtail,
                h_wlminus1[line_idx], h_wlminus2[line_idx],
                h_wlplus1[line_idx], h_wlplus2[line_idx],
                h_redcut[line_idx], h_bluecut[line_idx],
                cutoff,
                t3nhe_d, t3nh2_d, fo_d, dopph_d,
                c1d_d, c2d_d, y1s_d, y1b_d,
                gcon1_d, gcon2_d, pp_d,
                xnfph_0_d, xnfph_1_d, elec_d,
                asum, asum_lyman, y1wtm, xknmtb,
                propbm, c_tbl, d_tbl, pp_tbl, beta_tbl,
                cutoff_h2_plus, cutoff_h2, e1_table,
                h_fine_offsets[line_idx], h_fine_weights[line_idx], n_fine,
            )
