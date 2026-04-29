"""Frequency-grid and KAPCONT-like tabulation helpers."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .kapp import KappAtmosphereAdapter, compute_kapp
from .atlas_tables import load_atlas_tables
from .josh_math import _integ
from .trace_runtime import trace_emit, trace_enabled, trace_in_focus

_KAPCONT_WAVETAB_343 = np.array(
    [
        9.09, 9.35, 9.61, 9.77, 9.96, 10.20, 10.38, 10.56, 10.77, 11.04, 11.40, 11.78, 12.13, 12.48, 12.71, 12.84,
        13.05, 13.24, 13.39, 13.66, 13.98, 14.33, 14.72, 15.10, 15.52, 15.88, 16.20, 16.60, 17.03, 17.34, 17.68, 18.02,
        18.17, 18.61, 19.10, 19.39, 19.84, 20.18, 20.50, 21.05, 21.62, 21.98, 22.30, 22.68, 23.00, 23.40, 24.00, 24.65,
        25.24, 25.68, 26.00, 26.40, 26.85, 27.35, 27.85, 28.40, 29.0, 29.6, 30.1, 30.8, 31.8, 32.8, 33.8, 34.8,
        35.7, 36.6, 37.5, 38.5, 39.5, 40.5, 41.4, 42.2, 43.0, 44.1, 45.1, 46.0, 47.0, 48.0, 49.0, 50.0,
        50.6, 51.4, 53.0, 55.0, 56.7, 58.5, 60.5, 62.5, 64.5, 66.3, 68.0, 70.0, 71.6, 73.0, 75.0, 77.0,
        79.0, 81.0, 83.0, 85.0, 87.0, 89.0, 90.6, 92.6, 96.0, 100.0, 104.0, 108.0, 111.5, 114.5, 118.0, 122.0,
        126.0, 130.0, 134.0, 138.0, 142.0, 146.0, 150.0, 154.0, 160.0, 165.0, 169.0, 173.0, 177.5, 182.0, 186.0, 190.5,
        195.0, 200.0, 204.5, 208.5, 212.5, 217.5, 222.5, 227.5, 232.5, 237.5, 242.5, 248.0, 253.0, 257.5, 262.5, 267.5,
        272.5, 277.5, 282.5, 287.5, 295.0, 305.0, 315.0, 325.0, 335.0, 345.0, 355.0, 362.0, 367.0, 375.0, 385.0, 395.0,
        405.0, 415.0, 425.0, 435.0, 455.0, 465.0, 475.0, 485.0, 495.0, 505.0, 515.0, 525.0, 535.0, 545.0, 555.0, 565.0,
        575.0, 585.0, 595.0, 605.0, 615.0, 625.0, 635.0, 645.0, 655.0, 665.0, 675.0, 685.0, 695.0, 705.0, 715.0, 725.0,
        735.0, 745.0, 755.0, 765.0, 775.0, 785.0, 795.0, 805.0, 815.0, 825.0, 835.0, 845.0, 855.0, 865.0, 875.0, 885.0,
        895.0, 905.0, 915.0, 925.0, 935.0, 945.0, 955.0, 965.0, 975.0, 985.0, 995.0, 1012.5, 1037.5, 1062.5, 1087.5, 1112.5,
        1137.5, 1162.5, 1187.5, 1212.5, 1237.5, 1262.5, 1287.5, 1312.5, 1337.5, 1362.5, 1387.5, 1412.5, 1442.0, 1467.0, 1487.5, 1512.5,
        1537.5, 1562.5, 1587.5, 1620.0, 1660.0, 1700.0, 1740.0, 1780.0, 1820.0, 1860.0, 1900.0, 1940.0, 1980.0, 2025.0, 2075.0, 2125.0,
        2175.0, 2225.0, 2265.0, 2290.0, 2325.0, 2375.0, 2425.0, 2475.0, 2525.0, 2575.0, 2625.0, 2675.0, 2725.0, 2775.0, 2825.0, 2875.0,
        2925.0, 2975.0, 3025.0, 3075.0, 3125.0, 3175.0, 3240.0, 3340.0, 3450.0, 3550.0, 3650.0, 3750.0, 3850.0, 3950.0, 4050.0, 4150.0,
        4250.0, 4350.0, 4450.0, 4550.0, 4650.0, 4750.0, 4850.0, 4950.0, 5050.0, 5150.0, 5250.0, 5350.0, 5450.0, 5550.0, 5650.0, 5750.0,
        5850.0, 5950.0, 6050.0, 6150.0, 6250.0, 6350.0, 6500.0, 6700.0, 6900.0, 7100.0, 7300.0, 7500.0, 7700.0, 7900.0, 8100.0, 8300.0,
        8500.0, 8700.0, 8900.0, 9100.0, 9300.0, 9500.0, 9700.0, 9900.0, 10000.0, 20000.0, 40000.0, 60000.0, 80000.0, 100000.0, 120000.0, 140000.0,
        160000.0, 200000.0, 240000.0, 280000.0, 320000.0, 360000.0, 400000.0,
    ],
    dtype=np.float64,
)


@lru_cache(maxsize=16)
def build_waveset(teff: float) -> tuple[np.ndarray, np.ndarray]:
    """Construct WAVESET and RCOSET arrays following atlas12.for lines 188-214."""

    nulo = 1
    nuhi = 30000
    nustep = 1
    nuci = 11601
    nulyman = 9599
    nuhei = 7027
    nuheii = 3577
    nustart = 1
    if teff < 30000.0:
        nustart = nuheii
    if teff < 13000.0:
        nustart = nuhei
    if teff < 7250.0:
        nustart = nulyman
    if teff < 4500.0:
        nustart = nuci

    wave = np.zeros(nuhi, dtype=np.float64)
    rco = np.zeros(nuhi, dtype=np.float64)
    for nu in range(nulo, nuhi + 1, nustep):
        wave[nu - 1] = 10.0 ** (1.0 + 0.0001 * (nu + nustart - 1))

    c_nm = 2.99792458e17
    # Endpoints and interior trapezoid-like quadrature from Fortran.
    rco[0] = (c_nm / wave[0] - c_nm / wave[1]) * 1.5
    for nu in range(nulo + nustep, nuhi - nustep + 1, nustep):
        i = nu - 1
        rco[i] = (c_nm / wave[i - 1] - c_nm / wave[i + 1]) * 0.5
    rco[nuhi - 1] = (c_nm / wave[nuhi - 2] + c_nm / wave[nuhi - 1]) * 0.25
    return wave, rco


@lru_cache(maxsize=1)
def build_kapcont_wavetab() -> tuple[np.ndarray, np.ndarray]:
    """Build `WAVETAB(344)` and `IWAVETAB(344)` from atlas12.for KAPCONT."""

    wavetab = np.empty(344, dtype=np.float64)
    wavetab[:343] = _KAPCONT_WAVETAB_343
    wavetab[343] = wavetab[342]
    i_wavetab = np.empty(344, dtype=np.int64)
    ratio_lg = np.log(1.0 + 1.0 / 2_000_000.0)
    i_wavetab[:343] = np.asarray(np.log(wavetab[:343]) / ratio_lg + 0.5, dtype=np.int64)
    i_wavetab[343] = 2**30
    return wavetab, i_wavetab


def kapcont_table(
    *,
    adapter: KappAtmosphereAdapter,
    temperature_k: np.ndarray,
    teff: float,
    atlas_tables: dict[str, np.ndarray],
    ifop: list[int] | None = None,
    tcst=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute `TABCONT(kw,344)` on the atlas12 KAPCONT wavelength grid.

    Fortran reference: `atlas12.for` lines 14329-14421.
    """

    wave_set, _ = build_waveset(float(teff))
    hkt = 6.6256e-27 / np.maximum(np.asarray(temperature_k, dtype=np.float64) * 1.38054e-16, 1e-300)
    wavetab, i_wavetab = build_kapcont_wavetab()
    n_layers = temperature_k.size
    tabcont = np.zeros((n_layers, 344), dtype=np.float32)

    active = wavetab[:343] > float(wave_set[0])
    active_idx = np.nonzero(active)[0]
    active_acont = np.empty((n_layers, 0), dtype=np.float64)
    active_sigmac = np.empty((n_layers, 0), dtype=np.float64)
    if active_idx.size:
        active_freq = 2.99792458e17 / np.maximum(wavetab[active_idx], 1e-300)
        active_acont, active_sigmac, _ = compute_kapp(
            adapter=adapter,
            freq_hz=np.asarray(active_freq, dtype=np.float64),
            atlas_tables=atlas_tables,
            ifop=ifop,
            tcst=tcst,
        )
        stim = np.maximum(1.0 - np.exp(-np.outer(hkt, active_freq)), 1e-300)
        tabcont[:, active_idx] = np.asarray(
            (active_acont + active_sigmac) * 1.0e-3 / stim,
            dtype=np.float32,
        )

    inactive_idx = np.nonzero(~active)[0]
    if inactive_idx.size:
        inactive_freq = 2.99792458e17 / np.maximum(wavetab[inactive_idx], 1e-300)
        stim = np.maximum(1.0 - np.exp(-np.outer(hkt, inactive_freq)), 1e-300)
        tabcont[:, inactive_idx] = np.asarray(1.0e10 * 1.0e-3 / stim, dtype=np.float32)

    active_pos = {int(idx): pos for pos, idx in enumerate(active_idx)}
    for nu in range(343):
        wave = float(wavetab[nu])
        if active[nu]:
            pos = active_pos[nu]
            acont_col = np.asarray(active_acont[:, pos], dtype=np.float64)
            sigmac_col = np.asarray(active_sigmac[:, pos], dtype=np.float64)
        else:
            acont_col = np.full(n_layers, 1.0e10, dtype=np.float64)
            sigmac_col = np.zeros(n_layers, dtype=np.float64)
        if trace_enabled():
            for j0 in range(n_layers):
                if not trace_in_focus(wlvac_nm=wave, j0=j0):
                    continue
                trace_emit(
                    event="kapcont_tab",
                    iter_num=1,
                    line_num_1b=0,
                    depth_1b=j0 + 1,
                    nu_1b=nu + 1,
                    type_code=0,
                    wlvac_nm=wave,
                    center=float(acont_col[j0]),
                    adamp=float(sigmac_col[j0]),
                    cv=float(stim[j0]),
                    tabcont=float(tabcont[j0, nu]),
                    branch="kapcont",
                    reason="table_fill",
                )

    tabcont[:, 343] = tabcont[:, 342]
    return tabcont, wavetab, i_wavetab


def kapcont_baseline(
    adapter: KappAtmosphereAdapter,
    teff: float,
    atlas_tables: dict[str, np.ndarray],
    ifop: list[int] | None = None,
    tcst=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute baseline continuum opacity on the ATLAS12 wave grid."""

    wave_nm, rco = build_waveset(teff)
    freq_hz = 2.99792458e17 / np.maximum(wave_nm, 1e-300)
    acont, sigmac, scont = compute_kapp(
        adapter=adapter, freq_hz=freq_hz, atlas_tables=atlas_tables, ifop=ifop, tcst=tcst
    )
    return wave_nm, rco, acont, sigmac, scont


def rosseland_continuum_baseline(
    *,
    adapter: KappAtmosphereAdapter,
    temperature_k: np.ndarray,
    rhox: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute baseline ABROSS/TAUROS using continuum-only ABTOT.

    Fortran references:
    - Frequency loop setup: `atlas12.for` lines 326-338
    - ROSS accumulation/finalization: `atlas12.for` lines 1206-1214

    Notes:
    - This is an experimental Phase-2 baseline (continuum only).
    - Full parity requires line opacity and full JOSH coupling.
    """

    c_nm = 2.99792458e17
    h_planck = 6.6256e-27
    k_boltz = 1.38054e-16

    tables = load_atlas_tables()
    wave_nm, rco, acont, sigmac, _ = kapcont_baseline(
        adapter=adapter,
        teff=float(np.max(temperature_k)),
        atlas_tables=tables,
    )
    freq_hz = c_nm / np.maximum(wave_nm, 1e-300)
    # Fortran TEMP common: HKT = H / (K_B * T)
    hkt = h_planck / np.maximum(temperature_k * k_boltz, 1e-300)

    acc = np.zeros_like(temperature_k, dtype=np.float64)
    for i in range(freq_hz.size):
        freq = float(freq_hz[i])
        rcowt = float(rco[i])
        ehvkt = np.exp(-freq * hkt)
        stim = np.maximum(1.0 - ehvkt, 1e-300)
        freq15 = freq / 1.0e15
        bnu = 1.47439e-2 * (freq15**3) * ehvkt / stim
        dbdt = bnu * freq * hkt / np.maximum(temperature_k * stim, 1e-300)
        abtot = np.maximum(acont[:, i] + sigmac[:, i], 1e-300)
        acc += dbdt / abtot * rcowt

    sigma_over_pi = 4.0 * 5.6697e-5 / 3.14159
    abross = sigma_over_pi * np.power(temperature_k, 3.0) / np.maximum(acc, 1e-300)

    tauros = _integ(rhox, abross, abross[0] * rhox[0])
    return abross, tauros

