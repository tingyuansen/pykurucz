"""Helium I/II wing profile utilities ported from the legacy SYNTHE code."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .helium_tables import HeTables, HeLineTables, load_tables as load_bcs_tables
from .profiles.voigt import voigt_profile
from . import tables

from numba import jit

SQRT_PI = np.sqrt(np.pi)
HE_DATA_PATH = Path(__file__).resolve().parents[1] / "data"
# Fortran synthe.for GRIEM/DIMITRI lookup uses +/- 0.1 nm window:
# IF(WL.GT.WAVE(ILINE)-.1.AND.WL.LT.WAVE(ILINE)+.1)
HE_TABLE_MATCH_TOL_NM = 0.1

# Shared Voigt profile — single canonical JIT-compiled implementation
from synthe_py.physics.voigt_jit import voigt_profile_jit as _voigt_profile_jit


@jit(nopython=True, cache=True)
def _find_record_jit(wavelengths: np.ndarray, target: float, threshold: float) -> int:
    best_idx = -1
    best_delta = threshold
    for idx in range(wavelengths.size):
        delta = abs(wavelengths[idx] - target)
        if delta <= best_delta:
            best_delta = delta
            best_idx = idx
    return best_idx


@jit(nopython=True, cache=True)
def _profile_bcs_jit(
    delta_nm: float, dlam: np.ndarray, log_prof: np.ndarray, dlam_len: int
) -> float:
    delta_angstrom = delta_nm * 10.0
    if dlam_len <= 0:
        return 0.0
    if delta_angstrom <= dlam[0]:
        return 10.0 ** log_prof[0]
    if delta_angstrom >= dlam[dlam_len - 1]:
        return 10.0 ** log_prof[dlam_len - 1]
    hi = 1
    for idx in range(1, dlam_len):
        if dlam[idx] >= delta_angstrom:
            hi = idx
            break
    lo = hi - 1
    span = dlam[hi] - dlam[lo]
    if span <= 0.0:
        return 10.0 ** log_prof[lo]
    a = (dlam[hi] - delta_angstrom) / span
    b = (delta_angstrom - dlam[lo]) / span
    return 10.0 ** (a * log_prof[lo] + b * log_prof[hi])


@jit(nopython=True, cache=True)
def _evaluate_helium_profile_jit(
    line_type: int,
    depth_idx: int,
    delta_nm: float,
    line_wavelength: float,
    doppler_width: float,
    gamma_rad: float,
    gamma_stark: float,
    temperature: np.ndarray,
    electron_density: np.ndarray,
    xnfph: np.ndarray,
    xnf_he2: np.ndarray,
    has_xnfph: bool,
    has_xnfhe2: bool,
    bcs_dlam: np.ndarray,
    bcs_log_profile: np.ndarray,
    bcs_len: np.ndarray,
    griem_wavelength: np.ndarray,
    griem_log_ne: np.ndarray,
    griem_ttab: np.ndarray,
    griem_width: np.ndarray,
    griem_shift: np.ndarray,
    griem_alpha: np.ndarray,
    dim_wavelength: np.ndarray,
    dim_log_ne: np.ndarray,
    dim_ttab: np.ndarray,
    dim_width_e: np.ndarray,
    dim_shift_e: np.ndarray,
    dim_width_h: np.ndarray,
    dim_shift_h: np.ndarray,
    dim_width_he: np.ndarray,
    dim_shift_he: np.ndarray,
    h0tab: np.ndarray,
    h1tab: np.ndarray,
    h2tab: np.ndarray,
) -> float:
    doppler = doppler_width if doppler_width > 1e-40 else 1e-40
    wave = line_wavelength + delta_nm
    temp = temperature[depth_idx]
    if temp < 5000.0:
        temp = 5000.0
    if temp > 80000.0:
        temp = 80000.0
    e_density = electron_density[depth_idx]
    if e_density <= 0.0 or doppler <= 0.0:
        return 0.0
    xnfhp = 0.0
    if has_xnfph and xnfph.shape[1] > 1:
        xnfhp = xnfph[depth_idx, 1]
    xnfhep = 0.0
    if has_xnfhe2 and xnf_he2.size > 0:
        xnfhep = xnf_he2[depth_idx]

    if line_type in (-3, -4):
        if abs(line_wavelength - 447.15) < 0.4:
            if e_density > 1.0e13:
                return (
                    SQRT_PI
                    * doppler
                    * 10.0
                    * _profile_bcs_jit(
                        wave - line_wavelength,
                        bcs_dlam[0],
                        bcs_log_profile[depth_idx, 0],
                        int(bcs_len[0]),
                    )
                )
            ts = np.array([5.0e3, 1.0e4, 2.0e4, 4.0e4])
            ws = np.array([0.001460, 0.001269, 0.001079, 0.000898])
            ds = np.array([0.036, -0.005, -0.026, -0.034])
            alfs = np.array([0.107, 0.119, 0.134, 0.154])
            den = 1.0e13
            it = 1
            for i in range(1, 4):
                it = i
                if ts[i] >= temp:
                    break
            x = (temp - ts[it - 1]) / (ts[it] - ts[it - 1])
            xx = e_density / den
            width = xx * (x * ws[it] + (1.0 - x) * ws[it - 1])
            shift = x * ds[it] + (1.0 - x) * ds[it - 1]
            alf = xx**0.25 * (x * alfs[it] + (1.0 - x) * alfs[it - 1])
            xx_ratio = xnfhp / e_density if e_density > 0.0 else 0.0
            vm1 = 8.78 * (xx_ratio + 2.0 * (1.0 - xx_ratio)) / np.sqrt(temp)
            rhom = 1.0 / (4.19 * e_density) ** (1.0 / 3.0)
            sigma = 1.885e14 * width * rhom * vm1 / (line_wavelength * 10.0) ** 2
            sigma = max(sigma, 1e-40)
            ion_term = alf ** (8.0 / 9.0) / sigma ** (1.0 / 3.0)
            wtot = width * (1.0 + 1.36 * ion_term) * 0.1
            dtot = width * shift * (1.0 + 2.36 * ion_term / max(abs(shift), 1e-12)) * 0.1
            a = wtot / doppler
            wwd = wave - line_wavelength - dtot
            return (
                _voigt_profile_jit(abs(wwd - 0.0184) / doppler, a, h0tab, h1tab, h2tab)
                / 9.0
                + _voigt_profile_jit(
                    abs(wwd + 0.0013) / doppler, a, h0tab, h1tab, h2tab
                )
                / 12.0
                + _voigt_profile_jit(
                    abs(wwd + 0.0010) / doppler, a, h0tab, h1tab, h2tab
                )
                / 4.0
                + _voigt_profile_jit(
                    abs(wwd + 0.0029) / doppler, a, h0tab, h1tab, h2tab
                )
                / 180.0
                + _voigt_profile_jit(
                    abs(wwd + 0.0025) / doppler, a, h0tab, h1tab, h2tab
                )
                * 11.0
                / 20.0
            )
        if abs(line_wavelength - 402.62) < 0.4:
            if e_density > 1.0e14:
                return (
                    SQRT_PI
                    * doppler
                    * 10.0
                    * _profile_bcs_jit(
                        wave - line_wavelength,
                        bcs_dlam[1],
                        bcs_log_profile[depth_idx, 1],
                        int(bcs_len[1]),
                    )
                )
            ts = np.array([5.0e3, 1.0e4, 2.0e4, 4.0e4])
            ws = np.array([4.04, 3.49, 2.96, 2.47])
            ds = np.array([0.1339, 0.0960, 0.0780, 0.0709])
            alfs = np.array([0.969, 1.083, 1.225, 1.403])
            den = 1.0e16
            it = 1
            for i in range(1, 4):
                it = i
                if ts[i] >= temp:
                    break
            x = (temp - ts[it - 1]) / (ts[it] - ts[it - 1])
            xx = e_density / den
            width = xx * (x * ws[it] + (1.0 - x) * ws[it - 1])
            shift = x * ds[it] + (1.0 - x) * ds[it - 1]
            alf = xx**0.25 * (x * alfs[it] + (1.0 - x) * alfs[it - 1])
            xx_ratio = xnfhp / e_density if e_density > 0.0 else 0.0
            vm1 = 8.78 * (xx_ratio + 2.0 * (1.0 - xx_ratio)) / np.sqrt(temp)
            rhom = 1.0 / (4.19 * e_density) ** (1.0 / 3.0)
            sigma = 1.885e14 * width * rhom * vm1 / (line_wavelength * 10.0) ** 2
            sigma = max(sigma, 1e-40)
            ion_term = alf ** (8.0 / 9.0) / sigma ** (1.0 / 3.0)
            wtot = width * (1.0 + 1.36 * ion_term) * 0.1
            dtot = width * shift * (1.0 + 2.36 * ion_term / max(abs(shift), 1e-12)) * 0.1
            a = wtot / doppler
            wwd = wave - line_wavelength - dtot
            return (
                _voigt_profile_jit(abs(wwd - 0.0148) / doppler, a, h0tab, h1tab, h2tab)
                / 9.0
                + _voigt_profile_jit(
                    abs(wwd + 0.0012) / doppler, a, h0tab, h1tab, h2tab
                )
                / 12.0
                + _voigt_profile_jit(
                    abs(wwd + 0.0011) / doppler, a, h0tab, h1tab, h2tab
                )
                / 4.0
                + _voigt_profile_jit(
                    abs(wwd + 0.0025) / doppler, a, h0tab, h1tab, h2tab
                )
                / 180.0
                + _voigt_profile_jit(
                    abs(wwd + 0.0023) / doppler, a, h0tab, h1tab, h2tab
                )
                * 11.0
                / 20.0
            )
        if abs(line_wavelength - 438.79) < 0.4:
            if e_density > 1.0e14:
                return (
                    SQRT_PI
                    * doppler
                    * 10.0
                    * _profile_bcs_jit(
                        wave - line_wavelength,
                        bcs_dlam[2],
                        bcs_log_profile[depth_idx, 2],
                        int(bcs_len[2]),
                    )
                )
            ts = np.array([5.0e3, 1.0e4, 2.0e4, 4.0e4])
            ws = np.array([6.13, 5.15, 4.24, 3.45])
            ds = np.array([0.411, 0.363, 0.325, 0.293])
            alfs = np.array([1.159, 1.321, 1.527, 1.783])
            den = 1.0e16
            it = 1
            for i in range(1, 4):
                it = i
                if ts[i] >= temp:
                    break
            x = (temp - ts[it - 1]) / (ts[it] - ts[it - 1])
            xx = e_density / den
            width = xx * (x * ws[it] + (1.0 - x) * ws[it - 1])
            shift = x * ds[it] + (1.0 - x) * ds[it - 1]
            alf = xx**0.25 * (x * alfs[it] + (1.0 - x) * alfs[it - 1])
            xx_ratio = xnfhp / e_density if e_density > 0.0 else 0.0
            vm1 = 8.78 * (xx_ratio + 2.0 * (1.0 - xx_ratio)) / np.sqrt(temp)
            rhom = 1.0 / (4.19 * e_density) ** (1.0 / 3.0)
            sigma = 1.885e14 * width * rhom * vm1 / (line_wavelength * 10.0) ** 2
            sigma = max(sigma, 1e-40)
            ion_term = alf ** (8.0 / 9.0) / sigma ** (1.0 / 3.0)
            wtot = width * (1.0 + 1.36 * ion_term) * 0.1
            dtot = width * shift * (1.0 + 2.36 * ion_term / max(abs(shift), 1e-12)) * 0.1
            a = wtot / doppler
            return _voigt_profile_jit(
                abs(wave - line_wavelength - dtot) / doppler, a, h0tab, h1tab, h2tab
            )
        if abs(line_wavelength - 492.19) < 0.4:
            if e_density > 1.0e13:
                return (
                    SQRT_PI
                    * doppler
                    * 10.0
                    * _profile_bcs_jit(
                        wave - line_wavelength,
                        bcs_dlam[3],
                        bcs_log_profile[depth_idx, 3],
                        int(bcs_len[3]),
                    )
                )
            ts = np.array([5.0e3, 1.0e4, 2.0e4, 4.0e4])
            ws = np.array([0.002312, 0.001963, 0.001624, 0.001315])
            ds = np.array([0.3932, 0.3394, 0.2950, 0.2593])
            alfs = np.array([0.1207, 0.1365, 0.1564, 0.1844])
            den = 1.0e13
            it = 1
            for i in range(1, 4):
                it = i
                if ts[i] >= temp:
                    break
            x = (temp - ts[it - 1]) / (ts[it] - ts[it - 1])
            xx = e_density / den
            width = xx * (x * ws[it] + (1.0 - x) * ws[it - 1])
            shift = x * ds[it] + (1.0 - x) * ds[it - 1]
            alf = xx**0.25 * (x * alfs[it] + (1.0 - x) * alfs[it - 1])
            xx_ratio = xnfhp / e_density if e_density > 0.0 else 0.0
            vm1 = 8.78 * (xx_ratio + 2.0 * (1.0 - xx_ratio)) / np.sqrt(temp)
            rhom = 1.0 / (4.19 * e_density) ** (1.0 / 3.0)
            sigma = 1.885e14 * width * rhom * vm1 / (line_wavelength * 10.0) ** 2
            sigma = max(sigma, 1e-40)
            ion_term = alf ** (8.0 / 9.0) / sigma ** (1.0 / 3.0)
            wtot = width * (1.0 + 1.36 * ion_term) * 0.1
            dtot = width * shift * (1.0 + 2.36 * ion_term / max(abs(shift), 1e-12)) * 0.1
            a = wtot / doppler
            wwd = wave - line_wavelength - dtot
            return _voigt_profile_jit(abs(wwd) / doppler, a, h0tab, h1tab, h2tab)

    if line_type in (-3, -4, -6):
        idx = _find_record_jit(dim_wavelength, line_wavelength, HE_TABLE_MATCH_TOL_NM)
        if idx >= 0:
            record_ttab = dim_ttab[idx]
            it = 1
            for i in range(1, 4):
                it = i
                if record_ttab[i] >= temp:
                    break
            x = (temp - record_ttab[it - 1]) / (record_ttab[it] - record_ttab[it - 1])
            scale = 10.0 ** dim_log_ne[idx]
            xx = e_density / scale
            xxh = xnfhp / scale
            xxhe = xnfhep / scale
            width = xx * (
                x * dim_width_e[idx, it] + (1.0 - x) * dim_width_e[idx, it - 1]
            )
            width_h = xxh * (
                x * dim_width_h[idx, it] + (1.0 - x) * dim_width_h[idx, it - 1]
            )
            width_he = xxhe * (
                x * dim_width_he[idx, it] + (1.0 - x) * dim_width_he[idx, it - 1]
            )
            shift = x * dim_shift_e[idx, it] + (1.0 - x) * dim_shift_e[idx, it - 1]
            shift_h = x * dim_shift_h[idx, it] + (1.0 - x) * dim_shift_h[idx, it - 1]
            shift_he = x * dim_shift_he[idx, it] + (1.0 - x) * dim_shift_he[idx, it - 1]
            wtot_angstrom = width + width_h + width_he
            dtot_angstrom = width * shift + width_h * shift_h + width_he * shift_he
            wtot_nm = wtot_angstrom * 0.1 * 0.5
            dtot_nm = dtot_angstrom * 0.1
            a = wtot_nm / doppler + gamma_rad / (doppler / line_wavelength)
            return _voigt_profile_jit(
                abs(wave - line_wavelength - dtot_nm) / doppler, a, h0tab, h1tab, h2tab
            )

        idx = _find_record_jit(griem_wavelength, line_wavelength, HE_TABLE_MATCH_TOL_NM)
        if idx < 0:
            a = (gamma_rad + gamma_stark * e_density) / (doppler / line_wavelength)
            return _voigt_profile_jit(
                abs(wave - line_wavelength) / doppler, a, h0tab, h1tab, h2tab
            )
        record_ttab = griem_ttab[idx]
        it = 1
        for i in range(1, 4):
            it = i
            if record_ttab[i] >= temp:
                break
        x = (temp - record_ttab[it - 1]) / (record_ttab[it] - record_ttab[it - 1])
        xx = e_density / (10.0 ** griem_log_ne[idx])
        width = xx * (x * griem_width[idx, it] + (1.0 - x) * griem_width[idx, it - 1])
        shift = x * griem_shift[idx, it] + (1.0 - x) * griem_shift[idx, it - 1]
        alpha_val = xx**0.25 * (
            x * griem_alpha[idx, it] + (1.0 - x) * griem_alpha[idx, it - 1]
        )
        xx_ratio = xnfhp / e_density if e_density > 0.0 else 0.0
        vm1 = 8.78 * (xx_ratio + 2.0 * (1.0 - xx_ratio)) / np.sqrt(temp)
        rhom = 1.0 / (4.19 * e_density) ** (1.0 / 3.0)
        sigma = 1.885e14 * width * rhom * vm1 / (line_wavelength * 10.0) ** 2
        if sigma < 1e-40:
            sigma = 1e-40
        ion_term = alpha_val ** (8.0 / 9.0) / sigma ** (1.0 / 3.0)
        wtot = width * (1.0 + 1.36 * ion_term) * 0.1
        dtot = width * shift * (1.0 + 2.36 * ion_term / max(abs(shift), 1e-12)) * 0.1
        a = wtot / doppler + gamma_rad / (doppler / line_wavelength)
        return _voigt_profile_jit(
            abs(wave - line_wavelength - dtot) / doppler, a, h0tab, h1tab, h2tab
        )

    return 0.0


@dataclass(frozen=True)
class GriemRecord:
    wavelength: float
    log_ne: float
    ttab: np.ndarray  # (4,)
    width: np.ndarray
    shift: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray


@dataclass(frozen=True)
class DimitriRecord:
    wavelength: float
    log_ne: float
    ttab: np.ndarray
    width_e: np.ndarray
    shift_e: np.ndarray
    width_h: np.ndarray
    shift_h: np.ndarray
    width_he: np.ndarray
    shift_he: np.ndarray


def _load_aux_tables() -> Dict[str, np.ndarray]:
    path = HE_DATA_PATH / "helium_aux.npz"
    return dict(np.load(path, allow_pickle=False))


def _build_griem_records(aux: Dict[str, np.ndarray]) -> list[GriemRecord]:
    records: list[GriemRecord] = []
    for idx in range(aux["griem_wavelength"].shape[0]):
        records.append(
            GriemRecord(
                wavelength=float(aux["griem_wavelength"][idx]),
                log_ne=float(aux["griem_log_ne"][idx]),
                ttab=np.asarray(aux["griem_ttab"][idx], dtype=np.float64),
                width=np.asarray(aux["griem_width"][idx], dtype=np.float64),
                shift=np.asarray(aux["griem_shift"][idx], dtype=np.float64),
                alpha=np.asarray(aux["griem_alpha"][idx], dtype=np.float64),
                beta=np.asarray(aux["griem_beta"][idx], dtype=np.float64),
            )
        )
    return records


def _build_dimitri_records(aux: Dict[str, np.ndarray]) -> list[DimitriRecord]:
    records: list[DimitriRecord] = []
    for idx in range(aux["dimitri_wavelength"].shape[0]):
        records.append(
            DimitriRecord(
                wavelength=float(aux["dimitri_wavelength"][idx]),
                log_ne=float(aux["dimitri_log_ne"][idx]),
                ttab=np.asarray(aux["dimitri_ttab"][idx], dtype=np.float64),
                width_e=np.asarray(aux["dimitri_width"][idx], dtype=np.float64),
                shift_e=np.asarray(aux["dimitri_shift"][idx], dtype=np.float64),
                width_h=np.asarray(aux["dimitri_width_p"][idx], dtype=np.float64),
                shift_h=np.asarray(aux["dimitri_shift_p"][idx], dtype=np.float64),
                width_he=np.asarray(aux["dimitri_width_he"][idx], dtype=np.float64),
                shift_he=np.asarray(aux["dimitri_shift_he"][idx], dtype=np.float64),
            )
        )
    return records


def _find_record(records: list, wavelength: float, threshold: float) -> Optional[int]:
    best_idx = None
    best_delta = threshold
    for idx, record in enumerate(records):
        delta = abs(record.wavelength - wavelength)
        if delta <= best_delta:
            best_delta = delta
            best_idx = idx
    return best_idx


def _parabolic_coefficients(
    x: np.ndarray, f: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = x.size
    a = np.zeros(n, dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)
    c = np.zeros(n, dtype=np.float64)
    if n < 2:
        return a, b, c
    c[0] = 0.0
    b[0] = (f[1] - f[0]) / (x[1] - x[0])
    a[0] = f[0] - x[0] * b[0]
    n1 = n - 1
    c[-1] = 0.0
    b[-1] = (f[-1] - f[-2]) / (x[-1] - x[-2])
    a[-1] = f[-1] - x[-1] * b[-1]
    if n == 2:
        return a, b, c
    for j in range(1, n1):
        j1 = j - 1
        denom = x[j] - x[j1]
        if denom == 0.0:
            continue
        d = (f[j] - f[j1]) / denom
        c[j] = (
            f[j + 1] / ((x[j + 1] - x[j]) * (x[j + 1] - x[j1]))
            - f[j] / ((x[j] - x[j1]) * (x[j + 1] - x[j]))
            + f[j1] / ((x[j] - x[j1]) * (x[j + 1] - x[j1]))
        )
        b[j] = d - (x[j] + x[j1]) * c[j]
        a[j] = f[j1] - x[j1] * d + x[j] * x[j1] * c[j]
    # smoothing of coefficients as in legacy code
    if n > 2:
        c[1] = 0.0
        b[1] = (f[2] - f[1]) / (x[2] - x[1])
        a[1] = f[1] - x[1] * b[1]
    if n > 3:
        c[2] = 0.0
        b[2] = (f[3] - f[2]) / (x[3] - x[2])
        a[2] = f[2] - x[2] * b[2]
    for j in range(1, n1):
        if c[j] == 0.0:
            continue
        j1 = j + 1
        if j1 >= n:
            continue
        denom = abs(c[j1]) + abs(c[j])
        if denom == 0.0:
            continue
        wt = abs(c[j1]) / denom
        a[j] = a[j1] + wt * (a[j] - a[j1])
        b[j] = b[j1] + wt * (b[j] - b[j1])
        c[j] = c[j1] + wt * (c[j] - c[j1])
    if n > 2:
        a[n1 - 1] = a[-1]
        b[n1 - 1] = b[-1]
        c[n1 - 1] = c[-1]
    return a, b, c


def _parabolic_integral(x: np.ndarray, f: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    a, b, c = _parabolic_coefficients(x, f)
    total = 0.0
    for i in range(x.size - 1):
        x0 = x[i]
        x1 = x[i + 1]
        dx = x1 - x0
        total += (
            a[i] + 0.5 * b[i] * (x1 + x0) + c[i] / 3.0 * ((x1 + x0) * x1 + x0 * x0)
        ) * dx
    return total


class HeliumWingSolver:
    """Evaluate helium wing profiles for fort.19 special line types."""

    def __init__(
        self,
        temperature: np.ndarray,
        electron_density: np.ndarray,
        xnfph: Optional[np.ndarray],
        xnf_he2: Optional[np.ndarray],
    ) -> None:
        self.temperature = np.asarray(temperature, dtype=np.float64)
        self.electron_density = np.maximum(
            np.asarray(electron_density, dtype=np.float64), 1e-40
        )
        self.xnfph = np.asarray(xnfph, dtype=np.float64) if xnfph is not None else None
        self.xnf_he2 = (
            np.asarray(xnf_he2, dtype=np.float64) if xnf_he2 is not None else None
        )
        self.tables: HeTables = load_bcs_tables()
        aux = _load_aux_tables()
        self.griem_records = _build_griem_records(aux)
        self.dimitri_records = _build_dimitri_records(aux)
        self._bcs_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        self._numba_prepared = False
        self._bcs_dlam = None
        self._bcs_log_profile = None
        self._bcs_len = None
        self._griem_wavelength = None
        self._griem_log_ne = None
        self._griem_ttab = None
        self._griem_width = None
        self._griem_shift = None
        self._griem_alpha = None
        self._dimitri_wavelength = None
        self._dimitri_log_ne = None
        self._dimitri_ttab = None
        self._dimitri_width_e = None
        self._dimitri_shift_e = None
        self._dimitri_width_h = None
        self._dimitri_shift_h = None
        self._dimitri_width_he = None
        self._dimitri_shift_he = None
        self._voigt_h0tab = None
        self._voigt_h1tab = None
        self._voigt_h2tab = None

    @staticmethod
    def _clamp_temp(temp: float, tmin: float, tmax: float) -> float:
        return min(max(temp, tmin), tmax)

    def _xnfhp(self, depth_idx: int) -> float:
        if self.xnfph is None or self.xnfph.shape[1] < 2:
            return 0.0
        return float(self.xnfph[depth_idx, 1])

    def _xnfhep(self, depth_idx: int) -> float:
        if self.xnf_he2 is None:
            return 0.0
        return float(self.xnf_he2[depth_idx])

    def _prepare_numba_cache(self) -> None:
        if self._numba_prepared:
            return
        n_depths = self.temperature.shape[0]
        max_len = max(
            self.tables.line_4471.dlam.size,
            self.tables.line_4026.dlam.size,
            self.tables.line_4387.dlam.size,
            self.tables.line_4921.dlam.size,
        )
        bcs_dlam = np.zeros((4, max_len), dtype=np.float64)
        bcs_log_profile = np.zeros((n_depths, 4, max_len), dtype=np.float64)
        bcs_len = np.zeros(4, dtype=np.int64)

        for line_id in range(1, 5):
            dlam, _ = self._ensure_bcs(0, line_id)
            n = dlam.size
            bcs_len[line_id - 1] = n
            bcs_dlam[line_id - 1, :n] = dlam
            for depth_idx in range(n_depths):
                _, log_profile = self._ensure_bcs(depth_idx, line_id)
                bcs_log_profile[depth_idx, line_id - 1, :n] = log_profile

        self._bcs_dlam = bcs_dlam
        self._bcs_log_profile = bcs_log_profile
        self._bcs_len = bcs_len

        self._griem_wavelength = np.array(
            [record.wavelength for record in self.griem_records], dtype=np.float64
        )
        self._griem_log_ne = np.array(
            [record.log_ne for record in self.griem_records], dtype=np.float64
        )
        self._griem_ttab = np.array(
            [record.ttab for record in self.griem_records], dtype=np.float64
        )
        self._griem_width = np.array(
            [record.width for record in self.griem_records], dtype=np.float64
        )
        self._griem_shift = np.array(
            [record.shift for record in self.griem_records], dtype=np.float64
        )
        self._griem_alpha = np.array(
            [record.alpha for record in self.griem_records], dtype=np.float64
        )

        self._dimitri_wavelength = np.array(
            [record.wavelength for record in self.dimitri_records], dtype=np.float64
        )
        self._dimitri_log_ne = np.array(
            [record.log_ne for record in self.dimitri_records], dtype=np.float64
        )
        self._dimitri_ttab = np.array(
            [record.ttab for record in self.dimitri_records], dtype=np.float64
        )
        self._dimitri_width_e = np.array(
            [record.width_e for record in self.dimitri_records], dtype=np.float64
        )
        self._dimitri_shift_e = np.array(
            [record.shift_e for record in self.dimitri_records], dtype=np.float64
        )
        self._dimitri_width_h = np.array(
            [record.width_h for record in self.dimitri_records], dtype=np.float64
        )
        self._dimitri_shift_h = np.array(
            [record.shift_h for record in self.dimitri_records], dtype=np.float64
        )
        self._dimitri_width_he = np.array(
            [record.width_he for record in self.dimitri_records], dtype=np.float64
        )
        self._dimitri_shift_he = np.array(
            [record.shift_he for record in self.dimitri_records], dtype=np.float64
        )

        voigt_tables = tables.voigt_tables()
        self._voigt_h0tab = voigt_tables.h0tab
        self._voigt_h1tab = voigt_tables.h1tab
        self._voigt_h2tab = voigt_tables.h2tab
        self._numba_prepared = True

    def evaluate_numba(
        self,
        line_type: int,
        depth_idx: int,
        delta_nm: float,
        line_wavelength: float,
        doppler_width: float,
        gamma_rad: float,
        gamma_stark: float,
    ) -> float:
        self._prepare_numba_cache()
        result = _evaluate_helium_profile_jit(
            line_type=line_type,
            depth_idx=depth_idx,
            delta_nm=delta_nm,
            line_wavelength=line_wavelength,
            doppler_width=doppler_width,
            gamma_rad=gamma_rad,
            gamma_stark=gamma_stark,
            temperature=self.temperature,
            electron_density=self.electron_density,
            xnfph=(
                self.xnfph
                if self.xnfph is not None
                else np.zeros((0, 0), dtype=np.float64)
            ),
            xnf_he2=(
                self.xnf_he2
                if self.xnf_he2 is not None
                else np.zeros((0,), dtype=np.float64)
            ),
            has_xnfph=self.xnfph is not None,
            has_xnfhe2=self.xnf_he2 is not None,
            bcs_dlam=self._bcs_dlam,
            bcs_log_profile=self._bcs_log_profile,
            bcs_len=self._bcs_len,
            griem_wavelength=self._griem_wavelength,
            griem_log_ne=self._griem_log_ne,
            griem_ttab=self._griem_ttab,
            griem_width=self._griem_width,
            griem_shift=self._griem_shift,
            griem_alpha=self._griem_alpha,
            dim_wavelength=self._dimitri_wavelength,
            dim_log_ne=self._dimitri_log_ne,
            dim_ttab=self._dimitri_ttab,
            dim_width_e=self._dimitri_width_e,
            dim_shift_e=self._dimitri_shift_e,
            dim_width_h=self._dimitri_width_h,
            dim_shift_h=self._dimitri_shift_h,
            dim_width_he=self._dimitri_width_he,
            dim_shift_he=self._dimitri_shift_he,
            h0tab=self._voigt_h0tab,
            h1tab=self._voigt_h1tab,
            h2tab=self._voigt_h2tab,
        )
        return result

    def _ensure_bcs(
        self, depth_idx: int, line_id: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        key = (depth_idx, line_id)
        cached = self._bcs_cache.get(key)
        if cached is not None:
            return cached

        tables = self.tables
        line_table: HeLineTables
        if line_id == 1:
            line_table = tables.line_4471
        elif line_id == 2:
            line_table = tables.line_4026
        elif line_id == 3:
            line_table = tables.line_4387
        else:
            line_table = tables.line_4921

        temp = self._clamp_temp(self.temperature[depth_idx], 5000.0, 40000.0)
        log_temp = np.log10(temp)
        t_grid = np.log10(tables.temperatures)
        bt = (log_temp - t_grid[0]) / (t_grid[1] - t_grid[0])
        bt = np.clip(bt, 0.0, (tables.temperatures.size - 1) - 1e-8)
        it = int(np.floor(bt))
        wt = bt - it
        if it >= tables.temperatures.size - 1:
            it = tables.temperatures.size - 2
            wt = 1.0

        xne = self.electron_density[depth_idx]
        log_xne = np.log10(max(xne, 1e-40))
        log_grid = line_table.log_ne
        base_log = log_grid[0]
        step = 0.5
        bp = (max(log_xne, base_log) - base_log) / step
        bp = np.clip(bp, 0.0, (log_grid.size - 1) - 1e-8)
        ip = int(np.floor(bp))
        wp = bp - ip
        if ip >= log_grid.size - 1:
            ip = log_grid.size - 2
            wp = 1.0

        c1w1w = (1.0 - wp) * (1.0 - wt)
        c1ww = (1.0 - wp) * wt
        cw1w = wp * (1.0 - wt)
        cww = wp * wt

        def _bilinear(values: np.ndarray) -> np.ndarray:
            slice00 = values[it, ip]
            slice10 = values[it + 1, ip]
            slice01 = values[it, ip + 1]
            slice11 = values[it + 1, ip + 1]
            return c1w1w * slice00 + c1ww * slice10 + cw1w * slice01 + cww * slice11

        if line_id in (1, 4):
            xxh = self._xnfhp(depth_idx) / xne if xne > 0.0 else 0.0
            xxhe = self._xnfhep(depth_idx) / xne if xne > 0.0 else 0.0
            phi_h = 10.0 ** _bilinear(line_table.phi_h_plus)
            phi_he = 10.0 ** _bilinear(line_table.phi_he_plus)
            profile = xxh * phi_h + xxhe * phi_he
        else:
            profile = 10.0 ** _bilinear(line_table.phi)

        dlam = np.asarray(line_table.dlam, dtype=np.float64)  # in Angstrom offset
        normalization = _parabolic_integral(dlam, profile)
        if normalization <= 0.0:
            normalization = 1.0
        log_profile = np.log10(np.maximum(profile / normalization, 1e-99))
        cached = (dlam, log_profile)
        self._bcs_cache[key] = cached
        return cached

    def _profile_bcs(self, depth_idx: int, line_id: int, delta_nm: float) -> float:
        dlam, log_prof = self._ensure_bcs(depth_idx, line_id)
        delta_angstrom = delta_nm * 10.0
        if delta_angstrom <= dlam[0]:
            return 10.0 ** log_prof[0]
        if delta_angstrom >= dlam[-1]:
            return 10.0 ** log_prof[-1]
        idx = np.searchsorted(dlam, delta_angstrom)
        if idx == 0:
            return 10.0 ** log_prof[0]
        hi = min(idx, dlam.size - 1)
        lo = hi - 1
        span = dlam[hi] - dlam[lo]
        if span <= 0.0:
            return 10.0 ** log_prof[lo]
        a = (dlam[hi] - delta_angstrom) / span
        b = (delta_angstrom - dlam[lo]) / span
        return 10.0 ** (a * log_prof[lo] + b * log_prof[hi])

    def _hé4471(self, depth_idx: int, wave: float, wl: float, doppler: float) -> float:
        temp = self._clamp_temp(self.temperature[depth_idx], 5000.0, 40000.0)
        e_density = self.electron_density[depth_idx]
        xnfhp = self._xnfhp(depth_idx)
        if e_density <= 0.0 or doppler <= 0.0:
            return 0.0
        if e_density > 1.0e13:
            return SQRT_PI * doppler * 10.0 * self._profile_bcs(depth_idx, 1, wave - wl)
        ts = np.array([5.0e3, 1.0e4, 2.0e4, 4.0e4])
        ws = np.array([0.001460, 0.001269, 0.001079, 0.000898])
        ds = np.array([0.036, -0.005, -0.026, -0.034])
        alfs = np.array([0.107, 0.119, 0.134, 0.154])
        den = 1.0e13
        it = 1
        for idx in range(1, 4):
            it = idx
            if ts[idx] >= temp:
                break
        x = (temp - ts[it - 1]) / (ts[it] - ts[it - 1])
        xx = e_density / den
        width = xx * (x * ws[it] + (1.0 - x) * ws[it - 1])
        shift = x * ds[it] + (1.0 - x) * ds[it - 1]
        alf = xx**0.25 * (x * alfs[it] + (1.0 - x) * alfs[it - 1])
        xx_ratio = xnfhp / e_density
        vm1 = 8.78 * (xx_ratio + 2.0 * (1.0 - xx_ratio)) / np.sqrt(temp)
        rhom = 1.0 / (4.19 * e_density) ** (1.0 / 3.0)
        sigma = 1.885e14 * width * rhom * vm1 / (wl * 10.0) ** 2
        sigma = max(sigma, 1e-40)
        ion_term = alf ** (8.0 / 9.0) / sigma ** (1.0 / 3.0)
        wtot = width * (1.0 + 1.36 * ion_term) * 0.1
        dtot = width * shift * (1.0 + 2.36 * ion_term / max(abs(shift), 1e-12)) * 0.1
        a = wtot / doppler
        wwd = wave - wl - dtot
        components = (
            voigt_profile(abs(wwd - 0.0184) / doppler, a) / 9.0
            + voigt_profile(abs(wwd + 0.0013) / doppler, a) / 12.0
            + voigt_profile(abs(wwd + 0.0010) / doppler, a) / 4.0
            + voigt_profile(abs(wwd + 0.0029) / doppler, a) / 180.0
            + voigt_profile(abs(wwd + 0.0025) / doppler, a) * 11.0 / 20.0
        )
        return components

    def _hé4026(self, depth_idx: int, wave: float, wl: float, doppler: float) -> float:
        temp = self._clamp_temp(self.temperature[depth_idx], 5000.0, 40000.0)
        e_density = self.electron_density[depth_idx]
        xnfhp = self._xnfhp(depth_idx)
        if e_density <= 0.0 or doppler <= 0.0:
            return 0.0
        if e_density > 1.0e14:
            return SQRT_PI * doppler * 10.0 * self._profile_bcs(depth_idx, 2, wave - wl)
        ts = np.array([5.0e3, 1.0e4, 2.0e4, 4.0e4])
        ws = np.array([4.04, 3.49, 2.96, 2.47])
        ds = np.array([0.1339, 0.0960, 0.0780, 0.0709])
        alfs = np.array([0.969, 1.083, 1.225, 1.403])
        den = 1.0e16
        it = 1
        for idx in range(1, 4):
            it = idx
            if ts[idx] >= temp:
                break
        x = (temp - ts[it - 1]) / (ts[it] - ts[it - 1])
        xx = e_density / den
        width = xx * (x * ws[it] + (1.0 - x) * ws[it - 1])
        shift = x * ds[it] + (1.0 - x) * ds[it - 1]
        alf = xx**0.25 * (x * alfs[it] + (1.0 - x) * alfs[it - 1])
        xx_ratio = xnfhp / e_density
        vm1 = 8.78 * (xx_ratio + 2.0 * (1.0 - xx_ratio)) / np.sqrt(temp)
        rhom = 1.0 / (4.19 * e_density) ** (1.0 / 3.0)
        sigma = 1.885e14 * width * rhom * vm1 / (wl * 10.0) ** 2
        sigma = max(sigma, 1e-40)
        ion_term = alf ** (8.0 / 9.0) / sigma ** (1.0 / 3.0)
        wtot = width * (1.0 + 1.36 * ion_term) * 0.1
        dtot = width * shift * (1.0 + 2.36 * ion_term / max(abs(shift), 1e-12)) * 0.1
        a = wtot / doppler
        wwd = wave - wl - dtot
        components = (
            voigt_profile(abs(wwd - 0.0148) / doppler, a) / 9.0
            + voigt_profile(abs(wwd + 0.0012) / doppler, a) / 12.0
            + voigt_profile(abs(wwd + 0.0011) / doppler, a) / 4.0
            + voigt_profile(abs(wwd + 0.0025) / doppler, a) / 180.0
            + voigt_profile(abs(wwd + 0.0023) / doppler, a) * 11.0 / 20.0
        )
        return components

    def _hé4387(self, depth_idx: int, wave: float, wl: float, doppler: float) -> float:
        temp = self._clamp_temp(self.temperature[depth_idx], 5000.0, 40000.0)
        e_density = self.electron_density[depth_idx]
        xnfhp = self._xnfhp(depth_idx)
        if e_density <= 0.0 or doppler <= 0.0:
            return 0.0
        if e_density > 1.0e14:
            return SQRT_PI * doppler * 10.0 * self._profile_bcs(depth_idx, 3, wave - wl)
        ts = np.array([5.0e3, 1.0e4, 2.0e4, 4.0e4])
        ws = np.array([6.13, 5.15, 4.24, 3.45])
        ds = np.array([0.411, 0.363, 0.325, 0.293])
        alfs = np.array([1.159, 1.321, 1.527, 1.783])
        den = 1.0e16
        it = 1
        for idx in range(1, 4):
            it = idx
            if ts[idx] >= temp:
                break
        x = (temp - ts[it - 1]) / (ts[it] - ts[it - 1])
        xx = e_density / den
        width = xx * (x * ws[it] + (1.0 - x) * ws[it - 1])
        shift = x * ds[it] + (1.0 - x) * ds[it - 1]
        alf = xx**0.25 * (x * alfs[it] + (1.0 - x) * alfs[it - 1])
        xx_ratio = xnfhp / e_density
        vm1 = 8.78 * (xx_ratio + 2.0 * (1.0 - xx_ratio)) / np.sqrt(temp)
        rhom = 1.0 / (4.19 * e_density) ** (1.0 / 3.0)
        sigma = 1.885e14 * width * rhom * vm1 / (wl * 10.0) ** 2
        sigma = max(sigma, 1e-40)
        ion_term = alf ** (8.0 / 9.0) / sigma ** (1.0 / 3.0)
        wtot = width * (1.0 + 1.36 * ion_term) * 0.1
        dtot = width * shift * (1.0 + 2.36 * ion_term / max(abs(shift), 1e-12)) * 0.1
        a = wtot / doppler
        return voigt_profile(abs(wave - wl - dtot) / doppler, a)

    def _hé4921(self, depth_idx: int, wave: float, wl: float, doppler: float) -> float:
        temp = self._clamp_temp(self.temperature[depth_idx], 5000.0, 40000.0)
        e_density = self.electron_density[depth_idx]
        xnfhp = self._xnfhp(depth_idx)
        if e_density <= 0.0 or doppler <= 0.0:
            return 0.0
        if e_density > 1.0e13:
            return SQRT_PI * doppler * 10.0 * self._profile_bcs(depth_idx, 4, wave - wl)
        ts = np.array([5.0e3, 1.0e4, 2.0e4, 4.0e4])
        ws = np.array([0.002312, 0.001963, 0.001624, 0.001315])
        ds = np.array([0.3932, 0.3394, 0.2950, 0.2593])
        alfs = np.array([0.1207, 0.1365, 0.1564, 0.1844])
        den = 1.0e13
        it = 1
        for idx in range(1, 4):
            it = idx
            if ts[idx] >= temp:
                break
        x = (temp - ts[it - 1]) / (ts[it] - ts[it - 1])
        xx = e_density / den
        width = xx * (x * ws[it] + (1.0 - x) * ws[it - 1])
        shift = x * ds[it] + (1.0 - x) * ds[it - 1]
        alf = xx**0.25 * (x * alfs[it] + (1.0 - x) * alfs[it - 1])
        xx_ratio = xnfhp / e_density
        vm1 = 8.78 * (xx_ratio + 2.0 * (1.0 - xx_ratio)) / np.sqrt(temp)
        rhom = 1.0 / (4.19 * e_density) ** (1.0 / 3.0)
        sigma = 1.885e14 * width * rhom * vm1 / (wl * 10.0) ** 2
        sigma = max(sigma, 1e-40)
        ion_term = alf ** (8.0 / 9.0) / sigma ** (1.0 / 3.0)
        wtot = width * (1.0 + 1.36 * ion_term) * 0.1
        dtot = width * shift * (1.0 + 2.36 * ion_term / max(abs(shift), 1e-12)) * 0.1
        a = wtot / doppler
        wwd = wave - wl - dtot
        return voigt_profile(abs(wwd) / doppler, a)

    def _griem_profile(
        self,
        depth_idx: int,
        wave: float,
        wl: float,
        doppler: float,
        gamma_rad: float,
        gamma_stark: float,
    ) -> float:
        if doppler <= 0.0:
            return 0.0
        idx = _find_record(self.griem_records, wl, HE_TABLE_MATCH_TOL_NM)
        if idx is None:
            a = (gamma_rad + gamma_stark * self.electron_density[depth_idx]) / (
                doppler / wl
            )
            return voigt_profile(abs(wave - wl) / doppler, a)
        record = self.griem_records[idx]
        temp = self._clamp_temp(self.temperature[depth_idx], 5000.0, 80000.0)
        e_density = self.electron_density[depth_idx]
        xnfhp = self._xnfhp(depth_idx)
        it = 1
        for i in range(1, 4):
            it = i
            if record.ttab[i] >= temp:
                break
        x = (temp - record.ttab[it - 1]) / (record.ttab[it] - record.ttab[it - 1])
        xx = e_density / (10.0**record.log_ne)
        width = xx * (x * record.width[it] + (1.0 - x) * record.width[it - 1])
        shift = x * record.shift[it] + (1.0 - x) * record.shift[it - 1]
        alpha_val = xx**0.25 * (x * record.alpha[it] + (1.0 - x) * record.alpha[it - 1])
        xx_ratio = xnfhp / e_density if e_density > 0.0 else 0.0
        vm1 = 8.78 * (xx_ratio + 2.0 * (1.0 - xx_ratio)) / np.sqrt(temp)
        rhom = 1.0 / (4.19 * e_density) ** (1.0 / 3.0)
        sigma = 1.885e14 * width * rhom * vm1 / (wl * 10.0) ** 2
        sigma = max(sigma, 1e-40)
        ion_term = alpha_val ** (8.0 / 9.0) / sigma ** (1.0 / 3.0)
        wtot = width * (1.0 + 1.36 * ion_term) * 0.1
        dtot = width * shift * (1.0 + 2.36 * ion_term / max(abs(shift), 1e-12)) * 0.1
        a = wtot / doppler + gamma_rad / (doppler / wl)
        return voigt_profile(abs(wave - wl - dtot) / doppler, a)

    def _dimitri_profile(
        self,
        depth_idx: int,
        wave: float,
        wl: float,
        doppler: float,
        gamma_rad: float,
        gamma_stark: float,
    ) -> float:
        if doppler <= 0.0:
            return 0.0
        idx = _find_record(self.dimitri_records, wl, HE_TABLE_MATCH_TOL_NM)
        if idx is None:
            return self._griem_profile(
                depth_idx, wave, wl, doppler, gamma_rad, gamma_stark
            )
        record = self.dimitri_records[idx]
        temp = self._clamp_temp(self.temperature[depth_idx], 5000.0, 80000.0)
        e_density = self.electron_density[depth_idx]
        xnfhp = self._xnfhp(depth_idx)
        xnfhep = self._xnfhep(depth_idx)
        it = 1
        for i in range(1, 4):
            it = i
            if record.ttab[i] >= temp:
                break
        x = (temp - record.ttab[it - 1]) / (record.ttab[it] - record.ttab[it - 1])
        xx = e_density / (10.0**record.log_ne)
        xxh = xnfhp / (10.0**record.log_ne)
        xxhe = xnfhep / (10.0**record.log_ne)
        width = xx * (x * record.width_e[it] + (1.0 - x) * record.width_e[it - 1])
        width_h = xxh * (x * record.width_h[it] + (1.0 - x) * record.width_h[it - 1])
        width_he = xxhe * (
            x * record.width_he[it] + (1.0 - x) * record.width_he[it - 1]
        )
        shift = record.shift_e[it] + (1.0 - x) * record.shift_e[it - 1]
        shift_h = x * record.shift_h[it] + (1.0 - x) * record.shift_h[it - 1]
        shift_he = x * record.shift_he[it] + (1.0 - x) * record.shift_he[it - 1]
        wtot_angstrom = width + width_h + width_he
        dtot_angstrom = width * shift + width_h * shift_h + width_he * shift_he
        wtot_nm = wtot_angstrom * 0.1 * 0.5
        dtot_nm = dtot_angstrom * 0.1
        a = wtot_nm / doppler + gamma_rad / (doppler / wl)
        return voigt_profile(abs(wave - wl - dtot_nm) / doppler, a)

    def evaluate(
        self,
        line_type: int,
        depth_idx: int,
        delta_nm: float,
        line_wavelength: float,
        doppler_width: float,
        gamma_rad: float,
        gamma_stark: float,
    ) -> float:
        doppler = max(doppler_width, 1e-40)
        wave = line_wavelength + delta_nm
        if line_type in (-3, -4):
            if abs(line_wavelength - 447.15) < 0.4:
                return self._hé4471(depth_idx, wave, line_wavelength, doppler)
            if abs(line_wavelength - 402.62) < 0.4:
                return self._hé4026(depth_idx, wave, line_wavelength, doppler)
            if abs(line_wavelength - 438.79) < 0.4:
                return self._hé4387(depth_idx, wave, line_wavelength, doppler)
            if abs(line_wavelength - 492.19) < 0.4:
                return self._hé4921(depth_idx, wave, line_wavelength, doppler)
            # other He I lines
            if (
                _find_record(
                    self.dimitri_records, line_wavelength, HE_TABLE_MATCH_TOL_NM
                )
                is not None
            ):
                return self._dimitri_profile(
                    depth_idx, wave, line_wavelength, doppler, gamma_rad, gamma_stark
                )
            return self._griem_profile(
                depth_idx, wave, line_wavelength, doppler, gamma_rad, gamma_stark
            )
        if line_type == -6:
            return self._griem_profile(
                depth_idx, wave, line_wavelength, doppler, gamma_rad, gamma_stark
            )
        return 0.0
