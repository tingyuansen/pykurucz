"""Shared Numba-compiled Voigt profile function.

This module provides a single canonical JIT-compiled implementation of the
Voigt profile H(a,v) used throughout the synthesis pipeline.  All callers
should import ``voigt_profile_jit`` from here instead of maintaining
their own copy.
"""

from __future__ import annotations

import numpy as np

from numba import jit, prange


@jit(nopython=True, cache=True)
def accumulate_voigt_wings_jit(
    buffer: np.ndarray,
    continuum_row: np.ndarray,
    wavelength_grid: np.ndarray,
    center_index: int,
    line_wavelength: float,
    kappa_eff: float,
    doppler: float,
    adamp: float,
    cutoff: float,
    has_wcon: bool,
    wcon_val: float,
    has_wtail: bool,
    wtail_val: float,
    base_wave: float,
    h0tab: np.ndarray,
    h1tab: np.ndarray,
    h2tab: np.ndarray,
) -> None:
    """JIT-compiled Voigt wing accumulation for helium/normal lines (TYPE < -2).

    Replicates the Fortran 211/214 loop style from synthe.for XLINOP section.
    Red wing tests cutoff before accumulation (break on first sub-cutoff point);
    blue wing accumulates then tests cutoff (break after accumulation).
    """
    n_points = buffer.shape[0]
    clamped = max(0, min(center_index, n_points - 1))

    # Red wing (Fortran label 211 style)
    if line_wavelength <= wavelength_grid[n_points - 1]:
        for idx in range(clamped, n_points):
            wave = wavelength_grid[idx]
            if has_wcon and wave <= wcon_val:
                continue
            x_val = (wave - line_wavelength) if wave >= line_wavelength else (line_wavelength - wave)
            x_val = x_val / doppler
            iv = int(x_val * 200.0 + 0.5)
            if iv > h0tab.shape[0] - 1:
                iv = h0tab.shape[0] - 1
            if adamp < 0.2:
                if x_val > 10.0:
                    voigt_val = 0.5642 * adamp / (x_val * x_val)
                else:
                    voigt_val = (h2tab[iv] * adamp + h1tab[iv]) * adamp + h0tab[iv]
            elif adamp > 1.4 or (adamp + x_val) > 3.2:
                aa = adamp * adamp
                vv = x_val * x_val
                u = (aa + vv) * 1.4142
                voigt_val = adamp * 0.79788 / u
                if adamp <= 100.0:
                    aau = aa / u
                    vvu = vv / u
                    uu = u * u
                    voigt_val = (
                        (((aau - 10.0 * vvu) * aau * 3.0 + 15.0 * vvu * vvu) + 3.0 * vv - aa)
                        / uu
                        + 1.0
                    ) * voigt_val
            else:
                vv = x_val * x_val
                h0 = h0tab[iv]
                h1 = h1tab[iv] + h0 * 1.12838
                h2 = h2tab[iv] + h1 * 1.12838 - h0
                h3 = (1.0 - h2tab[iv]) * 0.37613 - h1 * 0.66667 * vv + h2 * 1.12838
                h4 = (3.0 * h3 - h1) * 0.37613 + h0 * 0.66667 * vv * vv
                poly_a = (((h4 * adamp + h3) * adamp + h2) * adamp + h1) * adamp + h0
                poly_b = ((-0.122727278 * adamp + 0.532770573) * adamp - 0.96284325) * adamp + 0.979895032
                voigt_val = poly_a * poly_b
            value = kappa_eff * voigt_val
            if has_wtail and wave < wtail_val:
                denom = wtail_val - base_wave
                if denom < 1e-12:
                    denom = 1e-12
                value = value * (wave - base_wave) / denom
            if value < continuum_row[idx] * cutoff:
                break
            buffer[idx] += value

    # Blue wing (Fortran label 214 style)
    if clamped > 0 and line_wavelength >= wavelength_grid[0]:
        for idx in range(clamped - 1, -1, -1):
            wave = wavelength_grid[idx]
            if has_wcon and wave <= wcon_val:
                break
            x_val = (line_wavelength - wave) if line_wavelength >= wave else (wave - line_wavelength)
            x_val = x_val / doppler
            iv = int(x_val * 200.0 + 0.5)
            if iv > h0tab.shape[0] - 1:
                iv = h0tab.shape[0] - 1
            if adamp < 0.2:
                if x_val > 10.0:
                    voigt_val = 0.5642 * adamp / (x_val * x_val)
                else:
                    voigt_val = (h2tab[iv] * adamp + h1tab[iv]) * adamp + h0tab[iv]
            elif adamp > 1.4 or (adamp + x_val) > 3.2:
                aa = adamp * adamp
                vv = x_val * x_val
                u = (aa + vv) * 1.4142
                voigt_val = adamp * 0.79788 / u
                if adamp <= 100.0:
                    aau = aa / u
                    vvu = vv / u
                    uu = u * u
                    voigt_val = (
                        (((aau - 10.0 * vvu) * aau * 3.0 + 15.0 * vvu * vvu) + 3.0 * vv - aa)
                        / uu
                        + 1.0
                    ) * voigt_val
            else:
                vv = x_val * x_val
                h0 = h0tab[iv]
                h1 = h1tab[iv] + h0 * 1.12838
                h2 = h2tab[iv] + h1 * 1.12838 - h0
                h3 = (1.0 - h2tab[iv]) * 0.37613 - h1 * 0.66667 * vv + h2 * 1.12838
                h4 = (3.0 * h3 - h1) * 0.37613 + h0 * 0.66667 * vv * vv
                poly_a = (((h4 * adamp + h3) * adamp + h2) * adamp + h1) * adamp + h0
                poly_b = ((-0.122727278 * adamp + 0.532770573) * adamp - 0.96284325) * adamp + 0.979895032
                voigt_val = poly_a * poly_b
            value = kappa_eff * voigt_val
            if has_wtail and wave < wtail_val:
                denom = wtail_val - base_wave
                if denom < 1e-12:
                    denom = 1e-12
                value = value * (wave - base_wave) / denom
            buffer[idx] += value
            if value < continuum_row[idx] * cutoff:
                break


@jit(nopython=True, cache=True)
def accumulate_voigt_wings_and_source_jit(
    wings_row: np.ndarray,
    sources_row: np.ndarray,
    bnu_row: np.ndarray,
    continuum_row: np.ndarray,
    wavelength_grid: np.ndarray,
    center_index: int,
    line_wavelength: float,
    kappa_eff: float,
    doppler: float,
    adamp: float,
    cutoff: float,
    has_wcon: bool,
    wcon_val: float,
    has_wtail: bool,
    wtail_val: float,
    base_wave: float,
    h0tab: np.ndarray,
    h1tab: np.ndarray,
    h2tab: np.ndarray,
) -> None:
    """Fused wing+source accumulation — avoids tmp buffer allocation.

    Same Voigt walk as ``accumulate_voigt_wings_jit`` but writes directly to
    both ``wings_row`` and ``sources_row = wings_row * bnu_row`` in a single
    pass, eliminating the per-line tmp array allocation and separate add loops.
    """
    n_points = wavelength_grid.shape[0]
    clamped = max(0, min(center_index, n_points - 1))

    # Red wing
    if line_wavelength <= wavelength_grid[n_points - 1]:
        for idx in range(clamped, n_points):
            wave = wavelength_grid[idx]
            if has_wcon and wave <= wcon_val:
                continue
            x_val = (wave - line_wavelength) if wave >= line_wavelength else (line_wavelength - wave)
            x_val = x_val / doppler
            iv = int(x_val * 200.0 + 0.5)
            if iv > h0tab.shape[0] - 1:
                iv = h0tab.shape[0] - 1
            if adamp < 0.2:
                if x_val > 10.0:
                    voigt_val = 0.5642 * adamp / (x_val * x_val)
                else:
                    voigt_val = (h2tab[iv] * adamp + h1tab[iv]) * adamp + h0tab[iv]
            elif adamp > 1.4 or (adamp + x_val) > 3.2:
                aa = adamp * adamp
                vv = x_val * x_val
                u = (aa + vv) * 1.4142
                voigt_val = adamp * 0.79788 / u
                if adamp <= 100.0:
                    aau = aa / u
                    vvu = vv / u
                    uu = u * u
                    voigt_val = (
                        (((aau - 10.0 * vvu) * aau * 3.0 + 15.0 * vvu * vvu) + 3.0 * vv - aa)
                        / uu
                        + 1.0
                    ) * voigt_val
            else:
                vv = x_val * x_val
                h0 = h0tab[iv]
                h1 = h1tab[iv] + h0 * 1.12838
                h2 = h2tab[iv] + h1 * 1.12838 - h0
                h3 = (1.0 - h2tab[iv]) * 0.37613 - h1 * 0.66667 * vv + h2 * 1.12838
                h4 = (3.0 * h3 - h1) * 0.37613 + h0 * 0.66667 * vv * vv
                poly_a = (((h4 * adamp + h3) * adamp + h2) * adamp + h1) * adamp + h0
                poly_b = ((-0.122727278 * adamp + 0.532770573) * adamp - 0.96284325) * adamp + 0.979895032
                voigt_val = poly_a * poly_b
            value = kappa_eff * voigt_val
            if has_wtail and wave < wtail_val:
                denom = wtail_val - base_wave
                if denom < 1e-12:
                    denom = 1e-12
                value = value * (wave - base_wave) / denom
            if value < continuum_row[idx] * cutoff:
                break
            wings_row[idx] += value
            sources_row[idx] += value * bnu_row[idx]

    # Blue wing
    if clamped > 0 and line_wavelength >= wavelength_grid[0]:
        for idx in range(clamped - 1, -1, -1):
            wave = wavelength_grid[idx]
            if has_wcon and wave <= wcon_val:
                continue
            x_val = (line_wavelength - wave) if line_wavelength >= wave else (wave - line_wavelength)
            x_val = x_val / doppler
            iv = int(x_val * 200.0 + 0.5)
            if iv > h0tab.shape[0] - 1:
                iv = h0tab.shape[0] - 1
            if adamp < 0.2:
                if x_val > 10.0:
                    voigt_val = 0.5642 * adamp / (x_val * x_val)
                else:
                    voigt_val = (h2tab[iv] * adamp + h1tab[iv]) * adamp + h0tab[iv]
            elif adamp > 1.4 or (adamp + x_val) > 3.2:
                aa = adamp * adamp
                vv = x_val * x_val
                u = (aa + vv) * 1.4142
                voigt_val = adamp * 0.79788 / u
                if adamp <= 100.0:
                    aau = aa / u
                    vvu = vv / u
                    uu = u * u
                    voigt_val = (
                        (((aau - 10.0 * vvu) * aau * 3.0 + 15.0 * vvu * vvu) + 3.0 * vv - aa)
                        / uu
                        + 1.0
                    ) * voigt_val
            else:
                vv = x_val * x_val
                h0 = h0tab[iv]
                h1 = h1tab[iv] + h0 * 1.12838
                h2 = h2tab[iv] + h1 * 1.12838 - h0
                h3 = (1.0 - h2tab[iv]) * 0.37613 - h1 * 0.66667 * vv + h2 * 1.12838
                h4 = (3.0 * h3 - h1) * 0.37613 + h0 * 0.66667 * vv * vv
                poly_a = (((h4 * adamp + h3) * adamp + h2) * adamp + h1) * adamp + h0
                poly_b = ((-0.122727278 * adamp + 0.532770573) * adamp - 0.96284325) * adamp + 0.979895032
                voigt_val = poly_a * poly_b
            value = kappa_eff * voigt_val
            if has_wtail and wave < wtail_val:
                denom = wtail_val - base_wave
                if denom < 1e-12:
                    denom = 1e-12
                value = value * (wave - base_wave) / denom
            if value < continuum_row[idx] * cutoff:
                break
            wings_row[idx] += value
            sources_row[idx] += value * bnu_row[idx]


@jit(nopython=True, cache=True)
def voigt_profile_jit(
    v: float, a: float, h0tab: np.ndarray, h1tab: np.ndarray, h2tab: np.ndarray
) -> float:
    """JIT-compiled Voigt profile H(a, v) matching the Fortran approximation.

    Parameters
    ----------
    v : float
        Frequency displacement in Doppler units.
    a : float
        Damping parameter (ratio of Lorentz to Doppler width).
    h0tab, h1tab, h2tab : np.ndarray
        Pre-computed Voigt coefficient tables (size 2001, step=1/200).

    Returns
    -------
    float
        Voigt function value H(a, v).
    """
    # Voigt function is symmetric in v — use abs(v) for table lookup.
    iv = int(abs(v) * 200.0 + 0.5)
    iv = max(0, min(iv, h0tab.size - 1))

    if a < 0.2:
        if abs(v) > 10.0:
            return 0.5642 * a / (v * v)
        else:
            return (h2tab[iv] * a + h1tab[iv]) * a + h0tab[iv]
    elif a > 1.4 or (a + abs(v)) > 3.2:
        aa = a * a
        vv = v * v
        u = (aa + vv) * 1.4142
        voigt_val = a * 0.79788 / u
        if a <= 100.0:
            aau = aa / u
            vvu = vv / u
            uu = u * u
            voigt_val = (
                (((aau - 10.0 * vvu) * aau * 3.0 + 15.0 * vvu * vvu) + 3.0 * vv - aa)
                / uu
                + 1.0
            ) * voigt_val
        return voigt_val
    else:
        vv = v * v
        h0 = h0tab[iv]
        h1 = h1tab[iv] + h0 * 1.12838
        h2 = h2tab[iv] + h1 * 1.12838 - h0
        h3 = (1.0 - h2tab[iv]) * 0.37613 - h1 * 0.66667 * vv + h2 * 1.12838
        h4 = (3.0 * h3 - h1) * 0.37613 + h0 * 0.66667 * vv * vv
        poly_a = (((h4 * a + h3) * a + h2) * a + h1) * a + h0
        poly_b = ((-0.122727278 * a + 0.532770573) * a - 0.96284325) * a + 0.979895032
        return poly_a * poly_b


@jit(nopython=True, parallel=True, cache=True)
def compute_helium_voigt_batch(
    wings_out: np.ndarray,    # (n_depths, n_wavelengths) — accumulated wings
    sources_out: np.ndarray,  # (n_depths, n_wavelengths) — accumulated sources
    bnu: np.ndarray,          # (n_depths, n_wavelengths)
    continuum: np.ndarray,    # (n_depths, n_wavelengths) — for cutoff
    wavelength_grid: np.ndarray,  # (n_wavelengths,)
    kappa0_2d: np.ndarray,    # (n_depths, n_he_lines)
    doppler_2d: np.ndarray,   # (n_depths, n_he_lines)
    adamp_2d: np.ndarray,     # (n_depths, n_he_lines)
    center_indices: np.ndarray,   # (n_he_lines,)
    line_wavelengths: np.ndarray, # (n_he_lines,)
    line_type_codes: np.ndarray,  # (n_he_lines,)
    wcon_2d: np.ndarray,      # (n_depths, n_he_lines)
    wtail_2d: np.ndarray,     # (n_depths, n_he_lines)
    cutoff: float,
    h0tab: np.ndarray,
    h1tab: np.ndarray,
    h2tab: np.ndarray,
) -> None:
    """Depth-parallel batch kernel for helium Voigt wing accumulation.

    Processes all helium lines (line_type_code in -3, -4, -6) across all
    depths using prange over depths.  Each depth writes to its own row of
    wings_out / sources_out so there are no write races.

    Mirrors the _apply_fort19_profile line_type_code < -2 branch exactly,
    including the TYPE=-4 (3He) kappa/doppler adjustment.

    Uses accumulate_voigt_wings_and_source_jit to write directly into
    wings_out[d] and sources_out[d] — no per-line tmp allocation needed.
    """
    n_depths = kappa0_2d.shape[0]
    n_he_lines = kappa0_2d.shape[1]

    for d in prange(n_depths):
        for l in range(n_he_lines):
            k0 = kappa0_2d[d, l]
            if k0 <= 0.0:
                continue
            dop = doppler_2d[d, l]
            if dop <= 0.0:
                continue
            ltc = int(line_type_codes[l])
            lw = line_wavelengths[l]
            ci = int(center_indices[l])

            # TYPE=-4 (3He) adjustments
            if ltc == -4:
                keff = k0 / 1.155
                deff = dop * 1.155
                ad = adamp_2d[d, l] / 1.155
            else:
                keff = k0
                deff = dop
                ad = adamp_2d[d, l]
            ad = max(ad, 1e-12)

            wcon_v = wcon_2d[d, l]
            wtail_v = wtail_2d[d, l]
            has_wcon = wcon_v > 0.0
            has_wtail = wtail_v > 0.0
            base = wcon_v if has_wcon else lw

            accumulate_voigt_wings_and_source_jit(
                wings_out[d], sources_out[d], bnu[d], continuum[d], wavelength_grid,
                ci, lw, keff, deff, ad, cutoff,
                has_wcon, wcon_v, has_wtail, wtail_v, base,
                h0tab, h1tab, h2tab,
            )





