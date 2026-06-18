"""Numba-compiled radiative transfer loop for SYNTHE.

Replaces the Python-level per-wavelength dispatch in ``radiative.py`` with a
single ``@njit(parallel=True)`` call that processes ALL wavelengths in compiled
code using ``prange``.  This eliminates ~537k Python function-call overhead
(the actual bottleneck — more threads made it *slower*).

The physics is identical to ``solve_lte_frequency`` + ``solve_josh_flux``;
only logging, try/except, and Optional handling are stripped for Numba
compatibility.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from synthe_py.physics.josh_solver import (
    _josh_iteration_kernel,
    _integ,
    _map1_kernel,
    _parcoe,
    _deriv,
    EPS,
    ITER_TOL,
    MAX_ITER,
    USE_FLOAT32_ITERATION,
)
from synthe_py.physics.josh_tables import (
    CH_WEIGHTS,
    XTAU_GRID,
    COEFJ_MATRIX,
    NXTAU,
)

# Module-level constants Numba can close over.
_COEFJ_DIAG = np.diag(COEFJ_MATRIX).copy()
_H_PLANCK = 6.62607015e-27   # erg·s
_K_BOLTZ  = 1.380649e-16     # erg/K
_C_LIGHT_NM = 2.99792458e17  # nm/s
_MAX_OPACITY_REAL4 = 3.4e38


# ---------------------------------------------------------------------------
# Lean Numba helpers
# ---------------------------------------------------------------------------

@njit(cache=True)
def _planck_bnu(freq: float, temperature: np.ndarray) -> np.ndarray:
    """Planck B_nu(T) matching Fortran formula. No np.errstate needed."""
    n = temperature.shape[0]
    bnu = np.zeros(n, dtype=np.float64)
    freq15 = freq / 1.0e15
    coeff = 1.47439e-2 * freq15 * freq15 * freq15
    for j in range(n):
        hkt_j = _H_PLANCK / (_K_BOLTZ * temperature[j])
        arg = -freq * hkt_j
        if arg < -700.0:
            bnu[j] = 0.0
        else:
            ehvkt = np.exp(arg)
            stim = 1.0 - ehvkt
            if abs(stim) < 1e-300:
                bnu[j] = 0.0
            else:
                bnu[j] = coeff * ehvkt / stim
                if not np.isfinite(bnu[j]):
                    bnu[j] = 0.0
    return bnu


@njit(cache=True)
def _map1_wrapper(xold, fold, xnew):
    """Thin wrapper around _map1_kernel returning (fnew, maxj)."""
    return _map1_kernel(xold, fold, xnew)


@njit(cache=True)
def _solve_josh_core(
    acont, scont, aline, sline, sigmac, sigmal, rho,
    coefj_matrix, coefj_diag, xtau_grid, ch_weights,
    nxtau,
):
    """Core JOSH flux solve for one frequency (no logging, no try/except).

    Returns emergent surface flux (HNU).
    """
    n = rho.shape[0]
    if n == 0:
        return 0.0

    # ABTOT, alpha, snubar -------------------------------------------------
    abtot = np.empty(n, dtype=np.float64)
    alpha = np.empty(n, dtype=np.float64)
    snubar = np.empty(n, dtype=np.float64)
    for j in range(n):
        ab = acont[j] + aline[j] + sigmac[j] + sigmal[j]
        if ab < EPS:
            ab = EPS
        abtot[j] = ab
        sc = sigmac[j] + sigmal[j]
        a = sc / ab if ab > 0.0 else 0.0
        if a < 0.0:
            a = 0.0
        if a > 1.0:
            a = 1.0
        alpha[j] = a
        denom = acont[j] + aline[j]
        if denom > 0.0:
            snubar[j] = (acont[j] * scont[j] + aline[j] * sline[j]) / denom
        else:
            snubar[j] = scont[j]

    # Ensure RHOX is increasing for INTEG -----------------------------------
    needs_reverse = n > 1 and rho[0] > rho[-1]
    if needs_reverse:
        rho_integ = rho[::-1].copy()
        abtot_integ = abtot[::-1].copy()
        start = abtot_integ[-1] * rho_integ[-1]
    else:
        rho_integ = rho.copy()
        abtot_integ = abtot.copy()
        start = abtot[0] * rho_integ[0]

    # TAUNU integration -----------------------------------------------------
    taunu = _integ(rho_integ, abtot_integ, start)

    if needs_reverse:
        snubar_ord = snubar[::-1].copy()
        alpha_ord = alpha[::-1].copy()
    else:
        snubar_ord = snubar
        alpha_ord = alpha

    # MAXJ-401 check --------------------------------------------------------
    maxj_force_401 = taunu.size > 0 and taunu[0] > xtau_grid[nxtau - 1]

    if maxj_force_401:
        # Label 401 path: iterate on physical TAUNU grid
        snu = snubar_ord.copy()
        hnu_profile = np.zeros(taunu.size, dtype=np.float64)

        dtau = np.empty(taunu.size - 1, dtype=np.float64)
        for k in range(taunu.size - 1):
            dtau[k] = taunu[k + 1] - taunu[k]
        min_dtau_deep = 1e30
        start_idx = min(2, dtau.size)
        for k in range(start_idx, dtau.size):
            v = abs(dtau[k])
            if v < min_dtau_deep:
                min_dtau_deep = v
        stab_thresh = 1e-4 * abs(taunu[0]) if taunu.size > 0 else 1e-4
        tau_unstable = min_dtau_deep < stab_thresh

        if tau_unstable:
            hnu_profile = _deriv(taunu, snubar_ord)
            for k in range(hnu_profile.size):
                hnu_profile[k] /= 3.0
            flux = hnu_profile[0] if hnu_profile.size > 0 else 0.0
        else:
            for l_iter in range(MAX_ITER):
                hnu_profile = _deriv(taunu, snu)
                for k in range(hnu_profile.size):
                    hnu_profile[k] /= 3.0
                jmins = _deriv(taunu, hnu_profile)
                max_corr = 0.0
                for k in range(jmins.size):
                    v = abs(alpha_ord[k] * jmins[k])
                    if v > max_corr:
                        max_corr = v
                max_snubar = EPS
                for k in range(snubar_ord.size):
                    v = abs(snubar_ord[k])
                    if v > max_snubar:
                        max_snubar = v
                if max_corr > max_snubar:
                    break
                jnu = jmins + snu
                snew = (1.0 - alpha_ord) * snubar_ord + alpha_ord * jnu
                total_rel = 0.0
                for k in range(snew.size):
                    denom_r = abs(snew[k])
                    if denom_r < EPS:
                        denom_r = EPS
                    total_rel += abs(snew[k] - snu[k]) / denom_r
                snu = snew
                if total_rel < ITER_TOL:
                    break
            flux = hnu_profile[0] if hnu_profile.size > 0 else 0.0
        return flux

    # Normal path: MAP1 interpolation onto XTAU_GRID -----------------------
    xsbar, _ = _map1_kernel(taunu, snubar_ord, xtau_grid)
    xalpha, _ = _map1_kernel(taunu, alpha_ord, xtau_grid)

    for k in range(nxtau):
        if xsbar[k] < EPS:
            xsbar[k] = EPS
        if xalpha[k] < 0.0:
            xalpha[k] = 0.0
        if xalpha[k] > 1.0:
            xalpha[k] = 1.0

    # Surface-value masking (xtau < taunu[0]) --------------------------------
    if taunu.size > 0:
        tau0 = taunu[0]
        snubar0 = snubar_ord[0] if snubar_ord[0] >= EPS else EPS
        alpha0 = alpha_ord[0]
        if alpha0 < 0.0:
            alpha0 = 0.0
        if alpha0 > 1.0:
            alpha0 = 1.0
        for k in range(nxtau):
            if xtau_grid[k] < tau0:
                xsbar[k] = snubar0
                xalpha[k] = alpha0

    xs = xsbar.copy()
    xsbar_modified = np.empty(nxtau, dtype=np.float64)
    for k in range(nxtau):
        xsbar_modified[k] = xsbar[k] * (1.0 - xalpha[k])

    # JOSH iteration --------------------------------------------------------
    if USE_FLOAT32_ITERATION:
        coefj_f32 = coefj_matrix.astype(np.float32)
        diag_f32 = coefj_diag.astype(np.float32)
        xs_f32 = xs.astype(np.float32)
        xalpha_f32 = xalpha.astype(np.float32)
        xsbar_mod_f32 = xsbar_modified.astype(np.float32)
        xs_result, _ = _josh_iteration_kernel(
            coefj_f32, xs_f32, xalpha_f32, xsbar_mod_f32, diag_f32,
            np.float32(ITER_TOL), MAX_ITER, np.float32(EPS),
        )
        for k in range(nxtau):
            xs[k] = float(xs_result[k])
    else:
        xs_result, _ = _josh_iteration_kernel(
            coefj_matrix, xs.copy(), xalpha, xsbar_modified, coefj_diag,
            ITER_TOL, MAX_ITER, EPS,
        )
        for k in range(nxtau):
            xs[k] = xs_result[k]

    # Emergent flux ---------------------------------------------------------
    flux = 0.0
    for k in range(nxtau):
        flux += ch_weights[k] * xs[k]
    return flux


@njit(cache=True)
def _solve_one_freq(
    wl_nm,
    temperature,
    column_mass,
    cont_abs_col,
    cont_scat_col,
    line_opacity_col,
    line_scattering_col,
    line_source_col,
    has_line_source,
    coefj_matrix,
    coefj_diag,
    xtau_grid,
    ch_weights,
    nxtau,
):
    """Solve LTE radiative transfer for a single wavelength (Numba-compiled).

    When *has_line_source* is True, ``line_source_col`` is used as the line
    source function (sline) instead of the Planck function.

    Returns (flux_total, flux_cont).
    """
    n_layers = temperature.shape[0]

    # Build validity mask ---------------------------------------------------
    count_valid = 0
    valid = np.empty(n_layers, dtype=np.bool_)
    for j in range(n_layers):
        ok = (
            column_mass[j] > 0.0
            and np.isfinite(column_mass[j])
            and np.isfinite(cont_abs_col[j])
            and np.isfinite(cont_scat_col[j])
            and np.isfinite(line_opacity_col[j])
            and np.isfinite(line_scattering_col[j])
            and cont_abs_col[j] < _MAX_OPACITY_REAL4
            and cont_scat_col[j] < _MAX_OPACITY_REAL4
            and line_opacity_col[j] < _MAX_OPACITY_REAL4
            and line_scattering_col[j] < _MAX_OPACITY_REAL4
        )
        valid[j] = ok
        if ok:
            count_valid += 1

    if count_valid == 0:
        return 0.0, 0.0

    # Extract valid layers --------------------------------------------------
    mass = np.empty(count_valid, dtype=np.float64)
    temp = np.empty(count_valid, dtype=np.float64)
    cont_a = np.empty(count_valid, dtype=np.float64)
    cont_s = np.empty(count_valid, dtype=np.float64)
    line_a = np.empty(count_valid, dtype=np.float64)
    line_sig = np.empty(count_valid, dtype=np.float64)
    line_src_raw = np.empty(count_valid, dtype=np.float64)
    idx = 0
    for j in range(n_layers):
        if valid[j]:
            mass[idx] = column_mass[j]
            temp[idx] = temperature[j]
            cont_a[idx] = cont_abs_col[j]
            cont_s[idx] = cont_scat_col[j]
            line_a[idx] = line_opacity_col[j]
            line_sig[idx] = line_scattering_col[j]
            if has_line_source:
                line_src_raw[idx] = line_source_col[j]
            idx += 1

    # Ensure RHOX is increasing (surface → deep) ---------------------------
    if count_valid > 1 and mass[0] > mass[-1]:
        mass = mass[::-1].copy()
        temp = temp[::-1].copy()
        cont_a = cont_a[::-1].copy()
        cont_s = cont_s[::-1].copy()
        line_a = line_a[::-1].copy()
        line_sig = line_sig[::-1].copy()
        if has_line_source:
            line_src_raw = line_src_raw[::-1].copy()

    # Planck function -------------------------------------------------------
    freq = _C_LIGHT_NM / max(wl_nm, 1e-12)
    planck = _planck_bnu(freq, temp)

    # Line source function: use provided or fall back to Planck -------------
    if has_line_source:
        line_src = np.empty(count_valid, dtype=np.float64)
        for j in range(count_valid):
            v = line_src_raw[j]
            if np.isfinite(v):
                line_src[j] = v
            else:
                line_src[j] = planck[j]  # replace NaN/INF with Planck
    else:
        line_src = planck

    # Continuum-only flux ---------------------------------------------------
    zero_line = np.zeros(count_valid, dtype=np.float64)
    flux_cont = _solve_josh_core(
        cont_a, planck, zero_line, planck, cont_s, zero_line,
        mass, coefj_matrix, coefj_diag, xtau_grid, ch_weights, nxtau,
    )

    # Total flux (continuum + lines) ----------------------------------------
    flux_total = _solve_josh_core(
        cont_a, planck, line_a, line_src, cont_s, line_sig,
        mass, coefj_matrix, coefj_diag, xtau_grid, ch_weights, nxtau,
    )

    return flux_total, flux_cont


@njit(parallel=True, cache=True)
def solve_all_wavelengths_prange(
    wavelength_nm: np.ndarray,
    temperature: np.ndarray,
    column_mass: np.ndarray,
    cont_abs: np.ndarray,
    cont_scat: np.ndarray,
    line_opacity: np.ndarray,
    line_scattering: np.ndarray,
    line_source: np.ndarray,
    has_line_source: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Process all wavelengths in a single compiled call with prange parallelism.

    Parameters match ``solve_lte_spectrum``.  When *has_line_source* is False,
    *line_source* is ignored (pass a dummy array).  ``NUMBA_NUM_THREADS``
    controls the number of parallel threads.

    Returns ``(flux_total, flux_cont)`` arrays of shape ``(n_wavelengths,)``.
    """
    n_wl = wavelength_nm.shape[0]
    flux_total = np.empty(n_wl, dtype=np.float64)
    flux_cont = np.empty(n_wl, dtype=np.float64)

    # Module-level constants are captured as closure variables by Numba.
    coefj = COEFJ_MATRIX
    diag = _COEFJ_DIAG
    xtau = XTAU_GRID
    ch = CH_WEIGHTS
    nxt = NXTAU

    for i in prange(n_wl):
        ls_col = line_source[:, i] if has_line_source else line_opacity[:, i]  # dummy
        ft, fc = _solve_one_freq(
            wavelength_nm[i],
            temperature,
            column_mass,
            cont_abs[:, i],
            cont_scat[:, i],
            line_opacity[:, i],
            line_scattering[:, i],
            ls_col,
            has_line_source,
            coefj, diag, xtau, ch, nxt,
        )
        flux_total[i] = ft
        flux_cont[i] = fc

    return flux_total, flux_cont
