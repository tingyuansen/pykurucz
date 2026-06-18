"""
Line opacity computation (TRANSP) for removing fort.9/fort.29 dependency.

This module computes line opacity at line center (TRANSP) from first principles,
following the Fortran XLINOP subroutine implementation.

Key formula from synthe.for line 692:
    KAPCEN = KAPPA0 * VOIGT(0., ADAMP)

Where:
    KAPPA0 = CGF * XNFDOP(NELION) * BOLT
    CGF = (0.026538/1.77245) * GF / FREQ  (from rgfall.for line 267)
    XNFDOP = XNFPEL / (RHO * DOPPLE) (population per unit mass per Doppler width)
    BOLT = exp(-ELO * HCKT) (Boltzmann factor)
    ADAMP = (GAMMAR + GAMMAS*XNE + GAMMAW*TXNXN) / DOPPLE
"""

from __future__ import annotations

import math
import os
import time
from typing import TYPE_CHECKING, Optional, Tuple, Dict
import logging
import numpy as np

from .profiles import voigt_profile
from . import tables

from numba import jit, prange

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..io.atmosphere import AtmosphereModel
    from ..io.lines.atomic import LineCatalog
    from ..physics.populations import Populations

# Constants matching Fortran
C_LIGHT_CM = 2.99792458e10  # cm/s
C_LIGHT_KM = 299792.458  # km/s
C_LIGHT_NM = 2.99792458e17  # nm/s (for frequency calculation)
H_PLANCK = 6.62607015e-27  # erg * s
K_BOLTZ = 1.380649e-16  # erg / K

# CGF conversion constants from rgfall.for line 267
CGF_CONSTANT = 0.026538 / 1.77245  # Factor for converting GF to CONGF
# Fortran synthe.for PARAMETER: MAXPROF=1000000
MAX_PROFILE_STEPS = 1_000_000


def _voigt_h_at_zero(adamp: np.ndarray, h0tab: np.ndarray, h1tab: np.ndarray, h2tab: np.ndarray) -> np.ndarray:
    """Vectorized Voigt H(a, 0) using the same piecewise polynomial as the wing kernel."""
    h0_0 = float(h0tab[0])
    h1_0 = float(h1tab[0])
    h2_0 = float(h2tab[0])
    h0v = h0_0
    h1v = h1_0 + h0v * 1.12838
    h2v = h2_0 + h1v * 1.12838 - h0v
    h3v = (1.0 - h2_0) * 0.37613 + h2v * 1.12838
    h4v = (3.0 * h3v - h1v) * 0.37613
    a_2d = adamp
    h_low = (h2_0 * a_2d + h1_0) * a_2d + h0_0
    poly_a_mid = (((h4v * a_2d + h3v) * a_2d + h2v) * a_2d + h1v) * a_2d + h0v
    poly_b_mid = ((-0.122727278 * a_2d + 0.532770573) * a_2d - 0.96284325) * a_2d + 0.979895032
    h_mid = poly_a_mid * poly_b_mid
    aa_2d = a_2d * a_2d
    u_2d = aa_2d * 1.4142
    safe_u_2d = np.maximum(u_2d, 1e-40)
    h_high_base = a_2d * 0.79788 / safe_u_2d
    aau_2d = aa_2d / safe_u_2d
    h_high = np.where(
        a_2d <= 100.0,
        ((aau_2d * aau_2d * 3.0 - aa_2d) / np.maximum(safe_u_2d * safe_u_2d, 1e-40) + 1.0) * h_high_base,
        h_high_base,
    )
    voigt_c = np.where(
        a_2d < 0.2,
        h_low,
        np.where((a_2d > 1.4) | (a_2d > 3.2), h_high, h_mid),
    )
    return np.maximum(voigt_c, 1e-30)


# Shared Voigt profile — single canonical JIT-compiled implementation
from synthe_py.physics.voigt_jit import voigt_profile_jit as _voigt_profile_jit


@jit(nopython=True, cache=True)
def _process_asynth_wing_pair_nb(
    asynth,
    wavelength_grid,
    depth_idx,
    line_idx,
    line_wavelength,
    center_idx,
    kappa0,
    adamp,
    doppler_width,
    stim_factor,
    kapmin_ref,
    wcon,
    wtail,
    use_cutoff,
    resolu,
    max_profile_steps,
    h0tab,
    h1tab,
    h2tab,
) -> None:
    """Accumulate Voigt wings for one (line, depth) pair into asynth[depth_idx, :]."""
    n_wavelengths = wavelength_grid.size

    if doppler_width <= 0.0:
        return

    dopple = doppler_width / line_wavelength if line_wavelength > 0.0 else 1e-10
    n10dop = int(10.0 * dopple * resolu)

    red_active = True
    blue_active = True
    offset = 1

    dvoigt = 1.0 / (dopple * resolu) if dopple > 0.0 else 1.0

    nstep_cutoff = n10dop
    profile_at_n10dop = 0.0
    tabstep = 200.0 * dvoigt
    tabi = 0.5
    for nstep in range(1, n10dop + 1):
        if adamp < 0.2:
            tabi += tabstep
            idx = int(tabi)
            if idx < 0:
                idx = 0
            x_step = float(nstep) * dvoigt
            if x_step > 10.0:
                profile_val = kappa0 * (0.5642 * adamp / (x_step * x_step))
            else:
                if idx >= h0tab.size:
                    idx = h0tab.size - 1
                profile_val = kappa0 * (h0tab[idx] + adamp * h1tab[idx])
        else:
            x_step = float(nstep) * dvoigt
            voigt_val = _voigt_profile_jit(x_step, adamp, h0tab, h1tab, h2tab)
            profile_val = kappa0 * voigt_val
        if nstep == n10dop:
            profile_at_n10dop = profile_val
        if use_cutoff and profile_val < kapmin_ref:
            nstep_cutoff = nstep
            break
    else:
        nstep_cutoff = -1

    if nstep_cutoff != -1:
        maxstep = nstep_cutoff
        use_far_wing = False
        x_far = 0.0
    else:
        use_far_wing = True
        if n10dop > 0 and profile_at_n10dop > 0.0:
            x_far = profile_at_n10dop * float(n10dop) ** 2
            if kapmin_ref > 0.0:
                maxstep = int(np.sqrt(x_far / kapmin_ref) + 1.0)
            elif kapmin_ref == 0.0:
                maxstep = max_profile_steps
            else:
                maxstep = 0
        else:
            x_far = 0.0
            maxstep = 0
        if maxstep > max_profile_steps:
            maxstep = max_profile_steps

    tabi_offset = 0.5
    while offset <= maxstep and (red_active or blue_active):
        if use_far_wing and offset > n10dop:
            profile_val = x_far / float(offset) ** 2
        else:
            if adamp < 0.2:
                tabi_offset += tabstep
                idx = int(tabi_offset)
                if idx < 0:
                    idx = 0
                x_offset = float(offset) * dvoigt
                if x_offset > 10.0:
                    profile_val = kappa0 * (0.5642 * adamp / (x_offset * x_offset))
                else:
                    if idx >= h0tab.size:
                        idx = h0tab.size - 1
                    profile_val = kappa0 * (h0tab[idx] + adamp * h1tab[idx])
            else:
                x_offset = float(offset) * dvoigt
                voigt_val = _voigt_profile_jit(x_offset, adamp, h0tab, h1tab, h2tab)
                profile_val = kappa0 * voigt_val
        profile_val = profile_val * stim_factor

        if profile_val == 0.0:
            break

        if red_active:
            idx = center_idx + offset
            if idx < 0:
                pass
            elif idx >= n_wavelengths:
                red_active = False
            else:
                wave = wavelength_grid[idx]
                skip_red = wcon > 0.0 and wave < wcon
                if not skip_red:
                    value_red = profile_val
                    if wtail > 0.0 and wcon > 0.0 and wave < wtail:
                        taper = (wave - wcon) / max(wtail - wcon, 1e-10)
                        value_red = value_red * taper
                    asynth[depth_idx, idx] += value_red

        if blue_active:
            idx = center_idx - offset
            if idx < 0:
                blue_active = False
            elif idx >= n_wavelengths:
                pass
            else:
                wave = wavelength_grid[idx]
                skip_blue = wcon > 0.0 and wave < wcon
                if not skip_blue:
                    value_blue = profile_val
                    if wtail > 0.0 and wcon > 0.0 and wave < wtail:
                        taper = (wave - wcon) / max(wtail - wcon, 1e-10)
                        value_blue = value_blue * taper
                    asynth[depth_idx, idx] += value_blue

        offset += 1


@jit(nopython=True, parallel=True, cache=True)
def _compute_asynth_wings_sparse_kernel(
    asynth,
    wavelength_grid,
    pair_line_idx,
    depth_starts,
    line_wavelengths,
    line_indices,
    kappa0_pair,
    adamp_pair,
    doppler_width_pair,
    kapmin_ref_pair,
    wcon_pair,
    wtail_pair,
    use_cutoff,
    resolu,
    max_profile_steps,
    h0tab,
    h1tab,
    h2tab,
    n_depths,
) -> None:
    """ASYNTH wings over compact (line, depth) pairs grouped by depth."""
    use_wcon = wcon_pair.size > 0
    use_kapmin = kapmin_ref_pair.size > 0

    for depth_idx in prange(n_depths):
        pi_start = int(depth_starts[depth_idx])
        pi_end = int(depth_starts[depth_idx + 1])
        for pi in range(pi_start, pi_end):
            line_idx = int(pair_line_idx[pi])
            center_idx = int(line_indices[line_idx])
            kappa0 = kappa0_pair[pi]
            adamp = adamp_pair[pi]
            doppler_width = doppler_width_pair[pi]
            kapmin_ref = kapmin_ref_pair[pi] if use_kapmin else 0.0

            wcon = -1.0
            wtail = -1.0
            if use_wcon:
                wcon_val = wcon_pair[pi]
                if wcon_val > 0.0:
                    wcon = wcon_val
                    if pi < wtail_pair.size:
                        wtail_val = wtail_pair[pi]
                        if wtail_val > 0.0:
                            wtail = wtail_val

            _process_asynth_wing_pair_nb(
                asynth,
                wavelength_grid,
                depth_idx,
                line_idx,
                line_wavelengths[line_idx],
                center_idx,
                kappa0,
                adamp,
                doppler_width,
                1.0,
                kapmin_ref,
                wcon,
                wtail,
                use_cutoff,
                resolu,
                max_profile_steps,
                h0tab,
                h1tab,
                h2tab,
            )


@jit(
    nopython=True,
    parallel=True,
    cache=True,
    fastmath=False,
)  # fastmath=False for bitwise reproducibility
def _compute_transp_numba_kernel(
    transp: np.ndarray,
    valid_mask: np.ndarray,
    process_mask: np.ndarray,
    element_idx: np.ndarray,
    ion_stage: np.ndarray,
    line_type: np.ndarray,
    wavelength: np.ndarray,
    gf: np.ndarray,
    cgf: np.ndarray,
    gamma_rad: np.ndarray,
    gamma_stark: np.ndarray,
    gamma_vdw: np.ndarray,
    center_indices: np.ndarray,
    center_indices_full: np.ndarray,
    boltzmann_factor: np.ndarray,
    population_per_ion: np.ndarray,
    doppler_per_ion: np.ndarray,
    mass_density: np.ndarray,
    electron_density: np.ndarray,
    txnxn: np.ndarray,
    continuum_absorption: np.ndarray,
    continuum_absorption_full: np.ndarray,
    n_wavelengths: int,
    cutoff: float,
    microturb_kms: float,
    c_light_km: float,
    h0tab: np.ndarray,
    h1tab: np.ndarray,
    h2tab: np.ndarray,
) -> None:
    """JIT-compiled TRANSP kernel. Matches Python compute_transp logic exactly."""
    n_lines = transp.shape[0]
    n_depths = transp.shape[1]
    n_elements = population_per_ion.shape[2]
    max_ion_stage = population_per_ion.shape[1]
    use_full_kapmin = continuum_absorption_full.shape[0] > 0 and continuum_absorption_full.shape[1] > 0
    micro = microturb_kms / c_light_km if microturb_kms > 0.0 else 0.0

    for line_idx in prange(n_lines):
        if not process_mask[line_idx]:
            continue

        elem_idx = element_idx[line_idx]
        if elem_idx < 0 or elem_idx >= n_elements:
            continue

        nelion = ion_stage[line_idx]
        if nelion <= 0 or nelion > max_ion_stage:
            continue

        line_wavelength = wavelength[line_idx]
        gf_linear = gf[line_idx]
        cgf_val = cgf[line_idx]
        gamma_rad_val = gamma_rad[line_idx]
        gamma_stark_val = gamma_stark[line_idx]
        gamma_vdw_val = gamma_vdw[line_idx]
        line_type_val = line_type[line_idx]
        center_idx = center_indices[line_idx]
        clamped_idx = max(0, min(center_idx, n_wavelengths - 1))

        for depth_idx in range(n_depths):
            pop_val = population_per_ion[depth_idx, nelion - 1, elem_idx]
            dop_val = doppler_per_ion[depth_idx, nelion - 1, elem_idx]
            if micro > 0.0:
                dop_val = np.sqrt(dop_val * dop_val + micro * micro)

            if pop_val <= 0.0 or dop_val <= 0.0:
                continue

            rho = mass_density[depth_idx]
            if rho <= 0.0:
                continue

            xnfdop = pop_val / (rho * dop_val)
            doppler_width = dop_val * line_wavelength
            boltz = boltzmann_factor[depth_idx, line_idx]

            if line_type_val == 1:
                kappa0_pre_boltz = gamma_vdw_val * gf_linear * (pop_val / rho)
            else:
                kappa0_pre_boltz = cgf_val * xnfdop

            if use_full_kapmin:
                full_idx = center_indices_full[line_idx]
                full_idx = max(0, min(full_idx, continuum_absorption_full.shape[1] - 1))
                kapmin = continuum_absorption_full[depth_idx, full_idx] * cutoff
            else:
                kapmin = continuum_absorption[depth_idx, clamped_idx] * cutoff

            if kappa0_pre_boltz < kapmin:
                continue

            post_candidate = kappa0_pre_boltz * boltz
            if post_candidate < kapmin:
                continue

            kappa0 = post_candidate
            if kappa0 <= 0.0:
                continue

            xne = electron_density[depth_idx]
            txnxn_val = txnxn[depth_idx]
            dopple = doppler_width / line_wavelength if line_wavelength > 0 else 1e-6

            if doppler_width > 0 and dopple > 0:
                gamma_total = gamma_rad_val + gamma_stark_val * xne + gamma_vdw_val * txnxn_val
                adamp = gamma_total / dopple
            else:
                adamp = 0.0

            if adamp >= 0.0 and kappa0 > 0.0:
                if line_type_val == 1:
                    kapcen = kappa0
                else:
                    if adamp < 0.2:
                        kapcen = kappa0 * (1.0 - 1.128 * adamp)
                    else:
                        voigt_center = _voigt_profile_jit(0.0, adamp, h0tab, h1tab, h2tab)
                        kapcen = kappa0 * voigt_center

                transp[line_idx, depth_idx] = kapcen
                valid_mask[line_idx, depth_idx] = True


@jit(
    nopython=True, parallel=True, cache=True
)
def _compute_asynth_wings_kernel(
    asynth: np.ndarray,
    wavelength_grid: np.ndarray,
    transp: np.ndarray,
    valid_mask: np.ndarray,
    line_wavelengths: np.ndarray,
    line_indices: np.ndarray,
    line_types: np.ndarray,
    stim_factors: np.ndarray,
    kappa0_values: np.ndarray,
    adamp_values: np.ndarray,
    doppler_widths: np.ndarray,
    gamma_rad_values: np.ndarray,
    gamma_stark_values: np.ndarray,
    gamma_vdw_values: np.ndarray,
    kapmin_ref_values: np.ndarray,
    continuum_absorption: np.ndarray,
    wcon_values: np.ndarray,
    wtail_values: np.ndarray,
    cutoff: float,
    max_profile_steps: int,
    h0tab: np.ndarray,
    h1tab: np.ndarray,
    h2tab: np.ndarray,
) -> None:
    """JIT-compiled kernel for computing ASYNTH wing contributions.

    CRITICAL FIX (Dec 2025): Match Fortran N10DOP logic exactly.
    Fortran synthe.for line 311: N10DOP = 10 * (DOPPLE * RESOLU)
    If N10DOP = 0 (which happens when DOPPLE*RESOLU < 0.1), NO wings are computed.
    This is critical for high-resolution spectra where Doppler widths are << grid spacing.

    Parallel strategy (May 2026): DEPTH-OUTER / LINE-INNER loop inversion.
    Outer prange over depths (ONE dispatch of ~80 tasks), inner sequential loop
    over all lines per depth.  Each prange worker owns asynth[depth_idx, :],
    so no write races.  Eliminates ~60k prange dispatches of the old approach
    and gives much better cache locality (each thread works on ~10 depth rows
    = ~2.3 MB, fits in L2).  Lines within each depth are processed in original
    order, preserving exact accumulation parity.
    """
    n_lines = transp.shape[0]
    n_depths = transp.shape[1]
    n_wavelengths = wavelength_grid.size

    use_cutoff = continuum_absorption.size > 0
    use_wcon = wcon_values.size > 0

    # Compute RESOLU from wavelength grid (matches Fortran)
    # RESOLU = 1 / (ratio - 1) where ratio = wavelength[i+1] / wavelength[i]
    resolu = 300000.0  # Default fallback
    if n_wavelengths > 1:
        ratio = wavelength_grid[1] / wavelength_grid[0]
        if ratio > 1.0:
            resolu = 1.0 / (ratio - 1.0)

    # Precompute depth-independent line masks to avoid redundant checks
    # inside the depth prange.  Only type-0 lines within reachable range
    # are processed.
    line_active = np.empty(n_lines, dtype=np.bool_)
    for li in range(n_lines):
        ci = line_indices[li]
        line_active[li] = (
            line_types[li] == 0
            and ci >= -max_profile_steps
            and ci <= n_wavelengths - 1 + max_profile_steps
        )

    for depth_idx in prange(n_depths):  # ONE prange dispatch for all depths
        for line_idx in range(n_lines):  # Sequential over lines — preserves accumulation order
            if not line_active[line_idx]:
                continue

            if not valid_mask[line_idx, depth_idx]:
                continue

            transp_val = transp[line_idx, depth_idx]
            if transp_val <= 0.0:
                continue

            line_wavelength = line_wavelengths[line_idx]
            center_idx = line_indices[line_idx]
            kappa0 = kappa0_values[line_idx, depth_idx]
            adamp = adamp_values[line_idx, depth_idx]
            doppler_width = doppler_widths[line_idx, depth_idx]
            stim_factor = stim_factors[line_idx, depth_idx]

            if doppler_width <= 0.0:
                continue

            # CRITICAL FIX: Compute N10DOP to match Fortran behavior exactly
            # Fortran synthe.for line 311: N10DOP = 10 * (DOPPLE * RESOLU)
            # DOPPLE is dimensionless: doppler_width / line_wavelength
            dopple = doppler_width / line_wavelength if line_wavelength > 0.0 else 1e-10
            n10dop = int(10.0 * dopple * resolu)

            # Get WCON/WTAIL for this line/depth (if available)
            wcon = -1.0  # Use -1.0 as sentinel for "not set"
            wtail = -1.0
            if use_wcon:
                idx_wcon = line_idx * n_depths + depth_idx
                if idx_wcon < wcon_values.size:
                    wcon_val = wcon_values[idx_wcon]
                    if wcon_val > 0.0:
                        wcon = wcon_val
                        if idx_wcon < wtail_values.size:
                            wtail_val = wtail_values[idx_wcon]
                            if wtail_val > 0.0:
                                wtail = wtail_val

            # Wing contributions (center contributions are added separately)
            red_active = True
            blue_active = True
            offset = 1

            # For MAXSTEP estimation, use depth-specific KAPMIN at line center.
            kapmin_ref = kapmin_ref_values[line_idx, depth_idx] if use_cutoff else 0.0

            # Pre-compute PROFILE array (matching Fortran's PROFILE(NSTEP))
            dvoigt = 1.0 / (dopple * resolu) if dopple > 0 else 1.0

            # Phase 1: Near-wing profile with KAPMIN check at line center
            nstep_cutoff = n10dop  # Max near-wing step before cutoff
            profile_at_n10dop = 0.0
            vsteps = 200.0
            tabstep = vsteps * dvoigt
            tabi = 0.5  # 0-based indexing (Fortran uses 1.5 for 1-based arrays)
            for nstep in range(1, n10dop + 1):
                if adamp < 0.2:
                    # Match Fortran's incremental TABI update to preserve rounding behavior.
                    tabi += tabstep
                    idx = int(tabi)
                    if idx < 0:
                        idx = 0
                    x_step = float(nstep) * dvoigt
                    if x_step > 10.0:
                        profile_val = kappa0 * (0.5642 * adamp / (x_step * x_step))
                    else:
                        if idx >= h0tab.size:
                            idx = h0tab.size - 1
                        profile_val = kappa0 * (h0tab[idx] + adamp * h1tab[idx])
                else:
                    x_step = float(nstep) * dvoigt
                    voigt_val = _voigt_profile_jit(x_step, adamp, h0tab, h1tab, h2tab)
                    profile_val = kappa0 * voigt_val  # No stim_factor here
                if nstep == n10dop:
                    profile_at_n10dop = profile_val
                # Check against KAPMIN at LINE CENTER (kapmin_ref)
                if use_cutoff and profile_val < kapmin_ref:
                    nstep_cutoff = nstep
                    break
            else:
                # Near-wing completed without cutoff - compute far-wing X
                nstep_cutoff = -1  # Flag: no early cutoff

            # Phase 2: Far-wing setup
            #
            # If near-wing cutoff triggers, Fortran skips far wings entirely.
            if nstep_cutoff != -1:
                maxstep = nstep_cutoff
                use_far_wing = False
                x_far = 0.0
            else:
                # Fortran far-wing (synthe.for lines 303-305):
                #   X = PROFILE(N10DOP) * FLOAT(N10DOP)**2
                #   MAXSTEP = SQRT(X / KAPMIN) + 1.
                #   MAXSTEP = MIN(MAXSTEP, MAXPROF)
                use_far_wing = True
                if n10dop > 0 and profile_at_n10dop > 0.0:
                    x_far = profile_at_n10dop * float(n10dop) ** 2
                    if kapmin_ref > 0.0:
                        maxstep = int(np.sqrt(x_far / kapmin_ref) + 1.0)
                    elif kapmin_ref == 0.0:
                        maxstep = max_profile_steps
                    else:
                        # NaN: int(sqrt(x/NaN)+1) = int(NaN) = 0 in Numba/Fortran
                        maxstep = 0
                else:
                    x_far = 0.0
                    maxstep = 0
                if maxstep > max_profile_steps:
                    maxstep = max_profile_steps

            # Phase 3: Apply profile to both red and blue wings
            tabi_offset = 0.5  # 0-based indexing (Fortran uses 1.5 for 1-based arrays)
            while offset <= maxstep and (red_active or blue_active):
                # Compute profile value for this offset (Fortran near-wing vs far-wing)
                if use_far_wing and offset > n10dop:
                    profile_val = x_far / float(offset) ** 2
                else:
                    if adamp < 0.2:
                        tabi_offset += tabstep
                        idx = int(tabi_offset)
                        if idx < 0:
                            idx = 0
                        x_offset = float(offset) * dvoigt
                        if x_offset > 10.0:
                            profile_val = kappa0 * (
                                0.5642 * adamp / (x_offset * x_offset)
                            )
                        else:
                            if idx >= h0tab.size:
                                idx = h0tab.size - 1
                            profile_val = kappa0 * (h0tab[idx] + adamp * h1tab[idx])
                    else:
                        x_offset = float(offset) * dvoigt
                        voigt_val = _voigt_profile_jit(
                            x_offset, adamp, h0tab, h1tab, h2tab
                        )
                        profile_val = kappa0 * voigt_val
                profile_val = profile_val * stim_factor

                # Early exit when profile has underflowed to zero
                if profile_val == 0.0:
                    break

                # Process red wing
                if red_active:
                    idx = center_idx + offset
                    if idx < 0:
                        pass  # Below grid, will reach it as offset increases
                    elif idx >= n_wavelengths:
                        red_active = False
                    else:
                        wave = wavelength_grid[idx]
                        skip_red = wcon > 0.0 and wave < wcon
                        if not skip_red:
                            value_red = profile_val

                            # Taper between WCON and WTAIL
                            if wtail > 0.0 and wcon > 0.0 and wave < wtail:
                                taper = (wave - wcon) / max(wtail - wcon, 1e-10)
                                value_red = value_red * taper

                            asynth[depth_idx, idx] += value_red

                # Process blue wing
                if blue_active:
                    idx = center_idx - offset
                    if idx < 0:
                        blue_active = False
                    elif idx >= n_wavelengths:
                        pass  # Center is above grid; wait for offset to bring idx in-range.
                    else:
                        wave = wavelength_grid[idx]
                        skip_blue = wcon > 0.0 and wave < wcon
                        if not skip_blue:
                            value_blue = profile_val

                            # Taper between WCON and WTAIL
                            if wtail > 0.0 and wcon > 0.0 and wave < wtail:
                                taper = (wave - wcon) / max(wtail - wcon, 1e-10)
                                value_blue = value_blue * taper

                            asynth[depth_idx, idx] += value_blue

                offset += 1


def compute_transp(
    catalog: "LineCatalog",
    populations: "Populations",
    atmosphere: "AtmosphereModel",
    cutoff: float = 1e-3,
    continuum_absorption: Optional[np.ndarray] = None,
    wavelength_grid: Optional[np.ndarray] = None,
    continuum_absorption_full: Optional[np.ndarray] = None,
    wavelength_grid_full: Optional[np.ndarray] = None,
    microturb_kms: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute TRANSP (line opacity at line center) for all lines and depths.

    This is the core computation that replaces fort.9/fort.29 dependency.

    Parameters
    ----------
    catalog:
        Line catalog containing line properties (gf, wavelength, excitation, etc.)
    populations:
        Pre-computed populations and Doppler widths for all depths
    atmosphere:
        Atmosphere model with temperature, electron density, etc.
    cutoff:
        Opacity cutoff factor (lines below this are ignored)
    continuum_absorption:
        Continuum absorption array, shape (n_depths, n_wavelengths).
        Used for KAPMIN = CONTINUUM * CUTOFF check (matches Fortran exactly).
    wavelength_grid:
        Wavelength grid for mapping lines to grid indices.
        Required when continuum_absorption is provided.

    Returns
    -------
    transp:
        Array of shape (n_lines, n_depths) containing line opacity at line center
    valid_mask:
        Boolean array of shape (n_lines, n_depths) indicating which lines/depths are valid
    line_indices:
        Array of line indices that contribute (for wavelength grid mapping)

    Notes
    -----
    TRANSP computation follows synthe.for XLINOP:
    1. KAPPA0 = gf * (population / doppler_width) * exp(-E/kT)
    2. ADAMP = (gamma_rad + gamma_stark*XNE + gamma_vdw*TXNXN) / doppler_width
    3. KAPCEN = KAPPA0 * VOIGT(0, ADAMP)
    """
    n_lines = len(catalog.records)
    n_depths = atmosphere.layers

    logger.info(f"Computing TRANSP for {n_lines:,} lines across {n_depths} depths...")

    # Initialize output arrays
    transp = np.zeros((n_lines, n_depths), dtype=np.float64)
    valid_mask = np.zeros((n_lines, n_depths), dtype=bool)

    # Progress logging in this hot loop can be expensive. Keep it opt-in.
    transp_progress = os.getenv("PY_TRANSP_PROGRESS", "0") == "1"
    log_interval = max(1, n_lines // 20) if transp_progress else n_lines + 1

    # Pre-compute center indices for all lines
    # This is used for KAPMIN = CONTINUUM(center_idx) * CUTOFF (matches Fortran exactly)
    # Fortran has no fallback - KAPMIN always uses CONTINUUM * CUTOFF
    if continuum_absorption is None or wavelength_grid is None:
        raise ValueError(
            "continuum_absorption and wavelength_grid are required for compute_transp. "
            "Fortran always uses KAPMIN = CONTINUUM * CUTOFF with no fallback."
        )

    from ..engine.opacity import _nearest_grid_indices

    index_wavelength = (
        catalog.index_wavelength
        if hasattr(catalog, "index_wavelength")
        else catalog.wavelength
    )
    center_indices = _nearest_grid_indices(wavelength_grid, index_wavelength)
    center_indices_full = None
    if continuum_absorption_full is not None and wavelength_grid_full is not None:
        center_indices_full = _nearest_grid_indices(
            wavelength_grid_full, index_wavelength
        )
    n_wavelengths = len(wavelength_grid)
    logger.info(f"Using dynamic KAPMIN = CONTINUUM * CUTOFF (Fortran-matching)")

    # Population data comes from NPZ (computed by pops_exact in convert_atm_to_npz.py)

    # Compute TXNXN if not available
    xnf_h = atmosphere.xnf_h if atmosphere.xnf_h is not None else np.zeros(n_depths)
    xnf_he1 = (
        atmosphere.xnf_he1 if atmosphere.xnf_he1 is not None else np.zeros(n_depths)
    )
    xnf_h2 = atmosphere.xnf_h2 if atmosphere.xnf_h2 is not None else np.zeros(n_depths)

    # Cache population computations per element to avoid redundant calculations
    # Format: {element: (pop_densities, dop_velocity)}
    population_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    include_h_lines = os.getenv("PY_INCLUDE_H_LINES") == "1"

    use_numba = os.getenv("PY_USE_NUMBA_TRANSP", "1") != "0"
    timing_ab = os.getenv("PY_TRANSP_TIMING_AB", "0") == "1"

    def _run_numba_transp() -> None:
        """Extract arrays and run Numba TRANSP kernel."""
        from ..engine.opacity import _element_atomic_number

        process_mask = np.zeros(n_lines, dtype=np.bool_)
        element_idx = np.full(n_lines, -1, dtype=np.int64)
        wavelength = np.zeros(n_lines, dtype=np.float64)
        gf = np.zeros(n_lines, dtype=np.float64)
        cgf = np.zeros(n_lines, dtype=np.float64)
        ion_stage = np.zeros(n_lines, dtype=np.int64)
        line_type_arr = np.zeros(n_lines, dtype=np.int64)
        gamma_rad = np.zeros(n_lines, dtype=np.float64)
        gamma_stark = np.zeros(n_lines, dtype=np.float64)
        gamma_vdw = np.zeros(n_lines, dtype=np.float64)

        for line_idx in range(n_lines):
            record = catalog.records[line_idx]
            elem_str = str(record.element).strip().upper()
            is_h = elem_str in {"H", "HI", "H I"}
            line_type_code = int(getattr(record, "line_type", 0) or 0)
            n_lower_abs = abs(int(getattr(record, "n_lower", 0) or 0))
            n_upper_abs = abs(int(getattr(record, "n_upper", 0) or 0))
            nb_sum = n_lower_abs + n_upper_abs
            routes_to_fort12 = bool(
                line_type_code != 2
                and line_type_code != 1
                and line_type_code <= 3
                and nb_sum == 0
            )
            if not include_h_lines and (line_type_code == -1 or (is_h and record.ion_stage == 1)):
                continue
            if not routes_to_fort12:
                continue
            anum = _element_atomic_number(record.element)
            if anum is None or atmosphere.population_per_ion is None:
                continue
            elem_idx = anum - 1
            if elem_idx >= atmosphere.population_per_ion.shape[2]:
                continue

            process_mask[line_idx] = True
            element_idx[line_idx] = elem_idx
            wavelength[line_idx] = float(
                index_wavelength[line_idx] if index_wavelength is not None else record.wavelength
            )
            gf[line_idx] = float(catalog.gf[line_idx])
            ion_stage[line_idx] = int(record.ion_stage)
            line_type_arr[line_idx] = line_type_code
            gamma_rad[line_idx] = float(catalog.gamma_rad[line_idx])
            gamma_stark[line_idx] = float(catalog.gamma_stark[line_idx])
            gamma_vdw[line_idx] = float(catalog.gamma_vdw[line_idx])

            freq_hz = C_LIGHT_NM / wavelength[line_idx]
            cgf_meta = None
            if record.metadata:
                cgf_meta = record.metadata.get("cgf")
            if cgf_meta is not None and cgf_meta > 0.0:
                cgf[line_idx] = float(cgf_meta)
            else:
                cgf[line_idx] = CGF_CONSTANT * gf[line_idx] / freq_hz

        boltzmann_factor = np.zeros((n_depths, n_lines), dtype=np.float64)
        for depth_idx in range(n_depths):
            state = populations.layers[depth_idx]
            boltzmann_factor[depth_idx, :] = state.boltzmann_factor

        pop_ion = np.asarray(atmosphere.population_per_ion, dtype=np.float64)
        dop_ion = np.asarray(atmosphere.doppler_per_ion, dtype=np.float64)
        mass_density = np.asarray(
            atmosphere.mass_density if atmosphere.mass_density is not None else np.ones(n_depths),
            dtype=np.float64,
        )
        electron_density = np.asarray(
            atmosphere.electron_density if atmosphere.electron_density is not None else np.zeros(n_depths),
            dtype=np.float64,
        )
        txnxn = np.zeros(n_depths, dtype=np.float64)
        for depth_idx in range(n_depths):
            txnxn[depth_idx] = populations.layers[depth_idx].txnxn

        cont_abs = np.asarray(continuum_absorption, dtype=np.float64)
        if continuum_absorption_full is not None and center_indices_full is not None:
            cont_abs_full = np.asarray(continuum_absorption_full, dtype=np.float64)
            center_full = np.asarray(center_indices_full, dtype=np.int64)
        else:
            cont_abs_full = np.zeros((0, 0), dtype=np.float64)
            center_full = np.zeros(n_lines, dtype=np.int64)

        voigt_tbl = tables.voigt_tables()
        h0tab = voigt_tbl.h0tab
        h1tab = voigt_tbl.h1tab
        h2tab = voigt_tbl.h2tab

        _compute_transp_numba_kernel(
            transp,
            valid_mask,
            process_mask,
            element_idx,
            ion_stage,
            line_type_arr,
            wavelength,
            gf,
            cgf,
            gamma_rad,
            gamma_stark,
            gamma_vdw,
            center_indices,
            center_full,
            boltzmann_factor,
            pop_ion,
            dop_ion,
            mass_density,
            electron_density,
            txnxn,
            cont_abs,
            cont_abs_full,
            n_wavelengths,
            cutoff,
            microturb_kms,
            C_LIGHT_KM,
            h0tab,
            h1tab,
            h2tab,
        )

    t_numba_start = time.perf_counter()
    _run_numba_transp()
    t_numba = time.perf_counter() - t_numba_start
    logger.info("Timing: TRANSP (Numba) in %.3fs", t_numba)
    logger.info(
        f"TRANSP computation complete: {np.sum(valid_mask):,} valid line-depth pairs"
    )
    if timing_ab:
        logger.info("TRANSP timing (Numba): %.3fs", t_numba)

    return transp, valid_mask, center_indices


def _compute_fortran_profile_steps(
    offset: int,
    kappa0: float,
    adamp: float,
    dopple: float,
    resolu: float,
    kapmin_ref: float,
    h0tab: np.ndarray,
    h1tab: np.ndarray,
    h2tab: np.ndarray,
    max_profile_steps: int,
) -> Tuple[Optional[float], int, int, Optional[float], bool]:
    """Compute per-offset profile using Fortran XLINOP steps (labels 320-323)."""
    offset_abs = abs(int(offset))
    if dopple <= 0.0:
        return None, 0, 0, None, False

    n10dop = int(10.0 * dopple * resolu)
    if n10dop <= 0:
        return None, n10dop, 0, None, False

    profile_at_offset = None
    profile_at_n10dop = None
    cutoff_hit = False

    dvoigt = 1.0 / (dopple * resolu)
    if adamp < 0.2:
        vsteps = 200.0
        tabstep = vsteps * dvoigt
        tabi = 0.5  # 0-based indexing (Fortran uses 1.5 for 1-based arrays)
        for nstep in range(1, n10dop + 1):
            tabi += tabstep
            idx_tab = int(tabi)
            if idx_tab < 0:
                idx_tab = 0
            x_step = float(nstep) * dvoigt
            if x_step > 10.0:
                profile = kappa0 * (0.5642 * adamp / (x_step * x_step))
            else:
                if idx_tab >= h0tab.size:
                    idx_tab = h0tab.size - 1
                profile = kappa0 * (h0tab[idx_tab] + adamp * h1tab[idx_tab])
            if nstep == offset_abs:
                profile_at_offset = profile
            if nstep == n10dop:
                profile_at_n10dop = profile
            if profile < kapmin_ref:
                cutoff_hit = True
                maxstep = nstep
                if offset_abs > maxstep:
                    return None, n10dop, maxstep, profile_at_n10dop, cutoff_hit
                return profile_at_offset, n10dop, maxstep, profile_at_n10dop, cutoff_hit
    else:
        for nstep in range(1, n10dop + 1):
            x_step = float(nstep) * dvoigt
            profile = kappa0 * _voigt_profile_jit(x_step, adamp, h0tab, h1tab, h2tab)
            if nstep == offset_abs:
                profile_at_offset = profile
            if nstep == n10dop:
                profile_at_n10dop = profile
            if profile < kapmin_ref:
                cutoff_hit = True
                maxstep = nstep
                if offset_abs > maxstep:
                    return None, n10dop, maxstep, profile_at_n10dop, cutoff_hit
                return profile_at_offset, n10dop, maxstep, profile_at_n10dop, cutoff_hit

    if profile_at_n10dop is None or kapmin_ref <= 0.0:
        return profile_at_offset, n10dop, 0, profile_at_n10dop, cutoff_hit

    x_far = profile_at_n10dop * float(n10dop) ** 2
    maxstep = int(np.sqrt(x_far / kapmin_ref) + 1.0)
    if maxstep > max_profile_steps:
        maxstep = max_profile_steps

    if offset_abs > n10dop:
        if offset_abs > maxstep or x_far <= 0.0:
            return None, n10dop, maxstep, profile_at_n10dop, cutoff_hit
        profile_at_offset = x_far / float(offset_abs) ** 2

    return profile_at_offset, n10dop, maxstep, profile_at_n10dop, cutoff_hit


def compute_asynth_from_transp(
    transp: np.ndarray,
    catalog: "LineCatalog",
    atmosphere: "AtmosphereModel",
    wavelength_grid: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    populations: Optional["Populations"] = None,
    cutoff: float = 1e-3,
    continuum_absorption: Optional[np.ndarray] = None,
    continuum_absorption_full: Optional[np.ndarray] = None,
    wavelength_grid_full: Optional[np.ndarray] = None,
    metal_tables: Optional["tables.MetalWingTables"] = None,
    grid_origin: Optional[float] = None,
) -> np.ndarray:
    """
    Compute ASYNTH from TRANSP using the stimulated emission correction.

    Formula from synthe.for line 368:
        ASYNTH(J) = TRANSP(J,I) * (1. - EXP(-FREQ*HKT(J)))

    CRITICAL: This function now includes wing contributions via Voigt profiles,
    matching Fortran's behavior where lines contribute to nearby wavelengths.

    Parameters
    ----------
    transp:
        Line opacity at line center, shape (n_lines, n_depths)
    catalog:
        Line catalog
    atmosphere:
        Atmosphere model
    wavelength_grid:
        Wavelength grid for output, shape (n_wavelengths,)
    valid_mask:
        Optional mask indicating valid lines/depths
    populations:
        Populations object (needed for computing damping and doppler widths)
    cutoff:
        Opacity cutoff factor for wing contributions (matches Fortran CUTOFF)
    continuum_absorption:
        Continuum absorption array, shape (n_depths, n_wavelengths).
        If None, cutoff check is skipped (wings extend to MAX_PROFILE_STEPS)

    Returns
    -------
    asynth:
        ASYNTH array, shape (n_depths, n_wavelengths)
    """
    n_wavelengths = wavelength_grid.size
    n_depths = atmosphere.layers

    # Initialize ASYNTH array
    asynth = np.zeros((n_depths, n_wavelengths), dtype=np.float64)

    # CRITICAL FIX: Match Fortran frequency calculation exactly
    # Fortran line 369: FREQ=2.99792458D17/WAVE (WAVE in nm, result in Hz)
    # C_LIGHT_NM = 2.99792458e17 nm/s = speed of light in nm/s
    # Frequency = C_LIGHT_NM / wavelength_nm (Hz)

    # Compute frequency grid
    freq_grid = C_LIGHT_NM / wavelength_grid  # Shape: (n_wavelengths,)

    # Compute HKT for each depth
    hkt = np.zeros(n_depths, dtype=np.float64)
    for depth_idx in range(n_depths):
        temp = atmosphere.temperature[depth_idx]
        if atmosphere.hckt is not None:
            # HKT = H_PLANCK / (K_BOLTZ * T) = hckt / T
            hkt[depth_idx] = H_PLANCK / (K_BOLTZ * max(temp, 1.0))
        else:
            hkt[depth_idx] = H_PLANCK / (K_BOLTZ * max(temp, 1.0))

    # Map lines to wavelength grid
    from ..engine.opacity import _nearest_grid_indices

    index_wavelength = (
        catalog.index_wavelength
        if hasattr(catalog, "index_wavelength")
        else catalog.wavelength
    )
    line_indices = _nearest_grid_indices(wavelength_grid, index_wavelength)

    # Raw (unclamped) indices for wing contributions so outside-center lines
    # still map to correct offset distances.
    def _nearest_grid_indices_raw(
        grid: np.ndarray, values: np.ndarray, origin_start: Optional[float] = None
    ) -> np.ndarray:
        if len(grid) < 2:
            return np.zeros(len(values), dtype=np.int64)
        ratio = grid[1] / grid[0]
        ratiolg = np.log(ratio)
        start_val = grid[0] if origin_start is None else origin_start
        ix_floor = int(np.floor(np.log(start_val) / ratiolg))
        wbegin = np.exp(ix_floor * ratiolg)
        if wbegin < start_val:
            ix_floor += 1
            wbegin = np.exp(ix_floor * ratiolg)
        with np.errstate(divide="ignore", invalid="ignore"):
            ix = np.rint(np.log(values / wbegin) / ratiolg).astype(np.int64)
        return ix

    line_indices_wing = _nearest_grid_indices_raw(
        wavelength_grid, index_wavelength, origin_start=grid_origin
    )
    if grid_origin is not None and wavelength_grid.size > 1:
        ratio = wavelength_grid[1] / wavelength_grid[0]
        ratiolg = np.log(ratio)
        ix_floor = int(np.floor(np.log(grid_origin) / ratiolg))
        wbegin = np.exp(ix_floor * ratiolg)
        if wbegin < grid_origin:
            ix_floor += 1
            wbegin = np.exp(ix_floor * ratiolg)
        grid_offset = int(np.rint(np.log(wavelength_grid[0] / wbegin) / ratiolg))
        line_indices_wing = line_indices_wing - grid_offset

    # Vectorized ASYNTH computation
    # Compute frequencies for all lines at once (matches Fortran line 369)
    line_freqs = C_LIGHT_NM / catalog.wavelength  # Shape: (n_lines,)

    # Fortran applies the stimulated emission factor after TRANSP is transposed
    # to each wavelength (synthe.for lines 439-443). Use grid frequency, not line centers.
    stim_grid = 1.0 - np.exp(-freq_grid[np.newaxis, :] * hkt[:, np.newaxis])

    # Kernel still expects stim_factors array; keep it as ones so wing profiles are unscaled here.
    stim_factors = np.ones((len(catalog.records), n_depths), dtype=np.float64)

    # Import needed functions
    from .profiles.voigt import voigt_profile
    from ..engine.opacity import MAX_PROFILE_STEPS, _element_atomic_number

    # Cache populations per element
    population_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    # Get continuum absorption for cutoff calculation (matches Fortran: KAPMIN = CONTINUUM * CUTOFF)
    # If not provided, skip cutoff check and extend wings to MAX_PROFILE_STEPS
    use_cutoff = continuum_absorption is not None and continuum_absorption.shape == (
        n_depths,
        n_wavelengths,
    )

    # ── Vectorized precomputation (replaces Python O(n_lines × n_depths) loops) ──
    n_lines = len(catalog.records)
    gamma_rad_array = np.asarray(catalog.gamma_rad, dtype=np.float64)
    gamma_stark_array = np.asarray(catalog.gamma_stark, dtype=np.float64)
    gamma_vdw_array = np.asarray(catalog.gamma_vdw, dtype=np.float64)
    line_types_array = (
        np.asarray(catalog.line_types, dtype=np.int8)
        if catalog.line_types is not None
        else np.zeros(n_lines, dtype=np.int8)
    )
    ion_stages_arr = np.asarray(catalog.ion_stages, dtype=np.int64)  # 1-based
    line_wavelengths_arr = np.asarray(catalog.wavelength, dtype=np.float64)

    # Build element → atomic-number mapping
    _atnum_cache: Dict[str, int] = {}
    elem_atnum_arr = np.zeros(n_lines, dtype=np.int64)
    for _i, _elem in enumerate(catalog.elements):
        _es = str(_elem)
        if _es not in _atnum_cache:
            _an = _element_atomic_number(_es)
            _atnum_cache[_es] = _an if _an is not None else 0
        elem_atnum_arr[_i] = _atnum_cache[_es]

    _pop_3d = atmosphere.population_per_ion  # (n_depths, n_ion, n_elem) or None
    _dop_3d = atmosphere.doppler_per_ion      # same shape
    n_elem_max = _pop_3d.shape[2] if _pop_3d is not None else 0
    n_ion_max  = _pop_3d.shape[1] if _pop_3d is not None else 0

    valid_lines = (
        (elem_atnum_arr > 0) & (elem_atnum_arr <= n_elem_max)
        & (ion_stages_arr >= 1) & (ion_stages_arr <= n_ion_max)
    )
    elem_idx_c = np.clip(elem_atnum_arr - 1, 0, max(n_elem_max - 1, 0))  # 0-based
    ion_idx_c  = np.clip(ion_stages_arr - 1, 0, max(n_ion_max  - 1, 0))  # 0-based

    voigt_tables = tables.voigt_tables()
    h0tab = voigt_tables.h0tab
    h1tab = voigt_tables.h1tab
    h2tab = voigt_tables.h2tab

    center_indices_full = None
    if continuum_absorption_full is not None and wavelength_grid_full is not None:
        center_indices_full = _nearest_grid_indices(
            wavelength_grid_full,
            (
                catalog.index_wavelength
                if hasattr(catalog, "index_wavelength")
                else catalog.wavelength
            ),
        )

    if populations is not None:
        txnxn_per_depth = np.array(
            [populations.layers[d].txnxn for d in range(n_depths)], dtype=np.float64
        )
    else:
        txnxn_per_depth = np.zeros(n_depths, dtype=np.float64)
    xne_arr_depth = np.asarray(atmosphere.electron_density, dtype=np.float64)

    max_profile_steps = int(MAX_PROFILE_STEPS)
    line_wavelengths_array = line_wavelengths_arr
    line_indices_array = np.asarray(line_indices_wing, dtype=np.int64)
    _vm = (
        valid_mask
        if valid_mask is not None
        else np.ones((n_lines, n_depths), dtype=np.bool_)
    )
    _use_cutoff_kernel = bool(use_cutoff)
    if n_wavelengths > 1:
        _ratio = wavelength_grid[1] / wavelength_grid[0]
        _resolu = 1.0 / (_ratio - 1.0) if _ratio > 1.0 else 300000.0
    else:
        _resolu = 300000.0

    _use_sparse = os.environ.get("SYNTHE_ASYNTH_SPARSE", "1") != "0"
    pair_line: Optional[np.ndarray] = None
    depth_starts: Optional[np.ndarray] = None
    kappa0_pair = np.empty(0, dtype=np.float64)
    adamp_pair = np.empty(0, dtype=np.float64)
    doppler_width_pair = np.empty(0, dtype=np.float64)
    kapmin_ref_pair = np.empty(0, dtype=np.float64)
    wcon_pair = np.empty(0, dtype=np.float64)
    wtail_pair = np.empty(0, dtype=np.float64)

    if _use_sparse:
        _line_active = (
            (line_types_array == 0)
            & (line_indices_array >= -max_profile_steps)
            & (line_indices_array <= n_wavelengths - 1 + max_profile_steps)
        )
        _pair_mask = _vm & (transp > 0.0) & _line_active[:, np.newaxis]
        _li, _di = np.nonzero(_pair_mask)
        if _li.size > 0:
            _order = np.argsort(_di, kind="mergesort")
            pair_line = _li[_order].astype(np.int32, copy=False)
            pair_depth = _di[_order].astype(np.int32, copy=False)
            depth_starts = np.searchsorted(
                pair_depth,
                np.arange(n_depths + 1, dtype=np.int32),
                side="left",
            ).astype(np.int64, copy=False)

            _t_pre = time.perf_counter()
            if _pop_3d is not None and n_elem_max > 0:
                dop_val = _dop_3d[pair_depth, ion_idx_c[pair_line], elem_idx_c[pair_line]]
            else:
                dop_val = np.zeros(pair_line.size, dtype=np.float64)

            gamma_total = (
                gamma_rad_array[pair_line]
                + gamma_stark_array[pair_line] * xne_arr_depth[pair_depth]
                + gamma_vdw_array[pair_line] * txnxn_per_depth[pair_depth]
            )
            safe_dop = np.where(dop_val > 0.0, dop_val, 1.0)
            adamp_pair = np.maximum(
                np.where(dop_val > 0.0, gamma_total / safe_dop, 0.0),
                1e-12,
            )
            doppler_width_pair = dop_val * line_wavelengths_arr[pair_line]
            transp_pair = transp[pair_line, pair_depth]

            voigt_c = _voigt_h_at_zero(adamp_pair, h0tab, h1tab, h2tab)
            _auto_pair = line_types_array[pair_line] == 1
            kappa0_pair = np.where(
                _auto_pair,
                transp_pair,
                np.where(transp_pair > 0.0, transp_pair / voigt_c, 0.0),
            )

            _KAPMIN_FLOOR = 1e-8
            if use_cutoff:
                if center_indices_full is not None and continuum_absorption_full is not None:
                    _ci = np.clip(
                        center_indices_full[pair_line].astype(np.int64),
                        0,
                        continuum_absorption_full.shape[1] - 1,
                    )
                    _cont = continuum_absorption_full[pair_depth, _ci]
                else:
                    _ci = np.clip(line_indices_wing[pair_line].astype(np.int64), 0, n_wavelengths - 1)
                    _cont = continuum_absorption[pair_depth, _ci]
                kapmin_ref_pair = np.maximum(_cont * cutoff, _cont * _KAPMIN_FLOOR)
            else:
                kapmin_ref_pair = np.zeros(pair_line.size, dtype=np.float64)

            wcon_pair = np.full(pair_line.size, -1.0, dtype=np.float64)
            wtail_pair = np.full(pair_line.size, -1.0, dtype=np.float64)
            if metal_tables is not None and populations is not None:
                from ..engine.opacity import _compute_continuum_limits

                _max_nv = min(int(np.max(ion_stages_arr)) if n_lines > 0 else 6, 10)
                wcon_tbl = np.zeros((n_depths, _max_nv + 1), dtype=np.float64)
                wtail_tbl = np.zeros((n_depths, _max_nv + 1), dtype=np.float64)
                for _d in range(n_depths):
                    _state = populations.layers[_d]
                    for _nv in range(1, _max_nv + 1):
                        _wcon_v, _wtail_v = _compute_continuum_limits(
                            ncon=getattr(_state, "ncon", 0),
                            nelion=_nv,
                            nelionx=getattr(_state, "nelionx", 0),
                            emerge_val=getattr(_state, "emerge", 0.0),
                            emerge_h_val=getattr(_state, "emerge_h", 0.0),
                            metal_tables=metal_tables,
                            ifvac=1,
                        )
                        if _wcon_v is not None and _wcon_v > 0.0:
                            wcon_tbl[_d, _nv] = _wcon_v
                        if _wtail_v is not None and _wtail_v > 0.0:
                            wtail_tbl[_d, _nv] = _wtail_v
                _ion_cw = np.clip(ion_stages_arr[pair_line], 1, _max_nv)
                wcon_sel = wcon_tbl[pair_depth, _ion_cw]
                wtail_sel = wtail_tbl[pair_depth, _ion_cw]
                wcon_pair = np.where(wcon_sel > 0.0, wcon_sel, -1.0)
                wtail_pair = np.where(wtail_sel > 0.0, wtail_sel, -1.0)

            logger.info(
                "Timing: ASYNTH precompute compact (%d pairs) in %.3fs",
                int(pair_line.size),
                time.perf_counter() - _t_pre,
            )
        else:
            _use_sparse = False

    kappa0_array = np.zeros((n_lines, n_depths), dtype=np.float64)
    adamp_array = np.zeros((n_lines, n_depths), dtype=np.float64)
    doppler_widths_array = np.zeros((n_lines, n_depths), dtype=np.float64)
    kapmin_ref_array = np.zeros((n_lines, n_depths), dtype=np.float64)
    wcon_array = np.zeros(n_lines * n_depths, dtype=np.float64)
    wtail_array = np.zeros(n_lines * n_depths, dtype=np.float64)

    if not _use_sparse:
        if _pop_3d is not None and n_elem_max > 0:
            _d_arange = np.arange(n_depths)
            pop_2d = _pop_3d[_d_arange[:, None], ion_idx_c[None, :], elem_idx_c[None, :]]
            dop_2d = _dop_3d[_d_arange[:, None], ion_idx_c[None, :], elem_idx_c[None, :]]
            pop_2d[:, ~valid_lines] = 0.0
            dop_2d[:, ~valid_lines] = 0.0
        else:
            pop_2d = np.zeros((n_depths, n_lines), dtype=np.float64)
            dop_2d = np.zeros((n_depths, n_lines), dtype=np.float64)

        doppler_widths_array = (dop_2d * line_wavelengths_arr[None, :]).T
        gamma_total_2d = (
            gamma_rad_array[:, None]
            + gamma_stark_array[:, None] * xne_arr_depth[None, :]
            + gamma_vdw_array[:, None] * txnxn_per_depth[None, :]
        )
        _safe_dop = np.where(dop_2d.T > 0, dop_2d.T, 1.0)
        adamp_array = np.where(
            (dop_2d.T > 0) & (line_wavelengths_arr[:, None] > 0),
            gamma_total_2d / _safe_dop,
            0.0,
        )
        adamp_array = np.maximum(adamp_array, 1e-12)
        _invalid_pair = (pop_2d.T <= 0.0) | (dop_2d.T <= 0.0)
        adamp_array[_invalid_pair] = 0.0
        doppler_widths_array[_invalid_pair] = 0.0

        voigt_c = _voigt_h_at_zero(adamp_array.ravel(), h0tab, h1tab, h2tab).reshape(
            adamp_array.shape
        )
        _auto_mask = (line_types_array == 1)[:, None]
        kappa0_array = np.where(
            _auto_mask | _invalid_pair | (transp <= 0.0),
            np.where(_auto_mask & ~_invalid_pair & (transp > 0.0), transp, 0.0),
            transp / voigt_c,
        )

        _KAPMIN_FLOOR = 1e-8
        if use_cutoff:
            if continuum_absorption_full is not None and center_indices_full is not None:
                _fi = np.clip(
                    center_indices_full.astype(np.int64), 0, continuum_absorption_full.shape[1] - 1
                )
                _cont_at_center = (continuum_absorption_full[:, _fi]).T
                kapmin_ref_array = np.maximum(
                    _cont_at_center * cutoff,
                    _cont_at_center * _KAPMIN_FLOOR,
                )
            else:
                _ci = np.clip(line_indices_wing.astype(np.int64), 0, n_wavelengths - 1)
                _cont_at_center = (continuum_absorption[:, _ci]).T
                kapmin_ref_array = np.maximum(
                    _cont_at_center * cutoff,
                    _cont_at_center * _KAPMIN_FLOOR,
                )

        if metal_tables is not None and populations is not None:
            from ..engine.opacity import _compute_continuum_limits

            _max_nv = min(int(np.max(ion_stages_arr)) if n_lines > 0 else 6, 10)
            wcon_tbl = np.zeros((n_depths, _max_nv + 1), dtype=np.float64)
            wtail_tbl = np.zeros((n_depths, _max_nv + 1), dtype=np.float64)
            for _d in range(n_depths):
                _state = populations.layers[_d]
                for _nv in range(1, _max_nv + 1):
                    _wcon_v, _wtail_v = _compute_continuum_limits(
                        ncon=getattr(_state, "ncon", 0),
                        nelion=_nv,
                        nelionx=getattr(_state, "nelionx", 0),
                        emerge_val=getattr(_state, "emerge", 0.0),
                        emerge_h_val=getattr(_state, "emerge_h", 0.0),
                        metal_tables=metal_tables,
                        ifvac=1,
                    )
                    if _wcon_v is not None and _wcon_v > 0.0:
                        wcon_tbl[_d, _nv] = _wcon_v
                    if _wtail_v is not None and _wtail_v > 0.0:
                        wtail_tbl[_d, _nv] = _wtail_v
            _ion_cw = np.clip(ion_stages_arr, 1, _max_nv)
            wcon_array = wcon_tbl[:, _ion_cw].T.ravel()
            wtail_array = wtail_tbl[:, _ion_cw].T.ravel()

    # ── Vectorized center accumulation (replaces Python scatter-add loop) ───
    _skip_types = np.array([-2, -1, 1, 2, 3, 4], dtype=np.int8)
    _center_keep = ~np.isin(line_types_array, _skip_types)  # (n_lines,)
    _ci_arr = line_indices  # (n_lines,) clamped grid indices
    _ci_valid = ((_ci_arr >= 0) & (_ci_arr < n_wavelengths)) & _center_keep
    _valid_li = np.where(_ci_valid)[0]   # valid line indices
    _ci_vals  = _ci_arr[_valid_li]       # their center grid indices
    for _d in range(n_depths):
        if valid_mask is not None:
            _contrib = np.where(
                valid_mask[_valid_li, _d], transp[_valid_li, _d], 0.0
            )
        else:
            _contrib = transp[_valid_li, _d]
        np.add.at(asynth[_d], _ci_vals, _contrib)

    if _use_sparse and pair_line is not None and depth_starts is not None:
        _t_sparse = time.perf_counter()
        _compute_asynth_wings_sparse_kernel(
            asynth,
            wavelength_grid,
            pair_line,
            depth_starts,
            line_wavelengths_array,
            line_indices_array,
            kappa0_pair,
            adamp_pair,
            doppler_width_pair,
            kapmin_ref_pair,
            wcon_pair,
            wtail_pair,
            _use_cutoff_kernel,
            _resolu,
            max_profile_steps,
            h0tab,
            h1tab,
            h2tab,
            n_depths,
        )
        logger.info(
            "Timing: ASYNTH wings sparse (%d pairs) in %.3fs",
            int(pair_line.size),
            time.perf_counter() - _t_sparse,
        )
    elif not _use_sparse:
        _compute_asynth_wings_kernel(
            asynth,
            wavelength_grid,
            transp,
            _vm,
            line_wavelengths_array,
            line_indices_array,
            line_types_array,
            stim_factors,
            kappa0_array,
            adamp_array,
            doppler_widths_array,
            gamma_rad_array,
            gamma_stark_array,
            gamma_vdw_array,
            kapmin_ref_array,
            (
                continuum_absorption
                if use_cutoff
                else np.zeros((n_depths, n_wavelengths), dtype=np.float64)
            ),
            wcon_array,
            wtail_array,
            cutoff,
            max_profile_steps,
            h0tab,
            h1tab,
            h2tab,
        )

    # Fortran synthe.for line 94: ASYNTH(J)=TRANSP(J,I)*(1.-EXP(-FREQ*HKT(J)))
    # Apply stimulated emission factor after center+wing accumulation.
    asynth *= stim_grid

    return asynth
