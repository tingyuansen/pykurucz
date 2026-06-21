#!/usr/bin/env python3
"""Pre-compile all Numba-jitted kernels to populate the on-disk cache.

Run this once after installation or code changes to avoid JIT compilation
overhead on the first synthesis run:

    python -m synthe_py.warmup

With ``cache=True`` on all kernels, Numba writes ``.nbi`` / ``.nbc`` files
into each module's ``__pycache__`` directory.  Subsequent imports load the
pre-compiled machine code directly, reducing first-call overhead from ~23 s
to < 1 s.

The script triggers compilation by calling each kernel with tiny dummy arrays
whose dtypes and shapes match the real call signatures.  No atmosphere file
or line catalog is needed.
"""

import sys
import time
import numpy as np


def _warmup_voigt_jit():
    """Compile Voigt profile helpers."""
    from synthe_py.physics.voigt_jit import (
        voigt_profile_jit,
        accumulate_voigt_wings_jit,
        compute_helium_voigt_batch,
    )

    h0 = np.zeros(201, dtype=np.float64)
    h1 = np.zeros(201, dtype=np.float64)
    h2 = np.zeros(201, dtype=np.float64)

    voigt_profile_jit(1.0, 0.1, h0, h1, h2)

    buf = np.zeros(10, dtype=np.float64)
    wl = np.linspace(500.0, 501.0, 10)
    accumulate_voigt_wings_jit(
        buf, np.ones(10), wl, 5, 500.5, 1.0, 0.1, 0.01, 1e-3,
        False, -1.0, False, -1.0, 500.0,
        h0, h1, h2,
    )

    # compute_helium_voigt_batch needs more complex setup; skip if it errors
    try:
        compute_helium_voigt_batch(
            np.zeros((1, 10), dtype=np.float64),  # out
            np.ones((1, 10), dtype=np.float64),    # continuum
            wl,
            np.array([5], dtype=np.int64),         # center_indices
            np.array([500.5]),                      # line_wavelengths
            np.array([[1.0]]),                      # kappa0
            np.array([[0.1]]),                      # adamp
            np.array([[0.01]]),                     # doppler_widths
            np.array([[True]]),                     # valid_mask
            np.array([[-1.0]]),                     # wcon
            np.array([[-1.0]]),                     # wtail
            1e-3,
            h0, h1, h2,
        )
    except Exception:
        pass


def _warmup_opacity_kernels():
    """Compile metal-wing profile and processing kernels."""
    from synthe_py.engine.opacity import (
        _accumulate_metal_profile_kernel,
        _process_metal_wings_kernel,
        MAX_PROFILE_STEPS,
    )

    n_wl = 10
    n_depths = 2
    n_lines = 1
    n_elements = 1
    max_ion = 3

    h0 = np.zeros(201, dtype=np.float64)
    h1 = np.zeros(201, dtype=np.float64)
    h2 = np.zeros(201, dtype=np.float64)
    profile_ws = np.empty(MAX_PROFILE_STEPS + 1, dtype=np.float64)

    # _accumulate_metal_profile_kernel
    buf = np.zeros(n_wl, dtype=np.float64)
    wl = np.linspace(500.0, 501.0, n_wl)
    _accumulate_metal_profile_kernel(
        buf, np.ones(n_wl), wl, 5, 500.5, 1.0, 0.1, 0.01, 1e-3,
        -1.0, -1.0, h0, h1, h2, profile_ws,
    )

    # _process_metal_wings_kernel
    _process_metal_wings_kernel(
        np.zeros((n_depths, n_wl)),       # metal_wings
        np.zeros((n_depths, n_wl)),       # metal_sources
        wl,                                # wavelength_grid
        np.array([5], dtype=np.int64),     # line_indices
        np.array([500.5]),                 # line_wavelengths
        np.array([1.0]),                   # line_cgf
        np.array([1e8]),                   # line_gamma_rad
        np.array([1e-5]),                  # line_gamma_stark
        np.array([1e-7]),                  # line_gamma_vdw
        np.array([0], dtype=np.int64),     # line_element_idx
        np.array([1], dtype=np.int64),     # line_nelion_eff
        np.array([0], dtype=np.int64),     # line_ncon
        np.array([0], dtype=np.int64),     # line_nelionx
        np.array([0.0]),                   # line_alpha
        np.array([0], dtype=np.int64),     # line_start_idx
        np.array([n_wl], dtype=np.int64),  # line_end_idx
        np.array([5], dtype=np.int64),     # line_center_local
        np.ones((n_elements, n_depths, max_ion)),  # pop_densities_all
        np.full((n_elements, n_depths), 1e-5),     # dop_velocity_all
        np.ones((n_depths, n_wl)),                 # continuum
        np.ones((n_depths, n_wl)),                 # bnu
        np.ones(n_depths) * 1e13,                  # electron_density
        np.ones(n_depths) * 5500.0,                # temperature
        np.ones(n_depths) * 1e-7,                  # mass_density
        np.ones(n_depths),                         # emerge
        np.ones(n_depths),                         # emerge_h
        np.ones(n_depths),                         # xnf_h
        np.ones(n_depths),                         # xnf_he1
        np.ones(n_depths),                         # xnf_h2
        np.ones(n_depths),                         # txnxn
        np.ones((n_depths, n_lines)),              # boltzmann_factor
        np.zeros((100, 100), dtype=np.float64),      # contx (2D: ncon × nelionx)
        np.array([55.845]),                        # atomic_masses
        1,                                         # ifvac
        1e-3,                                      # cutoff
        h0, h1, h2,
    )


def _warmup_hydrogen():
    """Compile hydrogen continuum + wing kernels."""
    from synthe_py.physics.hydrogen_wings import (
        _precompute_karsas_sigma,
        _compute_hydrogen_continuum_kernel,
        HIGH_LEVEL_TERMS_ARRAY,
        LOW_LEVEL_TERMS_ARRAY,
        LYMAN_LIMIT,
        BASE_LIMIT,
        C_LIGHT_CM,
        RYDBERG_CM,
    )
    from synthe_py.physics.karsas_tables import FREQ_LOG, XN_LOG, XL_LOG_ARRAY, EKARSAS, LN10

    n_wl = 10
    n_depths = 2
    freq = np.linspace(5e14, 6e14, n_wl)
    waveno = freq / C_LIGHT_CM
    xnfph = np.ones((n_depths, 6))
    rho = np.ones(n_depths) * 1e-7
    bhyd = np.ones((n_depths, 8))
    hkt = np.ones(n_depths) * 1e-4
    stim = np.ones((n_depths, n_wl))
    bnu = np.ones((n_depths, n_wl))

    HYDROGEN_CROSS_SECTION_COEFF = 2.815e29
    freq3 = HYDROGEN_CROSS_SECTION_COEFF / np.maximum(freq, 1e-30) ** 3
    ehvkt = np.ones((n_depths, n_wl))

    sigma_high, sigma_low, sigma_lyman, n_active_high, n_active_low, lyman_active = \
        _precompute_karsas_sigma(
            freq, waveno,
            HIGH_LEVEL_TERMS_ARRAY, LOW_LEVEL_TERMS_ARRAY,
            LYMAN_LIMIT,
            FREQ_LOG, XN_LOG, XL_LOG_ARRAY, EKARSAS, LN10,
        )

    _compute_hydrogen_continuum_kernel(
        freq, waveno, freq3,
        xnfph, rho, bhyd, hkt, ehvkt, stim, bnu,
        HIGH_LEVEL_TERMS_ARRAY, LOW_LEVEL_TERMS_ARRAY,
        C_LIGHT_CM, RYDBERG_CM, LYMAN_LIMIT, BASE_LIMIT,
        sigma_high, sigma_low, sigma_lyman,
        n_active_high, n_active_low, lyman_active,
    )


def _warmup_rt():
    """Compile radiative transfer kernel."""
    from synthe_py.engine._rt_numba import solve_all_wavelengths_prange

    n_wl = 10
    n_depths = 2
    solve_all_wavelengths_prange(
        np.linspace(500.0, 501.0, n_wl),             # wavelength_nm
        np.ones(n_depths) * 5500.0,                   # temperature
        np.linspace(0.0, 1.0, n_depths),              # column_mass
        np.ones((n_depths, n_wl)),                    # cont_abs
        np.ones((n_depths, n_wl)) * 0.1,             # cont_scat
        np.ones((n_depths, n_wl)),                    # line_opacity
        np.ones((n_depths, n_wl)) * 0.01,            # line_scattering
        np.ones((n_depths, n_wl)),                    # line_source
        False,                                        # has_line_source
    )


def _warmup_asynth():
    """Compile ASYNTH wing kernels."""
    from synthe_py.physics.line_opacity import (
        _compute_asynth_wings_kernel,
        _compute_asynth_wings_sparse_kernel,
    )

    n_wl = 10
    n_depths = 2
    n_lines = 1
    asynth = np.zeros((n_depths, n_wl))
    _compute_asynth_wings_kernel(
        asynth,
        np.linspace(500.0, 501.0, n_wl),                    # wavelength_grid
        np.ones((n_lines, n_depths)),                        # transp
        np.ones((n_lines, n_depths), dtype=np.bool_),        # valid_mask
        np.array([500.5]),                                   # line_wavelengths
        np.array([5], dtype=np.int64),                       # line_indices
        np.array([0], dtype=np.int64),                       # line_types
        np.ones((n_lines, n_depths)),                        # stim_factors
        np.ones((n_lines, n_depths)),                        # kappa0_values
        np.ones((n_lines, n_depths)) * 0.1,                 # adamp_values
        np.ones((n_lines, n_depths)) * 0.01,                # doppler_widths
        np.ones((n_lines, n_depths)) * 1e8,                 # gamma_rad_values
        np.ones((n_lines, n_depths)) * 1e-5,                # gamma_stark_values
        np.ones((n_lines, n_depths)) * 1e-7,                # gamma_vdw_values
        np.ones((n_lines, n_depths)) * 1e-3,                # kapmin_ref_values
        np.ones((n_depths, n_wl)),                           # continuum_absorption
        np.full(n_lines * n_depths, -1.0),                     # wcon_values (flat 1D)
        np.full(n_lines * n_depths, -1.0),                   # wtail_values (flat 1D)
        1e-3,                                                # cutoff
        5000,                                                # max_profile_steps
        np.zeros(201, dtype=np.float64),                     # h0tab
        np.zeros(201, dtype=np.float64),                     # h1tab
        np.zeros(201, dtype=np.float64),                     # h2tab
    )
    pair_line = np.array([0], dtype=np.int32)
    depth_starts = np.array([0, 1, 1], dtype=np.int64)
    _compute_asynth_wings_sparse_kernel(
        asynth,
        np.linspace(500.0, 501.0, n_wl),
        pair_line,
        depth_starts,
        np.array([500.5]),
        np.array([5], dtype=np.int64),
        np.ones(1),
        np.ones(1) * 0.1,
        np.ones(1) * 0.01,
        np.ones(1) * 1e-3,
        np.full(1, -1.0),
        np.full(1, -1.0),
        True,
        300000.0,
        5000,
        np.zeros(201, dtype=np.float64),
        np.zeros(201, dtype=np.float64),
        np.zeros(201, dtype=np.float64),
        n_depths,
    )


def warmup(verbose: bool = True):
    """Pre-compile all Numba kernels used in synthesis.

    Parameters
    ----------
    verbose : bool
        Print progress messages.
    """
    stages = [
        ("Voigt profile helpers", _warmup_voigt_jit),
        ("Metal wings kernel", _warmup_opacity_kernels),
        ("Hydrogen continuum", _warmup_hydrogen),
        ("Radiative transfer", _warmup_rt),
        ("ASYNTH wings", _warmup_asynth),
    ]

    t_total = time.perf_counter()
    for name, func in stages:
        if verbose:
            print(f"  Compiling {name}...", end="", flush=True)
        t0 = time.perf_counter()
        try:
            func()
            elapsed = time.perf_counter() - t0
            if verbose:
                print(f" {elapsed:.1f}s")
        except Exception as e:
            if verbose:
                print(f" FAILED: {e}")

    total = time.perf_counter() - t_total
    if verbose:
        print(f"  Total warmup: {total:.1f}s")
    return total


if __name__ == "__main__":
    print("Pre-compiling Numba kernels for pykurucz SYNTHE pipeline...")
    warmup(verbose=True)
    print("Done. Subsequent synthesis runs will skip JIT compilation.")
