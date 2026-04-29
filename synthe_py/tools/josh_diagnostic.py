#!/usr/bin/env python3
"""
JOSH Solver Diagnostic Script

This script outputs key intermediate values from the Python JOSH solver
at a specific wavelength for direct comparison with Fortran debug output.

Run this script, then compare output with Fortran's fort.99/fort.33 debug logs.
"""

import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from synthe_py.physics.josh_solver import (
    solve_josh_flux,
    _integ,
    _map1,
    XTAU_GRID,
    CH_WEIGHTS,
    COEFJ_MATRIX,
    COEFJ_DIAG,
    NXTAU,
    ITER_TOL,
    EPS,
)


def run_josh_diagnostic(wavelength_nm: float = 320.0):
    """Run JOSH solver with detailed diagnostics at a specific wavelength."""
    
    print("=" * 80)
    print(f"JOSH SOLVER DIAGNOSTIC - Wavelength: {wavelength_nm:.4f} nm")
    print("=" * 80)
    
    # Load atmosphere and output data
    atm_path = Path(__file__).parent.parent / "data" / "at12_aaaaa_t03750_fixed_v2.npz"
    out_path = Path(__file__).parent.parent / "out" / "test_fixed_3750.npz"
    
    if not atm_path.exists():
        print(f"ERROR: Atmosphere file not found: {atm_path}")
        return
    if not out_path.exists():
        print(f"ERROR: Output file not found: {out_path}")
        return
        
    atm = np.load(atm_path)
    py_out = np.load(out_path)
    
    # Find wavelength index
    py_wl = py_out['wavelength']
    wl_idx = np.argmin(np.abs(py_wl - wavelength_nm))
    actual_wl = py_wl[wl_idx]
    freq = 2.99792458e17 / actual_wl  # Hz
    
    print(f"\nWavelength: {actual_wl:.8f} nm (index {wl_idx})")
    print(f"Frequency: {freq:.8e} Hz")
    
    # Get opacities at this wavelength
    cont_abs = py_out['continuum_absorption'][:, wl_idx]
    cont_scat = py_out['continuum_scattering'][:, wl_idx]
    line_abs = py_out['line_opacity'][:, wl_idx]
    line_scat = py_out['line_scattering'][:, wl_idx]
    
    # Get atmosphere data
    rhox = atm['depth']  # Column mass (RHOX)
    temp = atm['temperature']
    
    # Compute Planck function
    freq15 = freq / 1e15
    hkt = 4.79927e-11 / temp  # h*freq / (k*T)
    ehvkt = np.exp(-freq * hkt)
    stim = 1.0 - ehvkt
    planck = 1.47439e-02 * freq15**3 * ehvkt / stim
    
    print("\n" + "=" * 80)
    print("SECTION 1: INPUT OPACITIES")
    print("=" * 80)
    print(f"{'Layer':<8} {'ACONT':>14} {'SIGMAC':>14} {'ALINE':>14} {'SIGMAL':>14}")
    for j in [0, 1, 39, 40, 78, 79]:
        if j < len(cont_abs):
            print(f"{j+1:<8} {cont_abs[j]:>14.6e} {cont_scat[j]:>14.6e} {line_abs[j]:>14.6e} {line_scat[j]:>14.6e}")
    
    # Compute ABTOT and ALPHA
    abtot = cont_abs + line_abs + cont_scat + line_scat
    scatter = cont_scat + line_scat
    alpha = np.divide(scatter, abtot, out=np.zeros_like(abtot), where=abtot > 0)
    
    print("\n" + "=" * 80)
    print("SECTION 2: ABTOT AND ALPHA (scattering fraction)")
    print("=" * 80)
    print(f"{'Layer':<8} {'ABTOT':>14} {'ALPHA':>14} {'1-ALPHA':>14}")
    for j in [0, 1, 39, 40, 78, 79]:
        if j < len(abtot):
            print(f"{j+1:<8} {abtot[j]:>14.6e} {alpha[j]:>14.8f} {1-alpha[j]:>14.8e}")
    
    # Compute SNUBAR
    denom = cont_abs + line_abs
    snubar = np.where(denom > 0, (cont_abs * planck + line_abs * planck) / denom, planck)
    
    print("\n" + "=" * 80)
    print("SECTION 3: SNUBAR (source function before scattering)")
    print("=" * 80)
    print(f"{'Layer':<8} {'SCONT=BNU':>14} {'SNUBAR':>14} {'ALPHA':>14}")
    for j in [0, 1, 39, 40, 78, 79]:
        if j < len(snubar):
            print(f"{j+1:<8} {planck[j]:>14.6e} {snubar[j]:>14.6e} {alpha[j]:>14.8f}")
    
    # Compute TAUNU (optical depth)
    # Match Fortran INTEG: TAUNU[0] = ABTOT[0] * RHOX[0]
    start = abtot[0] * rhox[0]
    taunu = _integ(rhox, abtot, start)
    
    print("\n" + "=" * 80)
    print("SECTION 4: TAUNU (optical depth)")
    print("=" * 80)
    print(f"Start value = ABTOT[0] * RHOX[0] = {abtot[0]:.8e} * {rhox[0]:.8e} = {start:.8e}")
    print(f"\n{'Layer':<8} {'RHOX':>14} {'ABTOT':>14} {'TAUNU':>14}")
    for j in [0, 1, 2, 39, 40, 78, 79]:
        if j < len(taunu):
            print(f"{j+1:<8} {rhox[j]:>14.6e} {abtot[j]:>14.6e} {taunu[j]:>14.6e}")
    
    print(f"\nTAUNU[0] = {taunu[0]:.8e}")
    print(f"TAUNU[-1] = {taunu[-1]:.8e}")
    print(f"XTAU_GRID[-1] = {XTAU_GRID[-1]:.8e}")
    print(f"TAUNU[0] > XTAU_GRID[-1]? {taunu[0] > XTAU_GRID[-1]} (MAXJ=1 condition)")
    
    # MAP1: Interpolate SNUBAR and ALPHA to XTAU_GRID
    print("\n" + "=" * 80)
    print("SECTION 5: MAP1 INTERPOLATION (SNUBAR → XSBAR, ALPHA → XALPHA)")
    print("=" * 80)
    
    xsbar, maxj_xsbar = _map1(taunu, snubar, XTAU_GRID)
    xalpha, maxj_xalpha = _map1(taunu, alpha, XTAU_GRID)
    
    print(f"MAXJ from SNUBAR MAP1: {maxj_xsbar}")
    print(f"MAXJ from ALPHA MAP1: {maxj_xalpha}")
    
    # Apply clipping
    xsbar = np.maximum(xsbar, EPS)
    xalpha = np.clip(xalpha, 0.0, 1.0)
    
    # Apply mask for XTAU < TAUNU[0]
    mask = XTAU_GRID < taunu[0]
    num_masked = np.sum(mask)
    if num_masked > 0:
        print(f"\nMask applied: {num_masked} points with XTAU < TAUNU[0]")
        print(f"  TAUNU[0] = {taunu[0]:.8e}")
        print(f"  Setting XSBAR[mask] = SNUBAR[0] = {snubar[0]:.8e}")
        print(f"  Setting XALPHA[mask] = ALPHA[0] = {alpha[0]:.8f}")
        xsbar[mask] = np.maximum(snubar[0], EPS)
        xalpha[mask] = np.clip(alpha[0], 0.0, 1.0)
    
    print(f"\n{'L':<5} {'XTAU':>12} {'XSBAR':>14} {'XALPHA':>14} {'1-XALPHA':>14}")
    for L in [1, 2, 10, 20, 25, 26, 30, 40, 50, 51]:
        idx = L - 1  # Convert to 0-based
        if idx < len(XTAU_GRID):
            print(f"{L:<5} {XTAU_GRID[idx]:>12.6e} {xsbar[idx]:>14.6e} {xalpha[idx]:>14.8f} {1-xalpha[idx]:>14.8e}")
    
    # Initialize XS from XSBAR
    xs = xsbar.copy()
    
    # Compute DIAG
    diag = 1.0 - xalpha * COEFJ_DIAG
    
    # Compute XSBAR_MODIFIED
    xsbar_modified = xsbar * (1.0 - xalpha)
    
    print("\n" + "=" * 80)
    print("SECTION 6: ITERATION INPUTS")
    print("=" * 80)
    print(f"\n{'L':<5} {'XS(init)':>14} {'XSBAR_MOD':>14} {'DIAG':>14} {'COEFJ(L,L)':>14}")
    for L in [1, 2, 10, 20, 25, 26, 30, 40, 50, 51]:
        idx = L - 1
        if idx < len(xs):
            print(f"{L:<5} {xs[idx]:>14.6e} {xsbar_modified[idx]:>14.6e} {diag[idx]:>14.8f} {COEFJ_DIAG[idx]:>14.8e}")
    
    # Perform iteration (simplified - just show first iteration)
    print("\n" + "=" * 80)
    print("SECTION 7: FIRST ITERATION (K=51 down to K=1)")
    print("=" * 80)
    
    # Show iteration for K=1 (surface, idx=0)
    k = 0
    delxs_sum = np.dot(COEFJ_MATRIX[k, :], xs)
    delxs = (delxs_sum * xalpha[k] + xsbar_modified[k] - xs[k]) / diag[k]
    xs_new = max(xs[k] + delxs, EPS)
    
    print(f"\nFor K=1 (surface):")
    print(f"  sum(COEFJ(1,M)*XS(M)) = {delxs_sum:.8e}")
    print(f"  XALPHA[0] = {xalpha[0]:.8f}")
    print(f"  XSBAR_MOD[0] = {xsbar_modified[0]:.8e}")
    print(f"  XS[0] (before) = {xs[0]:.8e}")
    print(f"  DIAG[0] = {diag[0]:.8f}")
    print(f"  DELXS = ({delxs_sum:.8e} * {xalpha[0]:.8f} + {xsbar_modified[0]:.8e} - {xs[0]:.8e}) / {diag[0]:.8f}")
    print(f"        = {delxs:.8e}")
    print(f"  XS[0] (after) = {xs_new:.8e}")
    
    # Now actually run full iteration using Numba kernel
    from synthe_py.physics.josh_solver import _josh_iteration_kernel
    
    xs_iter = xsbar.copy()
    xs_iter, num_iter = _josh_iteration_kernel(
        COEFJ_MATRIX, xs_iter, xalpha, xsbar_modified, COEFJ_DIAG,
        ITER_TOL, NXTAU, EPS
    )
    
    print(f"\nFull iteration completed in {num_iter} iterations")
    
    print("\n" + "=" * 80)
    print("SECTION 8: XS VALUES AFTER ITERATION")
    print("=" * 80)
    print(f"\n{'L':<5} {'XS(before)':>14} {'XS(after)':>14} {'Change':>14}")
    for L in [1, 2, 10, 20, 25, 26, 30, 40, 50, 51]:
        idx = L - 1
        if idx < len(xs_iter):
            change = (xs_iter[idx] - xsbar[idx]) / max(abs(xsbar[idx]), 1e-40)
            print(f"{L:<5} {xsbar[idx]:>14.6e} {xs_iter[idx]:>14.6e} {change:>14.4%}")
    
    # Compute flux
    flux = np.dot(CH_WEIGHTS, xs_iter)
    
    print("\n" + "=" * 80)
    print("SECTION 9: FLUX CALCULATION")
    print("=" * 80)
    print(f"\nFlux = sum(CH(M) * XS(M))")
    print(f"\n{'M':<5} {'CH':>14} {'XS':>14} {'CH*XS':>14}")
    for M in [1, 2, 10, 20, 25, 26, 30, 40, 50, 51]:
        idx = M - 1
        if idx < len(xs_iter):
            contrib = CH_WEIGHTS[idx] * xs_iter[idx]
            print(f"{M:<5} {CH_WEIGHTS[idx]:>14.6e} {xs_iter[idx]:>14.6e} {contrib:>14.6e}")
    
    print(f"\nTotal flux = {flux:.8e}")
    
    # Compare with stored Python output
    py_flux = py_out['flux_total'][wl_idx]
    py_flux_cont = py_out['flux_continuum'][wl_idx]
    
    print("\n" + "=" * 80)
    print("SECTION 10: COMPARISON WITH STORED OUTPUT")
    print("=" * 80)
    print(f"Computed flux (this script): {flux:.8e}")
    print(f"Stored Python flux_total: {py_flux:.8e}")
    print(f"Stored Python flux_continuum: {py_flux_cont:.8e}")
    
    # Load Fortran spectrum for comparison
    fort_file = Path(__file__).parent.parent.parent / "grids" / "at12_aaaaa" / "spec" / "at12_aaaaa_t03750g3.50.spec"
    if fort_file.exists():
        fort_data = np.loadtxt(fort_file)
        fort_wl = fort_data[:, 0]
        fort_flux = fort_data[:, 1]
        fort_flux_interp = np.interp(actual_wl, fort_wl, fort_flux)
        
        print(f"\nFortran flux (interpolated): {fort_flux_interp:.8e}")
        print(f"\nRatio (Python/Fortran): {py_flux / fort_flux_interp:.4f}")
        print(f"Difference: {(py_flux - fort_flux_interp) / fort_flux_interp * 100:.2f}%")
    
    print("\n" + "=" * 80)
    print("END OF DIAGNOSTIC OUTPUT")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="JOSH Solver Diagnostic")
    parser.add_argument("--wavelength", "-w", type=float, default=320.0,
                        help="Wavelength in nm (default: 320.0)")
    args = parser.parse_args()
    
    run_josh_diagnostic(args.wavelength)

