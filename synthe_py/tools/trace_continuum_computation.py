#!/usr/bin/env python3
"""Trace continuum computation to find where discrepancy occurs."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from synthe_py.io.atmosphere import load_cached
from synthe_py.physics import continuum

def main():
    """Trace continuum computation."""
    print("=" * 80)
    print("TRACING CONTINUUM COMPUTATION")
    print("=" * 80)
    
    atm_path = Path("synthe_py/data/at12_aaaaa_atmosphere_fixed.npz")
    diag_path = Path("synthe_py/out/diagnostics_rhox_fixed.npz")
    target_wl = 490.0
    
    # Load data
    atm = load_cached(atm_path)
    diag = np.load(diag_path)
    
    # Get diagnostics wavelength grid
    diag_wl = diag['wavelength']
    idx_490 = np.argmin(np.abs(diag_wl - target_wl))
    print(f"\nDiagnostics wavelength grid:")
    print(f"  Size: {len(diag_wl)}")
    print(f"  Range: {diag_wl[0]:.2f} - {diag_wl[-1]:.2f} nm")
    print(f"  Closest to {target_wl} nm: {diag_wl[idx_490]:.6f} nm (index {idx_490})")
    
    # Get diagnostics values
    diag_cont_abs = diag['continuum_absorption'][0, idx_490]
    diag_cont_scat = diag['continuum_scattering'][0, idx_490]
    
    print(f"\nDiagnostics values (layer 0, index {idx_490}):")
    print(f"  Continuum absorption: {diag_cont_abs:.6E} cm²/g")
    print(f"  Continuum scattering: {diag_cont_scat:.6E} cm²/g")
    print(f"  Total: {diag_cont_abs + diag_cont_scat:.6E} cm²/g")
    
    # Compute using diagnostics wavelength grid
    print(f"\nComputing using diagnostics wavelength grid...")
    cont_abs_grid, cont_scat_grid, _, _ = continuum.build_depth_continuum(atm, diag_wl)
    grid_cont_abs = cont_abs_grid[0, idx_490]
    grid_cont_scat = cont_scat_grid[0, idx_490]
    
    print(f"Computed values (layer 0, index {idx_490}):")
    print(f"  Continuum absorption: {grid_cont_abs:.6E} cm²/g")
    print(f"  Continuum scattering: {grid_cont_scat:.6E} cm²/g")
    print(f"  Total: {grid_cont_abs + grid_cont_scat:.6E} cm²/g")
    
    # Compare
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"Diagnostics absorption: {diag_cont_abs:.6E}")
    print(f"Computed absorption:    {grid_cont_abs:.6E}")
    print(f"Ratio (Diag/Comp):      {diag_cont_abs / grid_cont_abs:.6f}×")
    
    print(f"\nDiagnostics scattering: {diag_cont_scat:.6E}")
    print(f"Computed scattering:    {grid_cont_scat:.6E}")
    print(f"Ratio (Diag/Comp):      {diag_cont_scat / grid_cont_scat:.6f}×")
    
    # Check single wavelength
    print(f"\nComputing single wavelength {target_wl} nm...")
    cont_abs_single, cont_scat_single, _, _ = continuum.build_depth_continuum(atm, np.array([target_wl]))
    print(f"Single wavelength values (layer 0):")
    print(f"  Continuum absorption: {cont_abs_single[0, 0]:.6E} cm²/g")
    print(f"  Continuum scattering: {cont_scat_single[0, 0]:.6E} cm²/g")
    print(f"  Total: {cont_abs_single[0, 0] + cont_scat_single[0, 0]:.6E} cm²/g")
    
    # Check if diagnostics match computed grid
    if np.allclose(diag['continuum_absorption'], cont_abs_grid, rtol=1e-10):
        print("\n✓ Diagnostics absorption matches computed grid!")
    else:
        print("\n✗ Diagnostics absorption does NOT match computed grid")
        max_diff = np.max(np.abs(diag['continuum_absorption'] - cont_abs_grid))
        print(f"  Max difference: {max_diff:.6E}")
    
    if np.allclose(diag['continuum_scattering'], cont_scat_grid, rtol=1e-10):
        print("✓ Diagnostics scattering matches computed grid!")
    else:
        print("✗ Diagnostics scattering does NOT match computed grid")
        max_diff = np.max(np.abs(diag['continuum_scattering'] - cont_scat_grid))
        print(f"  Max difference: {max_diff:.6E}")
    
    # Check around 490nm
    print("\n" + "=" * 80)
    print("CHECKING AROUND 490nm")
    print("=" * 80)
    for i in range(max(0, idx_490-2), min(len(diag_wl), idx_490+3)):
        wl = diag_wl[i]
        diag_a = diag['continuum_absorption'][0, i]
        diag_s = diag['continuum_scattering'][0, i]
        comp_a = cont_abs_grid[0, i]
        comp_s = cont_scat_grid[0, i]
        print(f"Wavelength {wl:.6f} nm:")
        print(f"  Diagnostics: abs={diag_a:.6E}, scat={diag_s:.6E}, total={diag_a+diag_s:.6E}")
        print(f"  Computed:    abs={comp_a:.6E}, scat={comp_s:.6E}, total={comp_a+comp_s:.6E}")
        if not np.isclose(diag_a, comp_a, rtol=1e-10):
            print(f"  ✗ Absorption differs by {(diag_a-comp_a)/comp_a*100:.2f}%")
        if not np.isclose(diag_s, comp_s, rtol=1e-10):
            print(f"  ✗ Scattering differs by {(diag_s-comp_s)/comp_s*100:.2f}%")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

