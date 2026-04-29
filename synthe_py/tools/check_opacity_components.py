#!/usr/bin/env python3
"""Check opacity components to understand the 2.77× discrepancy."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from synthe_py.io.atmosphere import load_cached
from synthe_py.physics import continuum

def main():
    """Check opacity components."""
    print("=" * 80)
    print("OPACITY COMPONENTS ANALYSIS")
    print("=" * 80)
    
    atm_path = Path("synthe_py/data/at12_aaaaa_atmosphere_fixed.npz")
    if not atm_path.exists():
        print(f"ERROR: {atm_path} not found")
        return False
    
    atm = load_cached(atm_path)
    target_wl = 490.0
    layer_idx = 0
    
    # Compute continuum opacity
    cont_abs, cont_scat, _, _ = continuum.build_depth_continuum(atm, np.array([target_wl]))
    cont_total = cont_abs[layer_idx, 0] + cont_scat[layer_idx, 0]
    
    print(f"\nWavelength: {target_wl} nm")
    print(f"Layer: {layer_idx}")
    print(f"\nContinuum opacity (from fort.10 coefficients):")
    print(f"  Absorption: {cont_abs[layer_idx, 0]:.6E} cm²/g")
    print(f"  Scattering: {cont_scat[layer_idx, 0]:.6E} cm²/g")
    print(f"  Total:      {cont_total:.6E} cm²/g")
    
    # Expected values
    expected_fortran_kappa = 7.157E-04  # cm²/g (from τ[0]/RHOX[0])
    print(f"\nExpected Fortran κ[0] (from τ[0]/RHOX[0]): {expected_fortran_kappa:.6E} cm²/g")
    print(f"Ratio (Expected / Continuum): {expected_fortran_kappa / cont_total:.6f}×")
    
    # Check RHOX
    rhox = atm.depth
    print(f"\nRHOX[0]: {rhox[0]:.6E} g/cm²")
    
    # Compute START value
    start = cont_total * rhox[0]
    print(f"START (κ[0] × RHOX[0]): {start:.6E}")
    
    # Expected τ[0]
    expected_tau0 = 3.691E-07
    print(f"\nExpected Fortran τ[0]: {expected_tau0:.6E}")
    print(f"Ratio (Expected / START): {expected_tau0 / start:.6f}×")
    
    # What κ would give expected τ?
    implied_kappa = expected_tau0 / rhox[0]
    print(f"\nImplied κ[0] (from τ[0]/RHOX[0]): {implied_kappa:.6E} cm²/g")
    print(f"Ratio (Continuum / Implied): {cont_total / implied_kappa:.6f}×")
    
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print("If Fortran τ[0] = 3.691E-07 and RHOX[0] = 5.157E-04,")
    print(f"then Fortran κ[0] = {implied_kappa:.6E} cm²/g")
    print(f"\nBut fort.10 gives κ[0] = {cont_total:.6E} cm²/g")
    print(f"Ratio: {cont_total / implied_kappa:.6f}×")
    
    if abs(cont_total / implied_kappa - 2.77) < 0.1:
        print("\n✓ This matches the 2.77× discrepancy!")
        print("  → fort.10 coefficients are 2.77× too large")
        print("  → OR the expected Fortran value is wrong")
    else:
        print(f"\n? Ratio is {cont_total / implied_kappa:.3f}×, not 2.77×")
        print("  → Need to investigate further")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

