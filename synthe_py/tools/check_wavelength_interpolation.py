#!/usr/bin/env python3
"""Check wavelength interpolation to understand discrepancy."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from synthe_py.io.atmosphere import load_cached
from synthe_py.physics import continuum

def main():
    """Check wavelength interpolation."""
    print("=" * 80)
    print("CHECKING WAVELENGTH INTERPOLATION")
    print("=" * 80)
    
    atm_path = Path("synthe_py/data/at12_aaaaa_atmosphere_fixed.npz")
    diag_path = Path("synthe_py/out/diagnostics_rhox_fixed.npz")
    
    atm = load_cached(atm_path)
    diag = np.load(diag_path)
    
    # Get wavelength grids
    diag_wl = diag['wavelength']
    idx_490 = np.argmin(np.abs(diag_wl - 490.0))
    diag_wl_490 = diag_wl[idx_490]
    
    print(f"\nWavelengths:")
    print(f"  Diagnostics grid point: {diag_wl_490:.9f} nm (index {idx_490})")
    print(f"  Single computation:     490.000000 nm")
    print(f"  Difference:            {abs(diag_wl_490 - 490.0):.9f} nm")
    
    # Get edge table
    if atm.continuum_wledge is not None:
        wledge = np.asarray(atm.continuum_wledge, dtype=np.float64)
        print(f"\nWavelength edge table:")
        print(f"  Size: {len(wledge)} edges")
        print(f"  Range: {wledge[0]:.2f} - {wledge[-1]:.2f} nm")
        
        # Find which edge interval each wavelength falls into
        edge_idx_diag = np.searchsorted(wledge, diag_wl_490, side="right") - 1
        edge_idx_diag = np.clip(edge_idx_diag, 0, wledge.size - 2)
        
        edge_idx_single = np.searchsorted(wledge, 490.0, side="right") - 1
        edge_idx_single = np.clip(edge_idx_single, 0, wledge.size - 2)
        
        print(f"\nEdge intervals:")
        print(f"  Diagnostics wavelength ({diag_wl_490:.9f} nm):")
        print(f"    Interval: {edge_idx_diag} ({wledge[edge_idx_diag]:.6f} - {wledge[edge_idx_diag+1]:.6f} nm)")
        print(f"  Single wavelength (490.000000 nm):")
        print(f"    Interval: {edge_idx_single} ({wledge[edge_idx_single]:.6f} - {wledge[edge_idx_single+1]:.6f} nm)")
        
        if edge_idx_diag != edge_idx_single:
            print(f"\n⚠️  DIFFERENT EDGE INTERVALS!")
            print(f"  Diagnostics uses interval {edge_idx_diag}")
            print(f"  Single uses interval {edge_idx_single}")
        else:
            print(f"\n✓ Same edge interval: {edge_idx_diag}")
        
        # Show edge values around 490nm
        print(f"\nEdge values around 490nm:")
        for i in range(max(0, edge_idx_diag-2), min(len(wledge), edge_idx_diag+4)):
            marker = ""
            if i == edge_idx_diag:
                marker = " ← diagnostics"
            if i == edge_idx_single:
                marker = " ← single"
            print(f"  Edge[{i}]: {wledge[i]:.9f} nm{marker}")
    
    # Compute values
    print(f"\nComputing values...")
    cont_abs_diag, cont_scat_diag, _, _ = continuum.build_depth_continuum(atm, np.array([diag_wl_490]))
    cont_abs_single, cont_scat_single, _, _ = continuum.build_depth_continuum(atm, np.array([490.0]))
    
    print(f"\nComputed values (layer 0):")
    print(f"  Diagnostics wavelength ({diag_wl_490:.9f} nm):")
    print(f"    Absorption: {cont_abs_diag[0, 0]:.6E} cm²/g")
    print(f"    Scattering: {cont_scat_diag[0, 0]:.6E} cm²/g")
    print(f"    Total: {cont_abs_diag[0, 0] + cont_scat_diag[0, 0]:.6E} cm²/g")
    print(f"  Single wavelength (490.000000 nm):")
    print(f"    Absorption: {cont_abs_single[0, 0]:.6E} cm²/g")
    print(f"    Scattering: {cont_scat_single[0, 0]:.6E} cm²/g")
    print(f"    Total: {cont_abs_single[0, 0] + cont_scat_single[0, 0]:.6E} cm²/g")
    
    print(f"\nRatios:")
    print(f"  Absorption: {cont_abs_diag[0, 0] / cont_abs_single[0, 0]:.6f}×")
    print(f"  Scattering: {cont_scat_diag[0, 0] / cont_scat_single[0, 0]:.6f}×")
    print(f"  Total: {(cont_abs_diag[0, 0] + cont_scat_diag[0, 0]) / (cont_abs_single[0, 0] + cont_scat_single[0, 0]):.6f}×")
    
    # Check if this explains the discrepancy
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print("If diagnostics wavelength falls in a different edge interval,")
    print("it would use different parabolic coefficients, explaining the difference.")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

