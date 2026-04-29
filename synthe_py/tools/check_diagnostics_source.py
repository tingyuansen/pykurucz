#!/usr/bin/env python3
"""Check if diagnostics match what would be computed during synthesis."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from synthe_py.io.atmosphere import load_cached
from synthe_py.physics import continuum

def main():
    """Check diagnostics source."""
    print("=" * 80)
    print("CHECKING DIAGNOSTICS SOURCE")
    print("=" * 80)
    
    # Try interleaved reading fix first
    atm_path = Path("synthe_py/data/at12_aaaaa_atmosphere_fixed_interleaved.npz")
    if not atm_path.exists():
        atm_path = Path("synthe_py/data/at12_aaaaa_atmosphere_sorted.npz")
    if not atm_path.exists():
        atm_path = Path("synthe_py/data/at12_aaaaa_atmosphere_fixed.npz")
    # Try new diagnostics first, fall back to old
    diag_path = Path("synthe_py/out/diagnostics_interleaved.npz")
    if not diag_path.exists():
        diag_path = Path("synthe_py/out/diagnostics_sorted.npz")
    if not diag_path.exists():
        diag_path = Path("synthe_py/out/diagnostics_edge_fix.npz")
    if not diag_path.exists():
        diag_path = Path("synthe_py/out/diagnostics_rhox_fixed.npz")
    target_wl = 490.0
    
    # Load
    atm = load_cached(atm_path)
    diag = np.load(diag_path)
    
    # Get diagnostics wavelength grid
    diag_wl = diag['wavelength']
    idx_490 = np.argmin(np.abs(diag_wl - target_wl))
    
    print(f"\nRecomputing continuum using EXACT synthesis method...")
    print(f"Using diagnostics wavelength grid ({len(diag_wl)} points)")
    
    # Compute exactly as synthesis does
    cont_abs, cont_scat, _, _ = continuum.build_depth_continuum(atm, diag_wl)
    
    # Check if they match
    print(f"\nComparing at index {idx_490} (wavelength {diag_wl[idx_490]:.9f} nm):")
    print(f"  Diagnostics absorption: {diag['continuum_absorption'][0, idx_490]:.6E}")
    print(f"  Recomputed absorption:  {cont_abs[0, idx_490]:.6E}")
    print(f"  Match: {np.isclose(diag['continuum_absorption'][0, idx_490], cont_abs[0, idx_490], rtol=1e-10)}")
    
    print(f"\n  Diagnostics scattering: {diag['continuum_scattering'][0, idx_490]:.6E}")
    print(f"  Recomputed scattering:  {cont_scat[0, idx_490]:.6E}")
    print(f"  Match: {np.isclose(diag['continuum_scattering'][0, idx_490], cont_scat[0, idx_490], rtol=1e-10)}")
    
    # Check full arrays
    abs_match = np.allclose(diag['continuum_absorption'], cont_abs, rtol=1e-10)
    scat_match = np.allclose(diag['continuum_scattering'], cont_scat, rtol=1e-10)
    
    print(f"\nFull array comparison:")
    print(f"  Absorption arrays match: {abs_match}")
    print(f"  Scattering arrays match: {scat_match}")
    
    if not abs_match:
        max_diff = np.max(np.abs(diag['continuum_absorption'] - cont_abs))
        max_rel_diff = np.max(np.abs((diag['continuum_absorption'] - cont_abs) / np.maximum(cont_abs, 1e-40)))
        print(f"  Max absolute difference: {max_diff:.6E}")
        print(f"  Max relative difference: {max_rel_diff:.6E}")
        # Find where difference is largest
        diff = np.abs(diag['continuum_absorption'] - cont_abs)
        max_idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"  Max diff at layer {max_idx[0]}, wavelength index {max_idx[1]} ({diag_wl[max_idx[1]]:.6f} nm)")
        print(f"    Diagnostics: {diag['continuum_absorption'][max_idx]:.6E}")
        print(f"    Recomputed:  {cont_abs[max_idx]:.6E}")
    
    if not scat_match:
        max_diff = np.max(np.abs(diag['continuum_scattering'] - cont_scat))
        max_rel_diff = np.max(np.abs((diag['continuum_scattering'] - cont_scat) / np.maximum(cont_scat, 1e-40)))
        print(f"  Max absolute difference: {max_diff:.6E}")
        print(f"  Max relative difference: {max_rel_diff:.6E}")
        # Find where difference is largest
        diff = np.abs(diag['continuum_scattering'] - cont_scat)
        max_idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"  Max diff at layer {max_idx[0]}, wavelength index {max_idx[1]} ({diag_wl[max_idx[1]]:.6f} nm)")
        print(f"    Diagnostics: {diag['continuum_scattering'][max_idx]:.6E}")
        print(f"    Recomputed:  {cont_scat[max_idx]:.6E}")
    
    # Now check what fort.10 direct gives for 490nm
    print(f"\n" + "=" * 80)
    print("COMPARING WITH FORT.10 DIRECT (490nm)")
    print("=" * 80)
    cont_abs_490, cont_scat_490, _, _ = continuum.build_depth_continuum(atm, np.array([490.0]))
    print(f"Fort.10 direct at 490.0 nm:")
    print(f"  Absorption: {cont_abs_490[0, 0]:.6E} cm²/g")
    print(f"  Scattering: {cont_scat_490[0, 0]:.6E} cm²/g")
    print(f"  Total: {cont_abs_490[0, 0] + cont_scat_490[0, 0]:.6E} cm²/g")
    
    print(f"\nDiagnostics at {diag_wl[idx_490]:.9f} nm:")
    print(f"  Absorption: {diag['continuum_absorption'][0, idx_490]:.6E} cm²/g")
    print(f"  Scattering: {diag['continuum_scattering'][0, idx_490]:.6E} cm²/g")
    print(f"  Total: {diag['continuum_absorption'][0, idx_490] + diag['continuum_scattering'][0, idx_490]:.6E} cm²/g")
    
    print(f"\nRatio (Diagnostics/Fort.10 direct):")
    print(f"  Absorption: {diag['continuum_absorption'][0, idx_490] / cont_abs_490[0, 0]:.6f}×")
    print(f"  Scattering: {diag['continuum_scattering'][0, idx_490] / cont_scat_490[0, 0]:.6f}×")
    print(f"  Total: {(diag['continuum_absorption'][0, idx_490] + diag['continuum_scattering'][0, idx_490]) / (cont_abs_490[0, 0] + cont_scat_490[0, 0]):.6f}×")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

