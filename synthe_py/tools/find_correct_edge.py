#!/usr/bin/env python3
"""Find the correct edge index for a wavelength."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from synthe_py.io.atmosphere import load_cached

def main():
    """Find correct edge."""
    print("=" * 80)
    print("FINDING CORRECT EDGE INDEX")
    print("=" * 80)
    
    atm_path = Path("synthe_py/data/at12_aaaaa_atmosphere_fixed.npz")
    target_wl = 490.0
    
    atm = load_cached(atm_path)
    wledge = np.asarray(atm.continuum_wledge, dtype=np.float64)
    wledge_abs = np.abs(wledge)
    
    print(f"\nTarget wavelength: {target_wl} nm")
    print(f"Edge table size: {len(wledge)}")
    
    # Find edges that contain this wavelength
    print(f"\nEdges containing {target_wl} nm:")
    for i in range(len(wledge_abs) - 1):
        wl_left = wledge_abs[i]
        wl_right = wledge_abs[i + 1]
        if wl_left <= target_wl < wl_right:
            print(f"  Edge {i}: [{wl_left:.6f}, {wl_right:.6f}] ✓ CONTAINS")
        elif i < 10:  # Show first 10 for reference
            print(f"  Edge {i}: [{wl_left:.6f}, {wl_right:.6f}]")
    
    # Check what sequential search would find
    print(f"\nSequential search simulation:")
    edge = 0
    while edge < len(wledge_abs) - 1 and target_wl >= wledge_abs[edge + 1]:
        print(f"  Check edge {edge}: {target_wl} >= {wledge_abs[edge + 1]:.6f}? {target_wl >= wledge_abs[edge + 1]}")
        edge += 1
    print(f"  Final edge: {edge}")
    print(f"  Edge interval: [{wledge_abs[edge]:.6f}, {wledge_abs[edge + 1]:.6f}]")
    
    # Check what binary search would find
    edge_idx = np.searchsorted(wledge_abs, target_wl, side="right") - 1
    edge_idx = max(0, min(edge_idx, len(wledge_abs) - 2))
    print(f"\nBinary search (searchsorted):")
    print(f"  Edge index: {edge_idx}")
    print(f"  Edge interval: [{wledge_abs[edge_idx]:.6f}, {wledge_abs[edge_idx + 1]:.6f}]")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

