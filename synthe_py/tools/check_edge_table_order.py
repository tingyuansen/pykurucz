#!/usr/bin/env python3
"""Check if edge table needs to be sorted after reading."""

import numpy as np
import struct
from pathlib import Path

def read_fort10_edges(fort10_path):
    """Read edge table directly from fort.10."""
    with fort10_path.open('rb') as f:
        # Skip header
        header = f.read(4)
        size = struct.unpack('<i', header)[0]
        f.read(size)  # Skip header payload
        
        # Read edges record
        header = f.read(4)
        size = struct.unpack('<i', header)[0]
        rec_edges = f.read(size)
        f.read(4)  # Skip trailer
        
        idx = 0
        nedge = struct.unpack_from('<i', rec_edges, idx)[0]
        idx += 4
        frqedg = np.frombuffer(rec_edges, dtype='<f8', count=nedge, offset=idx)
        idx += nedge * 8
        wledge = np.frombuffer(rec_edges, dtype='<f8', count=nedge, offset=idx)
        idx += nedge * 8
        cmedge = np.frombuffer(rec_edges, dtype='<f8', count=nedge, offset=idx)
        
        return wledge, frqedg, cmedge

def main():
    """Check edge table order."""
    print("=" * 80)
    print("CHECKING EDGE TABLE ORDER IN fort.10")
    print("=" * 80)
    
    fort10_path = Path("synthe/stmp_at12_aaaaa/fort.10")
    if not fort10_path.exists():
        print(f"ERROR: fort.10 not found: {fort10_path}")
        return False
    
    wledge, frqedg, cmedge = read_fort10_edges(fort10_path)
    
    print(f"\nEdge table from fort.10:")
    print(f"  Size: {len(wledge)}")
    print(f"  Raw values - Min: {np.min(wledge):.6f}, Max: {np.max(wledge):.6f}")
    print(f"  Is sorted: {np.all(np.diff(wledge) >= 0)}")
    
    # Take ABS
    wledge_abs = np.abs(wledge)
    print(f"\nAfter ABS:")
    print(f"  Min: {np.min(wledge_abs):.6f}, Max: {np.max(wledge_abs):.6f}")
    print(f"  Is sorted: {np.all(np.diff(wledge_abs) >= 0)}")
    
    # Check if Fortran sorts by ABS
    # Fortran sorts by A(I) = ABS(WLEDGE(I)) in xnfpelsyn.for
    a = np.abs(wledge)
    sort_idx = np.argsort(a)
    wledge_sorted = a[sort_idx]
    print(f"\nAfter sorting by ABS (like Fortran):")
    print(f"  Is sorted: {np.all(np.diff(wledge_sorted) >= 0)}")
    
    # Show first 10 edges
    print(f"\nFirst 10 edges (raw):")
    for i in range(min(10, len(wledge))):
        print(f"  [{i}]: {wledge[i]:.6f}")
    
    print(f"\nFirst 10 edges (ABS):")
    for i in range(min(10, len(wledge_abs))):
        print(f"  [{i}]: {wledge_abs[i]:.6f}")
    
    print(f"\nFirst 10 edges (sorted by ABS):")
    for i in range(min(10, len(wledge_sorted))):
        print(f"  [{i}]: {wledge_sorted[i]:.6f}")
    
    return True

if __name__ == "__main__":
    success = main()
    import sys
    sys.exit(0 if success else 1)

