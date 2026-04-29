"""Count how many lines contribute to each wavelength bin.

This helps diagnose if Python includes more lines than Fortran per bin,
which could explain excessive cumulative weak line opacity.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from synthe_py.io.atmosphere import load_cached
from synthe_py.io.lines import atomic
from synthe_py.physics import populations
from synthe_py.physics.continuum import build_depth_continuum
from synthe_py.engine.opacity import _nearest_grid_indices, MAX_PROFILE_STEPS


def _build_wavelength_grid(start: float, end: float, resolution: float) -> np.ndarray:
    """Geometric wavelength grid matching run_synthesis spacing."""
    ratio = 1.0 + 1.0 / resolution
    rlog = math.log(ratio)
    ix_floor = math.floor(math.log(start) / rlog)
    if math.exp(ix_floor * rlog) < start:
        ix_floor += 1
    wbegin = math.exp(ix_floor * rlog)
    wavelengths = []
    wl = wbegin
    while wl <= end * (1.0 + 1e-9):
        wavelengths.append(wl)
        wl *= ratio
    return np.array(wavelengths, dtype=np.float64)


def count_lines_per_bin(
    catalog: atomic.LineCatalog,
    wavelength_grid: np.ndarray,
    resolution: float,
    dellim_margin_nm: float = 1.0,
) -> np.ndarray:
    """Count how many lines can contribute to each wavelength bin.
    
    This considers both:
    1. Lines with centers in the bin
    2. Lines with centers outside but wings reaching the bin (within MAX_PROFILE_STEPS)
    
    Returns
    -------
    counts : np.ndarray
        Number of potential contributing lines per wavelength bin
    """
    n_wavelengths = len(wavelength_grid)
    counts = np.zeros(n_wavelengths, dtype=np.int32)
    
    # Get line center indices
    center_indices = _nearest_grid_indices(wavelength_grid, catalog.wavelength)
    
    # Compute Doppler width in grid steps at each line wavelength
    # For weak lines, N10DOP = 10 * (DOPPLE * RESOLU)
    # If N10DOP = 0, no wings are computed (Fortran behavior)
    # For strong lines, wings can extend to MAX_PROFILE_STEPS
    
    for line_idx in range(len(catalog.records)):
        center_idx = center_indices[line_idx]
        
        # For simplicity, assume all lines can contribute within MAX_PROFILE_STEPS
        # (This is an upper bound; actual contribution depends on kappa0 and damping)
        start_idx = max(0, center_idx - MAX_PROFILE_STEPS)
        end_idx = min(n_wavelengths, center_idx + MAX_PROFILE_STEPS + 1)
        
        counts[start_idx:end_idx] += 1
    
    return counts


def analyze_line_distribution(
    atmos_path: Path,
    catalog_path: Path,
    wl_start: float = 300.0,
    wl_end: float = 500.0,
    resolution: float = 300_000.0,
):
    """Analyze line distribution in UV/Blue region."""
    
    print("=" * 80)
    print(f"ANALYZING LINE DISTRIBUTION: {wl_start}-{wl_end} nm")
    print("=" * 80)
    
    # Load catalog
    print(f"\nLoading catalog from {catalog_path}...")
    catalog = atomic.load_catalog(catalog_path)
    print(f"  Total lines in catalog: {len(catalog.records)}")
    
    # Filter to wavelength range
    wl_mask = (catalog.wavelength >= wl_start - 1.0) & (catalog.wavelength <= wl_end + 1.0)
    n_lines_in_range = np.sum(wl_mask)
    print(f"  Lines in {wl_start}-{wl_end} nm (±1 nm margin): {n_lines_in_range}")
    
    # Build wavelength grid
    wavelength_grid = _build_wavelength_grid(wl_start, wl_end, resolution)
    n_wavelengths = len(wavelength_grid)
    print(f"\nWavelength grid: {n_wavelengths} points")
    
    # Count by log(gf) ranges
    loggf_ranges = [
        (-10, -5, "Very weak (loggf < -5)"),
        (-5, -3, "Weak (-5 < loggf < -3)"),
        (-3, -1, "Moderate (-3 < loggf < -1)"),
        (-1, 0, "Strong (-1 < loggf < 0)"),
        (0, 3, "Very strong (loggf > 0)"),
    ]
    
    print(f"\nLine counts by log(gf):")
    total_gf_weighted = 0.0
    for low, high, label in loggf_ranges:
        mask = wl_mask & (catalog.log_gf >= low) & (catalog.log_gf < high)
        count = np.sum(mask)
        gf_sum = np.sum(catalog.gf[mask]) if count > 0 else 0.0
        total_gf_weighted += gf_sum
        print(f"  {label}: {count:,} lines (total gf = {gf_sum:.2e})")
    
    print(f"\nTotal gf-weighted contribution: {total_gf_weighted:.2e}")
    
    # Count lines per wavelength bin
    print("\nCounting lines per wavelength bin...")
    
    # Create a subset catalog for this range
    wl_indices = np.where(wl_mask)[0]
    subset_records = [catalog.records[i] for i in wl_indices]
    subset_catalog = atomic.LineCatalog.from_records(subset_records)
    
    counts = count_lines_per_bin(subset_catalog, wavelength_grid, resolution)
    
    print(f"\nLines per wavelength bin statistics:")
    print(f"  Min: {np.min(counts)}")
    print(f"  Max: {np.max(counts)}")
    print(f"  Mean: {np.mean(counts):.1f}")
    print(f"  Median: {np.median(counts):.1f}")
    
    # Analyze by wavelength region
    regions = [
        (300, 350, "UV"),
        (350, 400, "Near-UV"),
        (400, 450, "Blue"),
        (450, 500, "Green-Blue"),
    ]
    
    print(f"\nPer-region analysis:")
    for wl_lo, wl_hi, name in regions:
        if wl_lo >= wl_end or wl_hi <= wl_start:
            continue
        
        # Find grid indices for this region
        region_mask = (wavelength_grid >= wl_lo) & (wavelength_grid < wl_hi)
        region_counts = counts[region_mask]
        
        if len(region_counts) == 0:
            continue
            
        # Count lines centered in this region
        line_mask = wl_mask & (catalog.wavelength >= wl_lo) & (catalog.wavelength < wl_hi)
        n_lines = np.sum(line_mask)
        
        # Count very weak lines
        very_weak_mask = line_mask & (catalog.log_gf < -5)
        n_very_weak = np.sum(very_weak_mask)
        
        print(f"\n  {name} ({wl_lo}-{wl_hi} nm):")
        print(f"    Lines centered here: {n_lines:,}")
        print(f"    Very weak (loggf < -5): {n_very_weak:,} ({100*n_very_weak/max(n_lines,1):.1f}%)")
        print(f"    Lines per bin - mean: {np.mean(region_counts):.1f}, max: {np.max(region_counts)}")
        
        # Calculate total gf * Boltzmann for very weak lines
        # This shows if they can accumulate to significant opacity
        if n_very_weak > 0:
            vw_gf = catalog.gf[very_weak_mask]
            vw_e = catalog.excitation_energy[very_weak_mask]
            # Assume T = 3750 K for Boltzmann factor
            hckt = 1.4388 / 3750.0
            boltz = np.exp(-vw_e * hckt)
            gf_boltz_sum = np.sum(vw_gf * boltz)
            print(f"    Very weak gf*Boltz sum: {gf_boltz_sum:.3e}")
    
    # Show sample of lines near 307 nm (where we traced the Ce II line)
    print(f"\n" + "=" * 80)
    print("SAMPLE: Lines within 0.1 nm of 307.2 nm")
    print("=" * 80)
    
    sample_mask = wl_mask & (np.abs(catalog.wavelength - 307.2) < 0.1)
    sample_indices = np.where(sample_mask)[0]
    
    print(f"\nFound {len(sample_indices)} lines")
    
    if len(sample_indices) > 0:
        # Sort by gf (strongest first)
        sorted_idx = sample_indices[np.argsort(-catalog.gf[sample_indices])]
        
        print("\nTop 10 by gf:")
        print(f"{'Wavelength':>12} {'Element':>6} {'Ion':>3} {'log(gf)':>8} {'E_lo':>10}")
        print("-" * 45)
        for i in sorted_idx[:10]:
            rec = catalog.records[i]
            print(f"{rec.wavelength:12.6f} {rec.element:>6} {rec.ion_stage:>3} {rec.log_gf:8.4f} {rec.excitation_energy:10.2f}")
        
        print(f"\nVery weak lines (loggf < -5):")
        very_weak_sample = sample_indices[catalog.log_gf[sample_indices] < -5]
        print(f"  Count: {len(very_weak_sample)}")
        if len(very_weak_sample) > 0:
            print(f"  Total gf: {np.sum(catalog.gf[very_weak_sample]):.3e}")
            
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze line distribution per wavelength")
    parser.add_argument("--atmosphere", type=Path, required=True, help="NPZ atmosphere file")
    parser.add_argument("--catalog", type=Path, required=True, help="Line catalog (gfallvac)")
    parser.add_argument("--wl-start", type=float, default=300.0, help="Start wavelength (nm)")
    parser.add_argument("--wl-end", type=float, default=500.0, help="End wavelength (nm)")
    parser.add_argument("--resolution", type=float, default=300_000.0, help="Resolving power")
    args = parser.parse_args()
    
    analyze_line_distribution(
        atmos_path=args.atmosphere,
        catalog_path=args.catalog,
        wl_start=args.wl_start,
        wl_end=args.wl_end,
        resolution=args.resolution,
    )


if __name__ == "__main__":
    main()

