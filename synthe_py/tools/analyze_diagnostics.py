#!/usr/bin/env python3
"""Analyze diagnostics from a synthesis run."""

import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_diagnostics(diag_path: Path):
    """Load and analyze diagnostics from a .npz file."""
    
    print(f"Loading diagnostics from: {diag_path}")
    diag = np.load(diag_path)
    
    print("\n=== AVAILABLE KEYS ===")
    for key in sorted(diag.keys()):
        arr = diag[key]
        if isinstance(arr, np.ndarray):
            print(f"  {key}: shape {arr.shape}, dtype {arr.dtype}")
        else:
            print(f"  {key}: {type(arr)}")
    
    # Extract key arrays
    wavelength = diag.get('wavelength', None)
    if wavelength is None:
        print("\nERROR: 'wavelength' key not found!")
        return
    
    print(f"\n=== SPECTRUM OVERVIEW ===")
    print(f"Number of wavelength points: {wavelength.size}")
    print(f"Wavelength range: {wavelength[0]:.4f} - {wavelength[-1]:.4f} nm")
    
    # Check flux values
    flux_total = diag.get('flux_total', None)
    flux_continuum = diag.get('flux_continuum', None)
    
    if flux_total is not None and flux_continuum is not None:
        print(f"\n=== FLUX VALUES (first 10 points) ===")
        print(f"{'Wavelength (nm)':>15} {'Flux Total':>15} {'Flux Cont':>15} {'Ratio':>10}")
        print("-" * 60)
        for i in range(min(10, wavelength.size)):
            ratio = flux_total[i] / max(flux_continuum[i], 1e-40)
            print(f"{wavelength[i]:15.8f} {flux_total[i]:15.6E} {flux_continuum[i]:15.6E} {ratio:10.6f}")
        
        print(f"\n=== FLUX STATISTICS ===")
        print(f"Flux total - min: {np.min(flux_total):.6E}, max: {np.max(flux_total):.6E}, mean: {np.mean(flux_total):.6E}")
        print(f"Flux continuum - min: {np.min(flux_continuum):.6E}, max: {np.max(flux_continuum):.6E}, mean: {np.mean(flux_continuum):.6E}")
        
        ratio = flux_total / np.maximum(flux_continuum, 1e-40)
        print(f"Flux/Cont ratio - min: {np.min(ratio):.6f}, max: {np.max(ratio):.6f}, mean: {np.mean(ratio):.6f}")
    
    # Check opacity values
    cont_abs = diag.get('continuum_absorption', None)
    line_opacity = diag.get('line_opacity', None)
    
    if cont_abs is not None:
        print(f"\n=== CONTINUUM ABSORPTION ===")
        print(f"Shape: {cont_abs.shape}")
        print(f"Surface layer (first wavelength): {cont_abs[0, 0]:.6E}")
        if cont_abs.shape[0] > 1:
            print(f"Deepest layer (first wavelength): {cont_abs[-1, 0]:.6E}")
        print(f"Min: {np.min(cont_abs):.6E}, Max: {np.max(cont_abs):.6E}, Mean: {np.mean(cont_abs):.6E}")
    
    if line_opacity is not None:
        print(f"\n=== LINE OPACITY ===")
        print(f"Shape: {line_opacity.shape}")
        print(f"Surface layer (first wavelength): {line_opacity[0, 0]:.6E}")
        if line_opacity.shape[0] > 1:
            print(f"Deepest layer (first wavelength): {line_opacity[-1, 0]:.6E}")
        print(f"Min: {np.min(line_opacity):.6E}, Max: {np.max(line_opacity):.6E}, Mean: {np.mean(line_opacity):.6E}")
        
        # Check ratio
        if cont_abs is not None:
            ratio = line_opacity / np.maximum(cont_abs, 1e-40)
            print(f"\nLine/Continuum opacity ratio:")
            print(f"  Surface (first wavelength): {ratio[0, 0]:.6f}")
            print(f"  Min: {np.min(ratio):.6f}, Max: {np.max(ratio):.6f}, Mean: {np.mean(ratio):.6f}")
            print(f"  Points where line > cont: {np.sum(ratio > 1.0)} / {ratio.size} ({100*np.sum(ratio > 1.0)/ratio.size:.2f}%)")
    
    # Check source functions
    line_source = diag.get('line_source', None)
    if line_source is not None:
        print(f"\n=== LINE SOURCE FUNCTION ===")
        print(f"Shape: {line_source.shape}")
        print(f"Surface layer (first wavelength): {line_source[0, 0]:.6E}")
        print(f"Min: {np.min(line_source):.6E}, Max: {np.max(line_source):.6E}, Mean: {np.mean(line_source):.6E}")
    
    # Check specific wavelength point
    if wavelength.size > 0:
        idx_mid = wavelength.size // 2
        print(f"\n=== DETAILED CHECK (wavelength {wavelength[idx_mid]:.4f} nm, index {idx_mid}) ===")
        
        if cont_abs is not None:
            print(f"Continuum absorption (all layers):")
            print(f"  {cont_abs[:, idx_mid]}")
        
        if line_opacity is not None:
            print(f"Line opacity (all layers):")
            print(f"  {line_opacity[:, idx_mid]}")
        
        if flux_total is not None:
            print(f"Flux total: {flux_total[idx_mid]:.6E}")
        if flux_continuum is not None:
            print(f"Flux continuum: {flux_continuum[idx_mid]:.6E}")
    
    # Compare with ground truth if available
    ground_truth_path = Path("grids/at12_aaaaa/spec/at12_aaaaa_t05770g4.44.spec")
    if ground_truth_path.exists() and flux_total is not None:
        print(f"\n=== COMPARISON WITH GROUND TRUTH ===")
        gt_data = np.loadtxt(ground_truth_path)
        gt_wl = gt_data[:, 0]
        gt_flux = gt_data[:, 1]
        gt_cont = gt_data[:, 2]
        
        # Find matching wavelength points
        if wavelength.size == gt_wl.size:
            print(f"Matching {wavelength.size} wavelength points")
            
            # Compare first few points
            print(f"\nFirst 5 points comparison:")
            print(f"{'Wavelength':>15} {'Python Flux':>15} {'GT Flux':>15} {'Python Cont':>15} {'GT Cont':>15}")
            print("-" * 80)
            for i in range(min(5, wavelength.size)):
                print(f"{wavelength[i]:15.8f} {flux_total[i]:15.6E} {gt_flux[i]:15.6E} "
                      f"{flux_continuum[i]:15.6E} {gt_cont[i]:15.6E}")
            
            # Compare ratios
            python_ratio = flux_total / np.maximum(flux_continuum, 1e-40)
            gt_ratio = gt_flux / np.maximum(gt_cont, 1e-40)
            
            print(f"\nNormalized flux ratio comparison:")
            print(f"Python - min: {np.min(python_ratio):.6f}, max: {np.max(python_ratio):.6f}, mean: {np.mean(python_ratio):.6f}")
            print(f"Ground truth - min: {np.min(gt_ratio):.6f}, max: {np.max(gt_ratio):.6f}, mean: {np.mean(gt_ratio):.6f}")
            
            # Check magnitude differences
            flux_diff = np.abs(flux_total - gt_flux) / np.maximum(gt_flux, 1e-40)
            cont_diff = np.abs(flux_continuum - gt_cont) / np.maximum(gt_cont, 1e-40)
            
            print(f"\nRelative differences:")
            print(f"Flux - mean: {np.mean(flux_diff):.2%}, max: {np.max(flux_diff):.2%}")
            print(f"Continuum - mean: {np.mean(cont_diff):.2%}, max: {np.max(cont_diff):.2%}")
        else:
            print(f"Wavelength grid size mismatch: Python={wavelength.size}, GT={gt_wl.size}")
            print("Note: Python may be subsampled")
    
    print("\n=== ANALYSIS COMPLETE ===")

def main():
    if len(sys.argv) > 1:
        diag_path = Path(sys.argv[1])
    else:
        diag_path = Path("synthe_py/out/diagnostics_debug.npz")
    
    if not diag_path.exists():
        print(f"ERROR: Diagnostics file not found: {diag_path}")
        print("Usage: python analyze_diagnostics.py [path_to_diagnostics.npz]")
        sys.exit(1)
    
    analyze_diagnostics(diag_path)

if __name__ == "__main__":
    main()

