"""Trace TRANSP -> ASYNTH opacity loss for a specific line.

This script diagnoses the issue where TRANSP ~0.80 but ASYNTH ~7e-6.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from synthe_py.io.atmosphere import load_cached, AtmosphereModel
from synthe_py.io.lines import atomic
from synthe_py.physics import line_opacity, populations, tables
from synthe_py.physics.continuum import build_depth_continuum
from synthe_py.engine.opacity import _nearest_grid_indices, _element_atomic_number, C_LIGHT_NM

# Constants
H_PLANCK = 6.62607015e-27  # erg * s
K_BOLTZ = 1.380649e-16  # erg / K


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


def trace_line(
    atmos_path: Path,
    catalog_path: Path,
    target_wavelength: float = 307.1998,
    element: str = "CE",
    ion_stage: int = 2,
    window_nm: float = 1.0,
    resolution: float = 300_000.0,
    cutoff: float = 1e-3,
):
    """Trace a specific line through the TRANSP -> ASYNTH pipeline."""
    
    print("=" * 80)
    print(f"TRACING LINE: {element} {ion_stage} at {target_wavelength:.4f} nm")
    print("=" * 80)
    
    # Load atmosphere
    atmos = load_cached(atmos_path)
    print(f"\nAtmosphere: {atmos.layers} layers")
    print(f"  T[0] = {atmos.temperature[0]:.1f} K")
    print(f"  rho[0] = {atmos.mass_density[0]:.3e} g/cm³")
    print(f"  ne[0] = {atmos.electron_density[0]:.3e} cm⁻³")
    
    # Load catalog and find the line
    print(f"\nLoading catalog from {catalog_path}...")
    catalog_full = atomic.load_catalog(catalog_path)
    print(f"  Total lines: {len(catalog_full.records)}")
    
    # Find matching line
    deltas = np.abs(catalog_full.wavelength - target_wavelength)
    mask = deltas <= window_nm
    elem_mask = np.char.upper(catalog_full.elements.astype(str)) == element.upper()
    ion_mask = catalog_full.ion_stages == ion_stage
    combined_mask = mask & elem_mask & ion_mask
    
    candidates = np.where(combined_mask)[0]
    if len(candidates) == 0:
        print(f"ERROR: No {element} {ion_stage} lines found near {target_wavelength} nm")
        return
    
    best_idx = candidates[np.argmin(deltas[candidates])]
    rec = catalog_full.records[best_idx]
    
    print(f"\nSelected line:")
    print(f"  Wavelength: {rec.wavelength:.6f} nm")
    print(f"  Element: {rec.element}")
    print(f"  Ion stage: {rec.ion_stage}")
    print(f"  log(gf): {rec.log_gf:.4f}")
    print(f"  E_lower: {rec.excitation_energy:.2f} cm⁻¹")
    print(f"  gamma_rad: {rec.gamma_rad:.3e}")
    print(f"  gamma_stark: {rec.gamma_stark:.3e}")
    print(f"  gamma_vdw: {rec.gamma_vdw:.3e}")
    
    # Create single-line catalog
    catalog = atomic.LineCatalog.from_records([rec])
    
    # Build wavelength grid
    wl_start = target_wavelength - window_nm
    wl_end = target_wavelength + window_nm
    wavelength_grid = _build_wavelength_grid(wl_start, wl_end, resolution)
    n_wavelengths = len(wavelength_grid)
    n_depths = atmos.layers
    
    print(f"\nWavelength grid:")
    print(f"  Range: {wavelength_grid[0]:.6f} - {wavelength_grid[-1]:.6f} nm")
    print(f"  Points: {n_wavelengths}")
    print(f"  Grid spacing at center: {(wavelength_grid[n_wavelengths//2+1] - wavelength_grid[n_wavelengths//2]):.6f} nm")
    
    # Find where line maps to grid
    line_indices = _nearest_grid_indices(wavelength_grid, catalog.wavelength)
    center_idx = line_indices[0]
    
    print(f"\nLine-to-grid mapping:")
    print(f"  Line wavelength: {rec.wavelength:.6f} nm")
    print(f"  Center index: {center_idx}")
    if 0 <= center_idx < n_wavelengths:
        print(f"  Grid at center_idx: {wavelength_grid[center_idx]:.6f} nm")
        print(f"  Offset: {(wavelength_grid[center_idx] - rec.wavelength):.6f} nm")
    else:
        print(f"  WARNING: Line outside grid!")
    
    # Compute populations
    print("\nComputing populations...")
    pops = populations.compute_depth_state(
        atmosphere=atmos,
        line_wavelengths=catalog.wavelength,
        excitation_energy=catalog.excitation_energy,
        microturb_kms=0.0,
    )
    
    # Get element-specific population data
    atomic_num = _element_atomic_number(rec.element)
    print(f"\n{rec.element} (Z={atomic_num}) populations at depth 0:")
    if atomic_num is not None and atmos.population_per_ion is not None:
        elem_idx = atomic_num - 1
        if elem_idx < atmos.population_per_ion.shape[2]:
            pop_val = atmos.population_per_ion[0, rec.ion_stage - 1, elem_idx]
            dop_val = atmos.doppler_per_ion[0, rec.ion_stage - 1, elem_idx]
            print(f"  pop[0,{rec.ion_stage-1},{elem_idx}] = {pop_val:.3e}")
            print(f"  dop[0,{rec.ion_stage-1},{elem_idx}] = {dop_val:.3e}")
        else:
            print(f"  ERROR: elem_idx {elem_idx} >= shape[2] {atmos.population_per_ion.shape[2]}")
            return
    else:
        print("  ERROR: No population_per_ion array in atmosphere")
        return
    
    # Compute continuum
    print("\nComputing continuum...")
    cont_abs, cont_scat, _, _ = build_depth_continuum(atmos, wavelength_grid)
    print(f"  cont_abs shape: {cont_abs.shape}")
    if 0 <= center_idx < n_wavelengths:
        print(f"  cont_abs[0, center_idx] = {cont_abs[0, center_idx]:.3e}")
        kapmin = cont_abs[0, center_idx] * cutoff
        print(f"  KAPMIN = {kapmin:.3e} (cutoff={cutoff})")
    
    # Compute TRANSP
    print("\n" + "=" * 80)
    print("COMPUTING TRANSP")
    print("=" * 80)
    
    transp, valid_mask, _ = line_opacity.compute_transp(
        catalog=catalog,
        populations=pops,
        atmosphere=atmos,
        cutoff=cutoff,
        continuum_absorption=cont_abs,
        wavelength_grid=wavelength_grid,
    )
    
    print(f"\nTRANSP results:")
    print(f"  Shape: {transp.shape}")
    print(f"  valid_mask shape: {valid_mask.shape}")
    print(f"  Valid entries: {np.sum(valid_mask)}")
    print(f"  TRANSP[0, 0] (line 0, depth 0): {transp[0, 0]:.6e}")
    print(f"  TRANSP max: {np.max(transp):.6e}")
    print(f"  TRANSP non-zero count: {np.count_nonzero(transp)}")
    
    # Manual calculation to verify
    print("\n" + "-" * 40)
    print("MANUAL TRANSP VERIFICATION (depth 0)")
    print("-" * 40)
    
    rho = atmos.mass_density[0]
    xnfdop = pop_val / (rho * dop_val)
    gf_linear = 10.0 ** rec.log_gf
    freq_hz = C_LIGHT_NM / rec.wavelength
    CGF_CONSTANT = 0.026538 / 1.77245
    cgf = CGF_CONSTANT * gf_linear / freq_hz
    
    # Boltzmann factor
    temp = atmos.temperature[0]
    hckt = H_PLANCK * C_LIGHT_NM / (K_BOLTZ * temp)  # h*c/(k*T) with c in nm/s
    elo_cm = rec.excitation_energy  # E_lower in cm^-1
    boltz = math.exp(-elo_cm * hckt)  # hckt is dimensionless when multiplied by cm^-1
    
    # Actually, hckt should be in cm units for cm^-1 energies
    # h*c/(k*T) in cm = 1.4388 / T
    hckt_cm = 1.4388 / temp  # This is the correct Boltzmann constant for cm^-1
    boltz = math.exp(-elo_cm * hckt_cm)
    
    kappa0_pre_boltz = cgf * xnfdop
    kappa0 = kappa0_pre_boltz * boltz
    
    print(f"  rho = {rho:.3e}")
    print(f"  pop_val = {pop_val:.3e}")
    print(f"  dop_val = {dop_val:.3e}")
    print(f"  XNFDOP = pop/(rho*dop) = {xnfdop:.3e}")
    print(f"  gf = 10^{rec.log_gf:.4f} = {gf_linear:.3e}")
    print(f"  freq = {freq_hz:.3e} Hz")
    print(f"  CGF = {CGF_CONSTANT} * gf / freq = {cgf:.3e}")
    print(f"  kappa0_pre_boltz = CGF * XNFDOP = {kappa0_pre_boltz:.3e}")
    print(f"  E_lower = {elo_cm:.2f} cm^-1")
    print(f"  T = {temp:.1f} K")
    print(f"  hckt_cm = 1.4388/{temp} = {hckt_cm:.6f}")
    print(f"  boltz = exp(-E*hckt) = {boltz:.3e}")
    print(f"  kappa0 = kappa0_pre_boltz * boltz = {kappa0:.6e}")
    
    # Compute Voigt at center
    xne = atmos.electron_density[0]
    txnxn = atmos.txnxn[0] if atmos.txnxn is not None else 0.0
    gamma_total = rec.gamma_rad + rec.gamma_stark * xne + rec.gamma_vdw * txnxn
    doppler_width = dop_val * rec.wavelength
    delta_nu_doppler = (C_LIGHT_NM / rec.wavelength) * dop_val
    # Fortran: ADAMP = gamma_total / DOPPLE (gamma is pre-normalized by 4πν)
    adamp = gamma_total / dop_val if dop_val > 0 else 0.0
    
    print(f"\n  Damping:")
    print(f"    gamma_rad = {rec.gamma_rad:.3e}")
    print(f"    gamma_stark = {rec.gamma_stark:.3e}")
    print(f"    gamma_vdw = {rec.gamma_vdw:.3e}")
    print(f"    xne = {xne:.3e}")
    print(f"    txnxn = {txnxn:.3e}")
    print(f"    gamma_total = {gamma_total:.3e}")
    print(f"    doppler_width = {doppler_width:.6f} nm")
    print(f"    delta_nu_doppler = {delta_nu_doppler:.3e} Hz")
    print(f"    ADAMP = {adamp:.6f}")
    
    # Voigt at center (x=0)
    if adamp < 0.2:
        voigt_center = 1.0 - 1.128 * adamp
    else:
        from synthe_py.physics.profiles.voigt import voigt_profile
        voigt_center = voigt_profile(0.0, adamp)
    
    transp_manual = kappa0 * voigt_center
    print(f"\n  VOIGT(0, {adamp:.6f}) = {voigt_center:.6f}")
    print(f"  TRANSP_manual = kappa0 * voigt_center = {transp_manual:.6e}")
    print(f"  TRANSP_code   = {transp[0, 0]:.6e}")
    print(f"  Ratio: {transp[0, 0] / transp_manual if transp_manual > 0 else 'N/A':.3f}")
    
    # Now compute ASYNTH
    print("\n" + "=" * 80)
    print("COMPUTING ASYNTH")
    print("=" * 80)
    
    # Compute stimulated emission factor
    hkt = H_PLANCK / (K_BOLTZ * temp)
    stim_factor = 1.0 - math.exp(-freq_hz * hkt)
    asynth_expected = transp[0, 0] * stim_factor
    
    print(f"\nStimulated emission at depth 0:")
    print(f"  HKT = h/(k*T) = {hkt:.3e}")
    print(f"  freq * HKT = {freq_hz * hkt:.3f}")
    print(f"  stim_factor = 1 - exp(-freq*HKT) = {stim_factor:.6f}")
    print(f"  Expected ASYNTH[center] = TRANSP * stim = {asynth_expected:.6e}")
    
    # Run compute_asynth_from_transp
    asynth = line_opacity.compute_asynth_from_transp(
        transp=transp,
        catalog=catalog,
        atmosphere=atmos,
        wavelength_grid=wavelength_grid,
        valid_mask=valid_mask,
        populations=pops,
        cutoff=cutoff,
        continuum_absorption=cont_abs,
        metal_tables=tables.metal_wing_tables(),
    )
    
    print(f"\nASYNTH results:")
    print(f"  Shape: {asynth.shape}")
    print(f"  Non-zero count: {np.count_nonzero(asynth)}")
    print(f"  Max: {np.max(asynth):.6e}")
    
    if 0 <= center_idx < n_wavelengths:
        print(f"\n  At line center (depth 0, idx {center_idx}):")
        print(f"    ASYNTH[0, {center_idx}] = {asynth[0, center_idx]:.6e}")
        print(f"    Expected:               = {asynth_expected:.6e}")
        if asynth_expected > 0:
            ratio = asynth[0, center_idx] / asynth_expected
            print(f"    Ratio: {ratio:.3e}")
            if ratio < 0.99 or ratio > 1.01:
                print("    *** DISCREPANCY DETECTED! ***")
    
    # Check nearby grid points
    print(f"\n  ASYNTH around line center (depth 0):")
    for offset in range(-3, 4):
        idx = center_idx + offset
        if 0 <= idx < n_wavelengths:
            print(f"    [{idx}] wl={wavelength_grid[idx]:.6f} ASYNTH={asynth[0, idx]:.6e}")
    
    # Check all depths at center
    print(f"\n  ASYNTH at center (idx {center_idx}) across depths:")
    for depth in range(min(5, n_depths)):
        print(f"    depth {depth}: ASYNTH={asynth[depth, center_idx]:.6e} TRANSP={transp[0, depth]:.6e}")
    
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Trace ASYNTH loss for a specific line")
    parser.add_argument("--atmosphere", type=Path, required=True, help="NPZ atmosphere file")
    parser.add_argument("--catalog", type=Path, required=True, help="Line catalog (gfallvac)")
    parser.add_argument("--wavelength", type=float, default=307.1998, help="Target wavelength (nm)")
    parser.add_argument("--element", type=str, default="CE", help="Element symbol")
    parser.add_argument("--ion-stage", type=int, default=2, help="Ion stage")
    parser.add_argument("--window", type=float, default=1.0, help="Window half-width (nm)")
    parser.add_argument("--resolution", type=float, default=300_000.0, help="Resolving power")
    parser.add_argument("--cutoff", type=float, default=1e-3, help="Opacity cutoff")
    args = parser.parse_args()
    
    trace_line(
        atmos_path=args.atmosphere,
        catalog_path=args.catalog,
        target_wavelength=args.wavelength,
        element=args.element,
        ion_stage=args.ion_stage,
        window_nm=args.window,
        resolution=args.resolution,
        cutoff=args.cutoff,
    )


if __name__ == "__main__":
    main()

