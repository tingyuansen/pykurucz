#!/usr/bin/env python3
"""Compare XS (source function on XTAU grid) values between Python and Fortran."""

import numpy as np
from pathlib import Path
from synthe_py.io.atmosphere import load_cached
from synthe_py.physics.josh_solver import solve_josh_flux
from synthe_py.physics.josh_tables import XTAU_GRID
from synthe_py.physics.josh_solver import _map1
from synthe_py.engine.radiative import _planck_nu

_C_LIGHT_NM = 2.99792458e17  # nm/s

def main():
    """Compare XS values."""
    print("=" * 80)
    print("COMPARING XS (SOURCE FUNCTION ON XTAU GRID) VALUES")
    print("=" * 80)
    
    atm = load_cached('synthe_py/data/at12_aaaaa_atmosphere_fixed_interleaved.npz')
    wl = 490.0
    freq = _C_LIGHT_NM / wl
    
    # Get opacities
    from synthe_py.physics import continuum
    cont_abs, cont_scat, _, _ = continuum.build_depth_continuum(atm, np.array([wl]))
    
    # Reverse to match Fortran order (surface first)
    mass = atm.depth[::-1]
    temp = atm.temperature[::-1]
    cont_a = cont_abs[:, 0][::-1]
    cont_s = cont_scat[:, 0][::-1]
    line_a = np.zeros_like(cont_a)
    line_sig = np.zeros_like(cont_s)
    
    # Compute source function
    planck = _planck_nu(freq, temp)
    
    # Compute SNUBAR
    abtot = cont_a + line_a + cont_s + line_sig
    scatter = cont_s + line_sig
    alpha = scatter / np.maximum(abtot, 1e-38)
    denom = np.maximum(cont_a + line_a, 1e-38)
    snubar = (cont_a * planck + line_a * planck) / denom
    
    # Compute TAUNU
    from synthe_py.physics.josh_solver import _integ
    start = abtot[0] * mass[0] if mass.size else 0.0
    taunu = _integ(mass, abtot, start)
    if taunu.size:
        taunu = np.maximum.accumulate(taunu)
    
    # Map to XTAU grid
    xsbar, _ = _map1(taunu, snubar, XTAU_GRID)
    
    print(f'\nSource function values (first 10 on XTAU grid):')
    print(f'  XS[0:10]: {xsbar[:10]}')
    print(f'  BNU (surface): {planck[0]:.6E}')
    print(f'  SNUBAR (surface): {snubar[0]:.6E}')
    
    # Compute flux
    from synthe_py.physics.josh_tables import CH_WEIGHTS
    flux = np.dot(CH_WEIGHTS, xsbar)
    
    print(f'\nFlux computation:')
    print(f'  HNU(1) = sum(CH[i] * XS[i]): {flux:.6E}')
    print(f'  Expected (Fortran): 5.995358E-06')
    print(f'  Ratio: {flux / 5.995358E-06:.6f}×')
    
    # Check if multiplying CH_WEIGHTS by 4π helps
    ch_scaled = CH_WEIGHTS * (4 * np.pi)
    flux_scaled = np.dot(ch_scaled, xsbar)
    print(f'\nIf we multiply CH_WEIGHTS by 4π:')
    print(f'  HNU(1) = sum(4π*CH[i] * XS[i]): {flux_scaled:.6E}')
    print(f'  Ratio: {flux_scaled / 5.995358E-06:.6f}×')
    
    # Check if dividing by 4π helps
    flux_div = flux / (4 * np.pi)
    print(f'\nIf we divide by 4π:')
    print(f'  HNU(1) / 4π: {flux_div:.6E}')
    print(f'  Ratio: {flux_div / 5.995358E-06:.6f}×')

if __name__ == "__main__":
    main()

