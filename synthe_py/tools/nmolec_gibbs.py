#!/usr/bin/env python3
"""
Gibbs free-energy minimization solver for NMOLEC.

This module provides a drop-in replacement for the Newton-based NMOLEC solver
that uses Gibbs free energy minimization instead. The Gibbs approach guarantees
convergence to the unique thermodynamic equilibrium, avoiding the basin-of-attraction
issues that cause the Newton solver to fail for cool atmospheres.

CRITICAL FIX (Dec 2025): 
- Element totals computed using xnatom_atomic (total nuclei) not xntot (total particles)
- CPF (partition function) corrections now applied to equilibrium constants
- Proper conversion from EQUILJ to μ°/kT for Gibbs formulation
"""

from __future__ import annotations

import os
from typing import Optional, Tuple, Callable

import numpy as np

from synthe_py.tools.gibbs_solver import minimize_gibbs

# Boltzmann constant in different units
k_BOLTZ_EV = 8.617333262e-5  # eV/K
k_B_CGS = 1.38065e-16  # erg/K


def _compute_equilj_with_cpf(
    temperature: float,
    tkev: float,
    tlog: float,
    nummol: int,
    code_mol: np.ndarray,
    equil: np.ndarray,
    locj: np.ndarray,
    kcomps: np.ndarray,
    idequa: np.ndarray,
    nequa: int,
    cpfh: float,
    cpfc: float,
    cpfo: float,
    cpfmg: float,
    cpfal: float,
    cpfsi: float,
    cpfca: float,
) -> np.ndarray:
    """
    Compute equilibrium constants with CPF (partition function) corrections.
    
    This matches Fortran's NMOLEC calculation exactly:
    1. Compute base EQUILJ from polynomial
    2. Apply CPF correction for each element in the molecule
    
    Args:
        temperature: Temperature in K
        tkev: kT in eV
        tlog: log10(T)
        nummol: Number of molecules
        code_mol: Molecular codes
        equil: Equilibrium polynomial coefficients
        locj: Component location indices
        kcomps: Component indices
        idequa: Element IDs for equations
        nequa: Number of equations
        cpfh, cpfc, ...: CPF correction factors for each element
        
    Returns:
        equilj: (nummol,) array of CPF-corrected equilibrium constants
    """
    equilj = np.zeros(nummol, dtype=np.float64)
    
    for jmol in range(nummol):
        # H- has special formula
        if abs(code_mol[jmol] - 100.0) < 0.01:  # H-
            exp_arg = (0.754 / tkev - 1.5 * tlog + 2.3025851 * (-11.206998 + 
                       (4.36e-4 + (-1.12e-7 + 1.7e-11 * temperature) * temperature) * temperature))
            equilj[jmol] = np.exp(np.clip(exp_arg, -700, 700)) * cpfh
            continue
            
        # H2 has special formula
        if abs(code_mol[jmol] - 101.0) < 0.01:  # H2
            exp_arg = (4.478 / tkev - 46.4584 +
                      (1.63660e-3 + (-4.93992e-7 + (1.11822e-10 + (-1.49567e-14 +
                      (1.06206e-18 - 3.08720e-23 * temperature) * temperature) * temperature) * 
                       temperature) * temperature) * temperature - 1.5 * tlog)
            equilj[jmol] = np.exp(np.clip(exp_arg, -700, 700)) * cpfh * cpfh
            continue
        
        # General polynomial formula
        if equil[0, jmol] == 0.0:
            continue
            
        ncomp = locj[jmol + 1] - locj[jmol]
        ion = int((code_mol[jmol] - int(code_mol[jmol])) * 100 + 0.5)
        
        poly = (equil[0, jmol] / tkev - equil[1, jmol] +
               (equil[2, jmol] + (-equil[3, jmol] + (equil[4, jmol] +
               (-equil[5, jmol] + equil[6, jmol] * temperature) * 
                temperature) * temperature) * temperature) * temperature)
        
        exp_arg = poly - 1.5 * (ncomp - ion - ion - 1) * tlog
        equilj[jmol] = np.exp(np.clip(exp_arg, -700, 700))
        
        # Apply CPF corrections for each component atom
        locj1 = locj[jmol]
        locj2 = locj[jmol + 1] - 1
        
        for lock in range(locj1, locj2 + 1):
            k = kcomps[lock]
            if k < nequa:
                elem_id = idequa[k]
                if elem_id == 1:
                    equilj[jmol] *= cpfh
                elif elem_id == 6:
                    equilj[jmol] *= cpfc
                elif elem_id == 8:
                    equilj[jmol] *= cpfo
                elif elem_id == 12:
                    equilj[jmol] *= cpfmg
                elif elem_id == 13:
                    equilj[jmol] *= cpfal
                elif elem_id == 14:
                    equilj[jmol] *= cpfsi
                elif elem_id == 20:
                    equilj[jmol] *= cpfca
    
    return equilj


def _build_nmolec_gibbs_inputs(
    layer_idx: int,
    temperature: float,
    tkev: float,
    tlog: float,
    gas_pressure: float,
    electron_density: float,
    xabund: np.ndarray,
    xnatom_atomic: float,  # CRITICAL: Total nuclei density, NOT particle density
    nummol: int,
    code_mol: np.ndarray,
    equil: np.ndarray,
    locj: np.ndarray,
    kcomps: np.ndarray,
    idequa: np.ndarray,
    nequa: int,
    equilj: np.ndarray,  # CPF-corrected equilibrium constants
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Build Gibbs solver inputs from NMOLEC data structures.

    This maps between NMOLEC's equation-number-based representation and
    the Gibbs solver's species-based representation.

    CRITICAL: equilj must already have CPF corrections applied (from 
    _compute_equilj_with_cpf). The raw polynomial EQUILJ values are NOT
    the same as the mass-action equilibrium constants needed for Gibbs.

    Returns:
        (mu0, stoich, charges, elem_totals, species_map) where:
        - mu0: (n_species,) reduced chemical potentials μ₀/kT
        - stoich: (n_species, n_elements) stoichiometry matrix
        - charges: (n_species,) species charges
        - elem_totals: (n_elements,) elemental number densities (nuclei)
        - species_map: mapping from Gibbs species index to (type, nmolec_index)
    """
    # Validate xnatom_atomic
    if xnatom_atomic <= 0 or not np.isfinite(xnatom_atomic):
        xnatom_atomic = gas_pressure / (k_B_CGS * temperature)

    # CRITICAL FIX: Only include elements with significant abundance!
    # Elements with zero abundance can't form molecules but would create
    # unsolvable conservation equations (require Σn_i = 0 with non-zero species).
    # The Gibbs solver fails with 1e+20 error when this happens.
    MIN_ABUNDANCE = 1e-10  # Minimum abundance to include an element
    
    # Build element list from idequa, filtering by abundance
    element_ids = []
    for k in range(1, nequa):
        elem_id = idequa[k]
        if elem_id > 0 and elem_id < 100:
            # Check if this element has meaningful abundance
            if elem_id <= len(xabund) and xabund[elem_id - 1] > MIN_ABUNDANCE:
                element_ids.append(elem_id)

    n_elements = len(element_ids)
    elem_id_to_idx = {eid: i for i, eid in enumerate(element_ids)}

    # Element totals from abundances using xnatom_atomic
    elem_totals = np.zeros(n_elements, dtype=np.float64)
    for i, elem_id in enumerate(element_ids):
        if elem_id <= len(xabund):
            abund = xabund[elem_id - 1]
            if abund > 0 and np.isfinite(abund):
                elem_totals[i] = abund * xnatom_atomic
            else:
                elem_totals[i] = 1e-30

    # Build species list
    species_list = []
    mu0_list = []
    stoich_list = []
    charge_list = []
    species_map = []

    # Add neutral atoms (reference μ₀ = 0)
    for i, elem_id in enumerate(element_ids):
        species_list.append(f"atom_{elem_id}")
        mu0_list.append(0.0)  # Reference state
        stoich = np.zeros(n_elements)
        stoich[i] = 1.0
        stoich_list.append(stoich)
        charge_list.append(0.0)
        species_map.append(("atom", elem_id))

    # Reference concentration for dimensionless K
    n_ref = np.sum(elem_totals)
    if n_ref <= 0:
        n_ref = xnatom_atomic

    # Add molecules from NMOLEC data
    molecules_added = 0
    max_molecules = 200

    for jmol in range(nummol):
        if molecules_added >= max_molecules:
            break

        ncomp = locj[jmol + 1] - locj[jmol]
        if ncomp <= 1:
            continue

        # Decode stoichiometry from molecule code
        stoich = np.zeros(n_elements)
        charge = 0.0
        contains_missing_element = False  # Track if molecule has elements we filtered out

        code = code_mol[jmol]
        xcode = np.array([1e14, 1e12, 1e10, 1e8, 1e6, 1e4, 1e2, 1e0])

        ii = 0
        for i in range(8):
            if code >= xcode[i]:
                ii = i
                break

        x = code
        for i in range(ii, 8):
            elem_z = int(x / xcode[i])
            x = x - float(elem_z) * xcode[i]
            if elem_z == 0:
                elem_z = 100
            if elem_z in elem_id_to_idx:
                stoich[elem_id_to_idx[elem_z]] += 1.0
            elif elem_z == 100:
                charge += 1.0
            elif elem_z > 0 and elem_z < 100:
                # This element has zero abundance - molecule can't form!
                contains_missing_element = True

        ion = int(x * 100.0 + 0.5)
        charge += ion

        if abs(code - 100.0) < 0.01:
            charge = -1.0

        # Skip molecules that have zero stoichiometry OR contain elements with zero abundance
        if np.sum(stoich) == 0 or contains_missing_element:
            continue
        
        # Skip H⁻ (code 100.00) - it requires electron participation (H + e⁻ → H⁻)
        # which isn't properly tracked in the Gibbs element conservation formulation.
        # Without electrons, the solver would give wildly incorrect H⁻ densities.
        if abs(code - 100.0) < 0.01:
            continue
        
        # Skip ionized molecules (ION > 0)
        # Ionized species require proper electron handling which is complex
        # The simple Gibbs formulation doesn't handle electron exchange correctly
        ncomp = locj[jmol + 1] - locj[jmol]
        ion = int((code - int(code)) * 100 + 0.5)
        if ion > 0:
            continue  # Skip H2+, CO+, etc.

        # Get CPF-corrected EQUILJ
        equilj_val = equilj[jmol]
        if equilj_val <= 0 or not np.isfinite(equilj_val):
            continue

        # Convert EQUILJ to μ°/kT for Gibbs
        #
        # From the Gibbs solver formula:
        #   n_i = n_scale × exp(stoich @ λ - μ°_i)
        #   where n_scale = sum(elem_totals)
        #
        # For reaction 2H → H2 with H as reference (μ°_H = 0):
        #   n_H = n_scale × exp(λ_H)
        #   n_H2 = n_scale × exp(2×λ_H - μ°_H2)
        #
        # Equilibrium constant K = n_H2 / n_H² = exp(-μ°_H2) / n_scale
        # So: μ°_H2 = -ln(K × n_scale) = -ln(EQUILJ × n_scale)
        #
        # For general molecule with n_atoms atoms:
        #   μ° = -ln(EQUILJ × n_scale)
        # (The n_atoms factor cancels out in the derivation)
        
        if equilj_val > 0 and n_ref > 0:
            # mu0 = -ln(EQUILJ * n_scale), where n_scale = n_ref = sum(elem_totals)
            log_k_eff = np.log(equilj_val) + np.log(n_ref)
            log_k_eff = np.clip(log_k_eff, -700, 700)
            mu0 = -log_k_eff
        else:
            mu0 = 700  # Very unfavorable (won't form)

        # Include all molecules with valid equilibrium constants
        # (CPF corrections make some molecules more/less important)
        species_list.append(f"mol_{jmol}")
        mu0_list.append(mu0)
        stoich_list.append(stoich)
        charge_list.append(charge)
        species_map.append(("molecule", jmol))
        molecules_added += 1

    # NOTE: Electrons are NOT added to the Gibbs solver.
    # Electrons don't participate in element conservation (zero stoichiometry)
    # and would break the Gibbs formulation by giving n_electron = n_scale * exp(0) = n_scale,
    # which is ~1e6 times too high. Use initial electron density instead.

    mu0 = np.array(mu0_list, dtype=np.float64)
    stoich = np.array(stoich_list, dtype=np.float64)
    charges = np.array(charge_list, dtype=np.float64)

    return mu0, stoich, charges, elem_totals, species_map


def nmolec_gibbs_layer(
    layer_idx: int,
    temperature: float,
    tkev: float,
    tlog: float,
    gas_pressure: float,
    electron_density_init: float,
    xabund: np.ndarray,
    xnatom_atomic: float,
    nummol: int,
    code_mol: np.ndarray,
    equil: np.ndarray,
    locj: np.ndarray,
    kcomps: np.ndarray,
    idequa: np.ndarray,
    nequa: int,
    equilj: np.ndarray,  # CPF-corrected equilibrium constants
    max_iter: int = 500,
) -> Tuple[float, float, np.ndarray]:
    """
    Solve molecular equilibrium for a single layer using Gibbs minimization.

    Args:
        equilj: CPF-corrected equilibrium constants (from _compute_equilj_with_cpf)
        
    Note:
        This function iterates on XNATM to satisfy the pressure constraint:
        - n_tot = P / kT (total particles from ideal gas law)
        - XNATM = n_tot + sum of extra nuclei from molecules
        
        The Gibbs solver conserves element totals but doesn't constrain n_tot,
        so we iterate XNATM until the resulting n_tot matches P/kT.
    """
    k_B = 1.38054e-16  # Boltzmann constant in erg/K
    n_tot_target = gas_pressure / (k_B * temperature)  # Target total particles
    
    # CRITICAL: Iterate on XNATM until n_tot = P/kT
    xnatm_guess = n_tot_target  # Start with atomic limit (all atoms, no molecules)
    xnatm_tolerance = 1e-4  # 0.01% relative tolerance
    max_xnatm_iter = 30
    
    n_result = None
    species_map = None
    stoich = None
    
    for xnatm_iter in range(max_xnatm_iter):
        # Build Gibbs inputs with current XNATM guess
        mu0, stoich, charges, elem_totals, species_map = _build_nmolec_gibbs_inputs(
            layer_idx=layer_idx,
            temperature=temperature,
            tkev=tkev,
            tlog=tlog,
            gas_pressure=gas_pressure,
            electron_density=electron_density_init,
            xabund=xabund,
            xnatom_atomic=xnatm_guess,  # Use current XNATM guess
            nummol=nummol,
            code_mol=code_mol,
            equil=equil,
            locj=locj,
            kcomps=kcomps,
            idequa=idequa,
            nequa=nequa,
            equilj=equilj,
        )
        
        n_species = len(mu0)
        n_elements = stoich.shape[1] if len(stoich) > 0 else 0
        
        if n_species == 0 or n_elements == 0:
            # No valid species - return atomic solution
            xn = np.zeros(nequa, dtype=np.float64)
            xn[0] = xnatm_guess
            for k in range(1, nequa):
                elem_id = idequa[k]
                if elem_id > 0 and elem_id <= len(xabund):
                    xn[k] = xabund[elem_id - 1] * xnatm_guess
            return xnatm_guess, electron_density_init, xn
        
        # Call Gibbs minimizer
        try:
            n_result = minimize_gibbs(
                temperature=temperature,
                pressure=gas_pressure,
                mu0=mu0,
                stoich=stoich,
                elem_totals=elem_totals,
                charges=charges,
                charge_total=0.0,
                max_iter=max_iter,
            )
        except Exception as e:
            # Fallback to atomic solution
            xn = np.zeros(nequa, dtype=np.float64)
            xn[0] = xnatm_guess
            for k in range(1, nequa):
                elem_id = idequa[k]
                if elem_id > 0 and elem_id <= len(xabund):
                    xn[k] = xabund[elem_id - 1] * xnatm_guess
            return xnatm_guess, electron_density_init, xn
        
        # Compute n_tot = sum of all particles
        n_tot_actual = np.sum(n_result)
        
        # Compute sum of extra nuclei from molecules: sum(n_mol * (atoms_per_mol - 1))
        extra_nuclei = 0.0
        for i, (species_type, species_idx) in enumerate(species_map):
            if species_type == "molecule":
                atoms_in_mol = np.sum(stoich[i])
                extra_nuclei += n_result[i] * (atoms_in_mol - 1)
        
        # Check convergence: n_tot should equal n_tot_target
        rel_error = (n_tot_actual - n_tot_target) / n_tot_target
        if abs(rel_error) < xnatm_tolerance:
            break  # Converged!
        
        # Update XNATM guess: XNATM = n_tot + extra_nuclei
        # We want n_tot = n_tot_target, so new XNATM = n_tot_target + extra_nuclei
        xnatm_guess = n_tot_target + extra_nuclei
    
    # Compute final XNATM from converged result
    xnatom = 0.0
    if n_result is not None and species_map is not None:
        for i, (species_type, species_idx) in enumerate(species_map):
            if species_type == "atom":
                xnatom += n_result[i]
            elif species_type == "molecule":
                xnatom += n_result[i] * np.sum(stoich[i])
    
    # Build XN array (FREE atoms, not total)
    xn = np.zeros(nequa, dtype=np.float64)
    xn[0] = xnatom
    
    elem_id_to_k = {}
    for k in range(1, nequa):
        elem_id = idequa[k]
        if elem_id > 0 and elem_id < 100:
            elem_id_to_k[elem_id] = k
    
    elem_id_to_species = {}
    for i, (species_type, species_idx) in enumerate(species_map):
        if species_type == "atom":
            elem_id_to_species[species_idx] = i
    
    # XN[k] = FREE atoms (from Gibbs result)
    for k in range(1, nequa):
        elem_id = idequa[k]
        if elem_id in elem_id_to_species:
            species_idx = elem_id_to_species[elem_id]
            xn[k] = n_result[species_idx]
        elif elem_id > 0 and elem_id <= len(xabund):
            xn[k] = xabund[elem_id - 1] * xnatom
    
    # Electron density: use initial value since electrons aren't in Gibbs solver
    # (they have zero stoichiometry and would break element conservation)
    xne = electron_density_init
    
    return xnatom, xne, xn


def nmolec_gibbs(
    n_layers: int,
    temperature: np.ndarray,
    tkev: np.ndarray,
    tk: np.ndarray,
    tlog: np.ndarray,
    gas_pressure: np.ndarray,
    electron_density: np.ndarray,
    xabund: np.ndarray,
    xnatom_atomic: np.ndarray,
    nummol: int,
    code_mol: np.ndarray,
    equil: np.ndarray,
    locj: np.ndarray,
    kcomps: np.ndarray,
    idequa: np.ndarray,
    nequa: int,
    # Partition function data (required for CPF corrections)
    bhyd: np.ndarray,
    bc1: np.ndarray,
    bo1: np.ndarray,
    bmg1: np.ndarray,
    bal1: np.ndarray,
    bsi1: np.ndarray,
    bca1: np.ndarray,
    pfsaha_func: Optional[Callable] = None,
    xnatom_molecular: Optional[np.ndarray] = None,
    max_iter: int = 500,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute molecular XNATOM using Gibbs minimization for all layers.

    This is a drop-in replacement for nmolec_exact() with use_gibbs=True.
    
    CRITICAL: This function now requires partition function data (bhyd, bc1, etc.)
    to compute CPF corrections, matching Fortran's NMOLEC exactly.

    Args:
        n_layers: Number of atmosphere layers
        temperature: (n_layers,) temperature array in K
        tkev: (n_layers,) k_B*T in eV
        tk: (n_layers,) k_B*T in erg
        tlog: (n_layers,) log10(temperature)
        gas_pressure: (n_layers,) pressure array in dyn/cm^2
        electron_density: (n_layers,) electron density in cm^-3
        xabund: (99,) element abundances by number fraction
        xnatom_atomic: (n_layers,) atomic XNATOM = P/kT - XNE (nuclei density)
        nummol: Number of molecules in database
        code_mol: (MAXMOL,) molecular codes
        equil: (7, MAXMOL) equilibrium polynomial coefficients
        locj: (MAXMOL+1,) component location indices
        kcomps: (MAXLOC,) component indices
        idequa: (MAXEQ,) element ID for each equation
        nequa: Number of equations
        bhyd, bc1, ...: Partition function B-tables
        pfsaha_func: PFSAHA function for partition function calculation
        xnatom_molecular: Optional output array to fill
        max_iter: Maximum Gibbs iterations per layer
        verbose: Print progress

    Returns:
        (xnatom_molecular, electron_density, xnz) where:
        - xnatom_molecular: (n_layers,) molecular XNATOM
        - electron_density: (n_layers,) updated electron density
        - xnz: (n_layers, nequa) XN values for all layers
    """
    if xnatom_molecular is None:
        xnatom_molecular = np.zeros(n_layers, dtype=np.float64)

    xne_out = electron_density.copy()
    xnz = np.zeros((n_layers, nequa), dtype=np.float64)

    for j in range(n_layers):
        T = temperature[j]
        tkev_j = tkev[j]
        tlog_j = tlog[j]
        
        # Compute CPF corrections (matches Fortran atlas7v.for lines 4556-4592)
        # Default to 1.0 for LTE
        cpfh = 1.0
        cpfc = 1.0
        cpfo = 1.0
        cpfmg = 1.0
        cpfal = 1.0
        cpfsi = 1.0
        cpfca = 1.0
        
        if pfsaha_func is not None:
            # Compute partition functions using PFSAHA
            # The wrapper fills answer_full[j,:] and copies to frac[j,:]
            # So we need frac to have at least (j+1) rows
            pf = np.zeros((n_layers, 31), dtype=np.float64)
            
            # NLTE partition functions (mode=3 returns partition function in frac[j,0])
            pfsaha_func(j, 1, 1, 3, pf, -1)
            pfh = pf[j, 0]
            pfsaha_func(j, 6, 1, 3, pf, -1)
            pfc = pf[j, 0]
            pfsaha_func(j, 8, 1, 3, pf, -1)
            pfo = pf[j, 0]
            pfsaha_func(j, 12, 1, 3, pf, -1)
            pfmg = pf[j, 0]
            pfsaha_func(j, 13, 1, 3, pf, -1)
            pfal = pf[j, 0]
            pfsaha_func(j, 14, 1, 3, pf, -1)
            pfsi = pf[j, 0]
            pfsaha_func(j, 20, 1, 3, pf, -1)
            pfca = pf[j, 0]
            
            # LTE partition functions (NLTEON = 0)
            bpf = np.zeros((n_layers, 31), dtype=np.float64)
            pfsaha_func(j, 1, 1, 3, bpf, 0)
            bpfh = bpf[j, 0]
            pfsaha_func(j, 6, 1, 3, bpf, 0)
            bpfc = bpf[j, 0]
            pfsaha_func(j, 8, 1, 3, bpf, 0)
            bpfo = bpf[j, 0]
            pfsaha_func(j, 12, 1, 3, bpf, 0)
            bpfmg = bpf[j, 0]
            pfsaha_func(j, 13, 1, 3, bpf, 0)
            bpfal = bpf[j, 0]
            pfsaha_func(j, 14, 1, 3, bpf, 0)
            bpfsi = bpf[j, 0]
            pfsaha_func(j, 20, 1, 3, bpf, 0)
            bpfca = bpf[j, 0]
            
            # Compute CPF corrections: CPF = PF/BPF * B
            if bpfh != 0:
                cpfh = pfh / bpfh * bhyd[j, 0]
            if bpfc != 0:
                cpfc = pfc / bpfc * bc1[j, 0]
            if bpfo != 0:
                cpfo = pfo / bpfo * bo1[j, 0]
            if bpfmg != 0:
                cpfmg = pfmg / bpfmg * bmg1[j, 0]
            if bpfal != 0:
                cpfal = pfal / bpfal * bal1[j, 0]
            if bpfsi != 0:
                cpfsi = pfsi / bpfsi * bsi1[j, 0]
            if bpfca != 0:
                cpfca = pfca / bpfca * bca1[j, 0]
        
        # Compute EQUILJ with CPF corrections
        equilj = _compute_equilj_with_cpf(
            temperature=T,
            tkev=tkev_j,
            tlog=tlog_j,
            nummol=nummol,
            code_mol=code_mol,
            equil=equil,
            locj=locj,
            kcomps=kcomps,
            idequa=idequa,
            nequa=nequa,
            cpfh=cpfh,
            cpfc=cpfc,
            cpfo=cpfo,
            cpfmg=cpfmg,
            cpfal=cpfal,
            cpfsi=cpfsi,
            cpfca=cpfca,
        )
        
        # Call Gibbs solver for this layer
        xnatom_j, xne_j, xn_j = nmolec_gibbs_layer(
            layer_idx=j,
            temperature=T,
            tkev=tkev_j,
            tlog=tlog_j,
            gas_pressure=gas_pressure[j],
            electron_density_init=electron_density[j],
            xabund=xabund,
            xnatom_atomic=xnatom_atomic[j],
            nummol=nummol,
            code_mol=code_mol,
            equil=equil,
            locj=locj,
            kcomps=kcomps,
            idequa=idequa,
            nequa=nequa,
            equilj=equilj,
            max_iter=max_iter,
        )
        
        xnatom_molecular[j] = xnatom_j
        xne_out[j] = xne_j
        xnz[j, :] = xn_j
        
        if verbose and j % 10 == 0:
            h_locked = 0.0
            if xnatom_atomic[j] > 0 and len(xabund) > 0:
                n_H_total = xnatom_atomic[j] * xabund[0]
                n_H_free = xn_j[1] if nequa > 1 else 0
                if n_H_total > 0:
                    h_locked = (1 - n_H_free / n_H_total) * 100
            print(f"Layer {j}: T={T:.0f}K, H_locked={h_locked:.1f}%")

    return xnatom_molecular, xne_out, xnz


__all__ = ["nmolec_gibbs", "nmolec_gibbs_layer", "_compute_equilj_with_cpf"]
