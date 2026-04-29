#!/usr/bin/env python3
"""
Helpers to prepare Gibbs minimization inputs from existing Fortran data.

This scaffolds loading of thermo data (partition functions / dissociation
energies) and stoichiometry. It currently provides placeholders to keep the
pipeline explicit about required data and Fortran ground truth.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from synthe_py.tools.readmol_exact import readmol_exact

# Boltzmann in eV/K for ionization potentials
k_BOLTZ_EV = 8.617333262e-5

def load_molecules_stoich(molecules_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load molecule codes from molecules.dat (fort.2) and derive a stoichiometry
    matrix (species x elements). Species order matches code_mol from readmol_exact.

    NOTE: This is a scaffold. It decodes element IDs but does not yet map to
    full NMOLEC species ordering. It returns:
        stoich: (n_mol, max_elem_id) sparse-like matrix
        code_mol: (n_mol,) molecular codes
    """
    nummol, code_mol, _equil, locj, kcomps, _idequa, _nequa, _nloc = readmol_exact(
        molecules_path
    )
    max_elem = 102  # includes electron code=100, inverse electron=101 in Fortran
    stoich = np.zeros((nummol, max_elem), dtype=np.float64)

    for jm in range(nummol):
        start = locj[jm]
        end = locj[jm + 1]
        for iloc in range(start, end):
            eq_idx = kcomps[iloc]  # Fortran 1-based
            if 1 <= eq_idx <= max_elem:
                stoich[jm, eq_idx - 1] += 1.0
    return stoich, code_mol


def compute_molecular_mu0(
    temperature: float,
    tkev: float,
    tlog: float,
    equil: np.ndarray,
    code_mol: np.ndarray,
    locj: np.ndarray,
    nummol: int,
) -> np.ndarray:
    """
    Compute reduced chemical potentials mu0/kT for molecules from EQUILJ polynomial.

    For a dissociation reaction: AB → A + B (with equilibrium constant K)
    We have: ln(K) = (mu0_AB - mu0_A - mu0_B) / (-kT)
    If we set neutral atoms as reference (mu0 = 0):
        mu0_AB/kT = -ln(K) for a pure diatomic

    For molecules with multiple atoms and ions:
        mu0_mol/kT = -ln(EQUILJ) adjusted for stoichiometry

    Args:
        temperature: K
        tkev: temperature in keV (T / 11604.5)
        tlog: log10(temperature)
        equil: (7, MAXMOL) polynomial coefficients from molecules.dat
        code_mol: (nummol,) molecule codes
        locj: (nummol+1,) component location indices
        nummol: number of molecules

    Returns:
        mu0: (nummol,) reduced chemical potentials mu0/kT
    """
    mu0 = np.zeros(nummol, dtype=np.float64)

    for jmol in range(nummol):
        ncomp = locj[jmol + 1] - locj[jmol]

        if equil[0, jmol] == 0.0:
            # PFSAHA-based: will compute from ionization fractions later
            # For now, set to 0 (neutral atom reference)
            mu0[jmol] = 0.0
        else:
            # Use EQUIL polynomial to compute ln(EQUILJ)
            # Formula from atlas7v.for lines 4552-4555:
            # ln(EQUILJ) = E1/TKEV - E2 + (E3 + (-E4 + (E5 + (-E6 + E7*T)*T)*T)*T)*T
            #              - 1.5*(NCOMP-ION-ION-1)*TLOG
            ion = int((code_mol[jmol] - int(code_mol[jmol])) * 100 + 0.5)

            poly_term = (
                equil[0, jmol] / tkev
                - equil[1, jmol]
                + (
                    equil[2, jmol]
                    + (
                        -equil[3, jmol]
                        + (
                            equil[4, jmol]
                            + (-equil[5, jmol] + equil[6, jmol] * temperature)
                            * temperature
                        )
                        * temperature
                    )
                    * temperature
                )
                * temperature
            )

            ln_equilj = poly_term - 1.5 * (ncomp - ion - ion - 1) * tlog

            # For dissociation: AB → A + B with K_diss
            # ΔG° = μ°(A) + μ°(B) - μ°(AB) = -kT ln(K_diss)
            # If atoms are reference (μ°=0): μ°(AB) = kT ln(K_diss)
            # So μ₀/kT = ln(K_diss) = ln(EQUILJ)
            # NEGATIVE ln(EQUILJ) means molecule is MORE stable
            mu0[jmol] = ln_equilj

    return mu0


def placeholder_mu0_from_fortran(
    temperature: float,
    code_mol: np.ndarray,
    atlas_tables_path: Path | None = None,
) -> np.ndarray:
    """
    Placeholder: reduced chemical potentials mu0/kT for each species.

    To align with Fortran, this should be derived from:
      - Partition functions (pfsaha_levels / pfsaha_ion_pots, atlas_tables)
      - Dissociation energies for molecules
      - Ground state energies

    Currently returns zeros to keep the pipeline running. Replace with a
    proper Fortran-aligned implementation.
    """
    return np.zeros_like(code_mol, dtype=np.float64)


def load_ionization_potentials(path: Path) -> dict[str, np.ndarray]:
    """
    Load ionization potentials (eV) from pfsaha_ion_pots.npz.

    Returns a dict keyed by element symbol (or LO/LOLOG).
    This is a scaffold for building reduced mu0 for atomic/ionic species.
    """
    data = np.load(path)
    return {k: data[k] for k in data.files}


def compute_mu0_atomic_ions(
    temperature: float,
    element: str,
    max_ions: int = 3,
    ion_pots: dict[str, np.ndarray] | None = None,
) -> list[float]:
    """
    Compute reduced chemical potentials mu0/kT for atomic/ionic stages of one element.

    mu0_reduced(ion_stage) = +sum_{j=1..ion_stage} chi_j / (kT)
    where chi_j is the j-th ionization potential (eV), kT in eV.

    The POSITIVE sign means ions have HIGHER free energy than neutrals,
    which is correct because removing electrons costs energy.

    Notes:
      - This is a scaffold: proper mu0 should include partition functions.
      - ion_stage=0 is neutral, ion_stage=1 is singly ionized, etc.
    """
    if ion_pots is None:
        ion_pots = {}
    chi = ion_pots.get(element.upper())
    mu_list = []
    kt_ev = k_BOLTZ_EV * temperature
    for ion_stage in range(max_ions):
        if ion_stage == 0 or chi is None or len(chi) < ion_stage:
            mu_list.append(0.0)
        else:
            # cumulative ionization energy (POSITIVE: ions are less stable)
            mu_list.append(+np.sum(chi[:ion_stage]) / (kt_ev + 1e-30))
    return mu_list


def build_atomic_species_mu0(
    temperature: float,
    elements: list[str],
    max_ions: int,
    ion_pots: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build reduced mu0/kT, stoichiometry, and charges for atomic/ionic species.

    Species ordering: for each element in `elements`, include ion stages
    0..max_ions-1. For stage s, charge = s (neutral = 0).

    Returns:
        mu0: (n_species,) reduced mu0/kT
        stoich: (n_species, n_elements) with a 1.0 for the element of the species
        charges: (n_species,) charges (0,1,2,...)
    """
    n_species = len(elements) * max_ions
    n_elements = len(elements)
    mu0 = np.zeros(n_species, dtype=np.float64)
    stoich = np.zeros((n_species, n_elements), dtype=np.float64)
    charges = np.zeros(n_species, dtype=np.float64)

    idx = 0
    for elem_idx, elem in enumerate(elements):
        mu_list = compute_mu0_atomic_ions(
            temperature, elem, max_ions=max_ions, ion_pots=ion_pots
        )
        for stage in range(max_ions):
            mu0[idx] = mu_list[stage]
            stoich[idx, elem_idx] = 1.0
            charges[idx] = stage
            idx += 1

    return mu0, stoich, charges


def build_atomic_system_inputs(
    temperature: float,
    elements: list[str],
    elem_totals: np.ndarray,
    max_ions: int,
    ion_pots: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build mu0/kT, stoichiometry, charges, and elemental totals for an
    atomic/ionic-only system (no molecules).

    Args:
        temperature: K
        elements: list of element symbols matching elem_totals order
        elem_totals: array of elemental abundances (same order as elements)
        max_ions: number of ion stages per element (stage 0..max_ions-1)
        ion_pots: dict from load_ionization_potentials

    Returns:
        mu0 (n_species,), stoich (n_species, n_elements),
        charges (n_species,), elem_totals (n_elements,)
    """
    mu0, stoich, charges = build_atomic_species_mu0(
        temperature, elements, max_ions, ion_pots
    )
    elem_totals = np.asarray(elem_totals, dtype=np.float64)
    if elem_totals.shape[0] != len(elements):
        raise ValueError("elem_totals length must match elements list")
    return mu0, stoich, charges, elem_totals


def _decode_molecule_code(code: float) -> tuple[dict[int, int], int]:
    """
    Decode molecular code into element counts and ionization state.

    Args:
        code: Molecular code (e.g., 101.00 for H2, 608.00 for CO)

    Returns:
        (elem_counts, ion) where elem_counts is {atomic_number: count}
        and ion is the ionization state (0=neutral, 1=+1 ion, etc.)
    """
    xcode = np.array([1e14, 1e12, 1e10, 1e8, 1e6, 1e4, 1e2, 1e0])
    elem_counts: dict[int, int] = {}

    # Find starting power
    ii = 0
    for i in range(8):
        if code >= xcode[i]:
            ii = i
            break
    else:
        return elem_counts, 0  # Invalid code

    # Extract elements
    x = code
    for i in range(ii, 8):
        id_elem = int(x / xcode[i])
        x = x - float(id_elem) * xcode[i]
        if id_elem == 0:
            id_elem = 100  # Electron placeholder
        if id_elem not in elem_counts:
            elem_counts[id_elem] = 0
        elem_counts[id_elem] += 1

    # Extract ionization state (fractional part * 100)
    ion = int(x * 100.0 + 0.5)

    return elem_counts, ion


def build_full_system_inputs(
    temperature: float,
    elements: list[str],
    elem_totals: np.ndarray,
    max_ions: int,
    ion_pots: dict[str, np.ndarray],
    molecules_path: Path,
    include_pfsaha_molecules: bool = False,
) -> dict:
    """
    Build complete Gibbs solver inputs including atoms, ions, and molecules.

    This builds:
    - mu0/kT for neutral atoms (reference = 0)
    - mu0/kT for ions (from ionization potentials)
    - mu0/kT for molecules (from EQUILJ polynomial)
    - Combined stoichiometry matrix
    - Combined charge array

    Args:
        include_pfsaha_molecules: If False (default), only include molecules with
            polynomial equilibrium coefficients (equil[0] != 0). The PFSAHA-based
            entries (equil[0] == 0) in molecules.dat are typically atomic/ionic
            species which we build explicitly from ionization potentials.

    Returns a dict with keys:
        mu0, stoich, charges, elem_totals, species_codes, n_atomic, n_molecular
    """
    # Compute temperature-dependent values
    tkev = temperature * k_BOLTZ_EV  # kT in eV
    tlog = np.log10(temperature)

    from synthe_py.tools.readmol_exact import readmol_exact

    nummol, code_mol, equil, locj, kcomps, _idequa, _nequa, _nloc = readmol_exact(
        molecules_path
    )

    # Build atomic/ionic species
    mu0_atomic, stoich_atomic, charges_atomic = build_atomic_species_mu0(
        temperature, elements, max_ions, ion_pots
    )
    n_atomic = len(mu0_atomic)

    # Filter molecules: only include true molecules with polynomial coefficients
    # (equil[0] != 0) unless include_pfsaha_molecules is True
    if include_pfsaha_molecules:
        mol_indices = list(range(nummol))
    else:
        mol_indices = [i for i in range(nummol) if equil[0, i] != 0.0]

    n_mol_filtered = len(mol_indices)

    # Build molecular species mu0 from polynomial
    mu0_mol_all = compute_molecular_mu0(
        temperature, tkev, tlog, equil, code_mol, locj, nummol
    )
    mu0_mol = mu0_mol_all[mol_indices]

    # Build molecular stoichiometry by decoding molecule codes directly
    # (not using kcomps which has been remapped to equation numbers)
    n_elements = len(elements)
    elem_to_idx = {elem.upper(): i for i, elem in enumerate(elements)}

    # Element symbol lookup (atomic number -> symbol)
    z_to_sym = {
        1: "H",
        2: "HE",
        3: "LI",
        4: "BE",
        5: "B",
        6: "C",
        7: "N",
        8: "O",
        9: "F",
        10: "NE",
        11: "NA",
        12: "MG",
        13: "AL",
        14: "SI",
        15: "P",
        16: "S",
        17: "CL",
        18: "AR",
        19: "K",
        20: "CA",
        21: "SC",
        22: "TI",
        23: "V",
        24: "CR",
        25: "MN",
        26: "FE",
        27: "CO",
        28: "NI",
        29: "CU",
        30: "ZN",
    }

    mol_stoich_mapped = np.zeros((n_mol_filtered, n_elements), dtype=np.float64)
    mol_charges = np.zeros(n_mol_filtered, dtype=np.float64)
    mol_codes_filtered = []

    for out_idx, jmol in enumerate(mol_indices):
        mol_codes_filtered.append(code_mol[jmol])

        # Decode molecule code to get element counts and ionization
        elem_counts, ion = _decode_molecule_code(code_mol[jmol])

        # Map element counts to stoichiometry
        for z, count in elem_counts.items():
            if z == 100:
                # Electron in molecule structure (contributes to charge)
                continue
            sym = z_to_sym.get(z)
            if sym and sym in elem_to_idx:
                mol_stoich_mapped[out_idx, elem_to_idx[sym]] += count

        # Set charge based on ionization state
        # Positive ion: +ion charge; negative ion: we detect from code
        # Code 100.00 = H- (hydride ion), code ending in .01 = +1 ion
        if code_mol[jmol] == 100.0:
            # Special case: H- (hydride ion)
            mol_charges[out_idx] = -1.0
        else:
            mol_charges[out_idx] = float(ion)

    # Combine atomic and molecular
    mu0 = np.concatenate([mu0_atomic, mu0_mol])
    stoich = np.vstack([stoich_atomic, mol_stoich_mapped])
    charges = np.concatenate([charges_atomic, mol_charges])

    # Build species codes for identification
    species_codes = []
    for elem_idx, elem in enumerate(elements):
        for stage in range(max_ions):
            # Code format: Z.stage (e.g., 1.00 for H I, 1.01 for H II)
            z = list(z_to_sym.keys())[list(z_to_sym.values()).index(elem.upper())]
            species_codes.append(z + stage * 0.01)
    species_codes.extend(mol_codes_filtered)

    return {
        "mu0": mu0,
        "stoich": stoich,
        "charges": charges,
        "elem_totals": np.asarray(elem_totals, dtype=np.float64),
        "species_codes": np.array(species_codes),
        "n_atomic": n_atomic,
        "n_molecular": n_mol_filtered,
        "mol_indices": np.array(mol_indices),  # Indices into original molecules.dat
    }


__all__ = [
    "load_molecules_stoich",
    "placeholder_mu0_from_fortran",
    "load_ionization_potentials",
    "compute_mu0_atomic_ions",
    "build_atomic_species_mu0",
    "build_atomic_system_inputs",
    "compute_molecular_mu0",
    "build_full_system_inputs",
    "k_BOLTZ_EV",
]

