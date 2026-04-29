"""
Saha-Boltzmann population computation for computing XNFPEL from atmosphere properties.

This module implements the Fortran POPS/PFSAHA functionality to compute
ionization fractions and number densities from atmosphere properties,
removing the dependency on fort.10 pre-computed populations.

Key formulas:
1. Saha equation for ionization:
   N_{i+1}/N_i = (2*U_{i+1}/U_i) * (2πm_e kT/h²)^(3/2) * exp(-χ_i/kT) / n_e

2. Number density:
   N_i = (N_i/N_total) * N_atom * X_abund

Where:
- N_i = number density of ion i
- U_i = partition function of ion i
- χ_i = ionization potential
- n_e = electron density
- N_atom = total atomic number density
- X_abund = element abundance
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple
import numpy as np

if TYPE_CHECKING:
    from ..io.atmosphere import AtmosphereModel

# Physical constants (matching Fortran)
K_BOLTZ = 1.380649e-16  # erg / K
H_PLANCK = 6.62607015e-27  # erg * s
M_ELECTRON = 9.1093837e-28  # g
PI = 3.14159265358979323846

from .pfsaha_ion_pots import IONIZATION_POTENTIALS
from .pfsaha_partition import compute_pfsaha_partition_function


def _get_element_symbol(atomic_number: int) -> str:
    """Convert atomic number to element symbol."""
    elements = [
        "H",
        "HE",
        "LI",
        "BE",
        "B",
        "C",
        "N",
        "O",
        "F",
        "NE",
        "NA",
        "MG",
        "AL",
        "SI",
        "P",
        "S",
        "CL",
        "AR",
        "K",
        "CA",
        "SC",
        "TI",
        "V",
        "CR",
        "MN",
        "FE",
        "CO",
        "NI",
        "CU",
        "ZN",
        "GA",
        "GE",
        "AS",
        "SE",
        "BR",
        "KR",
        "RB",
        "SR",
        "Y",
        "ZR",
        "NB",
        "MO",
        "TC",
        "RU",
        "RH",
        "PD",
        "AG",
        "CD",
        "IN",
        "SN",
        "SB",
        "TE",
        "I",
        "XE",
        "CS",
        "BA",
        "LA",
        "CE",
        "PR",
        "ND",
        "PM",
        "SM",
        "EU",
        "GD",
        "TB",
        "DY",
        "HO",
        "ER",
        "TM",
        "YB",
        "LU",
        "HF",
        "TA",
        "W",
        "RE",
        "OS",
        "IR",
        "PT",
        "AU",
        "HG",
        "TL",
        "PB",
        "BI",
        "PO",
        "AT",
        "RN",
        "FR",
        "RA",
        "AC",
        "TH",
        "PA",
        "U",
        "NP",
        "PU",
        "AM",
        "CM",
        "BK",
        "CF",
        "ES",
        "FM",
        "MD",
        "NO",
        "LR",
        "RF",
        "DB",
        "SG",
        "BH",
        "HS",
        "MT",
        "DS",
    ]
    if 1 <= atomic_number <= len(elements):
        return elements[atomic_number - 1]
    return "UNKNOWN"


def compute_partition_function(
    element: str,
    ion_stage: int,
    temperature: float,
    electron_density: Optional[float] = None,
    gas_pressure: Optional[float] = None,
) -> float:
    """
    Compute partition function for element/ion at given temperature using PFSAHA tables.

    This uses the exact Fortran PFSAHA partition function logic from extracted NNN tables.

    Parameters
    ----------
    element:
        Element symbol (e.g., "FE", "C")
    ion_stage:
        Ionization stage (1 = neutral, 2 = once ionized, etc.)
    temperature:
        Temperature in K
    electron_density:
        Electron density in cm^-3 (for Debye screening, optional)
    gas_pressure:
        Gas pressure in dyn/cm^2 (for Debye screening, optional)

    Returns
    -------
    partition_function:
        Partition function value
    """
    # Normalise element key to uppercase to match internal tables
    key = element.strip().upper()

    # Get atomic number
    from ..engine.opacity import _element_atomic_number

    atomic_number = _element_atomic_number(key)

    if atomic_number is None:
        return 1.0

    # Use PFSAHA partition function calculation
    return compute_pfsaha_partition_function(
        atomic_number=atomic_number,
        ion_stage=ion_stage,
        temperature=temperature,
        electron_density=electron_density,
        gas_pressure=gas_pressure,
    )


def compute_ionization_fraction(
    element: str,
    ion_stage: int,
    temperature: np.ndarray,
    electron_density: np.ndarray,
) -> np.ndarray:
    """
    Compute ionization fraction using Saha equation.

    Formula:
        N_{i+1}/N_i = (2*U_{i+1}/U_i) * (2πm_e kT/h²)^(3/2) * exp(-χ_i/kT) / n_e

    Parameters
    ----------
    element:
        Element symbol
    ion_stage:
        Ionization stage (1 = neutral, 2 = once ionized, etc.)
    temperature:
        Temperature array (K)
    electron_density:
        Electron density array (cm^-3)

    Returns
    -------
    ionization_fraction:
        Fraction of atoms in ion_stage+1 relative to ion_stage
    """
    # Normalise element key to uppercase to match internal tables
    key = element.strip().upper()

    if key not in IONIZATION_POTENTIALS:
        # Unknown element - return zeros
        return np.zeros_like(temperature)

    ip_array = IONIZATION_POTENTIALS[key]
    # ion_stage is 1-based (1=neutral, 2=once ionized), but array is 0-indexed
    # For ion_stage=1 (Ni I -> Ni II), we need IP[0] (Ni I IP)
    if ion_stage - 1 >= len(ip_array) or ion_stage < 1:
        # Unknown ion stage - return zeros
        return np.zeros_like(temperature)

    chi = ip_array[ion_stage - 1]  # eV (0-indexed: IP[ion_stage-1])
    chi_erg = chi * 1.602176634e-12  # Convert to erg

    # Prefactor: (2πm_e kT/h²)^(3/2)
    # This is the electron pressure term
    prefactor = (2.0 * PI * M_ELECTRON * K_BOLTZ / (H_PLANCK**2)) ** 1.5

    # Compute for each depth (partition functions are temperature-dependent)
    n_depths = temperature.size
    ionization_ratio = np.zeros(n_depths, dtype=np.float64)

    # Get gas pressure if available (for Debye screening in partition functions)
    gas_pressure = None
    if hasattr(temperature, "gas_pressure"):
        gas_pressure = getattr(temperature, "gas_pressure", None)

    for idx in range(n_depths):
        T = max(temperature[idx], 1.0)
        n_e = max(electron_density[idx], 1e-40)

        # Compute partition functions at this depth
        u_i = compute_partition_function(element, ion_stage, T, n_e, gas_pressure)
        u_ip1 = compute_partition_function(element, ion_stage + 1, T, n_e, gas_pressure)

        # Saha equation
        # N_{i+1}/N_i = (2*U_{i+1}/U_i) * (2πm_e kT/h²)^(3/2) * exp(-χ_i/kT) / n_e
        saha_term = (
            (2.0 * u_ip1 / max(u_i, 1e-40))
            * (prefactor * (T**1.5))
            * np.exp(-chi_erg / (K_BOLTZ * T))
            / n_e
        )

        ionization_ratio[idx] = saha_term

    return ionization_ratio


def compute_population_densities(
    atmosphere: "AtmosphereModel",
    element: str,
    max_ion_stage: int = 6,
) -> np.ndarray:
    """
    Compute population densities for all ion stages of an element.

    This is equivalent to Fortran POPS subroutine.

    Parameters
    ----------
    atmosphere:
        Atmosphere model
    element:
        Element symbol (e.g., "FE", "C")
    max_ion_stage:
        Maximum ion stage to compute (default 6)

    Returns
    -------
    populations:
        Array of shape (n_depths, max_ion_stage) containing number densities
        for each ion stage (cm^-3)
    """
    n_depths = atmosphere.layers

    # Get element atomic number
    from ..engine.opacity import _element_atomic_number

    atomic_number = _element_atomic_number(element)

    if atomic_number is None:
        return np.zeros((n_depths, max_ion_stage), dtype=np.float64)

    # Get element abundance
    # CRITICAL FIX: Use xabund array if available (matches Fortran XABUND)
    x_abund = None
    # First, try to get xabund array directly from atmosphere model (99-element array, linear abundances)
    if hasattr(atmosphere, "xabund") and atmosphere.xabund is not None:
        xabund_array = atmosphere.xabund
        if isinstance(xabund_array, np.ndarray) and len(xabund_array) >= atomic_number:
            x_abund = float(xabund_array[atomic_number - 1])  # xabund is 0-indexed, atomic_number is 1-indexed
    # Fallback: try metadata
    elif hasattr(atmosphere, "metadata") and atmosphere.metadata:
        if "xabund" in atmosphere.metadata:
            xabund_array = atmosphere.metadata["xabund"]
            if isinstance(xabund_array, np.ndarray) and len(xabund_array) >= atomic_number:
                x_abund = float(xabund_array[atomic_number - 1])  # xabund is 0-indexed, atomic_number is 1-indexed
        # Fallback: reconstruct from abundances dict
        elif "abundances" in atmosphere.metadata:
            abund_dict = atmosphere.metadata.get("abundances", {})
            abundance_scale = float(atmosphere.metadata.get("abundance_scale", 1.0))
            if isinstance(abund_dict, dict) and atomic_number in abund_dict:
                # Abundance might be in log format or linear
                abund_val = abund_dict[atomic_number]
                if atomic_number == 1 or atomic_number == 2:
                    # H and He: if negative, it's a log abundance → convert to linear
                    # if positive, it's already a mass fraction → use directly
                    if abund_val < 0.0:
                        x_abund = 10.0**abund_val
                    else:
                        x_abund = abund_val
                else:
                    # Other elements: if positive, convert to log first, then back
                    if abund_val > 0.0:
                        log_abund = np.log10(abund_val)
                    else:
                        log_abund = abund_val
                    x_abund = abundance_scale * (10.0**log_abund)

    # Fallback to solar abundance table if not found in metadata
    if x_abund is None:
        # Solar abundances relative to H (log scale, from Asplund et al. 2009)
        # Format: {atomic_number: log10(abundance_relative_to_H)}
        SOLAR_ABUNDANCES_LOG = {
            1: 0.0,  # H
            2: -1.07,  # He
            3: -10.95,  # Li
            4: -10.62,  # Be
            5: -9.3,  # B
            6: -3.57,  # C
            7: -4.17,  # N
            8: -3.31,  # O
            9: -7.44,  # F
            10: -4.08,  # Ne
            11: -5.71,  # Na
            12: -4.40,  # Mg
            13: -5.55,  # Al
            14: -4.49,  # Si
            15: -6.59,  # P
            16: -4.88,  # S
            17: -6.50,  # Cl
            18: -5.60,  # Ar
            19: -6.91,  # K
            20: -5.66,  # Ca
            21: -8.87,  # Sc
            22: -7.05,  # Ti
            23: -8.04,  # V
            24: -6.36,  # Cr
            25: -6.57,  # Mn
            26: -4.50,  # Fe
            27: -7.01,  # Co
            28: -5.78,  # Ni
            29: -7.81,  # Cu
            30: -7.44,  # Zn
        }
        if atomic_number in SOLAR_ABUNDANCES_LOG:
            x_abund = 10.0 ** SOLAR_ABUNDANCES_LOG[atomic_number]
        else:
            # Default: very small abundance for unknown elements
            x_abund = 1e-10

    # Ensure non-zero
    x_abund = max(x_abund, 1e-30)

    # Initialize populations
    populations = np.zeros((n_depths, max_ion_stage), dtype=np.float64)

    # Compute ionization fractions using Saha equation
    # Start from neutral (ion_stage = 1)
    ionization_ratios = {}
    for ion_stage in range(1, max_ion_stage):
        ionization_ratios[ion_stage] = compute_ionization_fraction(
            element, ion_stage, atmosphere.temperature, atmosphere.electron_density
        )

    # Compute number densities for each depth
    # CRITICAL FIX (Dec 2025): Store N/U (number density / partition function)
    # to match Fortran's XNFPEL definition. Fortran PFSAHA MODE=1 returns
    # ionization_fraction/U, then POPS multiplies by XNATOM*XABUND to get N/U.
    for depth_idx in range(n_depths):
        # Get temperature and electron density for partition function
        T = atmosphere.temperature[depth_idx]
        n_e = atmosphere.electron_density[depth_idx]

        # Get total atomic number density
        # From Fortran: XNATOM = total number of atoms (excluding electrons)
        # Try to get from atmosphere if available (stored as xnatm in NPZ)
        if hasattr(atmosphere, "xnatm") and atmosphere.xnatm is not None:
            n_atom = atmosphere.xnatm[depth_idx]
        elif hasattr(atmosphere, "gas_pressure"):
            # Approximate from pressure and temperature
            # P = (N_atom + N_e) * kT
            # N_atom ≈ P / (kT) - N_e (approximate)
            P = atmosphere.gas_pressure[depth_idx]

            if P > 0 and T > 0:
                n_total = P / (K_BOLTZ * T)
                n_atom = n_total - n_e  # No 1e-40 guard (Fortran doesn't use it)
            else:
                # Fallback: use electron density as proxy
                n_atom = n_e * 10.0  # Rough approximation
        else:
            # Last resort: use electron density as proxy
            n_atom = n_e * 10.0  # Rough approximation

        # Compute relative populations using ionization fractions
        # N_1 (neutral) is the base
        # N_2 = N_1 * (N_2/N_1)
        # N_3 = N_2 * (N_3/N_2) = N_1 * (N_2/N_1) * (N_3/N_2)
        # etc.

        # Start with neutral population
        # Sum of all ion stages = N_atom * X_abund
        # N_1 + N_2 + N_3 + ... = N_atom * X_abund
        # N_1 * (1 + r_1 + r_1*r_2 + r_1*r_2*r_3 + ...) = N_atom * X_abund

        # Compute sum of ratios
        sum_ratios = 1.0
        product = 1.0
        for ion_stage in range(1, max_ion_stage):
            r = ionization_ratios[ion_stage][depth_idx]
            product *= r
            sum_ratios += product

        # Neutral population (N)
        n_neutral = (n_atom * x_abund) / sum_ratios  # No 1e-40 guard

        # Compute partition function for neutral (ion_stage=1 in PFSAHA convention)
        U_neutral = compute_partition_function(element, 1, T, n_e)

        # Store N/U (matching Fortran's XNFPEL)
        populations[depth_idx, 0] = n_neutral / U_neutral

        # Compute populations for higher ion stages
        product = 1.0
        for ion_stage in range(1, max_ion_stage):
            r = ionization_ratios[ion_stage][depth_idx]
            product *= r
            n_ion = n_neutral * product

            # Compute partition function for this ion stage
            # ion_stage in loop is 1-based (1=first ionized), PFSAHA uses ion_stage+1
            U_ion = compute_partition_function(element, ion_stage + 1, T, n_e)

            # Store N/U
            populations[depth_idx, ion_stage] = n_ion / U_ion

    return populations


def compute_doppler_velocity(
    atmosphere: "AtmosphereModel",
    element: str,
) -> np.ndarray:
    """
    Compute Doppler velocity for an element.

    From xnfpelsyn.for line 381-382:
        DOPPLE = sqrt(2*TK/mass/1.660e-24 + VTURB^2) / c

    Parameters
    ----------
    atmosphere:
        Atmosphere model
    element:
        Element symbol

    Returns
    -------
    doppler_velocity:
        Doppler velocity in units of c (dimensionless), shape (n_depths,)
    """
    from ..engine.opacity import _atomic_mass_lookup

    atomic_mass = _atomic_mass_lookup(element)
    if atomic_mass is None:
        atomic_mass = 1.0  # Default to hydrogen mass

    n_depths = atmosphere.layers
    doppler_velocity = np.zeros(n_depths, dtype=np.float64)

    C_LIGHT = 2.99792458e10  # cm/s
    AMU_TO_G = 1.66053906660e-24  # g (atomic mass unit)

    for depth_idx in range(n_depths):
        T = atmosphere.temperature[depth_idx]
        v_turb = (
            atmosphere.turbulent_velocity[depth_idx]
            if hasattr(atmosphere, "turbulent_velocity")
            else 0.0
        )

        # Thermal velocity: sqrt(2kT/m)
        # TK = kT in Fortran
        kT = K_BOLTZ * T
        mass = atomic_mass * AMU_TO_G
        v_thermal_sq = 2.0 * kT / max(mass, 1e-40)

        # Add turbulent velocity
        # CRITICAL FIX: turbulent_velocity from atmosphere is already in cm/s
        # (from .atm file VTURB field, which is in cm/s)
        # Do NOT multiply by 1e5 - that would make it 1e5x too large!
        v_turb_cm = v_turb  # Already in cm/s
        v_total_sq = v_thermal_sq + v_turb_cm**2

        # Doppler velocity in units of c
        doppler_velocity[depth_idx] = np.sqrt(v_total_sq) / C_LIGHT

    return doppler_velocity
