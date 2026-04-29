"""Population and state computations for atmosphere layers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

from . import tables

if TYPE_CHECKING:  # pragma: no cover
    from ..io.atmosphere import AtmosphereModel


@dataclass
class HydrogenDepthState:
    pp: float
    fo: float
    y1b: float
    y1s: float
    t3nhe: float
    t3nh2: float
    c1d: float
    c2d: float
    gcon1: float
    gcon2: float
    xnfph: np.ndarray
    dopph: float


@dataclass
class DepthState:
    """Derived physical quantities for one atmospheric depth."""

    boltzmann_factor: np.ndarray
    doppler_width: np.ndarray
    turbulence_width: float
    electron_density: float
    temperature: float
    continuum_opacity: np.ndarray
    hckt: float
    txnxn: float
    hydrogen: Optional[HydrogenDepthState] = None


@dataclass
class Populations:
    """Holds precomputed populations for all depths."""

    layers: Dict[int, DepthState]


HCKT_COEFF = 11604.51812155008  # 1/(k_B) in eV/K (for eV energies)
HCKT_CM_COEFF = 1.4388  # hc/k in cm·K (for cm⁻¹ energies, from Fortran)
# Note: The catalog stores excitation_energy in cm⁻¹ units. Use HCKT_CM_COEFF for correct Boltzmann.
# However, there are other bugs in the code that compensate for using the wrong HCKT.
# Until those are found and fixed, using the eV coefficient produces better results.
KBOLTZ = 1.380649e-16  # erg/K
AMU = 1.66054e-24  # g (atomic mass unit, from Fortran xnfpelsyn.for 1.660D-24)
C_LIGHT_KMS = 299_792.458
C_LIGHT_CMS = 2.99792458e10

# Atomic masses (amu) for Doppler width calculation
# From Fortran SYNTHE's ATMASS array. Index is atomic number (1=H, 2=He, ...)
# This matches xnfpelsyn.for lines 488, 502 which use ATMASS(NELEM)
ATOMIC_MASSES = {
    'H': 1.008, 'He': 4.003, 'Li': 6.941, 'Be': 9.012, 'B': 10.81,
    'C': 12.01, 'N': 14.01, 'O': 16.00, 'F': 19.00, 'Ne': 20.18,
    'Na': 22.99, 'Mg': 24.31, 'Al': 26.98, 'Si': 28.09, 'P': 30.97,
    'S': 32.07, 'Cl': 35.45, 'Ar': 39.95, 'K': 39.10, 'Ca': 40.08,
    'Sc': 44.96, 'Ti': 47.87, 'V': 50.94, 'Cr': 52.00, 'Mn': 54.94,
    'Fe': 55.85, 'Co': 58.93, 'Ni': 58.69, 'Cu': 63.55, 'Zn': 65.38,
    'Ga': 69.72, 'Ge': 72.64, 'As': 74.92, 'Se': 78.96, 'Br': 79.90,
    'Kr': 83.80, 'Rb': 85.47, 'Sr': 87.62, 'Y': 88.91, 'Zr': 91.22,
    'Nb': 92.91, 'Mo': 95.96, 'Tc': 98.00, 'Ru': 101.07, 'Rh': 102.91,
    'Pd': 106.42, 'Ag': 107.87, 'Cd': 112.41, 'In': 114.82, 'Sn': 118.71,
    'Sb': 121.76, 'Te': 127.60, 'I': 126.90, 'Xe': 131.29, 'Cs': 132.91,
    'Ba': 137.33, 'La': 138.91, 'Ce': 140.12, 'Pr': 140.91, 'Nd': 144.24,
    'Pm': 145.00, 'Sm': 150.36, 'Eu': 151.96, 'Gd': 157.25, 'Tb': 158.93,
    'Dy': 162.50, 'Ho': 164.93, 'Er': 167.26, 'Tm': 168.93, 'Yb': 173.05,
    'Lu': 174.97, 'Hf': 178.49, 'Ta': 180.95, 'W': 183.84, 'Re': 186.21,
    'Os': 190.23, 'Ir': 192.22, 'Pt': 195.08, 'Au': 196.97, 'Hg': 200.59,
    'Tl': 204.38, 'Pb': 207.2, 'Bi': 208.98, 'Po': 209.00, 'At': 210.00,
    'Rn': 222.00, 'Fr': 223.00, 'Ra': 226.00, 'Ac': 227.00, 'Th': 232.04,
    'Pa': 231.04, 'U': 238.03, 'Np': 237.00, 'Pu': 244.00, 'Am': 243.00,
    'Cm': 247.00, 'Bk': 247.00, 'Cf': 251.00, 'Es': 252.00, 'Fm': 257.00,
}
DEFAULT_ATOMIC_MASS = 56.0  # Default to Fe mass for unknown elements


def _hydrogen_state(
    temperature: float,
    electron_density: float,
    xnf_he1: float,
    xnf_h2: float,
    xnfph: np.ndarray,
    dopph: float,
) -> HydrogenDepthState:
    xne = max(electron_density, 1e-40)
    xne16 = xne ** (1.0 / 6.0)
    pp = xne16 * 0.08989 / np.sqrt(max(temperature, 1.0))
    fo = (xne16 ** 4) * 1.25e-9
    y1b = 2.0 / (1.0 + 0.012 / max(temperature, 1.0) * np.sqrt(xne / max(temperature, 1.0)))
    t4 = temperature / 10000.0
    t43 = t4 ** 0.3
    y1s = t43 / xne16
    t3nhe = t43 * xnf_he1
    t3nh2 = t43 * xnf_h2
    c1d = fo * 78940.0 / max(temperature, 1.0)
    c2d = (fo ** 2) / 5.96e-23 / xne
    gcon1 = 0.2 + 0.09 * np.sqrt(max(t4, 1e-12)) / (1.0 + xne / 1.0e13)
    gcon2 = 0.2 / (1.0 + xne / 1.0e15)
    return HydrogenDepthState(
        pp=float(pp),
        fo=float(fo),
        y1b=float(y1b),
        y1s=float(y1s),
        t3nhe=float(t3nhe),
        t3nh2=float(t3nh2),
        c1d=float(c1d),
        c2d=float(c2d),
        gcon1=float(gcon1),
        gcon2=float(gcon2),
        xnfph=np.asarray(xnfph, dtype=np.float64),
        dopph=float(dopph),
    )


def compute_depth_state(
    atmosphere: AtmosphereModel,
    line_wavelengths: np.ndarray,
    excitation_energy: np.ndarray,
    microturb_kms: float,
    elements: Optional[np.ndarray] = None,
) -> Populations:
    """Compute LTE-like populations and Doppler widths per depth.
    
    CRITICAL FIX (Dec 2025): Use element-specific atomic masses for thermal velocity.
    Fortran xnfpelsyn.for line 488: DOPPLE = SQRT(2*TK/ATMASS(NELEM)/1.660D-24 + ...)
    Previous Python code used hydrogen mass for all elements, causing:
    - Fe lines: Doppler widths 7.5x too large (sqrt(56) factor)
    - This affects wing profiles and total line opacity
    """

    layers: Dict[int, DepthState] = {}
    line_wavelengths = np.asarray(line_wavelengths, dtype=np.float64)
    excitation_energy = np.asarray(excitation_energy, dtype=np.float64)
    
    # Pre-compute atomic masses for each line
    n_lines = len(line_wavelengths)
    atomic_masses = np.ones(n_lines, dtype=np.float64) * DEFAULT_ATOMIC_MASS
    if elements is not None:
        for i, elem in enumerate(elements):
            elem_str = str(elem)
            atomic_masses[i] = ATOMIC_MASSES.get(elem_str, DEFAULT_ATOMIC_MASS)

    for idx in range(atmosphere.layers):
        temp = max(float(atmosphere.temperature[idx]), 1.0)

        # CRITICAL: Use HCKT from atmosphere file (Fortran synthe.for line 268)
        # HCKT = hc/kT in cm units (computed by atlas7v.for, stored in fort.10/NPZ)
        # ELO is in cm⁻¹, so ELO * HCKT is dimensionless
        # This matches Fortran: KAPPA0 = KAPPA0 * FASTEX(ELO * HCKT(J))
        if atmosphere.hckt is not None and len(atmosphere.hckt) > idx:
            hckt = float(atmosphere.hckt[idx])
        else:
            # Fallback: compute hc/kT = 1.4388/T (in cm)
            hckt = 1.4388 / temp

        boltz = tables.fast_ex_array(np.asarray(excitation_energy, dtype=np.float64) * hckt)

        # CRITICAL FIX: Element-specific thermal velocities
        # Fortran xnfpelsyn.for line 488: SQRT(2*TK/ATMASS(NELEM)/1.660D-24 + TURBV²)
        # thermal_velocity = sqrt(2*k*T / m) / c = sqrt(2*k*T / (mass_amu * AMU)) / c
        thermal_velocities = np.sqrt(2.0 * KBOLTZ * temp / (atomic_masses * AMU)) / C_LIGHT_CMS
        
        # CRITICAL FIX (Jan 2026): Include synthesis microturbulence in DOPPLE.
        # Fortran synthe.for line 244:
        #   QDOPPLE = SQRT(QDOPPLE**2 + (TURBV/299792.458)**2)
        # where TURBV is the microturbulent velocity from the synthesis input.
        vturb_model_cms = (
            float(atmosphere.turbulent_velocity[idx])
            if atmosphere.turbulent_velocity.size > idx
            else 0.0
        )
        vturb_model_kms = vturb_model_cms / 1e5  # cm/s to km/s
        vturb_model = vturb_model_kms / C_LIGHT_KMS
        microturb = microturb_kms / C_LIGHT_KMS if microturb_kms > 0 else 0.0
        total_turb = math.sqrt(vturb_model * vturb_model + microturb * microturb)
        
        # Doppler width per line (element-specific thermal + turbulent)
        doppler_width = line_wavelengths * np.sqrt(total_turb**2 + thermal_velocities**2)

        if atmosphere.txnxn is not None:
            txnxn = float(atmosphere.txnxn[idx])
        else:
            # Compute TXNXN (perturber density for van der Waals broadening)
            # Original formula included a temperature scaling factor
            xnf_h = float(atmosphere.xnf_h[idx]) if atmosphere.xnf_h is not None else 0.0
            xnf_he1 = float(atmosphere.xnf_he1[idx]) if atmosphere.xnf_he1 is not None else 0.0
            xnf_h2 = float(atmosphere.xnf_h2[idx]) if atmosphere.xnf_h2 is not None else 0.0
            txnxn = (xnf_h + 0.42 * xnf_he1 + 0.85 * xnf_h2) * (temp / 10_000.0) ** 0.3

        hydrogen_state = None
        if atmosphere.xnf_he1 is not None or atmosphere.xnf_h2 is not None:
            xnf_he1 = float(atmosphere.xnf_he1[idx]) if atmosphere.xnf_he1 is not None else 0.0
            xnf_h2 = float(atmosphere.xnf_h2[idx]) if atmosphere.xnf_h2 is not None else 0.0
            xnfph = atmosphere.xnfph[idx] if atmosphere.xnfph is not None else np.zeros(2, dtype=np.float64)
            if atmosphere.dopph is not None:
                dopph = float(atmosphere.dopph[idx])
            else:
                # Match Fortran DOPPLE(1): thermal + turbulent in units of v/c.
                # Use hydrogen atomic mass (same table as line doppler widths).
                mass_h = ATOMIC_MASSES.get("H", 1.008)
                thermal_vel_h = math.sqrt(2.0 * KBOLTZ * temp / (mass_h * AMU)) / C_LIGHT_CMS
                dopph = math.sqrt(thermal_vel_h * thermal_vel_h + total_turb * total_turb)
            hydrogen_state = _hydrogen_state(
                temperature=temp,
                electron_density=float(atmosphere.electron_density[idx]),
                xnf_he1=xnf_he1,
                xnf_h2=xnf_h2,
                xnfph=xnfph,
                dopph=dopph,
            )

        state = DepthState(
            boltzmann_factor=boltz,
            doppler_width=doppler_width,
            turbulence_width=float(total_turb * C_LIGHT_KMS),
            electron_density=float(atmosphere.electron_density[idx]),
            temperature=temp,
            continuum_opacity=np.zeros(line_wavelengths.size, dtype=np.float64),
            hckt=hckt,
            txnxn=txnxn,
            hydrogen=hydrogen_state,
        )
        layers[idx] = state
    return Populations(layers=layers)
