"""Gas constants and simple Saha-Boltzmann utilities."""

from __future__ import annotations

import numpy as np

KBOLTZ = 1.380649e-16  # erg/K
HPLANCK = 6.62607015e-27  # erg*s
M_E = 9.10938356e-28  # g
M_H = 1.6735575e-24  # g
C_LIGHT = 2.99792458e10  # cm/s
EV_TO_ERG = 1.602176634e-12


def saha_factor(temperature: float, electron_density: float, ionization_energy_ev: float) -> float:
    """Return the Saha factor (Saha-Boltzmann equation) for ionisation equilibrium."""

    T = max(temperature, 1.0)
    chi = ionization_energy_ev * EV_TO_ERG
    prefactor = (2 * np.pi * M_E * KBOLTZ * T / HPLANCK**2) ** 1.5
    return prefactor / electron_density * np.exp(-chi / (KBOLTZ * T))


def boltzmann_factor(energy_ev: np.ndarray, temperature: float) -> np.ndarray:
    """Boltzmann factor exp(-E/kT) for energy array in eV."""

    beta = EV_TO_ERG / (KBOLTZ * max(temperature, 1.0))
    return np.exp(-energy_ev * beta)
