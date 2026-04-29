"""Hydrostatic-equilibrium updates matching atlas12.for iteration section."""

from __future__ import annotations

import numpy as np

from ..io.atmosphere import AtlasAtmosphere


def integrate_hydrostatic_pressure(
    atm: AtlasAtmosphere,
    gravity_cgs: float,
    prad: np.ndarray,
    pturb: np.ndarray,
    pcon: float,
) -> np.ndarray:
    """Compute gas pressure from hydrostatic balance.

    Fortran reference (`atlas12.for`, lines 226-229):
      P(J)=GRAV*RHOX(J)-PRAD(J)-PTURB(J)-PCON

    Units:
    - gravity_cgs: cm s^-2
    - RHOX: g cm^-2
    - PRAD/PTURB/PCON/P: dyn cm^-2
    """

    p = gravity_cgs * atm.rhox - prad - pturb - pcon
    if np.any(p <= 0.0):
        bad = int(np.argmin(p))
        raise ValueError(
            f"Hydrostatic pressure became non-positive at layer {bad}: P={p[bad]:.6e}"
        )
    return p


def update_total_pressure(
    rhox: np.ndarray,
    gravity_cgs: float,
    pzero: float,
) -> np.ndarray:
    """Fortran reference (`atlas12.for`, line 245): PTOTAL=GRAV*RHOX+PZERO."""

    return gravity_cgs * rhox + pzero

