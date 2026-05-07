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
        # Hot, line-blanketed RSG atmospheres (especially with α-poor /
        # CN-rich abundances) can transiently produce non-positive gas
        # pressure during the radiative-equilibrium iteration when PRAD
        # overshoots due to a TCORR step that pushed T too high.  Rather
        # than raise — which kills the run mid-iteration and leaves an
        # unusable .atm — we floor at a small positive value and let the
        # next TCORR step pull T (and hence PRAD) back down.  If the
        # iteration is genuinely diverging the convergence-monitor in
        # driver.py will catch it and exit cleanly with a usable .atm.
        floor = np.maximum(1e-6 * gravity_cgs * atm.rhox, 1e-30)
        bad_mask = p <= 0.0
        n_bad = int(bad_mask.sum())
        worst = int(np.argmin(p))
        import warnings
        warnings.warn(
            f"Hydrostatic P non-positive at {n_bad} layer(s) "
            f"(worst layer {worst}: P={p[worst]:.3e}); flooring positive.",
            RuntimeWarning,
        )
        p = np.where(bad_mask, floor, p)
    return p


def update_total_pressure(
    rhox: np.ndarray,
    gravity_cgs: float,
    pzero: float,
) -> np.ndarray:
    """Fortran reference (`atlas12.for`, line 245): PTOTAL=GRAV*RHOX+PZERO."""

    return gravity_cgs * rhox + pzero

