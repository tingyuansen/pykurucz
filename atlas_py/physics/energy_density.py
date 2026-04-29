"""ATLAS12 atomic energy density (`SUBROUTINE ENERGYDENSITY`).

Fortran reference:
- `atlas12.for` lines 2927-3038 (`SUBROUTINE ENERGYDENSITY`)
- `atlas12.for` lines 12488-12505 (`POTIONSUM` construction)
"""

from __future__ import annotations

import numpy as np

from .nelect import _atomic_slot_start, _nion_for_atomic_number
from .pfsaha import POTION, pfsaha_depth
from .runtime import AtlasRuntimeState

_K_BOLTZ = 1.38054e-16
_H_PLANCK = 6.6256e-27
_C_LIGHT = 2.99792458e10
_MAX_ATOMIC_ION = 840  # Fortran ENERGYDENSITY loop bound.


def _build_potionsum() -> np.ndarray:
    """Build Fortran `POTIONSUM(999)` from `POTION(999)`."""
    if POTION is None:
        raise RuntimeError("POTION table unavailable for ENERGYDENSITY")
    potion = np.asarray(POTION, dtype=np.float64)
    if potion.size < 999:
        raise RuntimeError("POTION table has unexpected size")
    potionsum = np.zeros(999, dtype=np.float64)
    nelion = 0
    for iz in range(1, 31):
        nelion += 1
        potionsum[nelion - 1] = 0.0
        for _ion in range(2, iz + 2):
            nelion += 1
            potionsum[nelion - 1] = potion[nelion - 2] + potionsum[nelion - 2]
    for _iz in range(31, 100):
        nelion += 1
        potionsum[nelion - 1] = 0.0
        nelion += 1
        potionsum[nelion - 1] = potion[nelion - 2] + potionsum[nelion - 2]
        nelion += 1
        potionsum[nelion - 1] = potion[nelion - 2] + potionsum[nelion - 2]
        nelion += 1
        potionsum[nelion - 1] = potion[nelion - 2] + potionsum[nelion - 2]
        nelion += 1
        potionsum[nelion - 1] = potion[nelion - 2] + potionsum[nelion - 2]
    return potionsum


_POTIONSUM = _build_potionsum()


def compute_atomic_energy_density(
    *,
    temperature_k: np.ndarray,
    state: AtlasRuntimeState,
) -> np.ndarray:
    """Compute atomic `EDENS` in erg g^-1 for each depth.

    Uses PF partition finite difference around +/-0.1% in T, matching the
    Fortran ENERGYDENSITY atomic branch.
    """
    t = np.asarray(temperature_k, dtype=np.float64)
    n_layers = t.size
    edens = np.zeros(n_layers, dtype=np.float64)
    for j in range(n_layers):
        tj = max(float(t[j]), 1.0)
        tk = _K_BOLTZ * tj
        hckt = (_H_PLANCK * _C_LIGHT) / max(tk, 1e-300)
        xntot = float(state.xne[j] + state.xnatom[j])

        pfplus = np.ones(_MAX_ATOMIC_ION, dtype=np.float64)
        pfminus = np.ones(_MAX_ATOMIC_ION, dtype=np.float64)
        tplus = tj * 1.001
        tminus = tj * 0.999

        for z in range(1, 100):
            nion = _nion_for_atomic_number(z) if z <= 30 else 3
            slot0 = _atomic_slot_start(z)
            vals_plus = pfsaha_depth(
                temperature_k=tplus,
                electron_density_cm3=float(state.xne[j]),
                xnatom_cm3=float(state.xnatom[j]),
                xabund_linear=float(state.xabund[j, z - 1]),
                atomic_number=z,
                nion=nion,
                mode=13,  # MODE 3 + 10 => return all PART(ION)
                chargesq_cm3=float(state.chargesq[j]),
            )
            vals_minus = pfsaha_depth(
                temperature_k=tminus,
                electron_density_cm3=float(state.xne[j]),
                xnatom_cm3=float(state.xnatom[j]),
                xabund_linear=float(state.xabund[j, z - 1]),
                atomic_number=z,
                nion=nion,
                mode=13,
                chargesq_cm3=float(state.chargesq[j]),
            )
            ncopy = min(nion, _MAX_ATOMIC_ION - slot0)
            if ncopy > 0:
                pfplus[slot0 : slot0 + ncopy] = vals_plus[:ncopy]
                pfminus[slot0 : slot0 + ncopy] = vals_minus[:ncopy]

        e = 1.5 * xntot * tk
        frac_term = (pfplus - pfminus) / np.maximum(pfplus + pfminus, 1e-30) * 1000.0
        for nel in range(_MAX_ATOMIC_ION):
            e += state.xnf[j, nel] * tk * (_POTIONSUM[nel] * hckt + frac_term[nel])
        edens[j] = e / np.maximum(state.rho[j], 1e-300)
    return edens
