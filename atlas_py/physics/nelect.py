"""Electron-density iteration (NELECT analogue)."""

from __future__ import annotations

import numpy as np

from .pfsaha import pfsaha_depth
from .runtime import AtlasRuntimeState

_MION = 1006


def _nion_for_atomic_number(z: int) -> int:
    if z == 1:
        return 2
    if z == 2:
        return 3
    if z in (3, 4, 5):
        return 4
    if 6 <= z <= 16:
        return 6
    if 17 <= z <= 28:
        return 5
    if z in (29, 30):
        return 3
    if z >= 31:
        return 3
    return 5


def _atomic_slot_start(z: int) -> int:
    """Return 0-based XNF start slot for atomic number z."""

    # Fortran slots: Z<=30 packed triangular, Z>=31 starts at 496 with stride 5.
    if z <= 30:
        # 1-based triangular index start = 1 + sum_{k=1}^{z-1} (k+1)
        start_1 = 1 + ((z - 1) * (z + 2)) // 2
        return start_1 - 1
    return (496 + (z - 31) * 5) - 1


def nelect(
    temperature_k: np.ndarray,
    tk_erg: np.ndarray,
    state: AtlasRuntimeState,
    max_iter: int = 200,
    tol: float = 1e-4,
) -> None:
    """Iterate electron density and ion populations, matching NELECT flow.

    Fortran reference: `atlas12.for` lines 3039-3136.
    """

    n_layers = temperature_k.size
    if state.xnf.shape[0] != n_layers or state.xnf.shape[1] < _MION:
        raise ValueError("state.xnf must have shape (layers, >=1006)")

    for j in range(n_layers):
        xntot = state.p[j] / max(tk_erg[j], 1e-300)
        state.xnatom[j] = xntot - state.xne[j]

        converged = False
        for _ in range(max_iter):
            state.xnf[j, :] = 0.0
            xnenew = 0.0
            chargesquare = 0.0

            for z in range(1, 100):
                nion = _nion_for_atomic_number(z)
                vals = pfsaha_depth(
                    temperature_k=float(temperature_k[j]),
                    electron_density_cm3=float(state.xne[j]),
                    xnatom_cm3=float(state.xnatom[j]),
                    xabund_linear=float(state.xabund[j, z - 1]),
                    atomic_number=z,
                    nion=nion,
                    mode=12,
                    chargesq_cm3=float(max(state.chargesq[j], 1e-30)),
                )
                slot0 = _atomic_slot_start(z)
                for ion in range(nion):
                    v = vals[ion] * state.xnatom[j] * state.xabund[j, z - 1]
                    idx = slot0 + ion
                    if idx < state.xnf.shape[1]:
                        state.xnf[j, idx] = v
                    chargesquare += v * (ion**2)
                    xnenew += v * ion

            xnenew = max(xnenew, state.xne[j] * 0.5)
            xnenew = 0.5 * (xnenew + state.xne[j])
            err = abs((state.xne[j] - xnenew) / max(xnenew, 1e-300))

            state.xne[j] = xnenew
            state.xnatom[j] = xntot - state.xne[j]
            state.chargesq[j] = chargesquare + state.xne[j]
            if err < tol:
                converged = True
                break

        if not converged:
            raise RuntimeError(f"NELECT did not converge at depth index {j}")

        # Fortran line 3125
        state.rho[j] = state.xnatom[j] * state.wtmole[j] * 1.660e-24

