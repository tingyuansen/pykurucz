"""Fortran-faithful DOPPLE/XNFDOP updates (atlas12 main loop lines 272-279)."""

from __future__ import annotations
import numpy as np

from .runtime import AtlasRuntimeState

_AMU_GRAM = 1.660e-24
_C_LIGHT = 2.99792458e10


def update_doppler_populations(
    *,
    tk_erg: np.ndarray,
    vturb_cms: np.ndarray,
    state: AtlasRuntimeState,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute `DOPPLE` and `XNFDOP` from runtime populations.

    Fortran reference (`atlas12.for`):
    - `DOPPLE(J,NELION)=SQRT(2.*TK(J)/AMASSISO(1,NELION)/1.660D-24+VTURB(J)**2)/C`
    - `XNFDOP(J,NELION)=XNFP(J,NELION)/DOPPLE(J,NELION)/RHO(J)`
    with `NELION=1..MION-1`.
    """

    if state.amassiso_major is None:
        raise ValueError("AMASSISO major-isotope masses are required for DOPPLE/XNFDOP")

    tk = np.asarray(tk_erg, dtype=np.float64)
    vturb = np.asarray(vturb_cms, dtype=np.float64)
    rho = np.asarray(state.rho, dtype=np.float64)
    xnfp = np.asarray(state.xnfp, dtype=np.float64)
    amass = np.asarray(state.amassiso_major, dtype=np.float64)

    layers = tk.size
    mion = xnfp.shape[1]
    if amass.size < mion:
        raise ValueError(f"AMASSISO size {amass.size} is smaller than mion={mion}")

    dopple = np.zeros((layers, mion), dtype=np.float64)
    xnfdop = np.zeros((layers, mion), dtype=np.float64)
    if mion <= 1:
        state.dopple = dopple
        state.xnfdop = xnfdop
        return dopple, xnfdop

    mass = amass[: mion - 1]
    mass_term = np.divide(
        2.0 * tk[:, None],
        mass[None, :] * _AMU_GRAM,
        out=np.full((layers, mion - 1), np.inf, dtype=np.float64),
        where=mass[None, :] > 0.0,
    )
    v2 = vturb[:, None] ** 2
    dop = np.sqrt(mass_term + v2) / _C_LIGHT
    dopple[:, : mion - 1] = dop

    rho_safe = np.maximum(rho[:, None], 1e-300)
    xnfdop[:, : mion - 1] = np.divide(
        xnfp[:, : mion - 1],
        dop * rho_safe,
        out=np.zeros((layers, mion - 1), dtype=np.float64),
        where=dop > 0.0,
    )

    state.dopple = dopple
    state.xnfdop = xnfdop
    return dopple, xnfdop

