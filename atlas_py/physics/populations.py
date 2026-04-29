"""POPS dispatcher (`atlas12.for` lines 2889-2926)."""

from __future__ import annotations

import numpy as np

from .nelect import nelect
from .nmolec import molec, nmolec
from .pfsaha import pfsaha_depth
from .runtime import AtlasRuntimeState


def _decode_code(code: float) -> tuple[int, int]:
    """Decode Fortran CODE (e.g., 26.04 -> IZ=26, NION=5)."""

    iz = int(code)
    frac = code - float(iz)
    nion = int(frac * 100.0 + 1.5)
    nion = max(1, nion)
    return iz, nion


def pops(
    code: float,
    mode: int,
    out: np.ndarray,
    *,
    ifmol: bool,
    ifpres: bool,
    temperature_k: np.ndarray,
    tk_erg: np.ndarray,
    state: AtlasRuntimeState,
    itemp: int,
    itemp_cache: dict[str, int],
) -> None:
    """Populate `out` with PFSAHA/MOLEC results and number-density scaling.

    Fortran reference:
    - `atlas12.for` line 2903: call NELECT when pressure iteration active and new T
    - line 2913: PFSAHA call
    - lines 2916-2919: convert fractions to number density by XNATOM * XABUND
    """

    if not ifmol:
        last_itemp = itemp_cache.get("pops_itemp", -1)
        if ifpres and itemp != last_itemp:
            nelect(temperature_k=temperature_k, tk_erg=tk_erg, state=state)
            itemp_cache["pops_itemp"] = itemp

        if code == 0.0:
            return
        if code >= 100.0:
            raise ValueError("Molecule code requested while IFMOL=0")

        iz, nion = _decode_code(code)
        layers = temperature_k.size
        ncols = out.shape[1]
        for j in range(layers):
            vals = pfsaha_depth(
                temperature_k=float(temperature_k[j]),
                electron_density_cm3=float(state.xne[j]),
                xnatom_cm3=float(state.xnatom[j]),
                xabund_linear=float(state.xabund[j, iz - 1]),
                atomic_number=iz,
                nion=nion,
                mode=mode,
                chargesq_cm3=float(max(state.chargesq[j], 1e-30)),
            )
            if mode < 10:
                out[j, 0] = vals[0] * state.xnatom[j] * state.xabund[j, iz - 1]
            else:
                out[j, :] = 0.0
                ncopy = min(ncols, vals.size)
                scaled = vals[:ncopy] * state.xnatom[j] * state.xabund[j, iz - 1]
                out[j, :ncopy] = scaled
        return

    # IFMOL == 1 path (requires full NMOLEC/MOLEC port).
    if ifpres and itemp != itemp_cache.get("pops_itemp", -1):
        nmolec(mode=mode)
        itemp_cache["pops_itemp"] = itemp
    if code == 0.0:
        return
    molec(codout=code, mode=mode, number=out)

