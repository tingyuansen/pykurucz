"""POPS dispatcher (`atlas12.for` lines 2889-2926)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

try:
    import numba
except ImportError:  # pragma: no cover
    numba = None

from .nelect import nelect
from .nmolec import molec, nmolec
from .pfsaha import pfsaha_depth
from .pops_parallel import pops_parallel_enabled, pops_parallel_workers
from .runtime import AtlasRuntimeState

if numba is not None:
    @numba.njit(cache=True, parallel=True, nogil=True)
    def _scale_pops_out_mode_lt10_nb(
        out: np.ndarray,
        frac0: np.ndarray,
        xnatom: np.ndarray,
        xabund_col: np.ndarray,
    ) -> None:
        n = out.shape[0]
        for j in numba.prange(n):
            out[j, 0] = frac0[j] * xnatom[j] * xabund_col[j]

    @numba.njit(cache=True, parallel=True, nogil=True)
    def _scale_pops_out_mode_ge10_nb(
        out: np.ndarray,
        frac: np.ndarray,
        xnatom: np.ndarray,
        xabund_col: np.ndarray,
    ) -> None:
        n, ncols = out.shape[0], out.shape[1]
        ncopy = min(ncols, frac.shape[1])
        for j in numba.prange(n):
            out[j, :] = 0.0
            scale = xnatom[j] * xabund_col[j]
            for k in range(ncopy):
                out[j, k] = frac[j, k] * scale
else:  # pragma: no cover
    def _scale_pops_out_mode_lt10_nb(out, frac0, xnatom, xabund_col):
        out[:, 0] = frac0 * xnatom * xabund_col

    def _scale_pops_out_mode_ge10_nb(out, frac, xnatom, xabund_col):
        ncopy = min(out.shape[1], frac.shape[1])
        out[:, :] = 0.0
        out[:, :ncopy] = frac[:, :ncopy] * (xnatom * xabund_col)[:, np.newaxis]


def _decode_code(code: float) -> tuple[int, int]:
    """Decode Fortran CODE (e.g., 26.04 -> IZ=26, NION=5)."""

    iz = int(code)
    frac = code - float(iz)
    nion = int(frac * 100.0 + 1.5)
    nion = max(1, nion)
    return iz, nion


def _pops_atomic_element(
    *,
    iz: int,
    nion: int,
    mode: int,
    out: np.ndarray,
    temperature_k: np.ndarray,
    state: AtlasRuntimeState,
) -> None:
    """Fill one element's populations across all depths (IFMOL=0 path)."""
    layers = temperature_k.size
    ncols = out.shape[1]
    xabund_col = state.xabund[:, iz - 1]

    def _one_layer(j: int) -> np.ndarray:
        return pfsaha_depth(
            temperature_k=float(temperature_k[j]),
            electron_density_cm3=float(state.xne[j]),
            xnatom_cm3=float(state.xnatom[j]),
            xabund_linear=float(xabund_col[j]),
            atomic_number=iz,
            nion=nion,
            mode=mode,
            chargesq_cm3=float(max(state.chargesq[j], 1e-30)),
        )

    if pops_parallel_enabled() and layers > 1:
        workers = min(pops_parallel_workers(), layers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            vals_list = list(pool.map(_one_layer, range(layers)))
    else:
        vals_list = [_one_layer(j) for j in range(layers)]

    if mode < 10:
        frac0 = np.asarray([v[0] for v in vals_list], dtype=np.float64)
        _scale_pops_out_mode_lt10_nb(out, frac0, state.xnatom, xabund_col)
        return

    ncopy = min(ncols, max(len(v) for v in vals_list))
    frac = np.zeros((layers, ncopy), dtype=np.float64)
    for j, vals in enumerate(vals_list):
        frac[j, : min(ncopy, vals.size)] = vals[:ncopy]
    _scale_pops_out_mode_ge10_nb(out, frac, state.xnatom, xabund_col)


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
        _pops_atomic_element(
            iz=iz,
            nion=nion,
            mode=mode,
            out=out,
            temperature_k=temperature_k,
            state=state,
        )
        return

    # IFMOL == 1 path (requires full NMOLEC/MOLEC port).
    if ifpres and itemp != itemp_cache.get("pops_itemp", -1):
        nmolec(mode=mode)
        itemp_cache["pops_itemp"] = itemp
    if code == 0.0:
        return
    molec(codout=code, mode=mode, number=out)
