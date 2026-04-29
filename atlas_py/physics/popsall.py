"""POPSALL orchestration (`atlas12.for` line 4557+)."""

from __future__ import annotations

import numpy as np

from .populations import pops
from .runtime import AtlasRuntimeState


def _nn_from_code(code: float) -> int:
    frac = float(code) - float(int(code))
    return max(1, int(frac * 100.0 + 1.5))


def popsall(
    *,
    temperature_k: np.ndarray,
    tk_erg: np.ndarray,
    state: AtlasRuntimeState,
    ifmol: bool,
    ifpres: bool,
    itemp: int,
    itemp_cache: dict[str, int],
) -> None:
    """Populate XNF and XNFP for all species (early atomic path)."""

    n_layers = temperature_k.size
    if state.xnf.shape[0] != n_layers or state.xnfp.shape[0] != n_layers:
        raise ValueError("state arrays must match atmosphere layer count")

    # Atomic calls mirror atlas12.for POPSALL (lines 4747-4817), including
    # element-specific code tweaks (e.g., 4.03, 5.03, 20.09 in mode 11).
    mode12_calls: list[tuple[float, int]] = [
        (1.01, 1),
        (2.02, 3),
        (3.03, 6),
        (4.03, 10),
        (5.03, 15),
        (6.05, 21),
        (7.05, 28),
        (8.05, 36),
        (9.05, 45),
        (10.05, 55),
        (11.05, 66),
        (12.05, 78),
        (13.05, 91),
        (14.05, 105),
        (15.05, 120),
        (16.05, 136),
        (17.04, 153),
        (18.04, 171),
        (19.04, 190),
        (20.04, 210),
        (21.04, 231),
        (22.04, 253),
        (23.04, 276),
        (24.04, 300),
        (25.04, 325),
        (26.04, 351),
        (27.04, 378),
        (28.04, 406),
        (29.02, 435),
        (30.02, 465),
    ]
    mode11_calls: list[tuple[float, int]] = [
        (1.01, 1),
        (2.02, 3),
        (3.03, 6),
        (4.03, 10),
        (5.03, 15),
        (6.05, 21),
        (7.05, 28),
        (8.05, 36),
        (9.05, 45),
        (10.05, 55),
        (11.05, 66),
        (12.05, 78),
        (13.05, 91),
        (14.05, 105),
        (15.05, 120),
        (16.05, 136),
        (17.05, 153),
        (18.04, 171),
        (19.05, 190),
        (20.09, 210),
        (21.09, 231),
        (22.09, 253),
        (23.09, 276),
        (24.09, 300),
        (25.09, 325),
        (26.09, 351),
        (27.09, 378),
        (28.09, 406),
        (29.02, 435),
        (30.02, 465),
    ]

    for code, start_1based in mode12_calls:
        nout = _nn_from_code(code)
        slot0 = start_1based - 1
        sl = slice(slot0, slot0 + nout)
        pops(
            code=code,
            mode=12,
            out=state.xnf[:, sl],
            ifmol=ifmol,
            ifpres=ifpres,
            temperature_k=temperature_k,
            tk_erg=tk_erg,
            state=state,
            itemp=itemp,
            itemp_cache=itemp_cache,
        )
    for code, start_1based in mode11_calls:
        nout = _nn_from_code(code)
        slot0 = start_1based - 1
        sl = slice(slot0, slot0 + nout)
        pops(
            code=code,
            mode=11,
            out=state.xnfp[:, sl],
            ifmol=ifmol,
            ifpres=ifpres,
            temperature_k=temperature_k,
            tk_erg=tk_erg,
            state=state,
            itemp=itemp,
            itemp_cache=itemp_cache,
        )

    # Z=31..99 (Fortran starts each element at 496 + (Z-31)*5 and calls
    # POPS with code Z+0.02 for both mode 11 and 12).
    for z in range(31, 100):
        code = float(z) + 0.02
        slot0 = 495 + (z - 31) * 5
        nout = _nn_from_code(code)
        sl = slice(slot0, slot0 + nout)
        pops(
            code=code,
            mode=11,
            out=state.xnfp[:, sl],
            ifmol=ifmol,
            ifpres=ifpres,
            temperature_k=temperature_k,
            tk_erg=tk_erg,
            state=state,
            itemp=itemp,
            itemp_cache=itemp_cache,
        )
        pops(
            code=code,
            mode=12,
            out=state.xnf[:, sl],
            ifmol=ifmol,
            ifpres=ifpres,
            temperature_k=temperature_k,
            tk_erg=tk_erg,
            state=state,
            itemp=itemp,
            itemp_cache=itemp_cache,
        )

    if not ifmol:
        return

    # Molecular slots from atlas12.for POPSALL (lines 4831-4846), where
    # XNFP(:,NELION) is filled by POPS(CODE,1,...).
    mol_targets: list[tuple[float, int]] = [
        (101.00, 841),
        (106.00, 846),
        (107.00, 847),
        (108.00, 848),
        (112.00, 851),
        (114.00, 853),
        (120.00, 858),
        (124.00, 862),
        (126.00, 864),
        (606.00, 868),
        (607.00, 869),
        (608.00, 870),
        (814.00, 889),
        (822.00, 895),
        (823.00, 896),
        (10108.00, 940),
    ]
    for code, nelion_1based in mol_targets:
        col0 = nelion_1based - 1
        if col0 < 0 or col0 >= state.xnfp.shape[1]:
            continue
        sl = slice(col0, col0 + 1)
        pops(
            code=code,
            mode=1,
            out=state.xnfp[:, sl],
            ifmol=ifmol,
            ifpres=ifpres,
            temperature_k=temperature_k,
            tk_erg=tk_erg,
            state=state,
            itemp=itemp,
            itemp_cache=itemp_cache,
        )
        pops(
            code=code,
            mode=11,
            out=state.xnfp[:, sl],
            ifmol=ifmol,
            ifpres=ifpres,
            temperature_k=temperature_k,
            tk_erg=tk_erg,
            state=state,
            itemp=itemp,
            itemp_cache=itemp_cache,
        )

