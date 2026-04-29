"""ROSS integrator (`atlas12.for` lines 1193-1215)."""

from __future__ import annotations

import numpy as np

from .josh_math import _integ


def ross_step(
    abross: np.ndarray,
    *,
    mode: int,
    rcowt: float,
    bnu: np.ndarray,
    freq_hz: float,
    hkt: np.ndarray,
    temperature_k: np.ndarray,
    stim: np.ndarray,
    abtot: np.ndarray,
    numnu: int,
    rhox: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one ROSS mode and return updated (abross, tauros)."""

    tauros = np.zeros_like(abross)
    if mode == 1:
        abross[:] = 0.0
        return abross, tauros
    if mode == 2:
        dbdt = bnu * freq_hz * hkt / np.maximum(temperature_k * stim, 1e-300)
        if numnu == 1:
            dbdt = (4.0 * 5.6697e-5 / 3.14159) * np.power(temperature_k, 3.0)
        abross += dbdt / np.maximum(abtot, 1e-300) * rcowt
        return abross, tauros
    if mode != 3:
        raise ValueError(f"Unsupported ROSS mode: {mode}")

    abross[:] = (4.0 * 5.6697e-5 / 3.14159) * np.power(temperature_k, 3.0) / np.maximum(
        abross, 1e-300
    )
    # INTEG(RHOX,ABROSS,TAUROS,NRHOX,ABROSS(1)*RHOX(1))
    tauros = _integ(rhox, abross, abross[0] * rhox[0])
    return abross, tauros

