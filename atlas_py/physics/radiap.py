"""RADIAP integrator (`atlas12.for` lines 1152-1192)."""

from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np

from .josh_math import _integ


@dataclass
class RadiapState:
    """State accumulated by RADIAP across frequency integration."""

    h: np.ndarray
    raden: np.ndarray
    accrad: np.ndarray
    prad: np.ndarray
    pradk: np.ndarray
    pradk0: float


def init_radiap(n_layers: int) -> RadiapState:
    zeros = np.zeros(n_layers, dtype=np.float64)
    return RadiapState(
        h=zeros.copy(),
        raden=zeros.copy(),
        accrad=zeros.copy(),
        prad=zeros.copy(),
        pradk=zeros.copy(),
        pradk0=0.0,
    )


def radiap_accumulate(
    st: RadiapState,
    *,
    mode: int,
    rcowt: float,
    abtot: np.ndarray,
    hnu: np.ndarray,
    jnu: np.ndarray,
    knu_surface: float,
    freq_hz: float = 0.0,
    wave_nm: float = 0.0,
    flux: float,
    rhox: np.ndarray,
) -> None:
    """Apply one RADIAP mode step.

    Modes match Fortran:
    - 1: reset
    - 2: accumulate per-frequency
    - 3: finalize and integrate PRAD from ACCRAD over RHOX
    """

    if mode == 1:
        st.h[:] = 0.0
        st.raden[:] = 0.0
        st.accrad[:] = 0.0
        st.pradk0 = 0.0
        return
    if mode == 2:
        st.raden += jnu * rcowt
        st.h += hnu * rcowt
        st.accrad += abtot * hnu * rcowt
        st.pradk0 += knu_surface * rcowt
        radiap_log_path = os.getenv("ATLAS_RADIAP_LOG")
        if radiap_log_path:
            with open(radiap_log_path, "a", encoding="utf-8") as fh:
                fh.write(
                    f"RAD2,{float(wave_nm):.8e},{float(freq_hz):.8e},{float(rcowt):.8e},"
                    f"{float(knu_surface):.8e},{float(knu_surface * rcowt):.8e},{float(st.pradk0):.8e}\n"
                )
        return
    if mode != 3:
        raise ValueError(f"Unsupported RADIAP mode: {mode}")

    conv = 12.5664 / 2.99792458e10
    st.raden *= conv
    st.accrad *= conv
    ratio = st.h / max(flux, 1e-300)
    mask = ratio > 1.0
    st.accrad[mask] *= flux / np.maximum(st.h[mask], 1e-300)
    errormax = float(np.max(ratio))
    st.pradk0 *= conv
    if errormax > 1.0:
        st.pradk0 /= errormax

    # INTEG(RHOX, ACCRAD, PRAD, NRHOX, ACCRAD(1)*RHOX(1))
    st.prad[:] = _integ(rhox, st.accrad, st.accrad[0] * rhox[0])
    st.pradk[:] = st.prad + st.pradk0
    radiap_log_path = os.getenv("ATLAS_RADIAP_LOG")
    if radiap_log_path:
        with open(radiap_log_path, "a", encoding="utf-8") as fh:
            n = st.pradk.size
            j0 = 37
            j1 = min(41, n - 1)
            for j in range(j0, j1 + 1):
                fh.write(
                    f"RAD3,{j + 1:d},{st.prad[j]:.8e},{st.pradk0:.8e},{st.pradk[j]:.8e},{st.accrad[j]:.8e}\n"
                )

