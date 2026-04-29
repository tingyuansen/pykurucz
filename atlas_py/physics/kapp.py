"""Continuum opacity assembly for atlas_py (Phase 2 baseline)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kapp_continuum import compute_kapp_continuum


@dataclass
class KappAtmosphereAdapter:
    """Adapter mirroring the ATLAS12 COMMON blocks used by KAPP and its sub-routines.

    Every field corresponds to a Fortran array that is **always allocated and
    populated** before KAPP is called.  No field may be ``None`` unless noted;
    Fortran COMMON blocks have no concept of absent arrays.

    Units (matching Fortran):
    - temperature: K          — COMMON /TEMP/  T(kw)
    - mass_density: g cm^-3   — COMMON /STATE/ RHO(kw)
    - electron_density: cm^-3 — COMMON /STATE/ XNE(kw)
    - gas_pressure: dyn cm^-2 — COMMON /STATE/ P(kw)
    - xnfph: cm^-3 / partition — POPS(1.01D0,11,XNFPH)
    - bhyd: dimensionless     — COMMON /DEPART/ BHYD(kw,6), DATA 1.0
    - turbulent_velocity: cm s^-1 — COMMON /TURBPR/ VTURB(kw)
    """

    # Always-present arrays matching Fortran COMMON blocks.
    temperature: np.ndarray         # T(kw)
    mass_density: np.ndarray        # RHO(kw)
    electron_density: np.ndarray    # XNE(kw)
    gas_pressure: np.ndarray        # P(kw)
    xnfph: np.ndarray               # XNFPH(kw,2) from POPS mode-11
    xnf_h: np.ndarray               # XNF(J,1)  — total neutral H
    xnf_h_ionized: np.ndarray       # XNF(J,2)  — ionized H (H+)
    xnf_he1: np.ndarray             # XNFP(J,3) — neutral He
    xnf_he2: np.ndarray             # XNFP(J,4) — singly-ionized He
    xabund: np.ndarray              # XABUND(kw,99)
    bhyd: np.ndarray                # BHYD(kw,6), init to 1.0 (LTE)
    turbulent_velocity: np.ndarray  # VTURB(kw)
    xnf_all: np.ndarray             # XNF(kw,mion)
    xnfp_all: np.ndarray            # XNFP(kw,mion)
    # Molecular populations from POPSALL — zero when molecules are off
    # (Fortran: XNFP(:,846), XNFP(:,848) always exist even if zero).
    xnfpch: np.ndarray              # XNFP(:,846) — CH mode-1
    xnfpoh: np.ndarray              # XNFP(:,848) — OH mode-1

    @property
    def layers(self) -> int:
        return int(self.temperature.size)


def compute_kapp(
    adapter: KappAtmosphereAdapter,
    freq_hz: np.ndarray,
    atlas_tables: dict[str, np.ndarray],
    ifop: list[int],
    tcst=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute continuum opacity arrays (ACONT, SIGMAC, SCONT).

    ``ifop`` must be passed from the parsed .atm file or control deck —
    mirrors Fortran KAPP which reads IFOP directly from READIN output.
    """

    return compute_kapp_continuum(
        atmosphere=adapter,
        freq=np.asarray(freq_hz, dtype=np.float64),
        atlas_tables=atlas_tables,
        ifop=ifop,
        tcst=tcst,
    )

