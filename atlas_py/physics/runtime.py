"""Runtime state arrays mirroring key ATLAS12 COMMON blocks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AtlasRuntimeState:
    """Subset of ATLAS12 COMMON arrays used in early atlas_py phases.

    Units:
    - p: dyn cm^-2
    - xne: cm^-3
    - xnatom: cm^-3
    - rho: g cm^-3
    - chargesq: cm^-3
    - xabund: dimensionless number fraction
    - wtmole: dimensionless mean molecular weight in amu
    """

    p: np.ndarray
    xne: np.ndarray
    xnatom: np.ndarray
    rho: np.ndarray
    chargesq: np.ndarray
    xabund: np.ndarray  # shape (layers, 99), Fortran XABUND(J,IZ)
    wtmole: np.ndarray  # shape (layers,)
    xnf: np.ndarray  # shape (layers, mion)
    xnfp: np.ndarray  # shape (layers, mion)
    edens: np.ndarray  # shape (layers,), erg g^-1 (ATLAS /EDENS/)
    amassiso_major: np.ndarray | None = None  # shape (mion,), AMASSISO(1,:)
    dopple: np.ndarray | None = None  # shape (layers, mion), DOPPLE
    xnfdop: np.ndarray | None = None  # shape (layers, mion), XNFDOP
    # NLTE departure coefficients — COMMON /DEPART/ (atlas12.for line 45).
    # Fortran: DATA BHYD,BMIN/kw*1.,…,kw*1./ (line 1703). Always present,
    # always initialised to 1.0; STATEQ overwrites when NLTEON=1.
    bhyd: np.ndarray | None = None   # shape (layers, 6), defaults to 1.0 (LTE)
    bmin: np.ndarray | None = None   # shape (layers,), defaults to 1.0 (LTE)
    # Geometric height (COMMON /HEIGHT/) populated by CALL HIGH each iteration.
    # Units: cm (negative below atmosphere surface).
    height: np.ndarray | None = None

