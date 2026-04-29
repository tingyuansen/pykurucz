"""Helium I wing tables derived from he1tables.dat."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

_HE_DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "he1_tables.npz"


@dataclass(frozen=True)
class HeLineTables:
    log_ne: np.ndarray
    dlam: np.ndarray
    phi: np.ndarray
    phi_h_plus: Optional[np.ndarray] = None
    phi_he_plus: Optional[np.ndarray] = None


@dataclass(frozen=True)
class HeTables:
    temperatures: np.ndarray
    line_4471: HeLineTables
    line_4026: HeLineTables
    line_4387: HeLineTables
    line_4921: HeLineTables


_cached_tables: Optional[HeTables] = None


def load_tables(path: Optional[Path] = None) -> HeTables:
    global _cached_tables
    if _cached_tables is not None:
        return _cached_tables

    npz_path = path or _HE_DEFAULT_PATH
    with np.load(npz_path, allow_pickle=False) as data:
        temps = np.asarray(data["temperatures"], dtype=np.float64)

        def _build(prefix: str, has_species: bool) -> HeLineTables:
            log_ne = np.asarray(data[f"{prefix}_ne"], dtype=np.float64)
            dlam_plus = np.asarray(data[f"{prefix}_dlam"], dtype=np.float64)
            dlam = dlam_plus - 150.0
            if has_species:
                phi_h = np.asarray(data[f"{prefix}_phi_h_plus"], dtype=np.float64)
                phi_he = np.asarray(data[f"{prefix}_phi_he_plus"], dtype=np.float64)
                return HeLineTables(
                    log_ne=log_ne,
                    dlam=dlam,
                    phi=np.zeros_like(phi_h),
                    phi_h_plus=phi_h,
                    phi_he_plus=phi_he,
                )
            phi = np.asarray(data[f"{prefix}_phi"], dtype=np.float64)
            return HeLineTables(
                log_ne=log_ne,
                dlam=dlam,
                phi=phi,
            )

        tables = HeTables(
            temperatures=temps,
            line_4471=_build("line_4471", has_species=True),
            line_4026=_build("line_4026", has_species=False),
            line_4387=_build("line_4387", has_species=False),
            line_4921=_build("line_4921", has_species=True),
        )
    _cached_tables = tables
    return tables
