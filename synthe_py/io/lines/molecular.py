"""Molecular line-list handling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np


@dataclass
class MolecularBand:
    """Simplified representation of a molecular band list."""

    wavelength: np.ndarray
    intensity: np.ndarray
    metadata: dict


@dataclass
class MolecularCatalog:
    bands: List[MolecularBand]

    @property
    def total_lines(self) -> int:
        return sum(len(band.wavelength) for band in self.bands)


def load_catalog(paths: Sequence[Path]) -> MolecularCatalog:
    bands: List[MolecularBand] = []
    for path in paths:
        data = np.loadtxt(path, comments="#", usecols=(0, 1))
        bands.append(
            MolecularBand(
                wavelength=data[:, 0],
                intensity=data[:, 1],
                metadata={"source": str(path)},
            )
        )
    return MolecularCatalog(bands=bands)
