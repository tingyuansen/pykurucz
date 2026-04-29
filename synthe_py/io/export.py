"""Export utilities for SYNTHE outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..engine.buffers import SynthResult


def write_spec_file(result: SynthResult, destination: Path) -> None:
    """Write the synthesized spectrum to a `.spec` file.

    The legacy Fortran code emitted three columns per row: wavelength,
    emergent flux, and continuum flux. The Python reimplementation produces
    the same layout for drop-in comparability.
    """

    data = np.column_stack((result.wavelength, result.intensity, result.continuum))
    np.savetxt(destination, data, fmt="%15.8f %11.6E %11.6E")


