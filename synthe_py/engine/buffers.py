"""Spectrum buffer management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass
class SpectrumBuffers:
    """Holds working arrays used during synthesis."""

    wavelength: np.ndarray
    continuum: np.ndarray
    hydrogen_continuum: np.ndarray
    hydrogen_source: np.ndarray
    bfudge: np.ndarray
    slinec: np.ndarray
    line_opacity: np.ndarray
    line_scattering: np.ndarray
    profile: np.ndarray

    def zero(self) -> None:
        self.continuum.fill(0.0)
        self.hydrogen_continuum.fill(0.0)
        self.hydrogen_source.fill(0.0)
        self.bfudge.fill(0.0)
        self.slinec.fill(0.0)
        self.line_opacity.fill(0.0)
        self.line_scattering.fill(0.0)
        self.profile.fill(0.0)


@dataclass
class SynthResult:
    """Final synthesized spectrum for all depths combined."""

    wavelength: np.ndarray
    intensity: np.ndarray
    continuum: np.ndarray
    timings: Dict[str, float] = field(default_factory=dict)


def allocate_buffers(wavelength: np.ndarray, depth_layers: int) -> SpectrumBuffers:
    size = wavelength.size
    continuum = np.zeros((depth_layers, size), dtype=np.float64)
    hydrogen_continuum = np.zeros_like(continuum)
    hydrogen_source = np.zeros_like(continuum)
    bfudge = np.zeros(depth_layers, dtype=np.float64)
    slinec = np.zeros((depth_layers, size), dtype=np.float64)
    line_opacity = np.zeros((depth_layers, size), dtype=np.float64)
    line_scattering = np.zeros((depth_layers, size), dtype=np.float64)
    profile = np.zeros(size, dtype=np.float64)
    return SpectrumBuffers(
        wavelength=wavelength,
        continuum=continuum,
        hydrogen_continuum=hydrogen_continuum,
        hydrogen_source=hydrogen_source,
        bfudge=bfudge,
        slinec=slinec,
        line_opacity=line_opacity,
        line_scattering=line_scattering,
        profile=profile,
    )
