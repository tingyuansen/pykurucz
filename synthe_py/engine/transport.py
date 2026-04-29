"""Radiative transport post-processing."""

from __future__ import annotations

import numpy as np

from .buffers import SpectrumBuffers


def compute_emergent_intensity(buffers: SpectrumBuffers) -> np.ndarray:
    """Placeholder emergent intensity calculation."""

    return buffers.line_opacity.sum(axis=0)
