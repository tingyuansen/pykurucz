"""Helium line profile helper."""

from __future__ import annotations

import numpy as np

from ..populations import DepthState


def helium_profile(
    depth_state: DepthState,
    wavelength: float,
    offsets: np.ndarray,
) -> np.ndarray:
    """Return an approximate helium line profile."""

    sigma = 0.1 + 0.01 * depth_state.temperature / 10_000.0
    return np.exp(-(offsets / sigma) ** 2)
