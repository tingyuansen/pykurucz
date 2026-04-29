"""Line broadening utilities."""

from __future__ import annotations

import numpy as np

from .populations import DepthState


def damping_parameter(
    depth_state: DepthState,
    gamma_rad: np.ndarray,
    gamma_stark: np.ndarray,
    gamma_vdw: np.ndarray,
) -> np.ndarray:
    """Compute the damping parameter a for the Voigt profile."""

    # Placeholder: combine rates linearly and normalise by Doppler width
    total_gamma = gamma_rad + gamma_stark + gamma_vdw
    width = depth_state.doppler_width.copy()
    width[width == 0] = 1.0
    return total_gamma / width
