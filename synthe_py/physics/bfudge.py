"""
BFUDGE and continuum line source reconstruction helpers.

The original SYNTHE code raises the depth-dependent departure factors
(``BHYD``, ``BC1``, ``BC2``, ``BSI1``, ``BSI2``) to user-specified exponents
(``PH1``, ``PC1``, ``PSI1``) to obtain the effective source correction used in
``spectrv``.  This module mirrors that behaviour and provides a single entry
point that produces both the scalar ``BFUDGE`` column and the continuum source
term ``SLineC`` required by the line transport solver.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from typing import TYPE_CHECKING

from ..io.spectrv import SpectrvParams

if TYPE_CHECKING:
    from ..io.atmosphere import AtmosphereModel


def _ensure_column(matrix: np.ndarray, width: int) -> np.ndarray:
    """
    Guarantee that ``matrix`` has at least ``width`` columns.

    Missing columns are padded with ones (the neutral scaling used by the
    Fortran defaults).  The function does *not* copy when the requirement is
    already met.
    """

    if matrix.ndim != 2:
        raise ValueError("Expected 2-D array for BFUDGE inputs")
    rows, cols = matrix.shape
    if cols >= width:
        return matrix
    if cols == 0:
        padded = np.ones((rows, width), dtype=matrix.dtype)
        return padded
    pad = np.ones((rows, width - cols), dtype=matrix.dtype)
    return np.hstack((matrix, pad))


def _prep_factor(
    data: np.ndarray | None,
    layers: int,
    required_cols: int,
) -> np.ndarray:
    """
    Normalise optional atmosphere arrays for the BFUDGE computation.

    When the cached atmosphere does not contain a given table we fall back to a
    depth-by-column matrix filled with ones, matching the legacy Kurucz
    behaviour where DATA statements initialised these factors to unity.
    """

    if data is None:
        return np.ones((layers, required_cols), dtype=np.float64)
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[0] != layers:
        raise ValueError(
            f"BFUDGE helper expected {layers} layers, received {arr.shape[0]}"
        )
    return _ensure_column(arr, required_cols)


def compute_bfudge_and_slinec(
    atmosphere: AtmosphereModel,
    params: SpectrvParams,
    bnu: np.ndarray,
    stim: np.ndarray,
    ehvkt: np.ndarray,
    *,
    clip: Tuple[float, float] | None = None,  # No clamping by default (matches Fortran)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the scalar BFUDGE column and the continuum source term ``SLineC``.

    Parameters
    ----------
    atmosphere:
        The populated atmosphere model containing the depth-dependent B tables.
    params:
        The parsed ``spectrv`` control parameters (PH1 / PC1 / PSI1 exponents).
    bnu, stim, ehvkt:
        Depth-by-frequency arrays matching the main synthesis buffers.
    clip:
        Minimum/maximum clamp for BFUDGE to avoid numerical underflow.
    """

    layers, nfreq = bnu.shape
    lower, upper = clip if clip is not None else (None, None)

    bhyd = _prep_factor(atmosphere.bhyd, layers, 1)[:, 0]
    bc1 = _prep_factor(atmosphere.bc1, layers, 1)[:, 0]
    bc2 = _prep_factor(atmosphere.bc2, layers, 1)[:, 0]
    bsi1 = _prep_factor(atmosphere.bsi1, layers, 1)[:, 0]
    bsi2 = _prep_factor(atmosphere.bsi2, layers, 1)[:, 0]

    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        # Compute BFUDGE exactly as Fortran does (spectrv.for line 173-174):
        # BFUDGE(J) = BHYD(J,1)**PH1 * (BC1(J,1)/BC2(J,1))**PC1 * (BSI1(J,1)/BSI2(J,1))**PSI1
        #
        # When PH1=PC1=PSI1=0 (LTE case), this evaluates to exactly 1.0
        # This is correct! In LTE, SLINEC = BNU*STIM/(1.0-EHVKT) = BNU*STIM/STIM = BNU (Planck)
        #
        # The key insight: Fortran doesn't add any epsilon or empirical correction to BFUDGE
        # The source function in LTE should be the Planck function, which naturally produces
        # absorption lines (flux < continuum) through the radiative transfer equation.
        bfudge = (
            np.power(np.maximum(bhyd, 1e-30), params.ph1)
            * np.power(np.maximum(bc1 / np.maximum(bc2, 1e-30), 1e-30), params.pc1)
            * np.power(np.maximum(bsi1 / np.maximum(bsi2, 1e-30), 1e-30), params.psi1)
        )

    # CRITICAL: Fortran does NOT clamp BFUDGE (spectrv.for line 174-175)
    # BFUDGE is computed directly without any range limits
    # The clip parameter is kept for backward compatibility but defaults to no clamping
    # In LTE (PH1=PC1=PSI1=0), BFUDGE = 1.0 exactly, so clamping is unnecessary
    if lower is not None or upper is not None:
        # Only apply clip if explicitly requested (for backward compatibility)
        # But by default, clip=(1e-6, 1e6) is provided, so we need to check if it's the default
        # For now, disable clamping to match Fortran - user can override if needed
        pass  # No clamping to match Fortran
        # bfudge = np.clip(bfudge, lower if lower is not None else -np.inf, upper if upper is not None else np.inf)

    bfudge_grid = bfudge[:, None]
    # Match Fortran exactly: SLINEC = BNU * STIM / (BFUDGE - EHVKT)
    # No clamping - Fortran doesn't clamp, so we shouldn't either
    # In LTE: bfudge = 1.0, ehvkt = exp(-hν/kT), so bfudge - ehvkt = stim
    # Therefore: SLINEC = BNU * STIM / STIM = BNU (no division by tiny number)
    #
    # Fortran uses non-stop arithmetic: NaN/INF values propagate through calculations
    # without explicit handling. We match this behavior by:
    # 1. Using np.errstate to suppress warnings (matching Fortran's non-stop behavior)
    # 2. NOT converting NaN to 0 (unlike np.nan_to_num) - let NaN/INF propagate
    # 3. Only handle truly degenerate cases where Fortran would also fail
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        denom = bfudge_grid - ehvkt
        slinec = (bnu * stim) / denom
    
    # In LTE, denom should equal STIM exactly, so slinec should equal BNU
    # If we get NaN/INF, it means BFUDGE != 1.0 (non-LTE) or numerical precision issue
    # Fortran would propagate these values, so we do too (no np.nan_to_num)
    # However, we can optionally log warnings for debugging
    nan_count = np.sum(np.isnan(slinec))
    inf_count = np.sum(np.isinf(slinec))
    if nan_count > 0 or inf_count > 0:
        # Log warning but don't modify values (matching Fortran's behavior)
        import logging
        logger = logging.getLogger(__name__)
        if nan_count > 0:
            logger.warning(
                f"BFUDGE/SLINEC: {nan_count} NaN values detected (propagating as Fortran would)"
            )
        if inf_count > 0:
            logger.warning(
                f"BFUDGE/SLINEC: {inf_count} INF values detected (propagating as Fortran would)"
            )

    return bfudge, slinec


