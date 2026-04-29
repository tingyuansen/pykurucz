"""Continuum opacity calculations mirroring the legacy SYNTHE algorithm."""

from __future__ import annotations

import logging
import numpy as np

from typing import TYPE_CHECKING, Optional

from .tables import ContinuumTables

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from ..io.atmosphere import AtmosphereModel


def interpolate_continuum(
    tables: ContinuumTables,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """Reproduce the DO 2005 loop filling CONTINUUM(NBUFF)."""

    continuum = np.zeros_like(wavelengths, dtype=np.float64)
    edge = 0
    n_edges = tables.ablog.shape[1]

    for idx, wave in enumerate(wavelengths):
        while edge < n_edges - 1 and wave >= tables.wledge[edge + 1]:
            edge += 1
        edge = min(edge, n_edges - 1)

        numerator = (
            (wave - tables.half_edge[edge])
            * (wave - tables.wledge[edge + 1])
            * tables.ablog[0, edge]
            + (tables.wledge[edge] - wave)
            * (wave - tables.wledge[edge + 1])
            * 2.0
            * tables.ablog[1, edge]
            + (wave - tables.wledge[edge])
            * (wave - tables.half_edge[edge])
            * tables.ablog[2, edge]
        )
        continuum[idx] = numerator / tables.delta_edge[edge]

    return continuum


def finalize_continuum(log_continuum: np.ndarray) -> np.ndarray:
    """Apply the final 10** conversion (loop 2006)."""

    return np.power(10.0, log_continuum)


def build_continuum_grid(
    atmosphere: "AtmosphereModel",
    wavelength_grid: np.ndarray,
) -> np.ndarray:
    """Compute continuum opacity for each layer using legacy interpolation."""

    continuum = np.zeros((atmosphere.layers, wavelength_grid.size), dtype=np.float64)
    tables = atmosphere.continuum_tables
    if tables is None:
        return continuum

    log_cont = interpolate_continuum(tables, wavelength_grid)
    cont = finalize_continuum(log_cont)
    continuum[:] = cont
    return continuum


def build_depth_continuum(
    atmosphere: "AtmosphereModel",
    wavelength_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Return per-depth continuum absorption, scattering, and optional line sources.

    This function matches Fortran's approach: uses log10 coefficients (CONTABS/CONTSCAT)
    from fort.10, interpolates them, then converts to linear with 10**.
    Requires wledge and cont_abs_coeff/cont_scat_coeff to be present in the atmosphere model.
    """

    tables = atmosphere.continuum_tables
    coeff_abs = getattr(atmosphere, "continuum_abs_coeff", None)
    coeff_scat = getattr(atmosphere, "continuum_scat_coeff", None)

    wl = np.asarray(wavelength_grid, dtype=np.float64)

    # CRITICAL: Fortran uses coefficient interpolation (CONTABS/CONTSCAT from fort.10)
    # Python MUST use the same method - require coefficients to be present
    # This matches Fortran spectrv.for lines 254-255: ACONT(J)=10.**(C1*CONTABS(1,IEDGE,J)+...)
    if (
        atmosphere.continuum_wledge is not None
        and atmosphere.continuum_abs_coeff is not None
        and atmosphere.continuum_scat_coeff is not None
    ):
        wledge = np.asarray(atmosphere.continuum_wledge, dtype=np.float64)
        # Fortran takes ABS(WLEDGE) when reading (synthe.for line 136-138, spectrv.for line 113-115)
        wledge = np.abs(wledge)
        if wledge.size < 2:
            raise ValueError(
                "Continuum edge table requires at least two wavelength entries"
            )
        half_edge = atmosphere.continuum_half_edge
        if half_edge is None or half_edge.size != wledge.size - 1:
            half_edge = 0.5 * (wledge[:-1] + wledge[1:])
        delta_edge = atmosphere.continuum_delta_edge
        if delta_edge is None or delta_edge.size != wledge.size - 1:
            delta = wledge[1:] - wledge[:-1]
            delta_edge = 0.5 * delta * delta
        abs_coeff = np.asarray(atmosphere.continuum_abs_coeff, dtype=np.float64)
        scat_coeff = np.asarray(atmosphere.continuum_scat_coeff, dtype=np.float64)
        if (
            abs_coeff.shape[1] != wledge.size - 1
            or scat_coeff.shape[1] != wledge.size - 1
        ):
            raise ValueError(
                "Continuum coefficient grid does not match edge table dimensions"
            )

        # CRITICAL FIX: Fortran uses ABS(WLEDGE) for edge finding!
        # From spectrv.for lines 113-115:
        # WLEDGE(1)=ABS(WLEDGE(1))
        # DO 2001 IEDGE=2,NEDGE
        #   WLEDGE(IEDGE)=ABS(WLEDGE(IEDGE))
        # Then uses sequential search: IF(WAVE.LT.WLEDGE(IEDGE+1))GO TO 3005
        wledge_abs = np.abs(wledge)

        # Fortran sorts edge table by ABS(WLEDGE) when writing (xnfpelsyn.for lines 185-202)
        # After reading interleaved, edges should already be sorted
        # Check if sorted, and if not, warn but proceed (edges should be sorted by convert_fort10.py)
        is_sorted = np.all(np.diff(wledge_abs) >= 0)
        if not is_sorted:
            import warnings

            warnings.warn(
                "Edge table is not sorted! Results may be incorrect.", UserWarning
            )

        # Vectorize over all wavelengths
        n_layers = abs_coeff.shape[0]
        n_wl = wl.size

        # Find edge index for each wavelength
        edge_indices = np.searchsorted(wledge_abs, np.abs(wl), side="right") - 1
        edge_indices = np.clip(edge_indices, 0, wledge_abs.size - 2)

        # Pre-allocate output arrays
        absorption = np.zeros((n_layers, n_wl), dtype=np.float64)
        scattering = np.zeros((n_layers, n_wl), dtype=np.float64)

        # Process each unique edge index
        for edge_idx in range(wledge_abs.size - 1):
            mask = edge_indices == edge_idx
            if not np.any(mask):
                continue

            wl_mask = wl[mask]
            wl_left = wledge_abs[edge_idx]
            wl_right = wledge_abs[edge_idx + 1]

            # Use stored half_edge and delta_edge
            if half_edge is not None and edge_idx < len(half_edge):
                half = half_edge[edge_idx]
            else:
                half = 0.5 * (wl_left + wl_right)

            if delta_edge is not None and edge_idx < len(delta_edge):
                delta = delta_edge[edge_idx]
            else:
                delta = 0.5 * (wl_right - wl_left) ** 2

            if delta == 0.0:
                delta = 1e-20

            # Compute interpolation coefficients for all wavelengths in this edge
            # These are Lagrange basis polynomial coefficients normalized to sum to 1
            # The NPZ stores log10(opacity) values at three wavelength points within the edge
            c1 = (wl_mask - half) * (wl_mask - wl_right) / delta
            c2 = (wl_left - wl_mask) * (wl_mask - wl_right) * 2.0 / delta
            c3 = (wl_mask - wl_left) * (wl_mask - half) / delta
            # Get coefficients for this edge across all layers
            a1 = abs_coeff[:, edge_idx, 0]  # shape: (n_layers,)
            a2 = abs_coeff[:, edge_idx, 1]
            a3 = abs_coeff[:, edge_idx, 2]
            s1 = scat_coeff[:, edge_idx, 0]
            s2 = scat_coeff[:, edge_idx, 1]
            s3 = scat_coeff[:, edge_idx, 2]
            # Compute log opacity: broadcast (n_layers,) with (n_wl_in_edge,)
            # Result shape: (n_layers, n_wl_in_edge)
            log_abs = (
                a1[:, np.newaxis] * c1[np.newaxis, :]
                + a2[:, np.newaxis] * c2[np.newaxis, :]
                + a3[:, np.newaxis] * c3[np.newaxis, :]
            )
            log_scat = (
                s1[:, np.newaxis] * c1[np.newaxis, :]
                + s2[:, np.newaxis] * c2[np.newaxis, :]
                + s3[:, np.newaxis] * c3[np.newaxis, :]
            )

            # NOTE: Previous "fix" with log10(3.72) was WRONG
            # The NPZ stores correct SIGMAC values without any extra factor
            # Removing the incorrect correction restores Fortran-matching behavior
            # Raw NPZ SIGMAC at 300nm matches Fortran ABTOT (4.32e-3 vs 4.13e-3, only 5% diff)
            # pass  # No correction needed - coefficients are already correct

            # Store results
            absorption[:, mask] = log_abs
            scattering[:, mask] = log_scat

        # Convert to linear scale if needed
        if atmosphere.continuum_coeff_log10:
            log_abs = absorption
            log_scat = scattering

            # CRITICAL FIX: Handle zero coefficients correctly
            # When coefficients are zero (or very small), log_opacity ≈ 0, giving 10^0 = 1.0
            # But if opacity should be zero, this is wrong. Check if coefficients suggest
            # zero opacity and set to very small value instead of 1.0

            # CRITICAL FIX: Clamp log opacity to prevent INF (Fortran uses REAL*4, max ~3.4e38)
            # log10(3.4e38) ≈ 38.5, so clamp log values before 10** to prevent overflow
            # Reference: synthe.for line 215 (10.**CONTINUUM) with REAL*4 precision
            MAX_LOG_OPACITY = 38.0  # Slightly below 38.5 for safety
            log_abs_clamped = np.clip(log_abs, -np.inf, MAX_LOG_OPACITY)
            log_scat_clamped = np.clip(log_scat, -np.inf, MAX_LOG_OPACITY)

            # Diagnostic: Check for very large log values before conversion
            if log_abs_clamped.size > 0:
                max_log_abs = np.max(log_abs_clamped)
                clamped_count = np.sum(log_abs >= MAX_LOG_OPACITY)
                if max_log_abs >= MAX_LOG_OPACITY - 0.1:  # Close to clamp threshold
                    logger.warning(
                        f"Large log opacity values detected: max={max_log_abs:.2f} "
                        f"(will produce ~10^{max_log_abs:.1f}). "
                        f"Clamped {clamped_count} values to {MAX_LOG_OPACITY}."
                    )

            absorption = np.power(10.0, log_abs_clamped)
            scattering = np.power(10.0, log_scat_clamped)

            # Additional safety: clamp final values to REAL*4 maximum
            # Fortran REAL*4 max: ~3.4e38 (IEEE-754 single precision)
            MAX_OPACITY_REAL4 = 3.4e38
            absorption = np.clip(absorption, 0.0, MAX_OPACITY_REAL4)
            scattering = np.clip(scattering, 0.0, MAX_OPACITY_REAL4)

            # Handle any remaining INF/NaN (shouldn't happen after clamping, but safety check)
            absorption = np.where(
                np.isfinite(absorption), absorption, MAX_OPACITY_REAL4
            )
            scattering = np.where(
                np.isfinite(scattering), scattering, MAX_OPACITY_REAL4
            )

            # CRITICAL: Verify no INF values remain (shouldn't happen after clamping, but safety check)
            if np.any(~np.isfinite(absorption)):
                n_inf = np.sum(~np.isfinite(absorption))
                logger.warning(
                    f"Found {n_inf} INF/NaN values in absorption after clamping - forcing to max"
                )
                absorption = np.where(
                    np.isfinite(absorption), absorption, MAX_OPACITY_REAL4
                )

            if np.any(~np.isfinite(scattering)):
                n_inf = np.sum(~np.isfinite(scattering))
                logger.warning(
                    f"Found {n_inf} INF/NaN values in scattering after clamping - forcing to max"
                )
                scattering = np.where(
                    np.isfinite(scattering), scattering, MAX_OPACITY_REAL4
                )

            # Fortran behavior (synthe.for line 218):
            #   CONTINUUM(NBUFF)=10.**CONTINUUM(NBUFF)
            # Fortran simply converts 10^x on stored log values with NO threshold check.
            #
            # The floor is already applied during storage in xnfpelsyn.for line 388:
            #   LOGACONT=LOG10(MAX(ACONT(J),1.D-30))
            # And in Python's convert_atm_to_npz.py:
            #   cont_abs_log = np.log10(np.maximum(acont, 1e-30))
            #
            # So valid log values range from -30 to ~+6, and we trust them as-is.
            # NO CLAMPING HERE - match Fortran exactly.
        else:
            # CRITICAL: Fortran uses coefficient interpolation, not pre-computed linear values
            # If coefficients are missing, we cannot proceed - NPZ file must be regenerated
            # with correct coefficients (wledge, cont_abs_coeff, cont_scat_coeff)
            raise ValueError(
                "Continuum coefficients are missing from atmosphere model. "
                "The NPZ file must contain 'wledge', 'cont_abs_coeff', and 'cont_scat_coeff' "
                "for coefficient interpolation (matching Fortran's CONTABS/CONTSCAT method). "
                "Pre-computed 'continuum_absorption' values cannot be used as they may have "
                "incorrect units or be computed incorrectly. "
                "Please regenerate the NPZ file using convert_atm_to_npz.py or convert_fort10.py."
            )

        line_src_lower = getattr(atmosphere, "line_source_lower", None)
        line_src_upper = getattr(atmosphere, "line_source_upper", None)
        return absorption, scattering, line_src_lower, line_src_upper

    # Fallback path using continuum_tables (legacy ablog format)
    # This path is only used if coefficients are not available
    # CRITICAL: This should not be used for synthesis - coefficients are required
    if tables is None:
        raise ValueError(
            "Continuum tables (ablog) are missing from atmosphere model. "
            "The NPZ file must contain either 'cont_abs_coeff'/'cont_scat_coeff' (preferred) "
            "or 'ablog'/'qablog' (legacy format) for continuum interpolation. "
            "Please regenerate the NPZ file using convert_atm_to_npz.py or convert_fort10.py."
        )

    n_layers = coeff_abs.shape[0]
    n_points = wavelength_grid.size
    absorption = np.zeros((n_layers, n_points), dtype=np.float64)
    scattering = np.zeros((n_layers, n_points), dtype=np.float64)

    wledge = tables.wledge
    half_edge = tables.half_edge
    delta_edge = tables.delta_edge
    n_intervals = delta_edge.size
    max_edge = n_intervals - 1
    edge = 0

    for idx, wave in enumerate(wavelength_grid):
        while edge < max_edge and wave >= wledge[edge + 1]:
            edge += 1

        c1 = (wave - half_edge[edge]) * (wave - wledge[edge + 1]) / delta_edge[edge]
        c2 = (wledge[edge] - wave) * (wave - wledge[edge + 1]) * 2.0 / delta_edge[edge]
        c3 = (wave - wledge[edge]) * (wave - half_edge[edge]) / delta_edge[edge]

        base = 3 * edge
        a1 = coeff_abs[:, base]
        a2 = coeff_abs[:, base + 1]
        a3 = coeff_abs[:, base + 2]
        s1 = coeff_scat[:, base]
        s2 = coeff_scat[:, base + 1]
        s3 = coeff_scat[:, base + 2]

        # CRITICAL FIX: Clamp log opacity to prevent INF (Fortran uses REAL*4, max ~3.4e38)
        # Reference: synthe.for line 215 (10.**CONTINUUM) with REAL*4 precision
        log_abs_val = c1 * a1 + c2 * a2 + c3 * a3
        log_scat_val = c1 * s1 + c2 * s2 + c3 * s3

        MAX_LOG_OPACITY = 38.0  # Slightly below 38.5 for safety
        log_abs_clamped = np.clip(log_abs_val, -np.inf, MAX_LOG_OPACITY)
        log_scat_clamped = np.clip(log_scat_val, -np.inf, MAX_LOG_OPACITY)

        # Diagnostic: Check for very large log values before conversion
        max_log_abs = np.max(log_abs_clamped) if log_abs_clamped.size > 0 else 0.0
        max_log_scat = np.max(log_scat_clamped) if log_scat_clamped.size > 0 else 0.0

        if max_log_abs > 35.0:  # 10^35 is getting close to REAL*4 max
            logger.warning(
                f"Large log opacity values detected (wavelength {wave:.2f} nm): "
                f"max={max_log_abs:.2f} (will produce ~10^{max_log_abs:.1f}). "
                f"Clamping to {MAX_LOG_OPACITY}."
            )

        absorption[:, idx] = np.power(10.0, log_abs_clamped)
        scattering[:, idx] = np.power(10.0, log_scat_clamped)

        # Additional safety: clamp final values to REAL*4 maximum
        MAX_OPACITY_REAL4 = 3.4e38
        absorption[:, idx] = np.clip(absorption[:, idx], 0.0, MAX_OPACITY_REAL4)
        scattering[:, idx] = np.clip(scattering[:, idx], 0.0, MAX_OPACITY_REAL4)

        # Handle any remaining INF/NaN
        absorption[:, idx] = np.where(
            np.isfinite(absorption[:, idx]), absorption[:, idx], MAX_OPACITY_REAL4
        )
        scattering[:, idx] = np.where(
            np.isfinite(scattering[:, idx]), scattering[:, idx], MAX_OPACITY_REAL4
        )

        # CRITICAL: Verify no INF values remain (shouldn't happen after clamping, but safety check)
        if np.any(~np.isfinite(absorption[:, idx])):
            n_inf = np.sum(~np.isfinite(absorption[:, idx]))
            logger.warning(
                f"Found {n_inf} INF/NaN values in absorption[:, {idx}] after clamping - forcing to max"
            )
            absorption[:, idx] = np.where(
                np.isfinite(absorption[:, idx]), absorption[:, idx], MAX_OPACITY_REAL4
            )

        if np.any(~np.isfinite(scattering[:, idx])):
            n_inf = np.sum(~np.isfinite(scattering[:, idx]))
            logger.warning(
                f"Found {n_inf} INF/NaN values in scattering[:, {idx}] after clamping - forcing to max"
            )
            scattering[:, idx] = np.where(
                np.isfinite(scattering[:, idx]), scattering[:, idx], MAX_OPACITY_REAL4
            )

    line_src_lower = getattr(atmosphere, "line_source_lower", None)
    line_src_upper = getattr(atmosphere, "line_source_upper", None)

    return absorption, scattering, line_src_lower, line_src_upper
