"""LTE radiative transfer helpers using the Kurucz JOSH solver."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple

import numpy as np

from synthe_py.physics.josh_solver import solve_josh_flux

_H_PLANCK = 6.62607015e-27  # erg * s
_C_LIGHT = 2.99792458e10  # cm / s
_C_LIGHT_NM = 2.99792458e17  # nm / s
_K_BOLTZ = 1.380649e-16  # erg / K


def _planck_nu(freq: float, temperature: np.ndarray) -> np.ndarray:
    """Compute Planck function B_nu(T) using Fortran's exact formula.

    Fortran formula (atlas7v.for line 190):
    BNU(J) = 1.47439D-2 * FREQ15^3 * EHVKT(J) / STIM(J)
    Where FREQ15 = FREQ / 1.D15, EHVKT = exp(-FREQ*HKT), STIM = 1 - EHVKT

    This matches Fortran exactly to avoid any numerical precision differences.
    """
    # CRITICAL FIX: Match Fortran exactly - no clamping of temperature or STIM
    # Fortran (atlas7v.for line 186-187): EHVKT(J)=EXP(-FREQ*HKT(J)), STIM(J)=1.-EHVKT(J)
    # Fortran does NOT clamp temperature or STIM - use values directly
    freq15 = freq / 1.0e15
    hkt = _H_PLANCK / (_K_BOLTZ * temperature)
    ehvkt = np.exp(-freq * hkt)
    stim = 1.0 - ehvkt
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        bnu = 1.47439e-2 * freq15**3 * ehvkt / stim
    bnu[np.isnan(bnu)] = 0.0
    return bnu


def solve_lte_frequency(
    wavelength_nm: float,
    temperature: np.ndarray,
    column_mass: np.ndarray,
    cont_abs: np.ndarray,
    cont_scat: np.ndarray,
    line_opacity: np.ndarray,
    line_scattering: np.ndarray,
    line_source: Optional[np.ndarray] = None,
) -> Tuple[float, float]:

    # Filter INF/NaN values in continuum opacity (prevents TAUNU integration failure)
    # Fortran uses REAL*4 for CONTINUUM (max ~3.4e38), so filter values exceeding this
    # Reference: synthe.for line 9 (REAL*4) and line 215 (10.**CONTINUUM)
    MAX_OPACITY_REAL4 = 3.4e38
    mask = (
        (column_mass >= 0.0)
        & np.isfinite(column_mass)
        & np.isfinite(cont_abs)  # Filter INF in continuum absorption
        & np.isfinite(cont_scat)  # Filter INF in continuum scattering
        & np.isfinite(line_opacity)  # Filter INF in line opacity
        & np.isfinite(line_scattering)  # Filter INF in line scattering
        & (cont_abs < MAX_OPACITY_REAL4)  # Filter very large values (REAL*4 max)
        & (cont_scat < MAX_OPACITY_REAL4)  # Filter very large values
        & (line_opacity < MAX_OPACITY_REAL4)  # Filter very large line opacity
        & (line_scattering < MAX_OPACITY_REAL4)  # Filter very large line scattering
    )
    if not np.any(mask):
        return 0.0, 0.0

    # RHOX is now read correctly from fort.5 (not fort.10's wrong "depth" field)
    # Values are already in correct units (g/cm²) - no scaling needed
    mass_raw = np.asarray(column_mass[mask], dtype=np.float64)

    # Filter out zero-depth layers (invalid layers at the end)
    valid_mask = mass_raw > 0
    if not np.any(valid_mask):
        return 0.0, 0.0

    mass_valid = mass_raw[valid_mask]
    temp_valid = np.asarray(temperature[mask][valid_mask], dtype=np.float64)
    cont_a_valid = np.asarray(cont_abs[mask][valid_mask], dtype=np.float64)
    cont_s_valid = np.asarray(cont_scat[mask][valid_mask], dtype=np.float64)
    line_a_valid = np.asarray(line_opacity[mask][valid_mask], dtype=np.float64)
    line_sig_valid = np.asarray(line_scattering[mask][valid_mask], dtype=np.float64)

    # Fortran convention: J=1 is surface (small RHOX), J=NRHOX is deep (large RHOX)
    # Fortran assumes arrays are ALREADY in correct order (surface → deep, increasing RHOX)
    # No reversal logic needed - arrays from NPZ should already be in correct order
    # After masking: index 0 = surface (smallest RHOX), index -1 = deep (largest RHOX)
    mass = mass_valid
    temp = temp_valid
    cont_a = cont_a_valid
    cont_s = cont_s_valid
    line_a = line_a_valid
    line_sig = line_sig_valid

    # Fortran convention: J=1 is surface (small RHOX), J=NRHOX is deep (large RHOX)
    # INTEG requires RHOX to be monotonically increasing (surface → deep)
    # Fortran does NOT check array order - it assumes arrays are already correct
    # We should match Fortran: assume arrays are in correct order, only reverse if mass is decreasing

    # CRITICAL FIX: Match Fortran behavior - only reverse if mass is decreasing
    # Fortran doesn't check individual opacity arrays, it just uses them as-is
    # The only reversal needed is if RHOX itself is decreasing (which shouldn't happen with correct NPZ files)
    line_a_was_reversed = (
        False  # Track if line_a was reversed (for line_source alignment)
    )
    if mass.size > 1:
        mass_increasing = mass[0] < mass[-1]

        # Fortran does NOT check individual opacity arrays - it assumes they're already in correct order
        # Only reverse ALL arrays together if mass is decreasing (which shouldn't happen with correct NPZ files)
        # This matches Fortran's behavior: arrays are assumed to be in correct order (surface → deep)

        # Now ensure mass is in increasing order (surface → deep) for INTEG
        # Fortran's INTEG requires RHOX to be monotonically increasing
        if not mass_increasing:
            mass = mass[::-1]
            temp = temp[::-1]
            cont_a = cont_a[::-1]
            cont_s = cont_s[::-1]
            line_a = line_a[::-1]
            line_sig = line_sig[::-1]
            if line_source is not None:
                line_source = line_source[::-1]
            # Reset line_a_was_reversed since we've reversed everything together
            line_a_was_reversed = False

    # CRITICAL FIX: Match Fortran behavior - no clipping of opacity arrays
    # Fortran uses REAL*8 (double precision) and doesn't clip opacity arrays
    # Arrays are already float64 from masking/conversion above, matching Fortran REAL*8
    # Only ensure arrays are finite (masking already filtered INF/NaN)

    freq = _C_LIGHT_NM / max(wavelength_nm, 1e-12)
    planck = _planck_nu(freq, temp)

    if line_source is not None:
        ls_full = np.asarray(line_source, dtype=np.float64)

        # CRITICAL FIX: Reverse line_source in FULL array if line_a was reversed
        # This must happen BEFORE masking to keep them aligned after masking
        # We do this here (after getting the full array) rather than earlier to avoid
        # issues with parameter reassignment
        if line_a_was_reversed:
            ls_full = ls_full[::-1]

        # CRITICAL: Filter NaN/INF from line_source BEFORE masking
        # Fortran would propagate NaN/INF, but we need to filter them to prevent flux calculation failures
        # However, we should log warnings to match Fortran's behavior
        nan_mask_full = np.isnan(ls_full)
        inf_mask_full = np.isinf(ls_full)
        if np.any(nan_mask_full) or np.any(inf_mask_full):
            import logging

            logger = logging.getLogger(__name__)
            if np.any(nan_mask_full):
                logger.warning(
                    f"Line source contains {np.sum(nan_mask_full)} NaN values at wavelength {wavelength_nm:.6f} nm "
                    f"(replacing with Planck function to prevent flux failure)"
                )
            if np.any(inf_mask_full):
                logger.warning(
                    f"Line source contains {np.sum(inf_mask_full)} INF values at wavelength {wavelength_nm:.6f} nm "
                    f"(replacing with Planck function to prevent flux failure)"
                )
            # Compute Planck for full array to replace NaN/INF
            temp_full = np.asarray(temperature, dtype=np.float64)
            planck_full = _planck_nu(freq, temp_full)
            # Replace NaN/INF with Planck function (safer than propagating)
            # This is a compromise: Fortran would propagate, but we need to prevent flux failures
            ls_full = np.where(nan_mask_full | inf_mask_full, planck_full, ls_full)
        # Now apply masking
        ls = ls_full[mask][valid_mask]  # Already in correct order (aligned with line_a)
    else:
        ls = None

    line_src = ls if ls is not None else planck

    # CRITICAL FIX: line_source was already reversed in FULL array when line_a was reversed
    # So after masking, line_src should already be aligned with line_a
    # No need to reverse line_src again here - it's already in the correct order
    # The reversal happened in the full array before masking, so masking preserves alignment

    zero_line = np.zeros_like(line_a)
    zero_scatter = np.zeros_like(line_sig)
    # CRITICAL FIX: For continuum-only, SCONT should be Planck function, not scattering opacity!
    # In Fortran (atlas7v.for line 4477): SCONT(J) = BNU(J) for continuum-only
    # For continuum-only flux, use planck as scont (matching Fortran SCONT = BNU)
    # sigmac should be cont_s (scattering opacity), NOT planck!
    flux_cont = solve_josh_flux(
        cont_a,
        planck,  # scont: continuum source function (Planck function)
        zero_line,
        planck,  # sline: line source function (not used since aline=0)
        cont_s,  # sigmac: continuum scattering opacity (CRITICAL FIX!)
        zero_scatter,
        mass,
    )

    # CRITICAL FIX: Fortran spectrv.for passes ACONT (absorption only) to JOSH, not ABTOT
    # Line 254: ACONT(J) = 10.**(C1*CONTABS(...)) - absorption only
    # Line 256: SIGMAC(J) = 10.**(C1*CONTSCAT(...)) - scattering only
    # Line 270: CALL JOSH(IFSCAT,IFSURF) - passes ACONT, SIGMAC handled separately via IFSCAT
    # Python must match: pass cont_abs (ACONT) only, not cont_abs + cont_scat (ABTOT)
    # The scattering (cont_s) is passed separately and handled by the JOSH solver internally
    flux_total = solve_josh_flux(
        cont_a,  # ACONT only (absorption), NOT ABTOT (absorption + scattering)
        planck,
        line_a,
        line_src,
        cont_s,  # SIGMAC (scattering) - handled separately by JOSH solver
        line_sig,
        mass,
    )

    return flux_total, flux_cont


def _process_wavelength_batch(
    args: Tuple[
        int,
        float,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
    ],
) -> Tuple[int, float, float]:
    """Process a single wavelength (for parallel execution)."""
    (
        idx,
        wl,
        temperature,
        column_mass,
        cont_abs_col,
        cont_scat_col,
        line_opacity_col,
        line_scattering_col,
        line_source_col,
    ) = args

    ft, fc = solve_lte_frequency(
        wl,
        temperature,
        column_mass,
        cont_abs_col,
        cont_scat_col,
        line_opacity_col,
        line_scattering_col,
        line_source_col,
    )
    return idx, ft, fc


def _process_wavelength_chunk(
    chunk: list[
        Tuple[
            int,
            float,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            Optional[np.ndarray],
        ]
    ],
) -> list[Tuple[int, float, float]]:
    """Process a chunk of wavelengths with deterministic in-chunk ordering."""
    out: list[Tuple[int, float, float]] = []
    for args in chunk:
        out.append(_process_wavelength_batch(args))
    return out


def solve_lte_spectrum(
    wavelength_nm: np.ndarray,
    temperature: np.ndarray,
    column_mass: np.ndarray,
    cont_abs: np.ndarray,
    cont_scat: np.ndarray,
    line_opacity: np.ndarray,
    line_scattering: np.ndarray,
    line_source: Optional[np.ndarray] = None,
    n_workers: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve LTE radiative transfer for a spectrum.

    Parameters
    ----------
    wavelength_nm:
        Wavelength array (nm)
    temperature:
        Temperature array (K) for each depth
    column_mass:
        Column mass array (g/cm²) for each depth
    cont_abs:
        Continuum absorption opacity (n_depths, n_wavelengths)
    cont_scat:
        Continuum scattering opacity (n_depths, n_wavelengths)
    line_opacity:
        Line absorption opacity (n_depths, n_wavelengths)
    line_scattering:
        Line scattering opacity (n_depths, n_wavelengths)
    line_source:
        Optional line source function (n_depths, n_wavelengths)
    n_workers:
        Number of parallel workers. If None or 1, uses sequential processing.
        If > 1, uses multiprocessing.

    Returns
    -------
    flux_total:
        Total flux (HNU) for each wavelength
    flux_cont:
        Continuum flux (HNU) for each wavelength
    """
    n_points = wavelength_nm.size
    flux_total = np.zeros(n_points, dtype=np.float64)
    flux_cont = np.zeros(n_points, dtype=np.float64)

    logger = logging.getLogger(__name__)

    # Determine number of workers
    if n_workers is None:
        # Use max cores for large grids, sequential for small grids
        if n_points > 10000:
            import multiprocessing

            n_workers = max(1, multiprocessing.cpu_count())
        else:
            n_workers = 1
    elif n_workers < 1:
        n_workers = 1

    # Log initial status
    logger.info(f"Solving radiative transfer for {n_points:,} wavelengths...")
    if n_points > 10000:
        logger.warning(
            f"Large wavelength grid ({n_points:,} points) - "
            f"consider using --wavelength-subsample to reduce computation time"
        )

    if n_workers > 1:
        logger.info(f"Using {n_workers} parallel workers")
    else:
        logger.info("Using sequential processing")
    step2_rt_batching = os.getenv("PY_OPT_STEP2_RT_BATCH", "0") != "0"
    rt_batch_size = 32
    rt_batch_env = os.getenv("PY_RT_BATCH_SIZE")
    if rt_batch_env:
        try:
            rt_batch_size = max(1, int(rt_batch_env))
        except ValueError:
            rt_batch_size = 32

    # Process wavelengths
    if n_workers > 1 and n_points > 100:  # Only parallelize for large grids
        # Parallel processing using ThreadPoolExecutor — avoids the overhead of
        # pickling numpy arrays that ProcessPoolExecutor incurs.  The JOSH
        # solver's inner Numba kernels (_josh_iteration_kernel, _map1_kernel,
        # _integ, _parcoe, _deriv) release the GIL, enabling genuine
        # parallelism for the compute-heavy phases.
        log_interval = max(1, n_points // 100)  # Log every 1%
        completed = 0

        def _process_idx(idx: int) -> Tuple[int, float, float]:
            line_src_col = line_source[:, idx] if line_source is not None else None
            ft, fc = solve_lte_frequency(
                float(wavelength_nm[idx]),
                temperature,
                column_mass,
                cont_abs[:, idx],
                cont_scat[:, idx],
                line_opacity[:, idx],
                line_scattering[:, idx],
                line_src_col,
            )
            return idx, ft, fc

        def _process_idx_chunk(start_idx: int, end_idx: int) -> list[Tuple[int, float, float]]:
            out: list[Tuple[int, float, float]] = []
            for idx in range(start_idx, end_idx):
                out.append(_process_idx(idx))
            return out

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            if step2_rt_batching:
                chunks = [
                    (i, min(i + rt_batch_size, n_points))
                    for i in range(0, n_points, rt_batch_size)
                ]
                futures = {
                    executor.submit(_process_idx_chunk, chunk_start, chunk_end): (
                        chunk_start,
                        chunk_end,
                    )
                    for chunk_start, chunk_end in chunks
                }
            else:
                futures = {
                    executor.submit(_process_idx, idx): idx for idx in range(n_points)
                }

            for future in as_completed(futures):
                try:
                    result = future.result()
                    if step2_rt_batching:
                        for idx, ft, fc in result:
                            flux_total[idx] = ft
                            flux_cont[idx] = fc
                            completed += 1
                            if completed % log_interval == 0 or completed == n_points:
                                percent = 100.0 * completed / n_points
                                wl = float(wavelength_nm[idx]) if 0 <= idx < n_points else 0.0
                                logger.info(
                                    f"Progress: {completed:,}/{n_points:,} ({percent:.1f}%) - "
                                    f"wavelength {wl:.2f} nm"
                                )
                    else:
                        idx, ft, fc = result
                        flux_total[idx] = ft
                        flux_cont[idx] = fc
                        completed += 1
                        if completed % log_interval == 0 or completed == n_points:
                            percent = 100.0 * completed / n_points
                            wl = float(wavelength_nm[idx]) if 0 <= idx < n_points else 0.0
                            logger.info(
                                f"Progress: {completed:,}/{n_points:,} ({percent:.1f}%) - "
                                f"wavelength {wl:.2f} nm"
                            )
                except Exception as e:
                    task_payload = futures[future]
                    if step2_rt_batching:
                        first_idx = task_payload[0]
                        first_wl = (
                            float(wavelength_nm[first_idx])
                            if 0 <= first_idx < n_points
                            else 0.0
                        )
                        logger.error(
                            f"Error processing wavelength chunk starting at {first_idx} ({first_wl:.2f} nm): {e}"
                        )
                        for idx in range(task_payload[0], task_payload[1]):
                            flux_total[idx] = 0.0
                            flux_cont[idx] = 0.0
                            completed += 1
                    else:
                        idx = task_payload
                        wl = float(wavelength_nm[idx]) if 0 <= idx < n_points else 0.0
                        logger.error(
                            f"Error processing wavelength {idx} ({wl:.2f} nm): {e}"
                        )
                        flux_total[idx] = 0.0
                        flux_cont[idx] = 0.0
                        completed += 1
    else:
        # Sequential processing with progress logging
        log_interval = max(1, n_points // 100)  # Log every 1%

        for idx in range(n_points):
            wl = wavelength_nm[idx]

            line_src_col = line_source[:, idx] if line_source is not None else None
            ft, fc = solve_lte_frequency(
                wavelength_nm[idx],
                temperature,
                column_mass,
                cont_abs[:, idx],
                cont_scat[:, idx],
                line_opacity[:, idx],
                line_scattering[:, idx],
                line_src_col,
            )
            flux_total[idx] = ft
            flux_cont[idx] = fc

            # Progress logging
            if (idx + 1) % log_interval == 0 or idx == n_points - 1:
                percent = 100.0 * (idx + 1) / n_points
                logger.info(
                    f"Progress: {idx+1:,}/{n_points:,} ({percent:.1f}%) - "
                    f"wavelength {wl:.2f} nm"
                )

    logger.info(f"Completed radiative transfer for {n_points:,} wavelengths")
    return flux_total, flux_cont
