#!/usr/bin/env python3
"""Compare Python and Fortran spectra and compute statistics."""

import argparse
from pathlib import Path
import re
import sys
from typing import cast

import numpy as np

FLOAT_RE = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][+-]?\d+)?")


def load_spectrum(filepath: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load spectrum file with wavelength, flux, continuum columns.

    Uses regex tokenization so Fortran glued fields (e.g. "304.85-0.17E-15")
    are parsed correctly.
    """
    wavelengths: list[float] = []
    fluxes: list[float] = []
    continua: list[float] = []

    with open(filepath, "r") as f:
        for line in f:
            parts = FLOAT_RE.findall(line)
            if len(parts) >= 3:
                try:
                    wl = float(parts[0])
                    flux = float(parts[1])
                    cont = float(parts[2])
                    wavelengths.append(wl)
                    fluxes.append(flux)
                    continua.append(cont)
                except ValueError:
                    continue

    return np.array(wavelengths), np.array(fluxes), np.array(continua)


def compare_spectra(
    python_file: Path,
    fortran_file: Path,
    wl_range: tuple[float, float] | None = None,
    top_n: int | None = None,
    quiet: bool = False,
):
    """Compare two spectra and print statistics.

    Returns a dict with keys: flux_mean_rel, flux_median_rel, flux_rms_rel,
    flux_rms_rel_robust, cont_mean_rel, cont_median_rel, cont_rms_rel,
    norm_mean, norm_median, norm_rms, n_points, wl_min, wl_max.
    """

    if not quiet:
        print(f"Loading Python spectrum: {python_file}")
    py_wl, py_flux, py_cont = load_spectrum(python_file)
    if not quiet:
        print(
            f"  {len(py_wl)} points, wavelength range: {py_wl.min():.2f} - {py_wl.max():.2f} nm"
        )
        print(f"Loading Fortran spectrum: {fortran_file}")
    ft_wl, ft_flux, ft_cont = load_spectrum(fortran_file)
    if not quiet:
        print(
            f"  {len(ft_wl)} points, wavelength range: {ft_wl.min():.2f} - {ft_wl.max():.2f} nm"
        )

    # Find common wavelength range
    wl_min = max(py_wl.min(), ft_wl.min())
    wl_max = min(py_wl.max(), ft_wl.max())

    if wl_range:
        wl_min = max(wl_min, wl_range[0])
        wl_max = min(wl_max, wl_range[1])

    if not quiet:
        print(f"\nComparing in range: {wl_min:.2f} - {wl_max:.2f} nm")

    # Filter to common range
    py_mask = (py_wl >= wl_min) & (py_wl <= wl_max)
    ft_mask = (ft_wl >= wl_min) & (ft_wl <= wl_max)

    py_wl_common = py_wl[py_mask]
    py_flux_common = py_flux[py_mask]
    py_cont_common = py_cont[py_mask]

    ft_wl_common = ft_wl[ft_mask]
    ft_flux_common = ft_flux[ft_mask]
    ft_cont_common = ft_cont[ft_mask]

    # Interpolate Fortran to Python wavelengths
    ft_flux_interp = np.interp(py_wl_common, ft_wl_common, ft_flux_common)
    ft_cont_interp = np.interp(py_wl_common, ft_wl_common, ft_cont_common)

    n_points = len(py_wl_common)
    if not quiet:
        print(f"Comparing {n_points} wavelength points")

    # Relative differences in percent with explicit masked divide to avoid
    # divide-by-zero warnings and unstable tiny-denominator points.
    flux_rel = np.zeros_like(py_flux_common)
    flux_denom_mask = np.abs(ft_flux_interp) > 1e-30
    np.divide(
        100 * (py_flux_common - ft_flux_interp),
        ft_flux_interp,
        out=flux_rel,
        where=flux_denom_mask,
    )
    cont_rel = np.zeros_like(py_cont_common)
    cont_denom_mask = np.abs(ft_cont_interp) > 1e-30
    np.divide(
        100 * (py_cont_common - ft_cont_interp),
        ft_cont_interp,
        out=cont_rel,
        where=cont_denom_mask,
    )

    # Normalized flux (flux / continuum)
    py_norm = py_flux_common / np.maximum(py_cont_common, 1e-30)
    ft_norm = ft_flux_interp / np.maximum(ft_cont_interp, 1e-30)
    norm_diff = py_norm - ft_norm

    # Compute summary statistics
    flux_rms = np.sqrt(np.mean(flux_rel**2))
    cont_rms = np.sqrt(np.mean(cont_rel**2))
    cont_scale = float(np.median(np.abs(ft_cont_interp)))
    robust_flux_threshold = max(1e-30, 1e-3 * cont_scale)
    robust_flux_mask = np.abs(ft_flux_interp) > robust_flux_threshold
    flux_rms_robust = (
        float(np.sqrt(np.mean(flux_rel[robust_flux_mask] ** 2)))
        if np.any(robust_flux_mask)
        else float("nan")
    )

    if not quiet:
        print("\n" + "=" * 60)
        print("SPECTRUM COMPARISON SUMMARY")
        print("=" * 60)
        print(f"\n{'Column':<20} {'Mean':<12} {'Median':<12} {'RMS':<12}")
        print("-" * 56)
        print(
            f"{'Flux':<20} {np.mean(flux_rel):+.2f}%{'':<5} {np.median(flux_rel):+.2f}%{'':<5} {flux_rms:.2f}%"
        )
        print(
            f"{'Flux (robust)':<20} {np.mean(flux_rel[robust_flux_mask]):+.2f}%{'':<5} "
            f"{np.median(flux_rel[robust_flux_mask]):+.2f}%{'':<5} {flux_rms_robust:.2f}%"
        )
        print(
            f"{'Continuum':<20} {np.mean(cont_rel):+.2f}%{'':<5} {np.median(cont_rel):+.2f}%{'':<5} {cont_rms:.2f}%"
        )
        print(
            f"{'Normalized (F/C)':<20} {np.mean(norm_diff):+.4f}{'':<3} {np.median(norm_diff):+.4f}{'':<3} {np.sqrt(np.mean(norm_diff**2)):.4f}"
        )
        print()

        # Status check
        cont_status = "✅" if cont_rms < 1.0 else "❌"
        flux_status = "✅" if flux_rms < 1.0 else "❌"
        flux_status_robust = "✅" if flux_rms_robust < 1.0 else "❌"
        print(
            f"Sub-percent accuracy: Continuum {cont_status}  Flux {flux_status}  "
            + f"Flux(robust) {flux_status_robust}"
        )
        print(
            f"Robust flux uses |Fortran flux| > {robust_flux_threshold:.3e} "
            + f"({np.sum(robust_flux_mask)} / {len(robust_flux_mask)} points)"
        )
        print("=" * 60)

    if not quiet and top_n and top_n > 0:
        abs_frac = np.abs(flux_rel)
        order = np.argsort(abs_frac)[::-1]
        print("\nTop |fractional| flux outliers:")
        for rank, idx in enumerate(order[:top_n], start=1):
            if idx >= len(py_wl_common):
                continue
            wl_val = py_wl_common[idx]
            frac_val = (py_flux_common[idx] - ft_flux_interp[idx]) / max(
                ft_flux_interp[idx], 1e-30
            )
            msg = (
                f"{rank:2d} wl={wl_val:.6f} frac={frac_val:+.3f} "
                + f"py={py_flux_common[idx]:.3e} ft={ft_flux_interp[idx]:.3e}"
            )
            print(msg)

    return {
        "flux_mean_rel": np.mean(flux_rel),
        "flux_median_rel": np.median(flux_rel),
        "flux_rms_rel": flux_rms,
        "flux_rms_rel_robust": flux_rms_robust,
        "cont_mean_rel": np.mean(cont_rel),
        "cont_median_rel": np.median(cont_rel),
        "cont_rms_rel": cont_rms,
        "norm_mean": np.mean(norm_diff),
        "norm_median": np.median(norm_diff),
        "norm_rms": np.sqrt(np.mean(norm_diff**2)),
        "n_points": n_points,
        "wl_min": wl_min,
        "wl_max": wl_max,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Python and Fortran spectra and compute statistics."
    )
    _ = parser.add_argument(
        "python_file",
        nargs="?",
        default="synthe_py/out/at12_aaaaa_t02500g-1.0_300_1800_tfort.spec",
    )
    _ = parser.add_argument(
        "fortran_file",
        nargs="?",
        default="grids/at12_aaaaa/spec/at12_aaaaa_t02500g-1.0.spec",
    )
    _ = parser.add_argument(
        "--range",
        nargs=2,
        type=float,
        metavar=("WL_MIN", "WL_MAX"),
        help="Restrict comparison to wavelength range (nm).",
    )
    _ = parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="Print top N fractional flux outliers.",
    )
    args = parser.parse_args()

    python_file = Path(cast(str, args.python_file))
    fortran_file = Path(cast(str, args.fortran_file))

    if not python_file.exists():
        print(f"Error: Python spectrum not found: {python_file}")
        sys.exit(1)
    if not fortran_file.exists():
        print(f"Error: Fortran spectrum not found: {fortran_file}")
        sys.exit(1)

    args_range = cast(list[float] | None, args.range)
    wl_range = (float(args_range[0]), float(args_range[1])) if args_range else None
    top_val = int(cast(int, args.top)) if args.top is not None else None
    _ = compare_spectra(
        python_file,
        fortran_file,
        wl_range=wl_range,
        top_n=top_val,
    )
