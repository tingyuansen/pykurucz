#!/usr/bin/env python3
"""Analyze wavelength regions with consistent Python vs Fortran discrepancies.

Identifies wavelength bands that show elevated fractional error across multiple
atmospheres, indicating potential code bugs. Tracks denominator (|normalized flux|)
per band so you can filter out small-denominator bands later.

Usage:
  python -m synthe_py.tools.analyze_wavelength_errors
  python -m synthe_py.tools.analyze_wavelength_errors --band-width 25 --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

FLOAT_RE = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][+-]?\d+)?")

BASE_DIR = Path(__file__).resolve().parents[2] / "results" / "validation_100"
PYTHON_SPECS = BASE_DIR / "python_specs"
FORTRAN_SPECS = BASE_DIR / "fortran_specs"
CSV_PATH = BASE_DIR / "comparison_metrics_summary.csv"


def load_spectrum(filepath: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load spectrum: wavelength, flux, continuum."""
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


def compute_band_errors(
    band_width_nm: float = 50.0,
    wl_min: float = 300.0,
    wl_max: float = 1800.0,
    limit: int | None = None,
    peak_min_denom: float = 0.05,
    verbose: bool = False,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Compute per-band fractional error for each atmosphere.

    Returns:
        band_mean_err: dict[atmosphere] -> array of mean |fractional_error| per band
        band_peak_max: dict[atmosphere] -> array of max |fractional_error| per band
            (only points with |denom| >= peak_min_denom)
        band_peak_p95: dict[atmosphere] -> array of 95th pct |fractional_error| per band
        atmospheres: list of atmosphere names
        band_centers: center wavelength of each band
        band_median_denom: per-band median |normalized flux| (denominator)
        band_frac_small_denom: per-band fraction of points with |norm| < 0.05
    """
    py_files = sorted(PYTHON_SPECS.glob("*.spec"))
    if limit is not None:
        py_files = py_files[:limit]
    ft_files = {f.name: f for f in FORTRAN_SPECS.glob("*.spec")}

    band_edges = np.arange(wl_min, wl_max + 1e-9, band_width_nm)
    n_bands = len(band_edges) - 1
    band_centers = 0.5 * (band_edges[:-1] + band_edges[1:])

    band_mean_err: dict[str, np.ndarray] = {}
    band_peak_max: dict[str, np.ndarray] = {}
    band_peak_p95: dict[str, np.ndarray] = {}
    atmospheres: list[str] = []
    # Accumulate denominator stats across atmospheres for filtering
    all_denoms_by_band: list[list[float]] = [[] for _ in range(n_bands)]
    all_small_denom_counts: list[int] = [0] * n_bands
    all_point_counts: list[int] = [0] * n_bands

    for py_path in py_files:
        atm = py_path.name
        ft_path = ft_files.get(atm)
        if ft_path is None:
            continue

        py_wl, py_flux, py_cont = load_spectrum(py_path)
        ft_wl, ft_flux, ft_cont = load_spectrum(ft_path)

        py_norm = py_flux / np.maximum(py_cont, 1e-30)
        ft_norm = ft_flux / np.maximum(ft_cont, 1e-30)

        # Interpolate Fortran onto Python grid
        ft_norm_interp = np.interp(py_wl, ft_wl, ft_norm)

        # Fractional error: (Python - Fortran) / Python
        # Use tiny epsilon to avoid div-by-zero; track denominator for filtering
        denom = np.where(np.abs(py_norm) >= 1e-30, py_norm, np.nan)
        frac_err = np.divide(py_norm - ft_norm_interp, denom)
        abs_denom = np.abs(py_norm)

        # Robust mask: exclude small denominators for peak (avoids spurious inflation)
        robust = np.abs(py_norm) >= peak_min_denom

        # Bin by wavelength (vectorized)
        band_idx = np.searchsorted(band_edges[1:], py_wl, side="right")
        band_idx = np.clip(band_idx, 0, n_bands - 1)
        valid = np.isfinite(frac_err)
        abs_err = np.abs(frac_err)
        band_mean_abs_err = np.full(n_bands, np.nan)
        band_max_abs_err = np.full(n_bands, np.nan)
        band_p95_abs_err = np.full(n_bands, np.nan)
        for i in range(n_bands):
            mask = (band_idx == i) & valid
            if np.sum(mask) > 0:
                band_mean_abs_err[i] = np.mean(abs_err[mask])
            # Peak metrics: only over robust points (|denom| >= peak_min_denom)
            robust_mask = (band_idx == i) & valid & robust
            if np.sum(robust_mask) > 0:
                errs = abs_err[robust_mask]
                band_max_abs_err[i] = np.max(errs)
                band_p95_abs_err[i] = np.percentile(errs, 95)
            # Track denominator for filtering
            denom_mask = band_idx == i
            if np.sum(denom_mask) > 0:
                all_denoms_by_band[i].extend(abs_denom[denom_mask].tolist())
                all_small_denom_counts[i] += int(np.sum(abs_denom[denom_mask] < 0.05))
                all_point_counts[i] += int(np.sum(denom_mask))

        band_mean_err[atm] = band_mean_abs_err
        band_peak_max[atm] = band_max_abs_err
        band_peak_p95[atm] = band_p95_abs_err
        atmospheres.append(atm)

        if verbose:
            print(f"  {atm}: processed")

    # Per-band denominator stats (aggregated across atmospheres)
    band_median_denom = np.array(
        [
            np.median(d) if d else np.nan
            for d in all_denoms_by_band
        ]
    )
    band_frac_small_denom = np.array(
        [
            all_small_denom_counts[i] / all_point_counts[i]
            if all_point_counts[i] > 0 else np.nan
            for i in range(n_bands)
        ]
    )

    return (
        band_mean_err,
        band_peak_max,
        band_peak_p95,
        atmospheres,
        band_centers,
        band_median_denom,
        band_frac_small_denom,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find wavelength bands with consistent Python vs Fortran errors."
    )
    parser.add_argument(
        "--band-width",
        type=float,
        default=50.0,
        help="Wavelength band width in nm (default: 50)",
    )
    parser.add_argument(
        "--min-denom",
        type=float,
        default=None,
        help="Optional: filter bands with median |denom| < this (post-hoc filter)",
    )
    parser.add_argument(
        "--wl-min",
        type=float,
        default=300.0,
        help="Lower wavelength bound (nm)",
    )
    parser.add_argument(
        "--wl-max",
        type=float,
        default=1800.0,
        help="Upper wavelength bound (nm)",
    )
    parser.add_argument(
        "--threshold-pct",
        type=float,
        default=0.5,
        help="Fraction of atmospheres that must exceed threshold to flag a band (0-1)",
    )
    parser.add_argument(
        "--error-threshold",
        type=float,
        default=0.01,
        help="Fractional error threshold (0.01 = 1%%) to consider elevated",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top problematic bands to report",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Write per-band summary to CSV",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N atmospheres (for quick testing)",
    )
    parser.add_argument(
        "--peak-min-denom",
        type=float,
        default=0.05,
        help="Min |normalized flux| for peak metrics (exclude small-denom points)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print progress",
    )
    args = parser.parse_args()

    if not PYTHON_SPECS.exists() or not FORTRAN_SPECS.exists():
        print(
            f"Error: Missing {PYTHON_SPECS} or {FORTRAN_SPECS}",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Computing per-band fractional errors (mean + peak metrics)...")
    (
        band_mean_err,
        band_peak_max,
        band_peak_p95,
        atmospheres,
        band_centers,
        band_median_denom,
        band_frac_small_denom,
    ) = compute_band_errors(
        band_width_nm=args.band_width,
        wl_min=args.wl_min,
        wl_max=args.wl_max,
        limit=args.limit,
        peak_min_denom=args.peak_min_denom,
        verbose=args.verbose,
    )

    if not band_mean_err:
        print("No atmospheres to analyze.")
        sys.exit(0)

    n_atm = len(atmospheres)
    n_bands = len(band_centers)

    # Aggregate: mean stats + PEAK stats (max |error| within band, robust points only)
    band_mean_agg = np.zeros(n_bands)
    band_median_mean = np.zeros(n_bands)
    band_peak_max_agg = np.zeros(n_bands)  # max across atmospheres of peak in band
    band_peak_p95_agg = np.zeros(n_bands)
    n_peak_5pct = np.zeros(n_bands)  # atmospheres with peak >= 5%
    n_peak_10pct = np.zeros(n_bands)  # atmospheres with peak >= 10%

    for i in range(n_bands):
        mean_errs = [band_mean_err[atm][i] for atm in atmospheres]
        peak_errs = [band_peak_max[atm][i] for atm in atmospheres]
        p95_errs = [band_peak_p95[atm][i] for atm in atmospheres]
        valid_mean = [e for e in mean_errs if np.isfinite(e)]
        valid_peak = [e for e in peak_errs if np.isfinite(e)]
        if valid_mean:
            band_mean_agg[i] = np.mean(valid_mean)
            band_median_mean[i] = np.median(valid_mean)
        if valid_peak:
            band_peak_max_agg[i] = np.max(valid_peak)
            band_peak_p95_agg[i] = np.percentile(valid_peak, 95)
            n_peak_5pct[i] = sum(1 for e in valid_peak if e >= 0.05)
            n_peak_10pct[i] = sum(1 for e in valid_peak if e >= 0.10)

    # Optional post-hoc filter: exclude bands with small denominators
    if args.min_denom is not None:
        denom_mask = band_median_denom >= args.min_denom
        band_peak_max_agg = np.where(denom_mask, band_peak_max_agg, 0)
        print(f"Filtered bands with median |denom| < {args.min_denom}")

    # Rank by PEAK error (catches localized 10% spikes)
    rank_score = (
        band_peak_max_agg * 100  # primary: max peak in band
        + n_peak_10pct * 5  # secondary: how many atm show 10%+ peak
        + n_peak_5pct
    )
    order = np.argsort(-rank_score)

    print(f"\nAnalyzed {n_atm} atmospheres, {n_bands} bands ({args.band_width} nm)")
    print(f"Peak = max |fractional error| in band (points with |norm| >= {args.peak_min_denom})")
    print(f"Ranking by peak error to catch localized spikes (e.g. 10% in absorption lines)")
    print()

    print("=" * 100)
    print("TOP WAVELENGTH BANDS BY PEAK ERROR (localized spikes)")
    print("=" * 100)
    print(
        f"{'Band (nm)':<20} {'Peak max %':<12} {'N(peak>=10%)':<12} {'N(peak>=5%)':<12} "
        f"{'P95 peak %':<12} {'Mean err %':<10} {'Med denom':<10} {'Frac small':<10}"
    )
    print("-" * 100)

    rows: list[dict] = []
    for rank, idx in enumerate(order[: args.top], start=1):
        wl_lo = band_centers[idx] - args.band_width / 2
        wl_hi = band_centers[idx] + args.band_width / 2
        peak_max = band_peak_max_agg[idx] * 100
        n10 = int(n_peak_10pct[idx])
        n5 = int(n_peak_5pct[idx])
        p95 = band_peak_p95_agg[idx] * 100
        mean = band_mean_agg[idx] * 100
        med_denom = band_median_denom[idx]
        frac_small = band_frac_small_denom[idx]
        band_str = f"{wl_lo:.0f}-{wl_hi:.0f}"
        print(
            f"{band_str:<20} {peak_max:>10.2f}%  {n10:>10}/{n_atm}  {n5:>10}/{n_atm}  "
            f"{p95:>10.2f}%  {mean:>8.3f}%  {med_denom:.4f}  {frac_small:.2%}"
        )
        rows.append({
            "rank": rank,
            "band_nm": band_str,
            "band_center": band_centers[idx],
            "peak_max_pct": peak_max,
            "n_peak_10pct": n10,
            "n_peak_5pct": int(n_peak_5pct[idx]),
            "n_atmospheres": n_atm,
            "peak_p95_pct": p95,
            "mean_err_pct": mean,
            "median_denom": med_denom,
            "frac_small_denom": frac_small,
        })

    # Cross-reference with comparison_metrics_summary: which atmospheres have worst overall?
    if CSV_PATH.exists():
        print("\n" + "=" * 80)
        print("ATMOSPHERES WITH HIGHEST OVERALL NORM_RMS (from comparison_metrics_summary)")
        print("=" * 80)
        with open(CSV_PATH) as f:
            reader = csv.DictReader(f)
            metrics = list(reader)
        by_norm_rms = sorted(metrics, key=lambda r: float(r["norm_rms"]), reverse=True)
        for r in by_norm_rms[:15]:
            atm = r["atmosphere"]
            norm_rms = float(r["norm_rms"])
            flux_rms = float(r["flux_rms_pct"])
            print(f"  {atm}: norm_rms={norm_rms:.6f}  flux_rms={flux_rms:.2f}%")

    if args.csv:
        out_path = Path(args.csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "rank", "band_nm", "band_center", "peak_max_pct",
                    "n_peak_10pct", "n_peak_5pct", "n_atmospheres",
                    "peak_p95_pct", "mean_err_pct",
                    "median_denom", "frac_small_denom",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote band summary to {out_path}")

    # For top 5 bands, list atmospheres with highest PEAK error (for debugging)
    print("\n" + "=" * 100)
    print("ATMOSPHERES WITH HIGHEST PEAK ERROR IN TOP BANDS (for debugging)")
    print("=" * 100)
    for rank, idx in enumerate(order[:5], start=1):
        wl_lo = band_centers[idx] - args.band_width / 2
        wl_hi = band_centers[idx] + args.band_width / 2
        band_str = f"{wl_lo:.0f}-{wl_hi:.0f}"
        peak_atms = [
            (atm, band_peak_max[atm][idx])
            for atm in atmospheres
            if np.isfinite(band_peak_max[atm][idx])
        ]
        peak_atms.sort(key=lambda x: x[1], reverse=True)
        print(f"\nBand {band_str} nm (peak = max |frac err| in band, |denom|>={args.peak_min_denom}):")
        for atm, peak in peak_atms[:15]:
            print(f"  {atm}: peak={peak*100:.2f}%")


if __name__ == "__main__":
    main()
