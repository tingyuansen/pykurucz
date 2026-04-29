#!/usr/bin/env python3
"""Stage-by-stage comparison harness for Fortran vs Python synthesis pipelines.

Runs both Fortran and Python on the same atmosphere + wavelength range,
collects intermediate dumps at 5 stages, and reports the first stage
where relative error exceeds a threshold.

Usage:
    python -m synthe_py.tools.stage_compare \\
        --atm grids/at12_aaaaa/atm/at12_aaaaa_t05770g4.44.atm \\
        --wl-start 368 --wl-end 372 \\
        --fortran-dir synthe/stmp_at12_aaaaa_t05770 \\
        --threshold 1.0
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STAGES = [
    "npz_conversion",  # Stage 0: convert_atm_to_npz consistency
    "populations",  # XNFPEL, DOPPLE, T, XNE, RHO
    "continuum",  # ACONT, SIGMAC per depth
    "line_centers",  # TRANSP (line opacity at line center)
    "asynth",  # ASYNTH (full line opacity with wings)
    "rt_flux",  # HNU, SURF (JOSH output)
]

# Critical arrays that must match between the NPZ used by the pipeline
# and a freshly-converted reference NPZ from convert_atm_to_npz.py
NPZ_CRITICAL_KEYS = [
    "temperature",
    "electron_density",
    "gas_pressure",
    "mass_density",
    "depth",
    "tkev",
    "hkt",
    "hckt",
    "xnf_h",
    "xnf_he1",
    "xnf_he2",
    "xnf_h2",
    "xnatm",
    "xabund",
    "cont_abs_coeff",
    "cont_scat_coeff",
    "population_per_ion",
    "doppler_per_ion",
    "wledge",
    "frqedg",
    "freqset",
    "half_edge",
    "delta_edge",
]

DIAG_SUBDIR = "stage_diag"


# ---------------------------------------------------------------------------
# Data classes for stage results
# ---------------------------------------------------------------------------
@dataclass
class StageComparison:
    """Result of comparing a single stage between Fortran and Python."""

    stage: str
    max_rel_err: float  # worst-case relative error (%)
    mean_rel_err: float  # mean relative error (%)
    rms_rel_err: float  # RMS relative error (%)
    worst_index: Tuple  # (depth, wavelength_index) of worst element
    worst_py_val: float  # Python value at worst point
    worst_ft_val: float  # Fortran value at worst point
    n_compared: int  # total elements compared
    passed: bool  # True if max_rel_err < threshold


@dataclass
class HarnessResult:
    """Full result of running the comparison harness."""

    atmosphere: str
    wl_start: float
    wl_end: float
    stages: List[StageComparison]
    first_failure: Optional[str] = None
    wall_time_fortran: float = 0.0
    wall_time_python: float = 0.0
    timings: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Array comparison utilities
# ---------------------------------------------------------------------------
def compare_arrays(
    py_arr: np.ndarray,
    ft_arr: np.ndarray,
    name: str,
    threshold_pct: float = 1.0,
    abs_floor: float = 1e-30,
) -> StageComparison:
    """Compare two arrays and compute relative error statistics.

    Parameters
    ----------
    py_arr : array from Python pipeline
    ft_arr : array from Fortran pipeline
    name : stage name
    threshold_pct : pass/fail threshold in percent
    abs_floor : absolute floor below which values are skipped
    """
    # Flatten for comparison
    py_flat = py_arr.ravel()
    ft_flat = ft_arr.ravel()

    # Only compare where Fortran value is significant
    mask = np.abs(ft_flat) > abs_floor
    if not np.any(mask):
        return StageComparison(
            stage=name,
            max_rel_err=0.0,
            mean_rel_err=0.0,
            rms_rel_err=0.0,
            worst_index=(0,),
            worst_py_val=0.0,
            worst_ft_val=0.0,
            n_compared=0,
            passed=True,
        )

    py_masked = py_flat[mask]
    ft_masked = ft_flat[mask]
    rel_err_pct = (
        100.0 * np.abs(py_masked - ft_masked) / np.maximum(np.abs(ft_masked), abs_floor)
    )

    worst_flat_idx = int(np.argmax(rel_err_pct))
    # Map back to original shape
    orig_indices = np.where(mask)[0]
    worst_orig = int(orig_indices[worst_flat_idx])
    worst_nd = np.unravel_index(worst_orig, py_arr.shape)

    return StageComparison(
        stage=name,
        max_rel_err=float(np.max(rel_err_pct)),
        mean_rel_err=float(np.mean(rel_err_pct)),
        rms_rel_err=float(np.sqrt(np.mean(rel_err_pct**2))),
        worst_index=worst_nd,
        worst_py_val=float(py_flat[worst_orig]),
        worst_ft_val=float(ft_flat[worst_orig]),
        n_compared=int(np.sum(mask)),
        passed=bool(float(np.max(rel_err_pct)) < threshold_pct),
    )


# ---------------------------------------------------------------------------
# Stage 0: NPZ conversion validation
# ---------------------------------------------------------------------------
def validate_npz_conversion(
    atm_path: Path,
    kurucz_root: Path,
    threshold_pct: float = 1.0,
) -> List[StageComparison]:
    """Stage 0: Validate that the NPZ used by run_synthesis matches a fresh conversion.

    Re-runs convert_atm_to_npz.py on the .atm file and compares the resulting
    NPZ against the one _find_atmosphere_npz() would select. Any discrepancy
    here propagates silently into every downstream stage.

    Returns a list of StageComparison results (one per critical array).
    """
    import tempfile

    results: List[StageComparison] = []

    # 1. Find the NPZ the pipeline would actually use
    existing_npz = _find_atmosphere_npz(atm_path, kurucz_root)
    if existing_npz is None:
        print("  Stage 0: No existing NPZ found — will be freshly converted")
        return results

    print(f"  Stage 0: Existing NPZ = {existing_npz}")

    # 2. Re-run convert_atm_to_npz.py to produce a fresh reference
    with tempfile.TemporaryDirectory(prefix="stage0_") as tmpdir:
        fresh_npz = Path(tmpdir) / "fresh.npz"

        cmd = [
            sys.executable,
            str(kurucz_root / "synthe_py" / "tools" / "convert_atm_to_npz.py"),
            str(atm_path),
            str(fresh_npz),
        ]
        print(f"  Stage 0: Re-converting .atm → .npz ...")
        conv_result = subprocess.run(
            cmd,
            cwd=str(kurucz_root),
            capture_output=True,
            text=True,
        )

        if conv_result.returncode != 0:
            print(
                f"  Stage 0: convert_atm_to_npz FAILED (exit {conv_result.returncode})"
            )
            print(f"  STDERR: {conv_result.stderr[-1500:]}")
            # Return a single failure entry
            results.append(
                StageComparison(
                    stage="npz_conversion",
                    max_rel_err=float("inf"),
                    mean_rel_err=float("inf"),
                    rms_rel_err=float("inf"),
                    worst_index=(0,),
                    worst_py_val=0.0,
                    worst_ft_val=0.0,
                    n_compared=0,
                    passed=False,
                )
            )
            return results

        if not fresh_npz.exists():
            print(
                "  Stage 0: Fresh NPZ was not created (possible convert_atm_to_npz bug)"
            )
            return results

        # 3. Load both NPZs and compare critical arrays
        try:
            with np.load(existing_npz, allow_pickle=False) as existing_data:
                existing_keys = set(existing_data.files)
                existing_arrays = {k: existing_data[k] for k in existing_keys}
            with np.load(fresh_npz, allow_pickle=False) as fresh_data:
                fresh_keys = set(fresh_data.files)
                fresh_arrays = {k: fresh_data[k] for k in fresh_keys}
        except Exception as e:
            print(f"  Stage 0: Error loading NPZs: {e}")
            return results

        # Report key differences
        only_in_existing = existing_keys - fresh_keys
        only_in_fresh = fresh_keys - existing_keys
        if only_in_existing:
            print(
                f"  Stage 0: Keys only in existing NPZ: {sorted(only_in_existing)[:10]}"
            )
        if only_in_fresh:
            print(f"  Stage 0: Keys only in fresh NPZ: {sorted(only_in_fresh)[:10]}")

        # Compare critical arrays
        any_failure = False
        for key in NPZ_CRITICAL_KEYS:
            if key not in existing_arrays or key not in fresh_arrays:
                if key in existing_arrays:
                    print(f"  Stage 0 WARNING: '{key}' missing in fresh NPZ")
                elif key in fresh_arrays:
                    print(f"  Stage 0 WARNING: '{key}' missing in existing NPZ")
                continue

            ex_arr = np.asarray(existing_arrays[key], dtype=np.float64)
            fr_arr = np.asarray(fresh_arrays[key], dtype=np.float64)

            if ex_arr.shape != fr_arr.shape:
                print(
                    f"  Stage 0 FAIL: '{key}' shape mismatch: existing={ex_arr.shape} vs fresh={fr_arr.shape}"
                )
                results.append(
                    StageComparison(
                        stage=f"npz_{key}",
                        max_rel_err=float("inf"),
                        mean_rel_err=float("inf"),
                        rms_rel_err=float("inf"),
                        worst_index=(0,),
                        worst_py_val=0.0,
                        worst_ft_val=0.0,
                        n_compared=0,
                        passed=False,
                    )
                )
                any_failure = True
                continue

            cmp = compare_arrays(fr_arr, ex_arr, f"npz_{key}", threshold_pct)
            results.append(cmp)
            if not cmp.passed:
                any_failure = True
                print(
                    f"  Stage 0 FAIL: '{key}' max_rel_err={cmp.max_rel_err:.4f}% "
                    f"(threshold={threshold_pct:.1f}%)"
                )
            else:
                print(f"  Stage 0 OK: '{key}' max_rel_err={cmp.max_rel_err:.6f}%")

        if not any_failure:
            print("  Stage 0: All critical NPZ arrays match within threshold")
        else:
            print(
                "  Stage 0: *** NPZ INCONSISTENCY DETECTED — fix convert_atm_to_npz.py before proceeding ***"
            )

    return results


# ---------------------------------------------------------------------------
# Fortran pipeline runner
# ---------------------------------------------------------------------------
def run_fortran_pipeline(
    atm_path: Path,
    fortran_dir: Path,
    wl_start: float,
    wl_end: float,
    kurucz_root: Path,
) -> Dict[str, np.ndarray]:
    """Run the Fortran pipeline and extract intermediate dumps.

    For now, this parses the existing Fortran debug logs and the final
    spectrum to extract intermediate values.  Full binary dumps will be
    added when the Fortran source is instrumented (Phase 2).
    """
    dumps: Dict[str, np.ndarray] = {}

    # --- Stage 1: Populations (from xnfpelsyn debug log or btables.dat) ---
    btables_path = fortran_dir / "btables.dat"
    if btables_path.exists():
        bhyd, bmin = _parse_btables_bhyd(btables_path)
        if bhyd is not None:
            dumps["bhyd"] = bhyd
            dumps["bmin"] = bmin

    # --- Stage 5: Final spectrum ---
    spec_name = atm_path.stem.replace(".atm", "") + ".spec"

    # Prefer validation_100 Fortran specs (from run_validation_100.sh)
    spec_candidates = [
        kurucz_root / "results" / "validation_100" / "fortran_specs" / spec_name,
    ]
    # Prefer subrange-specific Fortran spec files (generated by run_subrange.sh)
    # over the full-range references to ensure an apples-to-apples comparison.
    spec_candidates.append(
        kurucz_root / "grids" / "at12_aaaaa" / "spec_subrange" / spec_name,
    )
    # Fall back to full-range spec directory
    spec_candidates.append(
        kurucz_root / "grids" / "at12_aaaaa" / "spec" / spec_name,
    )
    # Also try using the atm stem directly in any grid subdirectory
    atm_stem = atm_path.stem
    for parent_dir in (kurucz_root / "grids").iterdir():
        if parent_dir.is_dir():
            # subrange first, then full-range
            for subdir in ("spec_subrange", "spec"):
                candidate = parent_dir / subdir / (atm_stem + ".spec")
                if candidate.exists() and candidate not in spec_candidates:
                    spec_candidates.append(candidate)

    for spec_path in spec_candidates:
        if spec_path.exists():
            wl, flux, cont = _load_spectrum(spec_path, wl_start, wl_end)
            if wl is not None and len(wl) > 0:
                dumps["wavelength"] = wl
                dumps["flux"] = flux
                dumps["continuum"] = cont
                print(f"  Using Fortran spec: {spec_path}")
            break

    return dumps


def _parse_btables_bhyd(
    path: Path,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Parse btables.dat for BHYD and BMIN arrays."""
    bhyd_rows = []
    bmin_vals = []
    in_bhyd = False
    try:
        with open(path) as f:
            for line in f:
                if line.startswith("# BHYD"):
                    in_bhyd = True
                    continue
                if line.startswith("# BC1"):
                    in_bhyd = False
                    continue
                if in_bhyd and not line.startswith("#"):
                    parts = line.split()
                    if len(parts) >= 10:
                        # idx, bhyd(1..8), bmin
                        bhyd_rows.append([float(x) for x in parts[1:9]])
                        bmin_vals.append(float(parts[9]))
    except Exception:
        return None, None
    if bhyd_rows:
        return np.array(bhyd_rows), np.array(bmin_vals)
    return None, None


def _load_spectrum(
    path: Path, wl_start: float, wl_end: float
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Load a .spec file and filter to wavelength range."""
    wl_list, flux_list, cont_list = [], [], []
    try:
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        w = float(parts[0])
                        fl = float(parts[1])
                        co = float(parts[2])
                        if wl_start <= w <= wl_end:
                            wl_list.append(w)
                            flux_list.append(fl)
                            cont_list.append(co)
                    except ValueError:
                        continue
    except Exception:
        return None, None, None
    if wl_list:
        return np.array(wl_list), np.array(flux_list), np.array(cont_list)
    return None, None, None


# ---------------------------------------------------------------------------
# Python pipeline runner
# ---------------------------------------------------------------------------
def run_python_pipeline(
    atm_path: Path,
    wl_start: float,
    wl_end: float,
    kurucz_root: Path,
    diag_dir: Path,
    n_workers: Optional[int] = None,
    explicit_npz: Optional[Path] = None,
    python_timeout: Optional[float] = 600.0,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Run the Python synthesis pipeline with diagnostic dumps enabled.

    Returns (dumps_dict, timings_dict).
    """
    diag_dir.mkdir(parents=True, exist_ok=True)
    spec_path = diag_dir / "py_spectrum.spec"
    diag_path = diag_dir / "py_diagnostics.npz"

    # Always start from a clean diagnostic directory to avoid stale-file confusion.
    # Stage files from previous runs can otherwise be mixed with fresh outputs.
    for stale_stage in diag_dir.glob("stage_*.npz"):
        try:
            stale_stage.unlink()
        except OSError:
            pass
    for stale_file in (spec_path, diag_path):
        if stale_file.exists():
            try:
                stale_file.unlink()
            except OSError:
                pass

    # Determine atmosphere NPZ path
    # Prefer explicit NPZ > sibling .npz > data/ directory
    npz_path = (
        explicit_npz
        if explicit_npz is not None
        else _find_atmosphere_npz(atm_path, kurucz_root)
    )
    if npz_path is not None:
        print(f"  Using NPZ: {npz_path}")
    else:
        print("  WARNING: No NPZ found — pipeline will attempt auto-conversion")

    # Default to self-contained Python runtime input.
    # `tfort.*` remains ground-truth validation data, not Python runtime input.
    tfort12 = kurucz_root / "lines" / "gfallvac.latest"

    cmd = [
        sys.executable,
        "-m",
        "synthe_py",
        str(atm_path),
        str(tfort12),
        "--spec",
        str(spec_path),
        "--wl-start",
        str(wl_start),
        "--wl-end",
        str(wl_end),
        "--debug",
    ]
    # Only pass --n-workers if explicitly set; None means auto-detect
    if n_workers is not None:
        cmd.extend(["--n-workers", str(n_workers)])
    if npz_path is not None:
        cmd.extend(["--npz", str(npz_path)])

    env = os.environ.copy()
    # Enable stage dumps via environment variable
    env["SYNTHE_PY_STAGE_DUMPS"] = str(diag_dir)

    t0 = time.perf_counter()
    result = subprocess.run(
        cmd,
        cwd=str(kurucz_root),
        capture_output=True,
        text=True,
        env=env,
        timeout=python_timeout,
    )
    wall_time = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"  Python pipeline FAILED (exit code {result.returncode})")
        print(f"  STDERR: {result.stderr[-2000:]}")
        print(f"  STDOUT (last 2000 chars): {result.stdout[-2000:]}")
        return {}, {"total": wall_time}

    # Parse timing information from stdout
    timings = _parse_python_timings(result.stdout)
    timings["total"] = wall_time

    # Load diagnostics
    dumps: Dict[str, np.ndarray] = {}
    if diag_path.exists():
        with np.load(diag_path, allow_pickle=False) as data:
            for key in data:
                dumps[key] = data[key]

        # Prefer diagnostics RT arrays for stage comparison; they are produced directly
        # by the synthesis pipeline and avoid any spec parsing ambiguities.
        if (
            "wavelength" in dumps
            and "flux_total" in dumps
            and "flux_continuum" in dumps
        ):
            dumps["wavelength"] = dumps["wavelength"]
            dumps["flux"] = dumps["flux_total"]
            dumps["continuum"] = dumps["flux_continuum"]

    # Load spectrum
    if spec_path.exists():
        wl, flux, cont = _load_spectrum(spec_path, wl_start, wl_end)
        if wl is not None:
            # Keep spectrum as fallback if diagnostics flux arrays were not present.
            if "flux" not in dumps:
                dumps["wavelength"] = wl
                dumps["flux"] = flux
                dumps["continuum"] = cont

    # Load stage-specific dumps
    for stage_file in sorted(diag_dir.glob("stage_*.npz")):
        with np.load(stage_file, allow_pickle=False) as data:
            for key in data:
                dumps[f"{stage_file.stem}_{key}"] = data[key]

    return dumps, timings


def _find_atmosphere_npz(atm_path: Path, kurucz_root: Path) -> Optional[Path]:
    """Find the NPZ atmosphere file for a given .atm file."""
    # 1. Sibling .npz
    sibling = atm_path.with_suffix(".npz")
    if sibling.exists():
        return sibling

    # 2. results/validation_100/python_npz (from run_validation_100.sh)
    stem = atm_path.stem  # e.g., at12_aaaaa_t05770g4.44
    val_npz = kurucz_root / "results" / "validation_100" / "python_npz" / (stem + ".npz")
    if val_npz.exists():
        return val_npz

    # 3. data/ directory with various naming conventions
    data_dir = kurucz_root / "synthe_py" / "data"
    stem = atm_path.stem  # e.g., at12_aaaaa_t05770g4.44

    # Try exact match
    exact = data_dir / (stem + "_atmosphere_fixed_interleaved.npz")
    if exact.exists():
        return exact

    # Try base model name
    if "_t" in stem:
        base = stem.split("_t")[0]
        for suffix in [
            "_atmosphere_fixed_interleaved.npz",
            "_atmosphere.npz",
        ]:
            candidate = data_dir / (base + suffix)
            if candidate.exists():
                return candidate

    # Try partial match
    for npz_file in data_dir.glob("*.npz"):
        if "atmosphere" in npz_file.name:
            # Extract temperature from both names
            atm_temp = _extract_temp(stem)
            npz_temp = _extract_temp(npz_file.stem)
            if atm_temp and npz_temp and atm_temp == npz_temp:
                return npz_file

    return None


def _extract_temp(name: str) -> Optional[str]:
    """Extract temperature string like 't05770' from a filename."""
    import re

    m = re.search(r"(t\d{4,5})", name)
    return m.group(1) if m else None


def _parse_python_timings(stdout: str) -> Dict[str, float]:
    """Parse timing information from Python pipeline stdout."""
    timings = {}
    for line in stdout.split("\n"):
        if "Timing:" in line:
            # Format: "Timing: <stage> in <seconds>s"
            parts = line.split("Timing:")
            if len(parts) >= 2:
                rest = parts[1].strip()
                if " in " in rest and "s" in rest:
                    stage_part = rest.split(" in ")[0].strip()
                    time_part = rest.split(" in ")[1].strip().rstrip("s").strip()
                    try:
                        timings[stage_part] = float(time_part)
                    except ValueError:
                        pass
    return timings


def _parse_timeout_value(value: str | float | int | None) -> Optional[float]:
    """Parse timeout values from CLI/programmatic input.

    Accepts seconds as float/int/number-like string. Use 0/none/no/off for no timeout.
    """
    if value is None:
        return None
    if isinstance(value, (float, int)):
        numeric = float(value)
        return None if numeric <= 0.0 else numeric
    token = str(value).strip().lower()
    if token in {"none", "no", "off", "0", "0.0"}:
        return None
    numeric = float(token)
    return None if numeric <= 0.0 else numeric


# ---------------------------------------------------------------------------
# Main comparison logic
# ---------------------------------------------------------------------------
def run_comparison(
    atm_path: Path,
    wl_start: float,
    wl_end: float,
    fortran_dir: Path,
    kurucz_root: Path,
    threshold_pct: float = 1.0,
    n_workers: Optional[int] = None,
    explicit_npz: Optional[Path] = None,
    python_timeout: Optional[float] = 600.0,
) -> HarnessResult:
    """Run the full comparison pipeline."""

    atm_name = atm_path.stem
    print(f"\n{'='*70}")
    print(f"STAGE COMPARISON: {atm_name}")
    print(f"Wavelength range: {wl_start:.1f} - {wl_end:.1f} nm")
    print(f"Threshold: {threshold_pct:.1f}%")
    print(f"{'='*70}\n")

    diag_dir = kurucz_root / DIAG_SUBDIR / atm_name
    diag_dir.mkdir(parents=True, exist_ok=True)

    # --- Stage 0: Validate NPZ conversion ---
    print("Step 0: Validating NPZ conversion consistency...")
    stage0_results = validate_npz_conversion(atm_path, kurucz_root, threshold_pct)

    # --- Run Fortran ---
    print("\nStep 1: Collecting Fortran reference data...")
    t0 = time.perf_counter()
    ft_dumps = run_fortran_pipeline(
        atm_path, fortran_dir, wl_start, wl_end, kurucz_root
    )
    ft_time = time.perf_counter() - t0
    print(f"  Fortran data collected in {ft_time:.2f}s")
    print(f"  Available Fortran dumps: {list(ft_dumps.keys())}")

    # --- Run Python ---
    print("\nStep 2: Running Python pipeline...")
    py_dumps, py_timings = run_python_pipeline(
        atm_path,
        wl_start,
        wl_end,
        kurucz_root,
        diag_dir,
        n_workers,
        explicit_npz=explicit_npz,
        python_timeout=python_timeout,
    )
    print(f"  Python completed in {py_timings.get('total', 0):.2f}s")
    print(f"  Available Python dumps: {list(py_dumps.keys())}")

    # --- Compare stages ---
    print(f"\nStep 3: Comparing stages (threshold = {threshold_pct:.1f}%)...")
    stage_results: List[StageComparison] = list(stage0_results)
    first_failure = None

    # Check Stage 0 for failures
    for sc in stage0_results:
        if not sc.passed and first_failure is None:
            first_failure = sc.stage

    # Stage 5: Final spectrum (always available)
    if "flux" in ft_dumps and "flux" in py_dumps:
        ft_wl = ft_dumps["wavelength"]
        py_wl = py_dumps["wavelength"]

        # Interpolate to common grid
        if len(ft_wl) > 0 and len(py_wl) > 0:
            py_flux_interp = np.interp(ft_wl, py_wl, py_dumps["flux"])
            py_cont_interp = np.interp(ft_wl, py_wl, py_dumps["continuum"])

            flux_cmp = compare_arrays(
                py_flux_interp, ft_dumps["flux"], "rt_flux", threshold_pct
            )
            stage_results.append(flux_cmp)

            cont_cmp = compare_arrays(
                py_cont_interp, ft_dumps["continuum"], "rt_continuum", threshold_pct
            )
            stage_results.append(cont_cmp)

    # --- Report ---
    print(f"\n{'='*70}")
    print(f"RESULTS: {atm_name} ({wl_start:.1f}-{wl_end:.1f} nm)")
    print(f"{'='*70}")
    print(
        f"\n{'Stage':<20} {'Max Rel%':<12} {'Mean Rel%':<12} {'RMS Rel%':<12} {'N compared':<12} {'Status'}"
    )
    print("-" * 80)

    for sc in stage_results:
        status = "PASS" if sc.passed else "FAIL"
        print(
            f"{sc.stage:<20} {sc.max_rel_err:<12.4f} {sc.mean_rel_err:<12.4f} "
            f"{sc.rms_rel_err:<12.4f} {sc.n_compared:<12} {status}"
        )
        if not sc.passed and first_failure is None:
            first_failure = sc.stage
            print(f"  >>> FIRST FAILURE at {sc.stage}")
            print(f"      Worst index: {sc.worst_index}")
            print(f"      Python value: {sc.worst_py_val:.6e}")
            print(f"      Fortran value: {sc.worst_ft_val:.6e}")

    if first_failure is None:
        print(f"\nAll stages PASSED (threshold = {threshold_pct:.1f}%)")
    else:
        print(f"\nFirst failure at stage: {first_failure}")

    # --- Print timing table ---
    if py_timings:
        print(f"\n{'='*70}")
        print("PYTHON PIPELINE TIMING")
        print(f"{'='*70}")
        print(f"{'Stage':<40} {'Time (s)':<12}")
        print("-" * 52)
        for stage_name, t_val in sorted(py_timings.items(), key=lambda x: -x[1]):
            print(f"{stage_name:<40} {t_val:<12.3f}")

    return HarnessResult(
        atmosphere=atm_name,
        wl_start=wl_start,
        wl_end=wl_end,
        stages=stage_results,
        first_failure=first_failure,
        wall_time_fortran=ft_time,
        wall_time_python=py_timings.get("total", 0.0),
        timings=py_timings,
    )


# ---------------------------------------------------------------------------
# Multi-atmosphere runner
# ---------------------------------------------------------------------------
ATMOSPHERE_CONFIGS = {
    "t02500g-1.0": {
        "atm": "samples/at12_aaaaa_t02500g-1.0.atm",
        "npz": "results/validation_100/python_npz/at12_aaaaa_t02500g-1.0.npz",
        "fortran_dir": "synthe",
    },
    "t03200g5.50": {
        "atm": "samples/at12_aaaaa_t03200g5.50.atm",
        "npz": "results/validation_100/python_npz/at12_aaaaa_t03200g5.50.npz",
        "fortran_dir": "synthe",
    },
    "t05770g4.44": {
        "atm": "samples/at12_aaaaa_t05770g4.44.atm",
        "npz": "results/validation_100/python_npz/at12_aaaaa_t05770g4.44.npz",
        "fortran_dir": "synthe",
    },
    "t03750g3.50": {
        "atm": "samples/at12_aaaaa_t03750g3.50.atm",
        "npz": "results/validation_100/python_npz/at12_aaaaa_t03750g3.50.npz",
        "fortran_dir": "synthe",
    },
}


def run_all_atmospheres(
    kurucz_root: Path,
    wl_start: float = 368.0,
    wl_end: float = 372.0,
    threshold_pct: float = 1.0,
    n_workers: Optional[int] = None,
    atmospheres: Optional[List[str]] = None,
    python_timeout: Optional[float] = 600.0,
) -> Dict[str, HarnessResult]:
    """Run comparison for all (or selected) atmospheres."""
    results = {}
    atm_list = atmospheres or list(ATMOSPHERE_CONFIGS.keys())

    for atm_key in atm_list:
        if atm_key not in ATMOSPHERE_CONFIGS:
            print(f"WARNING: Unknown atmosphere '{atm_key}', skipping")
            continue

        cfg = ATMOSPHERE_CONFIGS[atm_key]
        atm_path = kurucz_root / cfg["atm"]
        fortran_dir = kurucz_root / cfg["fortran_dir"]
        npz_path = kurucz_root / cfg["npz"] if "npz" in cfg else None

        if not atm_path.exists():
            print(f"WARNING: Atmosphere file not found: {atm_path}, skipping")
            continue

        result = run_comparison(
            atm_path=atm_path,
            wl_start=wl_start,
            wl_end=wl_end,
            fortran_dir=fortran_dir,
            kurucz_root=kurucz_root,
            threshold_pct=threshold_pct,
            n_workers=n_workers,
            explicit_npz=npz_path,
            python_timeout=python_timeout,
        )
        results[atm_key] = result

    # --- Summary ---
    print(f"\n{'='*70}")
    print("SUMMARY ACROSS ALL ATMOSPHERES")
    print(f"{'='*70}")
    print(
        f"{'Atmosphere':<20} {'Flux RMS%':<12} {'Cont RMS%':<12} {'Status':<10} {'Py Time':<10}"
    )
    print("-" * 65)

    for atm_key, res in results.items():
        flux_rms = next(
            (s.rms_rel_err for s in res.stages if s.stage == "rt_flux"), float("nan")
        )
        cont_rms = next(
            (s.rms_rel_err for s in res.stages if s.stage == "rt_continuum"),
            float("nan"),
        )
        status = "PASS" if res.first_failure is None else f"FAIL@{res.first_failure}"
        print(
            f"{atm_key:<20} {flux_rms:<12.4f} {cont_rms:<12.4f} {status:<10} "
            f"{res.wall_time_python:<10.1f}s"
        )

    # Save results JSON
    results_json = {
        atm_key: {
            "atmosphere": res.atmosphere,
            "wl_start": res.wl_start,
            "wl_end": res.wl_end,
            "first_failure": res.first_failure,
            "wall_time_python": res.wall_time_python,
            "wall_time_fortran": res.wall_time_fortran,
            "stages": [
                {
                    "stage": s.stage,
                    "max_rel_err": s.max_rel_err,
                    "mean_rel_err": s.mean_rel_err,
                    "rms_rel_err": s.rms_rel_err,
                    "n_compared": s.n_compared,
                    "passed": s.passed,
                }
                for s in res.stages
            ],
            "timings": res.timings,
        }
        for atm_key, res in results.items()
    }
    results_path = kurucz_root / DIAG_SUBDIR / "comparison_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Stage-by-stage Fortran vs Python comparison harness"
    )
    parser.add_argument(
        "--atm",
        type=Path,
        default=None,
        help="Path to a single .atm file (overrides --all)",
    )
    parser.add_argument(
        "--fortran-dir",
        type=Path,
        default=None,
        help="Path to Fortran working directory (for single-atm mode)",
    )
    parser.add_argument(
        "--wl-start",
        type=float,
        default=368.0,
        help="Start wavelength in nm (default: 368)",
    )
    parser.add_argument(
        "--wl-end",
        type=float,
        default=372.0,
        help="End wavelength in nm (default: 372)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Pass/fail threshold in percent (default: 1.0)",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Number of workers for Python pipeline (default: auto-detect; use 1 for sequential debugging)",
    )
    parser.add_argument(
        "--validate-npz-only",
        action="store_true",
        help="Run only Stage 0 (NPZ conversion validation) without synthesis",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run comparison for all 3 atmospheres",
    )
    parser.add_argument(
        "--atmospheres",
        nargs="+",
        default=None,
        help="Specific atmospheres to compare (e.g., t05770g4.44 t03750g3.50)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Kurucz project root directory (default: auto-detect)",
    )
    parser.add_argument(
        "--python-timeout",
        type=str,
        default=None,
        help="Python stage timeout in seconds; use 'none' or '0' for no timeout",
    )
    args = parser.parse_args()
    python_timeout = _parse_timeout_value(args.python_timeout)

    # Auto-detect root
    if args.root:
        kurucz_root = args.root.resolve()
    else:
        # Walk up from this file to find project root
        kurucz_root = Path(__file__).resolve().parents[2]

    if not (kurucz_root / "synthe_py").is_dir():
        print(f"ERROR: Cannot find synthe_py/ in {kurucz_root}")
        sys.exit(1)

    print(f"Project root: {kurucz_root}")

    # --- NPZ-only validation mode ---
    if args.validate_npz_only:
        atm_list = args.atmospheres or list(ATMOSPHERE_CONFIGS.keys())
        all_passed = True
        for atm_key in atm_list:
            if atm_key not in ATMOSPHERE_CONFIGS:
                print(f"WARNING: Unknown atmosphere '{atm_key}', skipping")
                continue
            cfg = ATMOSPHERE_CONFIGS[atm_key]
            atm_path = kurucz_root / cfg["atm"]
            if not atm_path.exists():
                print(f"WARNING: {atm_path} not found, skipping")
                continue
            print(f"\n{'='*70}")
            print(f"NPZ VALIDATION: {atm_key}")
            print(f"{'='*70}")
            results = validate_npz_conversion(atm_path, kurucz_root, args.threshold)
            for r in results:
                if not r.passed:
                    all_passed = False
        if all_passed:
            print("\nAll NPZ validations PASSED")
        else:
            print("\nSome NPZ validations FAILED — fix convert_atm_to_npz.py")
        sys.exit(0 if all_passed else 1)

    if args.all or args.atmospheres:
        run_all_atmospheres(
            kurucz_root=kurucz_root,
            wl_start=args.wl_start,
            wl_end=args.wl_end,
            threshold_pct=args.threshold,
            n_workers=args.n_workers,
            atmospheres=args.atmospheres,
            python_timeout=python_timeout,
        )
    elif args.atm:
        if args.fortran_dir is None:
            print("ERROR: --fortran-dir is required when using --atm")
            sys.exit(1)
        run_comparison(
            atm_path=args.atm.resolve(),
            wl_start=args.wl_start,
            wl_end=args.wl_end,
            fortran_dir=args.fortran_dir.resolve(),
            kurucz_root=kurucz_root,
            threshold_pct=args.threshold,
            n_workers=args.n_workers,
            python_timeout=python_timeout,
        )
    else:
        # Default: run all atmospheres
        run_all_atmospheres(
            kurucz_root=kurucz_root,
            wl_start=args.wl_start,
            wl_end=args.wl_end,
            threshold_pct=args.threshold,
            n_workers=args.n_workers,
            python_timeout=python_timeout,
        )


if __name__ == "__main__":
    main()
