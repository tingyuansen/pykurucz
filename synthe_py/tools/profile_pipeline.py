#!/usr/bin/env python3
"""Profile the Python synthesis pipeline per-stage and per-function.

Runs the 368-372 nm diagnostic subrange for each atmosphere and produces:
1. Per-stage wall-clock timing table (parsed from logger output)
2. Function-level profiling via cProfile
3. Peak memory snapshot via tracemalloc

Usage:
    python -m synthe_py.tools.profile_pipeline [--atmospheres t05770 t03750 t02500]
    python -m synthe_py.tools.profile_pipeline --atmosphere t05770 --cprofile
    python -m synthe_py.tools.profile_pipeline --all
"""

from __future__ import annotations

import argparse
import cProfile
import io
import logging
import os
import pstats
import re
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from synthe_py.config import SynthesisConfig
from synthe_py.engine.opacity import run_synthesis

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WL_START = 368.0
WL_END = 372.0
RESOLUTION = 300_000.0
ATOMIC_CATALOG = _PROJECT_ROOT / "lines" / "gfallvac.latest"

ATMOSPHERE_CONFIGS: Dict[str, Dict[str, Path]] = {
    "t05770": {
        "atm": _PROJECT_ROOT / "grids" / "at12_aaaaa" / "atm" / "at12_aaaaa_t05770g4.44.atm",
        "npz": _PROJECT_ROOT / "grids" / "at12_aaaaa" / "at12_aaaaa_t05770g4.44.npz",
        "fortran_spec": _PROJECT_ROOT / "grids" / "at12_aaaaa" / "spec" / "at12_aaaaa_t05770g4.44.spec",
    },
    "t03750": {
        "atm": _PROJECT_ROOT / "grids" / "at12_aaaaa" / "at12_aaaaa_t03750g3.50.atm",
        "npz": _PROJECT_ROOT / "grids" / "at12_aaaaa" / "at12_aaaaa_t03750g3.50.npz",
        "fortran_spec": _PROJECT_ROOT / "grids" / "at12_aaaaa" / "spec" / "at12_aaaaa_t03750g3.50.spec",
    },
    "t02500": {
        "atm": _PROJECT_ROOT / "grids" / "at12_aaaaa" / "at12_aaaaa_t02500g-1.0.atm",
        "npz": _PROJECT_ROOT / "grids" / "at12_aaaaa" / "at12_aaaaa_t02500g-1.0.npz",
        "fortran_spec": _PROJECT_ROOT / "grids" / "at12_aaaaa" / "spec" / "at12_aaaaa_t02500g-1.0.spec",
    },
}


# ---------------------------------------------------------------------------
# Timing capture handler
# ---------------------------------------------------------------------------
_TIMING_PATTERN = re.compile(r"Timing:\s+(.+?)\s+in\s+([\d.]+)s")


class TimingCaptureHandler(logging.Handler):
    """Logging handler that captures 'Timing: <stage> in <seconds>s' lines."""

    def __init__(self) -> None:
        super().__init__()
        self.timings: Dict[str, float] = {}

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        m = _TIMING_PATTERN.search(msg)
        if m:
            stage_name = m.group(1).strip()
            seconds = float(m.group(2))
            self.timings[stage_name] = seconds


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ProfileResult:
    """Profiling result for a single atmosphere."""

    atmosphere: str
    stage_timings: Dict[str, float]
    total_time: float
    peak_memory_mb: float
    cprofile_stats: Optional[str] = None  # Top-N functions from cProfile


# ---------------------------------------------------------------------------
# Core profiling function
# ---------------------------------------------------------------------------
def profile_atmosphere(
    atm_key: str,
    do_cprofile: bool = False,
    do_tracemalloc: bool = True,
    n_workers: Optional[int] = None,
) -> ProfileResult:
    """Run synthesis for one atmosphere and capture profiling data."""

    cfg_paths = ATMOSPHERE_CONFIGS[atm_key]
    atm_path = cfg_paths["atm"]
    npz_path = cfg_paths["npz"]

    if not atm_path.exists():
        raise FileNotFoundError(f"Atmosphere file not found: {atm_path}")

    import tempfile

    out_dir = Path(tempfile.mkdtemp(prefix=f"profile_{atm_key}_"))
    spec_path = out_dir / f"{atm_key}.spec"
    diag_path = out_dir / f"{atm_key}_diag.npz"

    cfg = SynthesisConfig.from_cli(
        spec_path=spec_path,
        diagnostics_path=diag_path,
        atmosphere_path=atm_path,
        atomic_catalog=ATOMIC_CATALOG,
        wl_start=WL_START,
        wl_end=WL_END,
        resolution=RESOLUTION,
        npz_path=npz_path if npz_path.exists() else None,
        n_workers=n_workers,
    )
    cfg.log_level = "INFO"

    # Install timing capture handler
    root_logger = logging.getLogger("synthe_py")
    timing_handler = TimingCaptureHandler()
    timing_handler.setLevel(logging.INFO)
    root_logger.addHandler(timing_handler)

    # Start memory tracking
    if do_tracemalloc:
        tracemalloc.start()

    # Run synthesis with optional cProfile
    t0 = time.perf_counter()

    cprofile_text: Optional[str] = None
    if do_cprofile:
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            _result = run_synthesis(cfg)
        finally:
            profiler.disable()
        # Format top 30 functions
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats("cumulative")
        ps.print_stats(30)
        cprofile_text = s.getvalue()

        # Also save .prof file
        prof_path = out_dir / f"{atm_key}.prof"
        profiler.dump_stats(str(prof_path))
        print(f"  cProfile dump saved to {prof_path}")
    else:
        _result = run_synthesis(cfg)

    total_time = time.perf_counter() - t0

    # Capture peak memory
    peak_memory_mb = 0.0
    if do_tracemalloc:
        _current, peak = tracemalloc.get_traced_memory()
        peak_memory_mb = peak / (1024 * 1024)
        tracemalloc.stop()

    # Remove timing handler
    root_logger.removeHandler(timing_handler)

    # Clean up temp files
    import shutil

    shutil.rmtree(out_dir, ignore_errors=True)

    return ProfileResult(
        atmosphere=atm_key,
        stage_timings=dict(timing_handler.timings),
        total_time=total_time,
        peak_memory_mb=peak_memory_mb,
        cprofile_stats=cprofile_text,
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
# Canonical stage order for the timing table
_STAGE_ORDER = [
    "atmosphere load",
    "wavelength grid",
    "line catalog",
    "fort.19 metadata",
    "populations",
    "frequency quantities",
    "continuum",
    "full-grid KAPMIN",
    "hydrogen continuum",
    "populations (line opacity)",
    "TRANSP",
    "ASYNTH",
    "fort.19 add",
    "line opacity stage",
    "radiative transfer",
    "total pipeline",
]


def format_timing_table(results: List[ProfileResult]) -> str:
    """Format per-stage timings as a fixed-width table."""

    # Collect all stage names in order
    all_stages: List[str] = []
    for stage in _STAGE_ORDER:
        for r in results:
            if stage in r.stage_timings:
                if stage not in all_stages:
                    all_stages.append(stage)
                break
    # Add any stages not in canonical order
    for r in results:
        for stage in r.stage_timings:
            if stage not in all_stages:
                all_stages.append(stage)

    # Header
    col_width = 12
    header_cols = [f"{'Stage':<32}"]
    for r in results:
        header_cols.append(f"{r.atmosphere:>{col_width}}")
    header = " | ".join(header_cols)
    sep = "-" * len(header)

    lines = [sep, header, sep]

    for stage in all_stages:
        row_cols = [f"{stage:<32}"]
        for r in results:
            val = r.stage_timings.get(stage)
            if val is not None:
                row_cols.append(f"{val:>{col_width}.3f}")
            else:
                row_cols.append(f"{'—':>{col_width}}")
        lines.append(" | ".join(row_cols))

    # Total and memory rows
    lines.append(sep)
    row_total = [f"{'TOTAL (wall-clock)':<32}"]
    row_mem = [f"{'Peak memory (MB)':<32}"]
    for r in results:
        row_total.append(f"{r.total_time:>{col_width}.3f}")
        row_mem.append(f"{r.peak_memory_mb:>{col_width}.1f}")
    lines.append(" | ".join(row_total))
    lines.append(" | ".join(row_mem))
    lines.append(sep)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile the Python synthesis pipeline per-stage and per-function."
    )
    parser.add_argument(
        "--atmospheres",
        nargs="+",
        choices=list(ATMOSPHERE_CONFIGS.keys()),
        default=None,
        help="Atmosphere(s) to profile. Default: all available.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Profile all 3 atmospheres.",
    )
    parser.add_argument(
        "--cprofile",
        action="store_true",
        help="Enable cProfile function-level profiling (slower, more detail).",
    )
    parser.add_argument(
        "--no-tracemalloc",
        action="store_true",
        help="Disable tracemalloc memory tracking.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto).",
    )
    args = parser.parse_args()

    # Determine atmospheres to profile
    if args.all or args.atmospheres is None:
        atm_keys = [k for k in ATMOSPHERE_CONFIGS if ATMOSPHERE_CONFIGS[k]["atm"].exists()]
    else:
        atm_keys = args.atmospheres

    if not atm_keys:
        print("No atmosphere files found. Exiting.")
        return

    print("=" * 80)
    print("PIPELINE PROFILING REPORT")
    print(f"Wavelength range: {WL_START}-{WL_END} nm, R={RESOLUTION:.0f}")
    print(f"Atmospheres: {', '.join(atm_keys)}")
    print(f"cProfile: {'enabled' if args.cprofile else 'disabled'}")
    print(f"tracemalloc: {'disabled' if args.no_tracemalloc else 'enabled'}")
    print("=" * 80)

    results: List[ProfileResult] = []
    for atm_key in atm_keys:
        print(f"\n{'─'*40}")
        print(f"Profiling {atm_key} ...")
        print(f"{'─'*40}")
        try:
            result = profile_atmosphere(
                atm_key,
                do_cprofile=args.cprofile,
                do_tracemalloc=not args.no_tracemalloc,
                n_workers=args.workers,
            )
            results.append(result)
            print(f"  Total: {result.total_time:.3f}s, Peak memory: {result.peak_memory_mb:.1f} MB")
        except Exception as e:
            print(f"  ERROR profiling {atm_key}: {e}")
            import traceback

            traceback.print_exc()

    if not results:
        print("No successful profiles. Exiting.")
        return

    # Print timing table
    print(f"\n{'='*80}")
    print("PER-STAGE TIMING TABLE (seconds)")
    print(f"{'='*80}")
    print(format_timing_table(results))

    # Print cProfile results if available
    for r in results:
        if r.cprofile_stats:
            print(f"\n{'='*80}")
            print(f"cProfile TOP 30 FUNCTIONS — {r.atmosphere}")
            print(f"{'='*80}")
            print(r.cprofile_stats)


if __name__ == "__main__":
    main()




