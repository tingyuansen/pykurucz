"""Command-line entry point for the Python SYNTHE workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from . import config
from .engine.opacity import run_synthesis
from .io import persist
from .utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Python SYNTHE reimplementation")
    parser.add_argument(
        "model", type=Path, help="Path to the model atmosphere file (.atm or .npz)"
    )
    parser.add_argument("atomic", type=Path, help="Atomic line catalog (e.g. gfallvac)")
    parser.add_argument(
        "--npz",
        type=Path,
        default=None,
        help="Explicit path to .npz atmosphere file (overrides automatic lookup for .atm files)",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("spectrum.spec"),
        help="Destination path for the synthesized spectrum",
    )
    parser.add_argument(
        "--wl-start", type=float, default=300.0, help="Start wavelength (nm)"
    )
    parser.add_argument(
        "--wl-end", type=float, default=1800.0, help="End wavelength (nm)"
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=300_000.0,
        help="Resolving power lambda/dlambda",
    )
    parser.add_argument(
        "--no-vacuum",
        action="store_true",
        help="Treat wavelengths as air instead of vacuum",
    )
    parser.add_argument(
        "--cutoff", type=float, default=1e-3,
        help=(
            "Opacity cutoff factor.  KAPMIN = CUTOFF * CONTINUUM filters out lines and "
            "wing steps whose opacity falls below this fraction of the local continuum.  "
            "Fortran SYNBEG tfort.93 uses CUTOFF=0.001.  Default 0.001 matches Fortran."
        ),
    )
    parser.add_argument(
        "--linout", type=int, default=30, help="Line output control flag"
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Optional directory for cached line data",
    )
    parser.add_argument(
        "--allow-tfort-runtime",
        action="store_true",
        help="Allow using tfort.* files as runtime line input (compatibility/debug mode only).",
    )
    parser.add_argument(
        "--no-molecular-lines",
        action="store_true",
        help=(
            "Disable molecular line opacity: skip auto-discovered data/molecules "
            "and ignore --molecules-dir."
        ),
    )
    parser.add_argument(
        "--molecules-dir",
        type=Path,
        action="append",
        dest="molecules_dirs",
        default=None,
        metavar="DIR",
        help=(
            "Directory containing Kurucz ASCII molecular .dat/.asc files "
            "(e.g. data/molecules/). Can be repeated for multiple directories. "
            "If omitted, data/molecules/ inside the pykurucz repo is used when present "
            "(populated by scripts/setup_data.sh)."
        ),
    )
    parser.set_defaults(include_tio=True, include_h2o=False)
    parser.add_argument(
        "--no-tio",
        dest="include_tio",
        action="store_false",
        help="Exclude Schwenke TiO binary line list (default: include if binary exists).",
    )
    parser.add_argument(
        "--h2o",
        dest="include_h2o",
        action="store_true",
        help=(
            "Include Partridge-Schwenke H2O binary line list. "
            "Disabled by default: the Fortran reference tfort.12 has H2O records "
            "with broken NBUFF values (all out-of-range), so Fortran synthe.for "
            "processes zero H2O lines."
        ),
    )
    parser.add_argument(
        "--no-h2o",
        dest="include_h2o",
        action="store_false",
        help="Exclude Partridge-Schwenke H2O binary line list (this is already the default).",
    )
    parser.add_argument(
        "--tio-bin",
        type=Path,
        default=None,
        metavar="PATH",
        help="Explicit path to Schwenke TiO binary file (auto-located under --molecules-dir if omitted).",
    )
    parser.add_argument(
        "--h2o-bin",
        type=Path,
        default=None,
        metavar="PATH",
        help="Explicit path to Partridge-Schwenke H2O binary file (auto-located under --molecules-dir if omitted).",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help=(
            "Parallel workers for synthesis (metal line accumulation, RT when parallel). "
            "Default: all logical CPUs for maximum throughput; use 1 for sequential debugging."
        ),
    )
    parser.add_argument(
        "--nlte", action="store_true", help="Enable NLTE line source handling"
    )
    parser.add_argument(
        "--scat-iterations",
        type=int,
        default=8,
        help="Maximum scattering iterations per frequency",
    )
    parser.add_argument(
        "--scat-tol",
        type=float,
        default=1e-3,
        help="Relative tolerance for scattering iteration convergence",
    )
    parser.add_argument(
        "--rhoxj",
        type=float,
        default=0.0,
        help="Scattering scale height RHOXJ (cm^-2). Use 0 for LTE core.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Verbosity level",
    )
    parser.add_argument(
        "--diagnostics",
        type=Path,
        default=None,
        help="Optional path for diagnostics output",
    )
    parser.add_argument(
        "--microturb",
        type=float,
        default=0.0,
        help="Microturbulent velocity (km/s)",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    diagnostics_path = args.diagnostics

    resolved_molecular_dirs: List[Path]
    include_tio_eff: bool
    include_h2o_eff: bool
    if args.no_molecular_lines:
        resolved_molecular_dirs = []
        include_tio_eff = False
        include_h2o_eff = False
        if args.molecules_dirs:
            print(
                "synthe_py.cli: --no-molecular-lines set; ignoring --molecules-dir.",
                file=sys.stderr,
            )
    else:
        if args.molecules_dirs:
            resolved_molecular_dirs = list(args.molecules_dirs)
        else:
            resolved_molecular_dirs = config.discover_default_molecular_line_directories()
        include_tio_eff = args.include_tio
        include_h2o_eff = args.include_h2o

    cfg = config.SynthesisConfig.from_cli(
        spec_path=args.spec,
        diagnostics_path=diagnostics_path,
        atmosphere_path=args.model,
        atomic_catalog=args.atomic,
        wl_start=args.wl_start,
        wl_end=args.wl_end,
        resolution=args.resolution,
        velocity_microturb=args.microturb,
        vacuum=not args.no_vacuum,
        cutoff=args.cutoff,
        linout=args.linout,
        nlte=args.nlte,
        scattering_iterations=args.scat_iterations,
        scattering_tolerance=args.scat_tol,
        rhoxj_scale=args.rhoxj,
        npz_path=args.npz,
        n_workers=args.n_workers,
        allow_tfort_runtime=args.allow_tfort_runtime,
        molecular_line_dirs=resolved_molecular_dirs,
        include_tio=include_tio_eff,
        include_h2o=include_h2o_eff,
        tio_bin_path=args.tio_bin,
        h2o_bin_path=args.h2o_bin,
    )
    if args.cache:
        cfg.line_data.cache_directory = args.cache
    cfg.log_level = args.log_level

    configure_logging(cfg.log_level)
    persist.ensure_cache_dirs(cfg)
    run_synthesis(cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover - main guard
    raise SystemExit(main())
