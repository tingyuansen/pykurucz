"""Command-line interface for atlas_py."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from .config import AtlasConfig, AtlasInput, AtlasOutput
from .engine.driver import run_atlas


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Python ATLAS12 reimplementation (phase 1)")
    parser.add_argument("atm", type=Path, help="Input .atm file")
    parser.add_argument(
        "--output-atm",
        type=Path,
        required=True,
        help="Output .atm path",
    )
    parser.add_argument(
        "--deck",
        type=Path,
        default=None,
        help="ATLAS12 control deck (stdin input file). Sets TEFF, GRAVITY, OPACITY, etc.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of atlas_py outer iterations",
    )
    parser.add_argument(
        "--enable-molecules",
        action="store_true",
        help="Enable molecular POPS path (requires full NMOLEC port)",
    )
    parser.add_argument(
        "--molecules",
        type=Path,
        default=None,
        help="Path to molecules.new / molecules.dat for READMOL",
    )
    parser.add_argument(
        "--line-selection-bin",
        type=Path,
        default=None,
        help=(
            "Path to preselected line binary (Fortran fort.12 from READ LINES pass). "
            "Required for strict line-opacity parity."
        ),
    )
    parser.add_argument(
        "--nlteline-bin",
        type=Path,
        default=None,
        help=(
            "Path to nlteline binary used by XLINOP (Fortran fort.19). "
            "Required when IFOP(17)=1."
        ),
    )
    parser.add_argument("--fort11", type=Path, default=None, help="Kurucz lowlines binary (fort.11)")
    parser.add_argument("--fort111", type=Path, default=None, help="Kurucz lowlines-observed binary (fort.111)")
    parser.add_argument("--fort21", type=Path, default=None, help="Kurucz hilines binary (fort.21)")
    parser.add_argument("--fort31", type=Path, default=None, help="Kurucz diatomics binary (fort.31)")
    parser.add_argument("--fort41", type=Path, default=None, help="Kurucz TIO binary (fort.41)")
    parser.add_argument("--fort51", type=Path, default=None, help="Kurucz H2O binary (fort.51)")
    parser.add_argument("--fort61", type=Path, default=None, help="Kurucz H3+ binary (fort.61)")
    parser.add_argument(
        "--debug-state",
        type=Path,
        default=None,
        help="Optional .npz output path for internal EOS state arrays",
    )
    parser.add_argument(
        "--convergence-epsilon",
        type=float,
        default=None,
        help=(
            "Optional early-stop threshold on max normalized change across "
            "physical atmosphere columns (RHOX,T,P,XNE,ABROSS,VTURB). "
            "Disabled by default."
        ),
    )
    parser.add_argument(
        "--convergence-min-iterations",
        type=int,
        default=5,
        help="Minimum iterations before convergence early stopping can trigger (default: 5)",
    )
    parser.add_argument(
        "--convergence-consecutive",
        type=int,
        default=1,
        help="Consecutive converged iterations required before early stopping (default: 1)",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args(argv)
    cfg = AtlasConfig(
        inputs=AtlasInput(
            atmosphere_path=args.atm,
            control_deck_path=args.deck,
            molecules_path=args.molecules,
            line_selection_path=args.line_selection_bin,
            nlteline_path=args.nlteline_bin,
            fort11_path=args.fort11,
            fort111_path=args.fort111,
            fort21_path=args.fort21,
            fort31_path=args.fort31,
            fort41_path=args.fort41,
            fort51_path=args.fort51,
            fort61_path=args.fort61,
        ),
        outputs=AtlasOutput(
            output_atm_path=args.output_atm,
            debug_state_path=args.debug_state,
        ),
        iterations=args.iterations,
        enable_molecules=args.enable_molecules,
        convergence_epsilon=args.convergence_epsilon,
        convergence_min_iterations=args.convergence_min_iterations,
        convergence_consecutive=args.convergence_consecutive,
    )
    run_atlas(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

