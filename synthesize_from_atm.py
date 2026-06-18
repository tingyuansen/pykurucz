#!/usr/bin/env python
"""Synthesize a spectrum from a Kurucz .atm atmosphere file (Mode A).

This is the Python replacement for run_python_pipeline.sh. It runs the
two-step pipeline: atmosphere preprocessing -> spectrum synthesis.

Usage:
    python synthesize_from_atm.py <atm_file> [--wl-start 300] [--wl-end 1800]

Example:
    python synthesize_from_atm.py samples/at12_aaaaa_t08250g4.00.atm
    python synthesize_from_atm.py samples/at12_aaaaa_t02500g-1.0.atm --wl-start 500 --wl-end 510
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize a spectrum from a .atm file (Mode A)")
    parser.add_argument("atm_file", type=Path, help="Kurucz-format .atm file")
    parser.add_argument("--wl-start", type=float, default=300.0,
                        help="Start wavelength in nm (default: 300)")
    parser.add_argument("--wl-end", type=float, default=1800.0,
                        help="End wavelength in nm (default: 1800)")
    parser.add_argument("--resolution", type=float, default=300_000.0,
                        help="Resolving power (default: 300000)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: results/)")
    parser.add_argument(
        "--no-molecular-lines",
        action="store_true",
        help="Forward to synthe_py.cli: atomic lines only.",
    )
    parser.add_argument(
        "--no-tio",
        action="store_true",
        help="Forward to synthe_py.cli: exclude Schwenke TiO.",
    )
    parser.add_argument(
        "--no-h2o",
        action="store_true",
        help="Forward to synthe_py.cli: exclude Partridge-Schwenke H2O.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    atm_file = args.atm_file.resolve()
    if not atm_file.exists():
        print(f"ERROR: Atmosphere file not found: {atm_file}", file=sys.stderr)
        sys.exit(1)

    line_list = repo_root / "lines" / "gfallvac.latest"
    if not line_list.exists():
        print(f"ERROR: Missing line list: {line_list}", file=sys.stderr)
        sys.exit(1)

    out_root = Path(args.output_dir) if args.output_dir else repo_root / "results"
    npz_dir = out_root / "npz"
    spec_dir = out_root / "spec"
    log_dir = out_root / "logs"
    for d in (npz_dir, spec_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    stem = atm_file.stem
    wl_tag = f"{int(args.wl_start)}_{int(args.wl_end)}"
    npz_out = npz_dir / f"{stem}.npz"
    spec_out = spec_dir / f"{stem}_{wl_tag}.spec"
    log_out = log_dir / f"{stem}_{wl_tag}.log"

    n_workers = os.cpu_count() or 1

    print(f"[1/2] Preprocessing atmosphere: {atm_file.name}")
    cmd_convert = [
        sys.executable, str(repo_root / "synthe_py" / "tools" / "convert_atm_to_npz.py"),
        str(atm_file), str(npz_out),
        "--atlas-tables", str(repo_root / "synthe_py" / "data" / "atlas_tables.npz"),
    ]

    print(f"[2/2] Synthesizing spectrum: {args.wl_start}-{args.wl_end} nm, R={args.resolution:.0f}")
    cmd_synthe = [
        sys.executable, "-m", "synthe_py.cli",
        str(atm_file), str(line_list),
        "--npz", str(npz_out),
        "--spec", str(spec_out),
        "--wl-start", str(args.wl_start),
        "--wl-end", str(args.wl_end),
        "--resolution", str(args.resolution),
        "--n-workers", str(n_workers),
        "--log-level", "INFO",
    ]
    if args.no_molecular_lines:
        cmd_synthe.append("--no-molecular-lines")
    if args.no_tio:
        cmd_synthe.append("--no-tio")
    if args.no_h2o:
        cmd_synthe.append("--no-h2o")

    with open(log_out, "w") as log_f:
        for cmd in (cmd_convert, cmd_synthe):
            result = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT,
                                    cwd=str(repo_root))
            if result.returncode != 0:
                print(f"ERROR: Command failed (exit {result.returncode}). See {log_out}",
                      file=sys.stderr)
                sys.exit(result.returncode)

    print(f"\nDone!")
    print(f"  Spectrum: {spec_out}")
    print(f"  NPZ:     {npz_out}")
    print(f"  Log:     {log_out}")


if __name__ == "__main__":
    main()
