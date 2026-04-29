#!/usr/bin/env python3
"""Optional compatibility writer: emit `tfort.*` from Python-compiled metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

from synthe_py.io.lines import compiler, tfort_write


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write tfort.12/.14/.19/.20/.93 from gfall"
    )
    parser.add_argument("--gfall", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--wlbeg", type=float, required=True)
    parser.add_argument("--wlend", type=float, required=True)
    parser.add_argument("--resolution", type=float, default=300000.0)
    parser.add_argument("--cutoff", type=float, default=1.0e-3)
    args = parser.parse_args()

    compiled = compiler.compile_atomic_catalog(
        catalog_path=args.gfall,
        wlbeg=args.wlbeg,
        wlend=args.wlend,
        resolution=args.resolution,
        line_filter=True,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    tfort_write.write_tfort12(
        args.outdir / "tfort.12",
        nbuff=compiled.nbuff,
        cgf=compiled.cgf,
        nelion=compiled.nelion,
        elo_cm=compiled.elo_cm,
        gamma_rad=compiled.gamma_rad,
        gamma_stark=compiled.gamma_stark,
        gamma_vdw=compiled.gamma_vdw,
    )
    tfort_write.write_tfort14(args.outdir / "tfort.14", compiled.catalog)
    tfort_write.write_tfort19(args.outdir / "tfort.19", compiled.fort19_data)
    tfort_write.write_tfort20(args.outdir / "tfort.20", compiled.catalog)
    tfort_write.write_tfort93(
        args.outdir / "tfort.93",
        wlbeg=args.wlbeg,
        wlend=args.wlend,
        resolution=args.resolution,
        cutoff=args.cutoff,
    )
    print(f"Wrote compatibility tapes to: {args.outdir}")


if __name__ == "__main__":
    main()

