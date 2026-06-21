"""Extract hydrogen profile DATA tables from atlas12.for into atlas_py/data NPZ."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from atlas_py.physics.hydrogen_profile import load_hydrogen_profile_tables_from_atlas12


def _default_atlas12_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "data" / "src" / "atlas12.for"


def _default_output_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "hydrogen_profile_atlas12.npz"


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract hydrogen profile tables from atlas12.for")
    parser.add_argument(
        "--atlas12",
        type=Path,
        default=_default_atlas12_path(),
        help="Path to atlas12.for",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_default_output_path(),
        help="Output NPZ path",
    )
    args = parser.parse_args()

    atlas12_path = args.atlas12.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tables = load_hydrogen_profile_tables_from_atlas12(str(atlas12_path))
    np.savez_compressed(
        output_path,
        propbm=tables.propbm,
        c=tables.c,
        d=tables.d,
        pp=tables.pp,
        beta=tables.beta,
        stalph=tables.stalph,
        stwtal=tables.stwtal,
        istal=tables.istal,
        lnghal=tables.lnghal,
        stcomp=tables.stcomp,
        stcpwt=tables.stcpwt,
        lncomp=tables.lncomp,
        cutoff_h2_plus=tables.cutoff_h2_plus,
        cutoff_h2=tables.cutoff_h2,
        asumlyman=tables.asumlyman,
        asum=tables.asum,
        y1wtm=tables.y1wtm,
        xknmtb=tables.xknmtb,
        pf_h2=tables.pf_h2,
    )
    print(f"Wrote {output_path}")
    print(f"  propbm={tables.propbm.shape}, pf_h2={tables.pf_h2.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
