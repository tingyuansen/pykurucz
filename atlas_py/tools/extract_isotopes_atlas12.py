"""Extract ISOTOPE tables from atlas12.for into atlas_py/data NPZ."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from atlas_py.physics.isotopes import load_isotopes_from_atlas12


def _default_atlas12_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "data" / "src" / "atlas12.for"


def _default_output_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "isotopes_atlas12.npz"


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract isotopes table from atlas12.for")
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

    isotope, amassiso_major = load_isotopes_from_atlas12(str(atlas12_path))
    np.savez_compressed(output_path, isotope=isotope, amassiso_major=amassiso_major)
    print(f"Wrote {output_path}")
    print(f"  isotope shape={isotope.shape}, amassiso_major shape={amassiso_major.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
