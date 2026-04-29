"""Plot normalized flux vs wavelength for a spectrum.

Usage examples:
  python plot.py
  python plot.py --spec at12_aaaaa_t02500g-1.0_300_1800.spec
  python plot.py --wl-start 350 --wl-end 800 --no-show

Input spectrum should contain at least 3 numeric columns per row:
  wavelength_nm, flux, continuum
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_SPEC = "at12_aaaaa_t02500g-1.0_300_1800.spec"
DEFAULT_WL_START = 300.0
DEFAULT_WL_END = 1800.0
BASE_DIR = Path("results")

FLOAT_RE = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][+-]?\d+)?")


class _HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    """Formatter that keeps newlines and includes default values."""


def load_spectrum(path: Path) -> np.ndarray:
    """Parse 3-column spectrum rows (wavelength, flux, continuum)."""
    rows = []
    with open(path, "r") as f:
        for line in f:
            nums = FLOAT_RE.findall(line)
            if len(nums) >= 3:
                rows.append([float(nums[0]), float(nums[1]), float(nums[2])])
    if not rows:
        raise ValueError(f"No valid 3-column rows found in {path}")
    return np.array(rows, dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot normalized flux vs wavelength.",
        formatter_class=_HelpFormatter,
        epilog=(
            "Examples:\n"
            "  python plot.py\n"
            "  python plot.py --spec at12_aaaaa_t02500g-1.0_300_1800.spec\n"
            "  python plot.py --wl-start 400 --wl-end 700 --save spectrum.png --no-show\n"
        ),
    )
    parser.add_argument(
        "--spec",
        type=str,
        default=DEFAULT_SPEC,
        help="Spectrum filename under results/spec/ or full path.",
    )
    parser.add_argument(
        "--wl-start",
        type=float,
        default=DEFAULT_WL_START,
        help="Lower wavelength bound (nm).",
    )
    parser.add_argument(
        "--wl-end",
        type=float,
        default=DEFAULT_WL_END,
        help="Upper wavelength bound (nm).",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Output PNG path. If omitted, writes to results/plots/<spec>.png.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save plot without displaying.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    spec_path = Path(args.spec)
    if not spec_path.is_absolute():
        spec_path = BASE_DIR / "spec" / args.spec

    data = load_spectrum(spec_path)
    wavelength_all = data[:, 0]
    flux_all = data[:, 1]
    continuum_all = data[:, 2]

    normalized_flux_all = flux_all / continuum_all

    wl_mask = (wavelength_all >= args.wl_start) & (wavelength_all <= args.wl_end)
    wavelength = wavelength_all[wl_mask]
    normalized_flux = normalized_flux_all[wl_mask]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(wavelength, normalized_flux, color="steelblue", linewidth=0.8)
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.6)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized Flux")
    ax.set_title(f"Normalized Spectrum ({args.wl_start:g}–{args.wl_end:g} nm)")
    ax.set_xlim(args.wl_start, args.wl_end)
    ax.grid(True, alpha=0.3)

    if args.save is not None:
        out_png = Path(args.save)
    else:
        stem = spec_path.stem if spec_path.suffix else spec_path.name
        out_png = BASE_DIR / "plots" / f"{stem}.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {out_png}")
    if not args.no_show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
