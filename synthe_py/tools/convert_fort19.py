#!/usr/bin/env python3
"""Convert SYNTHE fort.19 wing records into a NumPy archive."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running directly from the repository checkout.
sys.path.append(str(Path(__file__).resolve().parents[2]))

from synthe_py.io.lines import fort19


def convert_fort19(input_path: Path, output_path: Path) -> None:
    data = fort19.load(input_path)
    np.savez(
        output_path,
        wavelength_vacuum=data.wavelength_vacuum,
        energy_lower=data.energy_lower,
        oscillator_strength=data.oscillator_strength,
        n_lower=data.n_lower,
        n_upper=data.n_upper,
        ion_index=data.ion_index,
        line_type=data.line_type,
        continuum_index=data.continuum_index,
        element_index=data.element_index,
        gamma_rad=data.gamma_rad,
        gamma_stark=data.gamma_stark,
        gamma_vdw=data.gamma_vdw,
        nbuff=data.nbuff,
        limb=data.limb,
        wing_type=np.array([wt.value for wt in data.wing_type], dtype=np.int16),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert fort.19 tape to NPZ")
    parser.add_argument("fort19", type=Path, help="Path to fort.19 binary")
    parser.add_argument("output", type=Path, help="Destination .npz file")
    args = parser.parse_args()
    convert_fort19(args.fort19, args.output)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
