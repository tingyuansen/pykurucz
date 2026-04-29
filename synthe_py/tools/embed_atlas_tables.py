#!/usr/bin/env python3
"""
Embed the extracted ATLAS DATA tables into an atmosphere NPZ archive.

This utility takes the ``atlas_tables.npz`` produced by
``extract_atlas_tables.py`` and injects its arrays into a cached atmosphere
file (e.g. ``at12_aaaaa_atmosphere.npz``).  The enriched archive carries all
constants required by the BFUDGE reconstruction, making the atmosphere
snapshot self-contained.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def embed_atlas_tables(
    atlas_path: Path,
    atmosphere_path: Path,
    *,
    output_path: Path | None = None,
) -> None:
    atlas_data = _load_npz(atlas_path)
    atmosphere_data = _load_npz(atmosphere_path)

    enriched = dict(atmosphere_data)
    enriched["atlas_tables_keys"] = np.array(sorted(atlas_data.keys()), dtype="<U32")

    for key, value in atlas_data.items():
        store_key = f"atlas_{key.lower()}"
        enriched[store_key] = np.asarray(value, dtype=np.float64)

    destination = output_path or atmosphere_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, **enriched)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed atlas_tables.npz into an atmosphere archive",
    )
    parser.add_argument(
        "atlas_tables",
        type=Path,
        help="Path to atlas_tables.npz",
    )
    parser.add_argument(
        "atmosphere",
        type=Path,
        help="Path to the atmosphere .npz file to update",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional destination path; defaults to in-place update",
    )
    args = parser.parse_args()

    embed_atlas_tables(args.atlas_tables, args.atmosphere, output_path=args.output)


if __name__ == "__main__":
    main()


