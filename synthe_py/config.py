"""Configuration models for the Python SYNTHE pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np


def discover_default_molecular_line_directories() -> List[Path]:
    """Return ``data/molecules`` inside the pykurucz repo if it exists.

    The data/ directory is populated by running ``scripts/setup_data.sh`` once.
    Falls back to the sibling ``../kurucz/molecules`` layout for backwards
    compatibility. If neither is present, returns an empty list (atomic-line
    synthesis only).
    """

    pykurucz_root = Path(__file__).resolve().parent.parent
    # Preferred: self-contained data/ tree inside pykurucz
    candidate = (pykurucz_root / "data" / "molecules").resolve()
    if candidate.is_dir() and any(candidate.iterdir()):
        return [candidate]
    # Fallback: sibling kurucz/ repo (legacy layout)
    fallback = (pykurucz_root.parent / "kurucz" / "molecules").resolve()
    if fallback.is_dir():
        return [fallback]
    return []


@dataclass
class WavelengthGrid:
    """Defines the spectral sampling to synthesise."""

    start: float
    end: float
    resolution: float
    velocity_microturb: float = 0.0
    vacuum: bool = True


@dataclass
class LineDataConfig:
    """Controls which line lists are included in the synthesis.

    Note: Populations are computed from Saha-Boltzmann equations (no fort.10 dependency).
    Line opacity is computed from first principles using the atomic catalog.
    """

    atomic_catalog: Path
    molecular_catalogs: List[Path] = field(default_factory=list)
    include_predicted: bool = False
    cache_directory: Optional[Path] = None
    allow_tfort_runtime: bool = False

    # Molecular line opacity (Kurucz rmolecasc / rschwenk / rh2ofast)
    molecular_line_dirs: List[Path] = field(default_factory=list)
    """Directories containing Kurucz ASCII molecular .dat/.asc files."""
    include_tio: bool = True
    """Include Schwenke TiO binary line list (rschwenk) when the binary is found."""
    include_h2o: bool = False
    """Include Partridge-Schwenke H2O binary line list (rh2ofast) when the binary is found.
    Disabled by default: the Fortran reference tfort.12 has all 12.6M H2O records
    with out-of-range NBUFF values, so Fortran synthe.for effectively skips every
    H2O line.  Enable with --h2o if using a corrected H2O compilation."""
    tio_bin_path: Optional[Path] = None
    """Explicit path to schwenke.bin (or eschwenke.bin). Auto-located if None."""
    h2o_bin_path: Optional[Path] = None
    """Explicit path to h2ofastfix.bin. Auto-located if None."""


@dataclass
class AtmosphereInput:
    """Describes the input model atmosphere."""

    model_path: Path
    format: str = "atlas12"  # could support atlas9, phoenix, etc.
    npz_path: Optional[Path] = None  # Optional explicit path to .npz file


@dataclass
class OutputConfig:
    """Specifies the artefacts produced by the pipeline."""

    spec_path: Path
    diagnostics_path: Optional[Path] = None


@dataclass
class SynthesisConfig:
    """Global settings for a SYNTHE run."""

    wavelength_grid: WavelengthGrid
    line_data: LineDataConfig
    atmosphere: AtmosphereInput
    output: OutputConfig
    cutoff: float = 1e-3
    linout: int = 30
    nlte: bool = False
    scattering_iterations: int = 8
    scattering_tolerance: float = 1e-3
    rhoxj_scale: float = 0.0
    log_level: str = "INFO"
    enable_helium_wings: bool = True
    skip_hydrogen_wings: bool = False
    line_filter: bool = True
    wavelength_subsample: int = 1
    n_workers: Optional[int] = (
        None  # Number of parallel workers for radiative transfer (None = auto, 1 = sequential)
    )

    @classmethod
    def from_cli(
        cls,
        spec_path: Path,
        diagnostics_path: Optional[Path],
        atmosphere_path: Path,
        atomic_catalog: Path,
        wl_start: float,
        wl_end: float,
        resolution: float,
        velocity_microturb: float = 0.0,
        vacuum: bool = True,
        cutoff: float = 1e-3,
        linout: int = 30,
        nlte: bool = False,
        scattering_iterations: int = 8,
        scattering_tolerance: float = 1e-3,
        rhoxj_scale: float = 0.0,
        enable_helium_wings: bool = True,
        skip_hydrogen_wings: bool = False,
        line_filter: bool = True,
        wavelength_subsample: int = 1,
        npz_path: Optional[Path] = None,
        n_workers: Optional[int] = None,
        allow_tfort_runtime: bool = False,
        molecular_line_dirs: Optional[List[Path]] = None,
        include_tio: bool = True,
        include_h2o: bool = True,
        tio_bin_path: Optional[Path] = None,
        h2o_bin_path: Optional[Path] = None,
    ) -> "SynthesisConfig":
        """Helper for the default CLI entry point."""

        return cls(
            wavelength_grid=WavelengthGrid(
                start=wl_start,
                end=wl_end,
                resolution=resolution,
                velocity_microturb=velocity_microturb,
                vacuum=vacuum,
            ),
            line_data=LineDataConfig(
                atomic_catalog=atomic_catalog,
                allow_tfort_runtime=allow_tfort_runtime,
                molecular_line_dirs=molecular_line_dirs or [],
                include_tio=include_tio,
                include_h2o=include_h2o,
                tio_bin_path=tio_bin_path,
                h2o_bin_path=h2o_bin_path,
            ),
            atmosphere=AtmosphereInput(model_path=atmosphere_path, npz_path=npz_path),
            output=OutputConfig(spec_path=spec_path, diagnostics_path=diagnostics_path),
            cutoff=cutoff,
            linout=linout,
            nlte=nlte,
            scattering_iterations=scattering_iterations,
            scattering_tolerance=scattering_tolerance,
            rhoxj_scale=rhoxj_scale,
            enable_helium_wings=enable_helium_wings,
            skip_hydrogen_wings=skip_hydrogen_wings,
            line_filter=line_filter,
            wavelength_subsample=wavelength_subsample,
            n_workers=n_workers,
        )


DEFAULT_WAVELENGTH: Tuple[float, float, float] = (300.0, 1800.0, 300_000.0)
"""Default wavelength grid (start, end, resolving power)."""
