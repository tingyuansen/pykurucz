"""Configuration models for atlas_py."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class AtlasInput:
    """Input sources for an ATLAS12-like run."""

    atmosphere_path: Path
    control_deck_path: Optional[Path] = None
    molecules_path: Optional[Path] = None
    line_selection_path: Optional[Path] = None
    nlteline_path: Optional[Path] = None
    # Optional line catalog files for Python SELECTLINES (fort.11, fort.111, etc.)
    fort11_path: Optional[Path] = None
    fort111_path: Optional[Path] = None
    fort21_path: Optional[Path] = None
    fort31_path: Optional[Path] = None
    fort41_path: Optional[Path] = None
    fort51_path: Optional[Path] = None
    fort61_path: Optional[Path] = None


@dataclass
class AtlasOutput:
    """Output targets for an ATLAS12-like run."""

    output_atm_path: Path
    diagnostics_path: Optional[Path] = None
    debug_state_path: Optional[Path] = None


@dataclass
class AtlasConfig:
    """Global runtime configuration for atlas_py."""

    inputs: AtlasInput
    outputs: AtlasOutput
    iterations: int = 1
    enable_molecules: bool = False
    enable_convection: bool = True
    enable_scattering: bool = True
    print_level: int = 1
    punch_level: int = 1
    log_level: str = "INFO"
    convergence_epsilon: Optional[float] = None
    convergence_min_iterations: int = 5
    convergence_consecutive: int = 1

