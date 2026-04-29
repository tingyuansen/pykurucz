"""Persistence helpers for cached inputs and outputs."""

from __future__ import annotations

from pathlib import Path

from ..config import SynthesisConfig


def ensure_cache_dirs(cfg: SynthesisConfig) -> None:
    """Create cache and output directories if requested."""

    if cfg.output.spec_path.parent:
        cfg.output.spec_path.parent.mkdir(parents=True, exist_ok=True)
    if cfg.output.diagnostics_path and cfg.output.diagnostics_path.parent:
        cfg.output.diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = cfg.line_data.cache_directory
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)


def cache_path(cfg: SynthesisConfig, name: str) -> Path:
    """Return the cache path for a given artefact name."""

    if not cfg.line_data.cache_directory:
        raise ValueError("Caching requested but no cache_directory configured")
    return cfg.line_data.cache_directory / name
