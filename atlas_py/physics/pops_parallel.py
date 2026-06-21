"""Shared helpers for Phase 1b POPS/NELECT depth parallelism."""

from __future__ import annotations

import os


def pops_parallel_enabled() -> bool:
    """Return True only when ATLAS_POPS_PARALLEL is explicitly enabled.

    Default is off: outer ThreadPoolExecutor competes with inner numba prange
    and regressed e2e on gate cases (Phase 1b).
    """
    val = os.environ.get("ATLAS_POPS_PARALLEL", "0").strip().lower()
    return val in ("1", "true", "yes", "on")


def pops_parallel_workers() -> int:
    """Worker count for POPS/NELECT thread pools."""
    raw = os.environ.get("ATLAS_POPS_WORKERS", "8").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 8
    return max(1, min(n, 16))
