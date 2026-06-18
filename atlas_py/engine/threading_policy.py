"""Unified thread-budget policy for pykurucz / atlas_py / synthe_py."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, replace
from typing import Optional

logger = logging.getLogger(__name__)


def estimate_wavelength_grid_size(
    wl_start: float,
    wl_end: float,
    resolution: float,
) -> int:
    """Estimate SYNTHE wavelength point count (matches geometric grid builder)."""
    if resolution <= 0.0:
        raise ValueError("resolution must be > 0")
    ratio = 1.0 + 1.0 / resolution
    rlog = math.log(ratio)
    ix_start = math.log(wl_start) / rlog
    ix_floor = math.floor(ix_start)
    if math.exp(ix_floor * rlog) < wl_start:
        ix_floor += 1
    wbegin = math.exp(ix_floor * rlog)
    count = 0
    wl = wbegin
    while wl <= wl_end * (1.0 + 1e-9):
        count += 1
        wl *= ratio
    return count


@dataclass(frozen=True)
class ThreadingPolicy:
    """Single thread budget applied across ATLAS + SYNTHE stages."""

    n_workers: int
    numba_threads: int
    rt_pool: int
    atlas_freq_pool: int
    convec_fd_pool: int
    linop1_serial: bool
    convec_fd_parallel: bool
    pops_parallel: bool

    @classmethod
    def from_n_workers(
        cls,
        n: Optional[int],
        *,
        linop1_serial: Optional[bool] = None,
        convec_fd_parallel: Optional[bool] = None,
        pops_parallel: Optional[bool] = None,
    ) -> "ThreadingPolicy":
        cpu = os.cpu_count() or 1
        n_eff = cpu if n is None else max(1, int(n))
        serial = n_eff == 1
        return cls(
            n_workers=n_eff,
            numba_threads=n_eff,
            rt_pool=n_eff,
            atlas_freq_pool=n_eff,
            convec_fd_pool=min(4, n_eff),
            linop1_serial=serial if linop1_serial is None else bool(linop1_serial),
            convec_fd_parallel=(not serial) if convec_fd_parallel is None else bool(convec_fd_parallel),
            pops_parallel=False if pops_parallel is None else bool(pops_parallel),
        )

    def adapt_for_grid(self, n_wavelengths: int) -> "ThreadingPolicy":
        """Coarse-vs-fine split for SYNTHE based on wavelength grid size."""
        n = self.n_workers
        if n_wavelengths < 50_000:
            return replace(self, numba_threads=n, rt_pool=1)
        if n_wavelengths < 200_000:
            rt = max(1, n // 4)
            numba = min(4, n)
            return replace(self, numba_threads=numba, rt_pool=rt)
        return replace(self, numba_threads=n, rt_pool=1)

    def env_updates(self) -> dict[str, str]:
        """Env vars for this policy (subprocess inheritance or pre-import setup)."""
        return {
            "NUMBA_NUM_THREADS": str(self.numba_threads),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "ATLAS_LINOP1_SERIAL": "1" if self.linop1_serial else "0",
            "ATLAS_CONVEC_FD_PARALLEL": "1" if self.convec_fd_parallel else "0",
            "ATLAS_POPS_PARALLEL": "1" if self.pops_parallel else "0",
        }

    def apply_to_env(self, env: dict[str, str]) -> None:
        """Merge policy env into *env* (use before spawning subprocesses)."""
        env.update(self.env_updates())

    def apply(self) -> None:
        """Set process env and Numba thread count (call before numba JIT)."""
        self.apply_to_env(os.environ)
        try:
            import numba

            numba.set_num_threads(self.numba_threads)
        except Exception:
            pass

    @classmethod
    def for_synthe_grid(
        cls,
        n_workers: Optional[int],
        *,
        wl_start: float,
        wl_end: float,
        resolution: float,
        linop1_serial: Optional[bool] = None,
        convec_fd_parallel: Optional[bool] = None,
        pops_parallel: Optional[bool] = None,
    ) -> "ThreadingPolicy":
        """Base budget + coarse/fine split for a SYNTHE wavelength window."""
        base = cls.from_n_workers(
            n_workers,
            linop1_serial=linop1_serial,
            convec_fd_parallel=convec_fd_parallel,
            pops_parallel=pops_parallel,
        )
        n_wl = estimate_wavelength_grid_size(wl_start, wl_end, resolution)
        return base.adapt_for_grid(n_wl)

    def banner(self, *, stage: str = "e2e") -> str:
        cpu = os.cpu_count() or 1
        blas = os.environ.get("OPENBLAS_NUM_THREADS", "?")
        return (
            f"[parallelism:{stage}] cpu={cpu} n_workers={self.n_workers} "
            f"numba={self.numba_threads} rt_pool={self.rt_pool} "
            f"atlas_freq_pool={self.atlas_freq_pool} convec_fd_pool={self.convec_fd_pool} "
            f"linop1_serial={self.linop1_serial} convec_fd_parallel={self.convec_fd_parallel} "
            f"pops_parallel={self.pops_parallel} blas={blas}"
        )


def log_threading_banner(
    policy: Optional[ThreadingPolicy] = None,
    *,
    stage: str = "e2e",
    n_workers: Optional[int] = None,
) -> None:
    """Log effective threading (uses policy or reads current env)."""
    if policy is None:
        cpu = os.cpu_count() or 1
        n = n_workers if n_workers is not None else cpu
        policy = ThreadingPolicy.from_n_workers(n)
    msg = policy.banner(stage=stage)
    logger.info(msg)
    print(msg, flush=True)
