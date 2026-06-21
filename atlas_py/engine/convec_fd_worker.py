"""Worker for parallel CONVEC FD perturbations (Phase 1a).

Each worker process runs ONE of the four POPS/NMOLEC perturbations from
`_convec_fd_samples` (atlas12.for line 4882+) in isolation, returning the
resulting (edens, rho) arrays to the parent.

Design:
- Worker process is created with `multiprocessing.get_context("spawn").Pool`,
  to avoid forking with numpy/numba state already present.
- Worker persists across all ATLAS iterations to amortize spawn cost (which is
  ~1-2 s per process on macOS with `spawn`).
- Static READMOL tables are copied once per worker at Pool init (Tier 2);
  per-call payloads send only perturbed T/P and state snapshots.
- Per-call payload sends only the minimal state arrays needed by NELECT /
  NMOLEC; other state fields (xnf, xnfp, dopple, ...) are reallocated in
  the worker.

Fortran reference for the perturbations (atlas12.for 4882+):
    +0.1% T:  T(J) -> T(J)*1.001          ; EDENS += pradk*(1+dilut*(1.001^4-1))
    -0.1% T:  T(J) -> T(J)*0.999          ; EDENS += pradk*(1+dilut*(0.999^4-1))
    +0.1% P:  P(J) -> P(J)*1.001          ; EDENS += pradk
    -0.1% P:  P(J) -> P(J)*0.999          ; EDENS += pradk
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


# Module-level marker that Pool initializer was run.
_WORKER_READY = False
# Static READMOL tables copied once at Pool init (Tier 2 worker cache).
_WORKER_MOL_STATIC: dict[str, Any] | None = None


def init_worker(mol_static: dict[str, Any] | None = None) -> None:
    """Worker-side init. Called once per process by Pool initializer.

    Pre-warms imports of the heavy physics modules so the first per-call
    perturbation does not pay first-import cost.  When ``mol_static`` is
    provided, READMOL tables are stored module-wide so each perturbation
    job avoids re-pickling the same molecular metadata.
    """
    global _WORKER_READY, _WORKER_MOL_STATIC
    _WORKER_MOL_STATIC = mol_static
    # Import the heavy modules now so subsequent calls are fast.
    # Importing here avoids paying import cost on the parent's `pool.map` call.
    from ..physics import populations  # noqa: F401
    from ..physics import nelect  # noqa: F401
    from ..physics import nmolec  # noqa: F401
    from ..physics import pfsaha  # noqa: F401

    # Limit BLAS / numexpr thread spawn inside each worker; the parent
    # already drives parallelism via the Pool, so per-worker BLAS threading
    # would oversubscribe the CPU.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("ATLAS_POPS_PARALLEL", "0")

    _WORKER_READY = True


def _mol_fields_from_static() -> dict[str, Any]:
    """Return molecular table fields from the worker-global cache."""
    if _WORKER_MOL_STATIC is None:
        raise RuntimeError("NMOLEC static tables missing in CONVEC FD worker")
    return _WORKER_MOL_STATIC


@dataclass
class PerturbationJob:
    """All data a worker needs to run one FD perturbation.

    All numpy arrays are sent as float64 (or int32 for index arrays). The
    parent uses `.copy()` on every array before queuing to ensure isolation.
    Static READMOL tables live in the worker-global cache when available.
    """

    pert_type: str  # 'TPLUS' | 'TMINUS' | 'PPLUS' | 'PMINUS'
    ifmol: bool
    itemp: int

    # atm fields
    temp_baseline: np.ndarray  # shape (n,)
    # state baseline (deep copies)
    p: np.ndarray
    xne: np.ndarray
    xnatom: np.ndarray
    rho: np.ndarray
    chargesq: np.ndarray
    xabund: np.ndarray  # shape (n, 99)
    wtmole: np.ndarray
    edens: np.ndarray
    xnf_shape1: int = 1006

    # extra fields for FD output
    pradk: np.ndarray = None  # type: ignore[assignment]
    dilut: np.ndarray = None  # type: ignore[assignment]

    # Warm-start from parent (CONVEC IFEDNS=1); populated only when ifmol=True.
    xnsave: Optional[np.ndarray] = None

    # Fallback molecular metadata when worker cache is unavailable (tests).
    nummol: int = 0
    code_mol: Optional[np.ndarray] = None
    equil: Optional[np.ndarray] = None
    locj: Optional[np.ndarray] = None
    kcomps: Optional[np.ndarray] = None
    idequa: Optional[np.ndarray] = None
    nequa: int = 0


def run_perturbation(job: PerturbationJob) -> tuple[np.ndarray, np.ndarray]:
    """Apply one perturbation and return (edens_with_pradk, rho).

    Mirrors `_recompute` + the FD edens math from `_convec_fd_samples`.
    Each worker runs in process isolation; mutating its own _CTX / state
    cannot affect any other perturbation or the parent process.
    """
    from ..physics.populations import pops
    from ..physics.nmolec import (
        set_nmolec_context,
        clear_nmolec_context,
        set_nmolec_ifedns,
    )
    from ..physics.runtime import AtlasRuntimeState

    n = int(job.temp_baseline.size)
    temperature = job.temp_baseline.copy()
    p = job.p.copy()

    # Apply perturbation in-place.
    if job.pert_type == "TPLUS":
        temperature *= 1.001
        edens_scale = 1.001
    elif job.pert_type == "TMINUS":
        temperature *= 0.999
        edens_scale = 0.999
    elif job.pert_type == "PPLUS":
        p *= 1.001
        edens_scale = None  # P perturbation: pradk term has no T^4 scaling
    elif job.pert_type == "PMINUS":
        p *= 0.999
        edens_scale = None
    else:
        raise ValueError(f"Unknown perturbation type {job.pert_type!r}")

    # Build a worker-local AtlasRuntimeState from the snapshot. Allocate
    # xnf / xnfp fresh — NELECT/NMOLEC will fill them.
    state = AtlasRuntimeState(
        p=p,
        xne=job.xne.copy(),
        xnatom=job.xnatom.copy(),
        rho=job.rho.copy(),
        chargesq=job.chargesq.copy(),
        xabund=job.xabund,  # read-only across pops; safe to share
        wtmole=job.wtmole,
        xnf=np.zeros((n, int(job.xnf_shape1)), dtype=np.float64),
        xnfp=np.zeros((n, int(job.xnf_shape1)), dtype=np.float64),
        edens=job.edens.copy(),
    )

    # Compute tk_erg from the perturbed temperature (Fortran TK = 1.38054e-16*T).
    tk_erg = temperature * 1.38054e-16
    tlog = np.log(np.maximum(temperature, 1e-300))

    if job.ifmol:
        if _WORKER_MOL_STATIC is not None:
            mol = _mol_fields_from_static()
        else:
            mol = {
                "nummol": int(job.nummol),
                "code_mol": job.code_mol,
                "equil": job.equil,
                "locj": job.locj,
                "kcomps": job.kcomps,
                "idequa": job.idequa,
                "nequa": int(job.nequa),
            }
        # Re-establish NMOLEC context for this worker. set_nmolec_context
        # zeroes xnmol/xnfpmol/xnz/xnsave, so we reseed xnsave from the
        # parent's baseline before flipping IFEDNS=1 (CONVEC FD path).
        set_nmolec_context(
            temperature_k=temperature,
            tk_erg=tk_erg,
            tlog=tlog,
            gas_pressure=p,
            state=state,
            nummol=int(mol["nummol"]),
            code_mol=mol["code_mol"],
            equil=mol["equil"],
            locj=mol["locj"],
            kcomps=mol["kcomps"],
            idequa=mol["idequa"],
            nequa=int(mol["nequa"]),
        )
        if job.xnsave is not None:
            from ..physics import nmolec as _nm

            if _nm._CTX is not None and job.xnsave.shape == _nm._CTX.xnsave.shape:
                _nm._CTX.xnsave[:] = job.xnsave
        set_nmolec_ifedns(1)
    else:
        clear_nmolec_context()

    dummy = np.zeros((n, 1), dtype=np.float64)
    itemp_cache: Dict[str, int] = {}
    pops(
        code=0.0,
        mode=1,
        out=dummy,
        ifmol=bool(job.ifmol),
        ifpres=True,
        temperature_k=temperature,
        tk_erg=tk_erg,
        state=state,
        itemp=int(job.itemp),
        itemp_cache=itemp_cache,
    )

    pradk = np.asarray(job.pradk, dtype=np.float64)
    rho_safe = np.maximum(state.rho, 1e-300)
    if edens_scale is not None:
        # T perturbation
        dilut = np.asarray(job.dilut, dtype=np.float64)
        edens_out = state.edens + 3.0 * pradk / rho_safe * (1.0 + dilut * (edens_scale**4 - 1.0))
    else:
        # P perturbation — Fortran adds 3*pradk/rho with no T^4 dilution term.
        edens_out = state.edens + 3.0 * pradk / rho_safe

    return edens_out.copy(), state.rho.copy()


def convec_fd_parallel_enabled(teff: float) -> bool:
    """Whether to run the four FD perturbations in a spawn Pool.

    Default ``auto``: enabled for all Teff when IFCONV=1 (parity-validated on
    t04000 and t08250; ~3.9x FD speedup on cool stars). Set
    ``ATLAS_CONVEC_FD_PARALLEL=0`` to force serial FD.
    Override with ATLAS_CONVEC_FD_PARALLEL=0|1|auto.
    """
    val = os.environ.get("ATLAS_CONVEC_FD_PARALLEL", "auto").strip().lower()
    if val in ("0", "false", "no", "off"):
        return False
    if val in ("1", "true", "yes", "on"):
        return True
    # auto: always parallel (legacy Teff>=6000 gate removed after t04000 parity)
    return True


def build_worker_mol_static(*, ifmol: bool) -> dict[str, Any] | None:
    """Snapshot static READMOL tables for worker Pool initialization."""
    if not ifmol:
        return None
    from ..physics import nmolec as _nm

    ctx = _nm._CTX
    if ctx is None:
        raise RuntimeError("NMOLEC context required for parallel CONVEC FD (IFMOL=1)")
    return {
        "nummol": int(ctx.nummol),
        "code_mol": np.asarray(ctx.code_mol, dtype=np.float64).copy(),
        "equil": np.asarray(ctx.equil, dtype=np.float64).copy(),
        "locj": np.asarray(ctx.locj, dtype=np.int32).copy(),
        "kcomps": np.asarray(ctx.kcomps, dtype=np.int32).copy(),
        "idequa": np.asarray(ctx.idequa, dtype=np.int32).copy(),
        "nequa": int(ctx.nequa),
    }


def _build_perturbation_jobs(
    *,
    temp0: np.ndarray,
    state,
    pradk: np.ndarray,
    dilut: np.ndarray,
    ifmol: bool,
    itemp_seed: int,
) -> list[PerturbationJob]:
    """Build the four FD jobs from baseline atmosphere/state snapshots."""
    n = int(temp0.size)
    xnsave = None
    if ifmol:
        from ..physics import nmolec as _nm

        if _nm._CTX is None:
            raise RuntimeError("NMOLEC context required for parallel CONVEC FD (IFMOL=1)")
        xnsave = _nm.save_nmolec_xnsave()
        if xnsave is not None:
            xnsave = np.asarray(xnsave, dtype=np.float64).copy()

    base = dict(
        ifmol=ifmol,
        temp_baseline=temp0.copy(),
        p=state.p.copy(),
        xne=state.xne.copy(),
        xnatom=state.xnatom.copy(),
        rho=state.rho.copy(),
        chargesq=state.chargesq.copy(),
        xabund=np.asarray(state.xabund, dtype=np.float64).copy(),
        wtmole=np.asarray(state.wtmole, dtype=np.float64).copy(),
        edens=state.edens.copy(),
        xnf_shape1=int(state.xnf.shape[1]) if state.xnf is not None else 1006,
        pradk=np.asarray(pradk, dtype=np.float64).copy(),
        dilut=np.asarray(dilut, dtype=np.float64).copy(),
        xnsave=xnsave,
    )
    return [
        PerturbationJob(pert_type="TPLUS", itemp=itemp_seed + 1, **base),
        PerturbationJob(pert_type="TMINUS", itemp=itemp_seed + 2, **base),
        PerturbationJob(pert_type="PPLUS", itemp=itemp_seed + 3, **base),
        PerturbationJob(pert_type="PMINUS", itemp=itemp_seed + 4, **base),
    ]


def run_convec_fd_parallel(
    *,
    temp0: np.ndarray,
    state,
    pradk: np.ndarray,
    tauros: np.ndarray,
    ifmol: bool,
    itemp_seed: int,
    pool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run four FD perturbations via an existing spawn Pool."""
    dilut = 1.0 - np.exp(-np.asarray(tauros, dtype=np.float64))
    jobs = _build_perturbation_jobs(
        temp0=temp0,
        state=state,
        pradk=pradk,
        dilut=dilut,
        ifmol=ifmol,
        itemp_seed=itemp_seed,
    )
    results = pool.map(run_perturbation, jobs)
    (ed1, r1), (ed2, r2), (ed3, r3), (ed4, r4) = results
    return ed1, ed2, ed3, ed4, r1, r2, r3, r4
