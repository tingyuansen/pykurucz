"""Top-level atlas_py execution driver."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)

import numpy as np
from ..config import AtlasConfig
from ..io.atmosphere import AtlasAtmosphere, load_atm, write_atm
from ..io.molecules import find_default_molecules_file, readmol_atlas12
from ..io.readin import AtlasDeck, parse_readin_deck
from ..physics.convec import convec, high_from_rhox
from ..physics.doppler import update_doppler_populations
from ..physics.energy_density import compute_atomic_energy_density
from ..physics.isotopes import load_isotopes_from_atlas12
from ..physics.josh import josh_depth_profiles
from ..physics.kapcont import kapcont_baseline, kapcont_table
from ..physics.hydrogen_wings import compute_hydrogen_wings
from ..physics.line_opacity import linop1, xlinop
from ..physics.line_selection import read_nlteline_records, read_selected_lines
from ..physics.selectlines import selectlines as run_selectlines
from ..physics.kapp import KappAtmosphereAdapter
from ..physics.atlas_tables import load_atlas_tables
from ..physics.nmolec import (
    clear_nmolec_context, get_nmolec_snapshot, set_nmolec_context,
    compute_nmolec_edens, set_nmolec_ifedns, save_nmolec_xnsave,
    restore_nmolec_xnsave,
)
from ..physics.popsall import popsall
from ..physics.populations import pops
from ..physics.radiap import init_radiap, radiap_accumulate
from ..physics.ross import ross_step
from ..physics.runtime import AtlasRuntimeState
from ..physics.tcorr import init_tcorr, rosstab_ingest, tcorr_step
from ..physics.josh_math import _map1 as _josh_map1
from ..physics.turb import turb as compute_turb, vturbstandard
from .hydrostatic import integrate_hydrostatic_pressure


def _build_xabund_layers(atm: AtlasAtmosphere) -> np.ndarray:
    """Construct Fortran-like XABUND(J,IZ) from parsed abundance metadata."""

    layers = atm.layers
    xab = np.full((layers, 99), 1e-30, dtype=np.float64)
    # Basic default close to H/He dominated composition.
    xab[:, 0] = 0.92
    xab[:, 1] = 0.08
    for z, val in atm.abundances.items():
        if z < 1 or z > 99:
            continue
        if z <= 2:
            x = float(val)
        else:
            x = 10.0 ** float(val)
        xab[:, z - 1] = max(x, 1e-30)
    return xab


def _mean_molecular_weight_amu(xab: np.ndarray) -> np.ndarray:
    """WTMOLE(J) = Σ XABUND(J,IZ)*ATMASS(IZ) — atlas12.for lines 2218-2220."""

    # Exact Fortran ATMASS table (atlas12.for DATA ATMASS, line 1652-1662).
    mass = np.array([
         1.008,  4.003,  6.939,  9.013, 10.81,  12.01,  14.01,  16.00,
        19.00,  20.18,  22.99,  24.31,  26.98,  28.09,  30.98,  32.07,
        35.45,  39.95,  39.10,  40.08,  44.96,  47.90,  50.94,  52.00,
        54.94,  55.85,  58.94,  58.71,  63.55,  65.37,  69.72,  72.60,
        74.92,  78.96,  79.91,  83.80,  85.48,  87.63,  88.91,  91.22,
        92.91,  95.95,  99.00, 101.1,  102.9,  106.4,  107.9,  112.4,
       114.8,  118.7,  121.8,  127.6,  126.9,  131.3,  132.9,  137.4,
       138.9,  140.1,  140.9,  144.3,  147.0,  150.4,  152.0,  157.3,
       158.9,  162.5,  164.9,  167.3,  168.9,  173.0,  175.0,  178.5,
       181.0,  183.9,  186.3,  190.2,  192.2,  195.1,  197.0,  200.6,
       204.4,  207.2,  209.0,  210.0,  211.0,  222.0,  223.0,  226.1,
       227.1,  232.0,  231.0,  238.0,  237.0,  244.0,  243.0,  247.0,
       247.0,  251.0,  254.0,
    ], dtype=np.float64)
    return np.sum(xab * mass[None, :], axis=1)


def _runtime_from_atm(atm: AtlasAtmosphere) -> AtlasRuntimeState:
    layers = atm.layers
    xab = _build_xabund_layers(atm)
    wtmole = _mean_molecular_weight_amu(xab)
    _, amassiso_major = load_isotopes_from_atlas12()
    tk = atm.tk
    xntot = atm.gas_pressure / np.maximum(tk, 1e-300)
    xnatom = xntot - atm.electron_density
    rho = xnatom * wtmole * 1.660e-24
    return AtlasRuntimeState(
        p=atm.gas_pressure.copy(),
        xne=atm.electron_density.copy(),
        xnatom=xnatom,
        rho=rho,
        chargesq=np.maximum(2.0 * atm.electron_density, 1e-30),
        xabund=xab,
        wtmole=wtmole,
        xnf=np.zeros((layers, 1006), dtype=np.float64),
        xnfp=np.zeros((layers, 1006), dtype=np.float64),
        edens=np.zeros(layers, dtype=np.float64),
        amassiso_major=np.asarray(amassiso_major, dtype=np.float64),
        # Fortran: DATA BHYD,BMIN/kw*1.,…,kw*1./ (atlas12.for line 1703).
        bhyd=np.ones((layers, 6), dtype=np.float64),
        bmin=np.ones(layers, dtype=np.float64),
    )


def _gravity_from_atm_metadata(atm: AtlasAtmosphere) -> float:
    grav = float(atm.metadata.get("grav", "4.44"))
    # .atm stores log10(g), Fortran GRAV is cgs acceleration.
    return 10.0**grav


def _parse_pradk0_from_metadata(atm: AtlasAtmosphere) -> float:
    raw = atm.metadata.get("pradk_line", "")
    m = re.search(r"[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?", raw)
    if m is None:
        return 0.0
    return float(m.group(0).replace("D", "E").replace("d", "e"))


def _parse_ifop_flags(atm: AtlasAtmosphere) -> list[int]:
    raw = atm.metadata.get("ifop", "")
    vals = [int(x) for x in re.findall(r"-?\d+", raw)]
    if len(vals) >= 20:
        return vals[-20:]
    # Default used in .atm writer: IFOP(15)=1, IFOP(17)=1.
    return [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0]


def _require_line_opacity_inputs(*, cfg: AtlasConfig, ifop: list[int]) -> None:
    if_line = ifop[14] == 1
    if_xline = ifop[16] == 1
    if not (if_line or if_xline):
        return
    # If line_selection_path is provided, use it directly.
    if cfg.inputs.line_selection_path is not None:
        if if_xline and cfg.inputs.nlteline_path is None:
            raise NotImplementedError(
                "IFOP(17)=1 requires nlteline binary input (fort.19). "
                "Pass --nlteline-bin with the atlas12 nlteline file."
            )
        return
    # If any line catalog is provided, Python SELECTLINES can generate fort.12.
    has_catalogs = any(
        getattr(cfg.inputs, attr) is not None
        for attr in ("fort11_path", "fort111_path", "fort21_path",
                     "fort31_path", "fort41_path", "fort51_path", "fort61_path")
    )
    if has_catalogs:
        if if_xline and cfg.inputs.nlteline_path is None:
            raise NotImplementedError(
                "IFOP(17)=1 requires nlteline binary input (fort.19). "
                "Pass --nlteline-bin with the atlas12 nlteline file."
            )
        return
    raise NotImplementedError(
        "Strict ATLAS12 RT parity requires preselected line data (fort.12). "
        "Either pass --line-selection-bin (precomputed fort.12) or supply "
        "line catalog files (--fort11, --fort111, --fort21, etc.) for "
        "Python SELECTLINES."
    )


def _build_kapp_adapter(
    atm: AtlasAtmosphere,
    state: AtlasRuntimeState,
) -> KappAtmosphereAdapter:
    """Build continuum-opacity adapter from runtime state arrays.

    Every field maps to a Fortran COMMON array that is always present.
    """
    n = atm.layers
    xnfph = np.column_stack([state.xnfp[:, 0], state.xnfp[:, 1]])
    # Fortran XNFP is always (kw, mion=1006), so columns 845/847 always exist.
    xnfpch = np.asarray(state.xnfp[:, 845], dtype=np.float64)
    xnfpoh = np.asarray(state.xnfp[:, 847], dtype=np.float64)
    return KappAtmosphereAdapter(
        temperature=np.asarray(atm.temperature, dtype=np.float64),
        mass_density=np.asarray(state.rho, dtype=np.float64),
        electron_density=np.asarray(state.xne, dtype=np.float64),
        gas_pressure=np.asarray(state.p, dtype=np.float64),
        xnfph=np.asarray(xnfph, dtype=np.float64),
        xnf_h=np.asarray(state.xnf[:, 0], dtype=np.float64),
        xnf_h_ionized=np.asarray(state.xnf[:, 1], dtype=np.float64),
        xnf_he1=np.asarray(state.xnfp[:, 2], dtype=np.float64),
        xnf_he2=np.asarray(state.xnfp[:, 3], dtype=np.float64),
        xabund=np.asarray(state.xabund, dtype=np.float64),
        bhyd=np.asarray(state.bhyd, dtype=np.float64),
        turbulent_velocity=np.asarray(atm.vturb, dtype=np.float64),
        xnf_all=np.asarray(state.xnf, dtype=np.float64),
        xnfp_all=np.asarray(state.xnfp, dtype=np.float64),
        xnfpch=xnfpch,
        xnfpoh=xnfpoh,
    )


def _max_normalized_column_delta(
    before: np.ndarray,
    after: np.ndarray,
    *,
    floor: float = 1.0e-300,
    symmetric: bool = False,
) -> float:
    """Max layer-wise normalized change for convergence diagnostics."""
    before_arr = np.asarray(before, dtype=np.float64)
    after_arr = np.asarray(after, dtype=np.float64)
    if before_arr.shape != after_arr.shape or before_arr.size == 0:
        return float("nan")
    if symmetric:
        denom = np.maximum.reduce(
            [np.abs(before_arr), np.abs(after_arr), np.full(before_arr.shape, floor)]
        )
    else:
        denom = np.maximum(np.abs(before_arr), floor)
    return float(np.max(np.abs(after_arr - before_arr) / denom))


def _convec_fd_samples(
    *,
    atm: AtlasAtmosphere,
    state: AtlasRuntimeState,
    pradk: np.ndarray,
    tauros: np.ndarray,
    ifmol: bool,
    itemp_seed: int,
    itemp_cache: Dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute CONVEC finite-difference EDENS/RHO samples (Fortran lines 4882+)."""
    n = atm.layers
    temp0 = atm.temperature.copy()
    p0 = state.p.copy()
    xne0 = state.xne.copy()
    xnatom0 = state.xnatom.copy()
    rho0 = state.rho.copy()
    xnf0 = state.xnf.copy()
    xnfp0 = state.xnfp.copy()
    edens0 = state.edens.copy()
    cache0 = dict(itemp_cache)
    dilut = 1.0 - np.exp(-np.asarray(tauros, dtype=np.float64))
    dummy = np.zeros((n, 1), dtype=np.float64)
    nmolec_xnsave0 = None
    convec_log_path = os.getenv("ATLAS_CONVEC_FD_LOG")

    if ifmol:
        # Fortran CONVEC sets IFEDNS=1 before POPS calls (atlas12.for line 4882),
        # and NMOLEC then warm-starts from XNSAVE (line 4190+). Mirror that mode.
        nmolec_xnsave0 = save_nmolec_xnsave()
        set_nmolec_ifedns(1)

    def _recompute(itemp: int) -> None:
        pops(
            code=0.0,
            mode=1,
            out=dummy,
            ifmol=ifmol,
            ifpres=True,
            temperature_k=atm.temperature,
            tk_erg=atm.tk,
            state=state,
            itemp=itemp,
            itemp_cache=itemp_cache,
        )
        # Fortran CONVEC FD (atlas12.for 4882+) calls POPS(0,1,...) only.
        # - IFMOL=0: POPS->NELECT already computes EDENS with atomic ionization terms.
        #   Do not overwrite it with a simplified expression.
        # - IFMOL=1: POPS routes through NMOLEC; recompute molecular EDENS terms
        #   through the NMOLEC-equivalent helper.
        if ifmol:
            state.edens[:] = compute_nmolec_edens(
                temperature_k=atm.temperature,
                tk_erg=atm.tk,
                gas_pressure=state.p,
                state=state,
            )

    def _log_fd(label: str) -> None:
        if not convec_log_path:
            return
        with Path(convec_log_path).open("a", encoding="utf-8") as fh:
            for j in range(37, min(42, n - 1) + 1):
                fh.write(
                    f"CFD,{label},{j + 1:d},{state.edens[j]:.8e},{state.rho[j]:.8e},"
                    f"{atm.temperature[j]:.8e},{state.p[j]:.8e},{state.xne[j]:.8e}\n"
                )

    def _set_nmolec_label(label: str) -> str | None:
        prev = os.getenv("ATLAS_NMOLEC_EDENS_LABEL")
        if ifmol and os.getenv("ATLAS_NMOLEC_EDENS_LOG"):
            os.environ["ATLAS_NMOLEC_EDENS_LABEL"] = label
        return prev

    def _restore_nmolec_label(prev: str | None) -> None:
        if ifmol and os.getenv("ATLAS_NMOLEC_EDENS_LOG"):
            if prev is None:
                os.environ.pop("ATLAS_NMOLEC_EDENS_LABEL", None)
            else:
                os.environ["ATLAS_NMOLEC_EDENS_LABEL"] = prev

    try:
        # +0.1% T
        atm.temperature[:] = temp0 * 1.001
        prev_label = _set_nmolec_label("TPLUS")
        _recompute(itemp_seed + 1)
        _log_fd("TPLUS")
        _restore_nmolec_label(prev_label)
        edens1 = state.edens + 3.0 * pradk / np.maximum(state.rho, 1e-300) * (1.0 + dilut * (1.001**4 - 1.0))
        rho1 = state.rho.copy()

        # -0.1% T
        atm.temperature[:] = temp0 * 0.999
        prev_label = _set_nmolec_label("TMINUS")
        _recompute(itemp_seed + 2)
        _log_fd("TMINU")
        _restore_nmolec_label(prev_label)
        edens2 = state.edens + 3.0 * pradk / np.maximum(state.rho, 1e-300) * (1.0 + dilut * (0.999**4 - 1.0))
        rho2 = state.rho.copy()

        # restore T and perturb P +0.1%
        atm.temperature[:] = temp0
        state.p[:] = p0 * 1.001
        prev_label = _set_nmolec_label("PPLUS")
        _recompute(itemp_seed + 3)
        _log_fd("PPLUS")
        _restore_nmolec_label(prev_label)
        edens3 = state.edens + 3.0 * pradk / np.maximum(state.rho, 1e-300)
        rho3 = state.rho.copy()

        # perturb P -0.1%
        state.p[:] = p0 * 0.999
        prev_label = _set_nmolec_label("PMINUS")
        _recompute(itemp_seed + 4)
        _log_fd("PMINU")
        _restore_nmolec_label(prev_label)
        edens4 = state.edens + 3.0 * pradk / np.maximum(state.rho, 1e-300)
        rho4 = state.rho.copy()
    finally:
        # Restore baseline state exactly.
        atm.temperature[:] = temp0
        state.p[:] = p0
        state.xne[:] = xne0
        state.xnatom[:] = xnatom0
        state.rho[:] = rho0
        state.xnf[:] = xnf0
        state.xnfp[:] = xnfp0
        if np.any(edens0 != 0.0):
            state.edens[:] = edens0
        else:
            state.edens[:] = compute_atomic_energy_density(temperature_k=atm.temperature, state=state)
        itemp_cache.clear()
        itemp_cache.update(cache0)
        if ifmol:
            restore_nmolec_xnsave(nmolec_xnsave0)
            set_nmolec_ifedns(0)

    return edens1, edens2, edens3, edens4, rho1, rho2, rho3, rho4


def _write_debug_state_npz(
    path: Path,
    *,
    atm: AtlasAtmosphere,
    state: AtlasRuntimeState,
    iterations: int,
    enable_molecules: bool,
    abross_out: np.ndarray | None = None,
    tauros_out: np.ndarray | None = None,
    accrad_out: np.ndarray | None = None,
    prad_out: np.ndarray | None = None,
    flxrad_out: np.ndarray | None = None,
    rjmins_out: np.ndarray | None = None,
    rdabh_out: np.ndarray | None = None,
    rdiagj_out: np.ndarray | None = None,
    flxerr_out: np.ndarray | None = None,
    flxdrv_out: np.ndarray | None = None,
    dtflux_out: np.ndarray | None = None,
    dtlamb_out: np.ndarray | None = None,
    t1_out: np.ndarray | None = None,
    flxcnv_out: np.ndarray | None = None,
    vconv_out: np.ndarray | None = None,
    grdadb_out: np.ndarray | None = None,
    hscale_out: np.ndarray | None = None,
) -> None:
    """Dump internal state arrays for depth-by-depth EOS validation."""
    arrays: dict[str, np.ndarray] = {
        "rhox": np.asarray(atm.rhox, dtype=np.float64),
        "temperature": np.asarray(atm.temperature, dtype=np.float64),
        "tk": np.asarray(atm.tk, dtype=np.float64),
        "p": np.asarray(state.p, dtype=np.float64),
        "xne": np.asarray(state.xne, dtype=np.float64),
        "xnatom": np.asarray(state.xnatom, dtype=np.float64),
        "rho": np.asarray(state.rho, dtype=np.float64),
        "chargesq": np.asarray(state.chargesq, dtype=np.float64),
        "edens": np.asarray(state.edens, dtype=np.float64),
        "wtmole": np.asarray(state.wtmole, dtype=np.float64),
        "xabund": np.asarray(state.xabund, dtype=np.float64),
        "xnf": np.asarray(state.xnf, dtype=np.float64),
        "xnfp": np.asarray(state.xnfp, dtype=np.float64),
        "abross_out": np.asarray(
            atm.abross if abross_out is None else abross_out, dtype=np.float64
        ),
        "accrad_out": np.asarray(
            atm.accrad if accrad_out is None else accrad_out, dtype=np.float64
        ),
        "iterations": np.asarray([iterations], dtype=np.int32),
        "enable_molecules": np.asarray([1 if enable_molecules else 0], dtype=np.int32),
    }
    if state.amassiso_major is not None:
        arrays["amassiso_major"] = np.asarray(state.amassiso_major, dtype=np.float64)
    if state.dopple is not None:
        arrays["dopple"] = np.asarray(state.dopple, dtype=np.float64)
    if state.xnfdop is not None:
        arrays["xnfdop"] = np.asarray(state.xnfdop, dtype=np.float64)
    if tauros_out is not None:
        arrays["tauros_out"] = np.asarray(tauros_out, dtype=np.float64)
    if prad_out is not None:
        arrays["prad_out"] = np.asarray(prad_out, dtype=np.float64)
    if flxrad_out is not None:
        arrays["flxrad_out"] = np.asarray(flxrad_out, dtype=np.float64)
    if rjmins_out is not None:
        arrays["rjmins_out"] = np.asarray(rjmins_out, dtype=np.float64)
    if rdabh_out is not None:
        arrays["rdabh_out"] = np.asarray(rdabh_out, dtype=np.float64)
    if rdiagj_out is not None:
        arrays["rdiagj_out"] = np.asarray(rdiagj_out, dtype=np.float64)
    if flxerr_out is not None:
        arrays["flxerr_out"] = np.asarray(flxerr_out, dtype=np.float64)
    if flxdrv_out is not None:
        arrays["flxdrv_out"] = np.asarray(flxdrv_out, dtype=np.float64)
    if dtflux_out is not None:
        arrays["dtflux_out"] = np.asarray(dtflux_out, dtype=np.float64)
    if dtlamb_out is not None:
        arrays["dtlamb_out"] = np.asarray(dtlamb_out, dtype=np.float64)
    if t1_out is not None:
        arrays["t1_out"] = np.asarray(t1_out, dtype=np.float64)
    if flxcnv_out is not None:
        arrays["flxcnv_out"] = np.asarray(flxcnv_out, dtype=np.float64)
    if vconv_out is not None:
        arrays["vconv_out"] = np.asarray(vconv_out, dtype=np.float64)
    if grdadb_out is not None:
        arrays["grdadb_out"] = np.asarray(grdadb_out, dtype=np.float64)
    if hscale_out is not None:
        arrays["hscale_out"] = np.asarray(hscale_out, dtype=np.float64)
    if enable_molecules:
        snap = get_nmolec_snapshot()
        if snap is not None:
            arrays.update(snap)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def _apply_deck_to_atm_abundances(atm: AtlasAtmosphere, deck: AtlasDeck) -> None:
    """Override atm.abundances from deck ABUNDANCE cards (atlas12.for READIN 810-2003)."""
    import math
    for iz, val in deck.abund.items():
        if 1 <= iz <= 99:
            atm.abundances[iz] = val
    # XRELATIVE offsets are log10 scale changes applied on top of base abundance.
    for iz, offset in deck.xrelative.items():
        if 1 <= iz <= 99:
            base = atm.abundances.get(iz, 0.0)
            atm.abundances[iz] = base + offset


def run_atlas(cfg: AtlasConfig) -> AtlasAtmosphere:
    """Run one or more atlas_py iterations and write output `.atm`."""

    _t_total = time.perf_counter()
    _t_stage = time.perf_counter()

    atm = load_atm(cfg.inputs.atmosphere_path)
    state = _runtime_from_atm(atm)
    itemp_cache: Dict[str, int] = {"pops_itemp": -1}

    # ------------------------------------------------------------------
    # Parse control deck (READIN equivalent) if provided.
    # Deck values override the corresponding config / atm defaults.
    # ------------------------------------------------------------------
    deck: Optional[AtlasDeck] = None
    if cfg.inputs.control_deck_path is not None:
        deck = parse_readin_deck(cfg.inputs.control_deck_path)

    # Resolve per-run parameters (deck overrides cfg).
    numits = max(1, cfg.iterations)
    if deck is not None and deck.numits > 0:
        numits = deck.numits
    convergence_epsilon = (
        float(cfg.convergence_epsilon)
        if cfg.convergence_epsilon is not None and cfg.convergence_epsilon > 0.0
        else None
    )
    convergence_min_iterations = max(1, int(cfg.convergence_min_iterations))
    convergence_consecutive_required = max(1, int(cfg.convergence_consecutive))
    convergence_consecutive_count = 0
    completed_iterations = 0

    gravity_cgs = _gravity_from_atm_metadata(atm)
    if deck is not None and deck.grav > 0.0:
        gravity_cgs = deck.grav

    # IFOP flags: deck > atm metadata > driver default.
    ifop: list[int]
    if deck is not None:
        ifop = list(deck.ifop)
    else:
        ifop = _parse_ifop_flags(atm)

    # Molecular equilibrium.
    ifmol: bool = cfg.enable_molecules
    if deck is not None:
        ifmol = deck.molecules_on

    # Convection parameters (atlas12.for COMMON /CONV/).
    ifconv: int = 1 if cfg.enable_convection else 0
    mixlth: float = 1.25
    # Validation decks use `CONVECTION OVER 1.25 0 36` (overshoot OFF).
    # Keep the no-deck default consistent with that Fortran reference path.
    overwt: float = 0.0
    nconv: int = 0
    if deck is not None:
        ifconv = deck.ifconv
        mixlth = deck.mixlth
        overwt = deck.overwt
        nconv = deck.nconv

    # Turbulence parameters (atlas12.for COMMON /TURBPR/).
    ifturb: int = 0
    trbfdg: float = 0.0
    trbpow: float = 0.0
    trbsnd: float = 0.0
    trbcon: float = 0.0
    if deck is not None:
        ifturb = deck.ifturb
        trbfdg = deck.trbfdg
        trbpow = deck.trbpow
        trbsnd = deck.trbsnd
        trbcon = deck.trbcon

    # Apply abundance overrides from deck.
    if deck is not None and (deck.abund or deck.xrelative):
        _apply_deck_to_atm_abundances(atm, deck)
        # Rebuild state with updated abundances.
        state = _runtime_from_atm(atm)

    prad = np.zeros(atm.layers, dtype=np.float64)
    pturb = np.zeros(atm.layers, dtype=np.float64)
    pcon = 0.0
    # Fortran PTOTAL path (atlas12.for lines 238-245):
    #   PZERO = PCON + PRADK0 + PTURB0
    #   PTOTAL(J) = GRAV * RHOX(J) + PZERO
    # Seed PRADK0 from the input .atm PRADK line for iteration 1.
    pradk0_prev = _parse_pradk0_from_metadata(atm)

    # Initialise VTURB.
    # Priority: (1) explicit VTURB card from deck, (2) VTURBSTANDARD from deck or
    # Teff/logg defaults, (3) values already in the .atm.
    teff_init = float(atm.metadata.get("teff", 5778.0))
    if deck is not None and deck.teff > 0.0:
        teff_init = deck.teff
    glog_init = float(atm.metadata.get("glog", atm.metadata.get("logg", 4.44)))
    if deck is not None and deck.glog != 0.0:
        glog_init = deck.glog
    taustd_init = 10.0 ** (-6.875 + np.arange(atm.layers, dtype=np.float64) * 0.125)

    if deck is not None and not deck.vturb_standard and deck.vturb_value is not None:
        # VTURB card with explicit positive value: uniform profile.
        atm.vturb[:] = deck.vturb_value
    elif deck is not None and deck.vturb_standard:
        # VTURB card with negative value: call VTURBSTANDARD.
        atm.vturb[:] = vturbstandard(
            teff=teff_init,
            glog=glog_init,
            taustd=taustd_init,
            vnew=deck.vturb_standard_vnew,
        )
    elif not np.any(atm.vturb > 0):
        # No vturb set in .atm: generate via VTURBSTANDARD (atlas12.for default).
        atm.vturb[:] = vturbstandard(teff=teff_init, glog=glog_init, taustd=taustd_init)

    if ifmol:
        mol_path = cfg.inputs.molecules_path or find_default_molecules_file()
        if mol_path is None:
            raise FileNotFoundError(
                "Molecular mode enabled but no molecules file found "
                "(expected molecules.new/molecules.dat)"
            )
        mol = readmol_atlas12(mol_path)
        set_nmolec_context(
            temperature_k=atm.temperature,
            tk_erg=atm.tk,
            tlog=atm.tlog,
            gas_pressure=state.p,
            state=state,
            nummol=mol.nummol,
            code_mol=mol.code_mol,
            equil=mol.equil,
            locj=mol.locj,
            kcomps=mol.kcomps,
            idequa=mol.idequa,
            nequa=mol.nequa,
        )
    else:
        clear_nmolec_context()

    # IFPRES flag: atlas12.for line 224 `IF(IFPRES.EQ.0)GO TO 12`.
    # When IFPRES=0 (PRESSURE OFF): P and XNE are kept from the input model,
    # and POPS/POPSALL are skipped entirely in the main iteration loop.
    # Priority: deck > atm metadata > default (1 = ON).
    ifpres: int = 1
    if deck is not None:
        ifpres = deck.ifpres
    else:
        _ifpres_meta = atm.metadata.get("ifpres")
        if _ifpres_meta is not None:
            ifpres = int(_ifpres_meta)

    tcst = init_tcorr(atm.layers)
    itemp_counter = 0
    _require_line_opacity_inputs(cfg=cfg, ifop=ifop)
    tables = load_atlas_tables()
    selected_line_records = None
    nlteline_records = None
    fort12_path = cfg.inputs.line_selection_path
    if ifop[14] == 1:
        if fort12_path is None:
            import tempfile
            _tmp = tempfile.NamedTemporaryFile(
                suffix="_fort12.bin", delete=False, prefix="atlas_py_"
            )
            fort12_path = Path(_tmp.name)
            _tmp.close()
        elif fort12_path.exists():
            selected_line_records = read_selected_lines(fort12_path)
    if ifop[16] == 1:
        nlteline_records = read_nlteline_records(cfg.inputs.nlteline_path)

    for iter_idx in range(numits):
        iter_no = iter_idx + 1
        completed_iterations = iter_no
        itemp_counter += iter_no
        itemp = itemp_counter
        _t_stage = time.perf_counter()
        temp_before_iter = atm.temperature.copy()
        convergence_before = {
            "RHOX": atm.rhox.copy(),
            "T": atm.temperature.copy(),
            "P": state.p.copy(),
            "XNE": state.xne.copy(),
            "ABROSS": atm.abross.copy(),
            "VTURB": atm.vturb.copy(),
            "ACCRAD": atm.accrad.copy(),
        }
        # Turbulence pressure update (atlas12.for TURB subroutine, lines 5199-5214).
        # When IFTURB=1: call TURB to recompute VTURB and PTURB each iteration.
        # When IFTURB=0: PTURB stays at initial model values (zero from CALCULATE).
        if ifturb == 1:
            # Approximate sound speed: velsnd ≈ sqrt(gamma * P / rho).
            velsnd = np.sqrt(1.667 * state.p / np.maximum(state.rho, 1e-300))
            vturb_new, pturb_new = compute_turb(
                rho=state.rho,
                velsnd=velsnd,
                trbfdg=trbfdg,
                trbpow=trbpow,
                trbsnd=trbsnd,
                trbcon=trbcon,
            )
            atm.vturb[:] = vturb_new
            pturb[:] = pturb_new
        # atlas12.for lines 224-249: when IFPRES=0, jump to label 12, skipping
        # the hydrostatic P integration and the POPS/POPSALL calls entirely.
        if ifpres != 0:
            # atlas12.for line 225: IF(ITEMP.EQ.1)GO TO 111
            # On the first iteration, keep P from the input model (already includes
            # radiation pressure from a prior converged run). Recomputing P = g*RHOX
            # without PRAD would destroy the converged pressure balance.
            if itemp > 1:
                state.p[:] = integrate_hydrostatic_pressure(
                    atm=atm, gravity_cgs=gravity_cgs, prad=prad, pturb=pturb, pcon=pcon
                )

            # atlas12.for label 111, lines 239-243: init CHARGESQ with He II correction.
            tk_erg = atm.tk
            state.chargesq[:] = 2.0 * state.xne
            excess = 2.0 * state.xne - state.p / np.maximum(tk_erg, 1e-300)
            state.chargesq[excess > 0] += 2.0 * excess[excess > 0]

            dummy = np.zeros((atm.layers, 1), dtype=np.float64)
            pops(
                code=0.0,
                mode=1,
                out=dummy,
                ifmol=ifmol,
                ifpres=True,
                temperature_k=atm.temperature,
                tk_erg=atm.tk,
                state=state,
                itemp=itemp,
                itemp_cache=itemp_cache,
            )
            popsall(
                temperature_k=atm.temperature,
                tk_erg=atm.tk,
                state=state,
                ifmol=ifmol,
                ifpres=True,
                itemp=itemp,
                itemp_cache=itemp_cache,
            )
        update_doppler_populations(
            tk_erg=atm.tk,
            vturb_cms=atm.vturb,
            state=state,
        )

        logger.info("Timing: setup (load + POPS + POPSALL + DOPPLER) in %.3fs", time.perf_counter() - _t_stage)
        _t_stage = time.perf_counter()

        abross_out = atm.abross.copy()
        tauros_out = None
        accrad_out = atm.accrad.copy()
        prad_out = None
        flxrad_out = None
        rjmins_out = None
        rdabh_out = None
        rdiagj_out = None
        flxerr_out = None
        flxdrv_out = None
        dtflux_out = None
        dtlamb_out = None
        t1_out = None
        flxcnv_out = None
        vconv_out = None
        grdadb_out = None
        hscale_out = None
        adapter = _build_kapp_adapter(atm, state)
        teff = teff_init
        # Initialize TCORR state early so it can be passed to kapcont_baseline for XCONOP.
        # On the first iteration the ROSSTAB table is empty (nross=0), so XCONOP returns
        # 1.0 — matching Fortran behaviour: static ROSS(1)=0 → ROSSTAB=10.**0=1.0.
        _t_sub = time.perf_counter()
        wave_nm, rco, acont_all, sigmac_all, scont_all = kapcont_baseline(
            adapter=adapter,
            teff=teff,
            atlas_tables=tables,
            ifop=ifop,
            tcst=tcst,
        )
        logger.info("Timing: kapcont_baseline continuum grid in %.3fs", time.perf_counter() - _t_sub)
        if ifop[14] == 1:
            if state.xnfdop is None or state.dopple is None:
                raise ValueError("DOPPLE/XNFDOP arrays are required for LINOP1")
            _t_sub = time.perf_counter()
            tabcont, _wavetab, i_wavetab = kapcont_table(
                adapter=adapter,
                temperature_k=np.asarray(atm.temperature, dtype=np.float64),
                teff=teff,
                atlas_tables=tables,
                ifop=ifop,
                tcst=tcst,
            )
            logger.info("Timing: KAPCONT tabulation in %.3fs", time.perf_counter() - _t_sub)
            # Resolve fort.12: use precomputed path or run Python SELECTLINES.
            if selected_line_records is None:
                _t_sub = time.perf_counter()
                run_selectlines(
                    xnfdop=np.asarray(state.xnfdop, dtype=np.float32),
                    tabcont=np.asarray(tabcont, dtype=np.float32),
                    iwavetab=np.asarray(i_wavetab, dtype=np.int64),
                    hckt=np.asarray(atm.hckt, dtype=np.float64),
                    fort11_path=cfg.inputs.fort11_path,
                    fort111_path=cfg.inputs.fort111_path,
                    fort21_path=cfg.inputs.fort21_path,
                    fort31_path=cfg.inputs.fort31_path,
                    fort41_path=cfg.inputs.fort41_path,
                    fort51_path=cfg.inputs.fort51_path,
                    fort61_path=cfg.inputs.fort61_path,
                    fort12_output=fort12_path,
                )
                selected_line_records = read_selected_lines(fort12_path)
                logger.info("Timing: SELECTLINES/read fort.12 in %.3fs", time.perf_counter() - _t_sub)
            records = selected_line_records
            _t_sub = time.perf_counter()
            line_state = linop1(
                records=records,
                wave_set_nm=np.asarray(wave_nm, dtype=np.float64),
                i_wavetab=np.asarray(i_wavetab, dtype=np.int64),
                tabcont=np.asarray(tabcont, dtype=np.float64),
                temperature_k=np.asarray(atm.temperature, dtype=np.float64),
                hckt=np.asarray(atm.hckt, dtype=np.float64),
                xne=np.asarray(state.xne, dtype=np.float64),
                xnf=np.asarray(state.xnf, dtype=np.float64),
                xnfdop=np.asarray(state.xnfdop, dtype=np.float64),
                dopple=np.asarray(state.dopple, dtype=np.float64),
                nulo=1,
                nuhi=int(wave_nm.size),
            )
            logger.info("Timing: LINOP1 kernel in %.3fs", time.perf_counter() - _t_sub)
            xlines = line_state.xlines
        else:
            xlines = np.zeros((atm.layers, wave_nm.size), dtype=np.float32)
        logger.info("Timing: kapcont_baseline in %.3fs", time.perf_counter() - _t_stage)
        _t_stage = time.perf_counter()
        if ifop[16] == 1:
            if state.xnfdop is None or state.dopple is None:
                raise ValueError("DOPPLE/XNFDOP arrays are required for XLINOP")
            if ifop[14] != 1:
                _t_sub = time.perf_counter()
                tabcont, _wavetab, i_wavetab = kapcont_table(
                    adapter=adapter,
                    temperature_k=np.asarray(atm.temperature, dtype=np.float64),
                    teff=teff,
                    atlas_tables=tables,
                    ifop=ifop,
                    tcst=tcst,
                )
                logger.info("Timing: KAPCONT tabulation for XLINOP in %.3fs", time.perf_counter() - _t_sub)
            _t_sub = time.perf_counter()
            xline_state = xlinop(
                records=nlteline_records,
                wave_set_nm=np.asarray(wave_nm, dtype=np.float64),
                i_wavetab=np.asarray(i_wavetab, dtype=np.int64),
                tabcont=np.asarray(tabcont, dtype=np.float64),
                temperature_k=np.asarray(atm.temperature, dtype=np.float64),
                hckt=np.asarray(atm.hckt, dtype=np.float64),
                xne=np.asarray(state.xne, dtype=np.float64),
                xnf=np.asarray(state.xnf, dtype=np.float64),
                xnfp=np.asarray(state.xnfp, dtype=np.float64),
                rho=np.asarray(state.rho, dtype=np.float64),
                xnfdop=np.asarray(state.xnfdop, dtype=np.float64),
                dopple=np.asarray(state.dopple, dtype=np.float64),
                ifop15_enabled=(ifop[14] == 1),
                base_xlines=np.asarray(xlines, dtype=np.float32),
                nulo=1,
                nuhi=int(wave_nm.size),
            )
            logger.info("Timing: XLINOP kernel in %.3fs", time.perf_counter() - _t_sub)
            xlines = xline_state.xlines

        logger.info("Timing: LINOP (selectlines + line opacity) in %.3fs", time.perf_counter() - _t_stage)
        _t_stage = time.perf_counter()

        freq_all = 2.99792458e17 / np.maximum(wave_nm, 1e-300)
        hkt = np.asarray(atm.hkt, dtype=np.float64)
        temp = np.asarray(atm.temperature, dtype=np.float64)
        rhox = np.asarray(atm.rhox, dtype=np.float64)
        n_layers = atm.layers
        sigmal = np.zeros(n_layers, dtype=np.float64)
        abross_work = np.zeros(n_layers, dtype=np.float64)

        # HLINOP (IFOP(14)): precompute hydrogen line-wing opacity for all frequencies.
        # Fortran atlas12.for line 5247: IF(IFOP(14).EQ.1)CALL HLINOP
        # Returns ahline_all, shline_all of shape (n_layers, n_freq).
        if ifop[13] == 1:  # IFOP(14) in Fortran (0-based Python index)
            ehvkt_2d = np.exp(-np.outer(hkt, freq_all))           # (n_layers, n_freq)
            stim_2d = np.maximum(1.0 - ehvkt_2d, 1e-300)
            bnu_2d = 1.47439e-2 * ((freq_all / 1.0e15) ** 3)[None, :] * ehvkt_2d / stim_2d
            adapter_hw = _build_kapp_adapter(atm, state)
            ahline_all, shline_all = compute_hydrogen_wings(
                adapter_hw, freq_all, bnu_2d, ehvkt_2d, stim_2d, hkt
            )
        else:
            ahline_all = None
            shline_all = None
        logger.info("Timing: hydrogen wings (HLINOP) in %.3fs", time.perf_counter() - _t_stage)
        _t_stage = time.perf_counter()
        _ = ross_step(
            abross_work,
            mode=1,
            rcowt=0.0,
            bnu=np.zeros(n_layers, dtype=np.float64),
            freq_hz=0.0,
            hkt=hkt,
            temperature_k=temp,
            stim=np.ones(n_layers, dtype=np.float64),
            abtot=np.ones(n_layers, dtype=np.float64),
            numnu=int(freq_all.size),
            rhox=rhox,
        )
        flux = 5.6697e-5 / 12.5664 * teff**4
        radst = init_radiap(n_layers)
        # Reinitialise tcst for the frequency loop (was pre-created above for XCONOP;
        # mode=1 call below resets its accumulators to zero regardless).
        radiap_accumulate(
            radst,
            mode=1,
            rcowt=0.0,
            abtot=np.ones(n_layers, dtype=np.float64),
            hnu=np.zeros(n_layers, dtype=np.float64),
            jnu=np.zeros(n_layers, dtype=np.float64),
            knu_surface=0.0,
            flux=flux,
            rhox=rhox,
        )
        tcorr_step(
            tcst,
            mode=1,
            rcowt=0.0,
            rhox=rhox,
            abtot=np.ones(n_layers, dtype=np.float64),
            hnu=np.zeros(n_layers, dtype=np.float64),
            jmins=np.zeros(n_layers, dtype=np.float64),
            taunu=np.zeros(n_layers, dtype=np.float64),
            bnu=np.zeros(n_layers, dtype=np.float64),
            freq_hz=0.0,
            hkt=hkt,
            temperature_k=temp,
            stim=np.ones(n_layers, dtype=np.float64),
            alpha=np.zeros(n_layers, dtype=np.float64),
            flux=flux,
            teff=teff,
            numnu=int(freq_all.size),
        )
        # Initialize NLTE STATEQ accumulator if NLTE is active.
        nlteon: int = 0
        if deck is not None:
            nlteon = deck.nlteon
        stateq_acc = None
        if nlteon == 1:
            from ..physics.stateq import (
                StateqAccumulator, stateq_init, stateq_accumulate, stateq_solve,
            )
            stateq_acc = StateqAccumulator(nrhox=n_layers)
            stateq_init(stateq_acc, temp)
        logger.info("Timing: pre-loop setup (ROSS/RADIAP/TCORR init) in %.3fs", time.perf_counter() - _t_stage)
        _t_stage = time.perf_counter()
        for inu in range(freq_all.size):
            freq = float(freq_all[inu])
            rcowt = float(rco[inu])
            ehvkt = np.exp(-freq * hkt)
            stim = np.maximum(1.0 - ehvkt, 1e-300)
            freq15 = freq / 1.0e15
            bnu = 1.47439e-2 * (freq15**3) * ehvkt / stim
            acont = acont_all[:, inu]
            sigmac = sigmac_all[:, inu]
            scont = scont_all[:, inu]
            alines = xlines[:, inu] * stim
            # ALINE assembly: atlas12.for line 5260
            # ALINE(J) = AHLINE(J) + ALINES(J) + AXLINE(J)
            # AXLINE is zero (no separate NLTE-xline path in current driver).
            if ahline_all is not None:
                ahline = ahline_all[:, inu]
                shline = shline_all[:, inu]
                aline = ahline + alines
            else:
                ahline = np.zeros(n_layers, dtype=np.float64)
                aline = alines.copy()
            # SLINE assembly: atlas12.for lines 5261-5263
            # SLINE(J) = BNU(J)
            # IF(ALINE(J).GT.0.) SLINE(J)=(AHLINE*SHLINE+ALINES*BNU)/ALINE(J)
            sline = bnu.copy()
            mask = aline > 0.0
            if np.any(mask) and ahline_all is not None:
                sline[mask] = (
                    ahline[mask] * shline[mask] + alines[mask] * bnu[mask]
                ) / aline[mask]
            knu_log_path = os.getenv("ATLAS_KNU_LOG")
            if knu_log_path and inu == 2213:
                im1 = max(inu - 1, 0)
                ip1 = min(inu + 1, wave_nm.size - 1)
                with Path(knu_log_path).open("a", encoding="utf-8") as fh:
                    fh.write(
                        f"DRI,{float(abtot:=np.maximum(acont[0] + aline[0] + sigmac[0] + sigmal[0], 1e-300)):.8e},"
                        f"{float((sigmac[0] + sigmal[0]) / abtot):.8e},{float(acont[0]):.8e},{float(aline[0]):.8e},"
                        f"{float(sigmac[0]):.8e},{float(sigmal[0]):.8e},{float(ahline[0] if ahline_all is not None else 0.0):.8e},"
                        f"{float(alines[0]):.8e},{0.0:.8e}\n"
                    )
                    fh.write(
                        f"DRN,{int(im1):d},{float(wave_nm[im1]):.8e},{float(xlines[0, im1]):.8e},"
                        f"{int(inu):d},{float(wave_nm[inu]):.8e},{float(xlines[0, inu]):.8e},"
                        f"{int(ip1):d},{float(wave_nm[ip1]):.8e},{float(xlines[0, ip1]):.8e},"
                        f"{float(stim[0]):.8e}\n"
                    )
            if knu_log_path and 2212 <= inu <= 2214:
                with Path(knu_log_path).open("a", encoding="utf-8") as fh:
                    fh.write(
                        f"DRW,{int(inu):d},{float(wave_nm[inu]):.8e},{float(xlines[0, inu]):.8e},"
                        f"{float(stim[0]):.8e},{float(alines[0]):.8e},{float(aline[0]):.8e}\n"
                    )
            jres = josh_depth_profiles(
                ifscat=1,
                ifsurf=0,
                acont=acont,
                scont=scont,
                aline=aline,
                sline=sline,
                sigmac=sigmac,
                sigmal=sigmal,
                rhox=rhox,
                bnu=bnu,
                freq_hz=freq,
                wave_nm=float(wave_nm[inu]),
            )
            # Fortran safety clamp (atlas12.for lines 355-376): if any HNU(J)<0
            # after JOSH, clamp HNU, JNU, SNU to at least 1e-99 to prevent
            # RADIAP from accumulating negative radiative accelerations.
            if np.any(jres.hnu < 0.0):
                jres.hnu[:] = np.maximum(jres.hnu, 1e-99)
                jres.jnu[:] = np.maximum(jres.jnu, 1e-99)
                jres.snu[:] = np.maximum(jres.snu, 1e-99)
            radiap_accumulate(
                radst,
                mode=2,
                rcowt=rcowt,
                abtot=jres.abtot,
                hnu=jres.hnu,
                jnu=jres.jnu,
                knu_surface=jres.knu_surface,
                freq_hz=freq,
                wave_nm=float(wave_nm[inu]),
                flux=flux,
                rhox=rhox,
            )
            tcorr_step(
                tcst,
                mode=2,
                rcowt=rcowt,
                rhox=rhox,
                abtot=jres.abtot,
                hnu=jres.hnu,
                jmins=jres.jmins,
                taunu=jres.taunu,
                bnu=bnu,
                freq_hz=freq,
                hkt=hkt,
                temperature_k=temp,
                stim=stim,
                alpha=jres.alpha,
                flux=flux,
                teff=teff,
                numnu=int(freq_all.size),
            )
            abross_work, _ = ross_step(
                abross_work,
                mode=2,
                rcowt=rcowt,
                bnu=bnu,
                freq_hz=freq,
                hkt=hkt,
                temperature_k=temp,
                stim=stim,
                abtot=jres.abtot,
                numnu=int(freq_all.size),
                rhox=rhox,
            )
            # STATEQ MODE=2: accumulate H photo-ionization rates at this frequency.
            if stateq_acc is not None:
                stateq_accumulate(
                    stateq_acc,
                    freq=freq,
                    rcowt=rcowt,
                    jnu=jres.jnu,
                    hkt=hkt,
                    temperature_k=temp,
                )
        logger.info("Timing: frequency loop (%d freqs) in %.3fs", freq_all.size, time.perf_counter() - _t_stage)
        _t_stage = time.perf_counter()
        abross_out, tauros_out = ross_step(
            abross_work,
            mode=3,
            rcowt=0.0,
            bnu=np.zeros(n_layers, dtype=np.float64),
            freq_hz=0.0,
            hkt=hkt,
            temperature_k=temp,
            stim=np.ones(n_layers, dtype=np.float64),
            abtot=np.ones(n_layers, dtype=np.float64),
            numnu=int(freq_all.size),
            rhox=rhox,
        )
        radiap_accumulate(
            radst,
            mode=3,
            rcowt=0.0,
            abtot=np.ones(n_layers, dtype=np.float64),
            hnu=np.zeros(n_layers, dtype=np.float64),
            jnu=np.zeros(n_layers, dtype=np.float64),
            knu_surface=0.0,
            flux=flux,
            rhox=rhox,
        )
        accrad_out = radst.accrad.copy()
        prad_out = radst.prad.copy()
        flxrad_out = tcst.flxrad.copy()
        rjmins_out = tcst.rjmins.copy()
        rdabh_out = tcst.rdabh.copy()
        rdiagj_out = tcst.rdiagj.copy()
        rosstab_ingest(tcst, temp, state.p, abross_out)
        # CALL HIGH: compute geometric height from RHOX / RHO (atlas12.for line 388).
        state.height = high_from_rhox(rhox=rhox, rho=state.rho)
        pturb0 = float(pturb[0]) if pturb.size > 0 else 0.0
        pzero = pcon + float(pradk0_prev) + pturb0
        ptotal = gravity_cgs * rhox + pzero
        if ifconv == 1:
            ed1, ed2, ed3, ed4, r1, r2, r3, r4 = _convec_fd_samples(
                atm=atm,
                state=state,
                pradk=radst.pradk,
                tauros=tauros_out,
                ifmol=ifmol,
                itemp_seed=itemp * 10,
                itemp_cache=itemp_cache,
            )
        else:
            ed1 = ed2 = ed3 = ed4 = None
            r1 = r2 = r3 = r4 = None
        convec_log_path = os.getenv("ATLAS_CONVEC_LOG")
        cv = convec(
            tcst=tcst,
            rhox=rhox,
            tauros=tauros_out,
            temperature_k=temp,
            gas_pressure=state.p,
            mass_density=state.rho,
            abross=abross_out,
            vturb=atm.vturb,
            pradk=radst.pradk,
            ptotal=ptotal,
            gravity_cgs=gravity_cgs,
            flux=flux,
            mixlth=mixlth,
            overwt=overwt,
            ifconv=ifconv,
            nconv=nconv if nconv > 0 else 36,
            edens1=ed1,
            edens2=ed2,
            edens3=ed3,
            edens4=ed4,
            rho1=r1,
            rho2=r2,
            rho3=r3,
            rho4=r4,
            convec_log_path=convec_log_path,
        )
        flxcnv_out = cv.flxcnv.copy()
        vconv_out = cv.vconv.copy()
        grdadb_out = cv.grdadb.copy()
        hscale_out = cv.hscale.copy()
        tcres = tcorr_step(
            tcst,
            mode=3,
            rcowt=0.0,
            rhox=rhox,
            abtot=np.ones(n_layers, dtype=np.float64),
            hnu=np.zeros(n_layers, dtype=np.float64),
            jmins=np.zeros(n_layers, dtype=np.float64),
            taunu=np.zeros(n_layers, dtype=np.float64),
            bnu=np.zeros(n_layers, dtype=np.float64),
            freq_hz=0.0,
            hkt=hkt,
            temperature_k=temp,
            stim=np.ones(n_layers, dtype=np.float64),
            alpha=np.zeros(n_layers, dtype=np.float64),
            flux=flux,
            teff=teff,
            numnu=int(freq_all.size),
            tauros=tauros_out,
            abross=abross_out,
            iter_index=iter_no,
            ifconv=ifconv,
            flxcnv=cv.flxcnv,
            flxcnv0=cv.flxcnv0,
            dltdlp=cv.dltdlp,
            grdadb=cv.grdadb,
            hscale=cv.hscale,
            ptotal=ptotal,
            rho=state.rho,
            dlrdlt=cv.dlrdlt,
            heatcp=cv.heatcp,
            mixlth=mixlth,
            prad=prad_out if prad_out is not None else np.zeros(n_layers, dtype=np.float64),
            pturb=pturb,
            gravity_cgs=gravity_cgs,
            steplg=0.125,
            tau1lg=-6.875,
        )
        pradk0_prev = float(radst.pradk0)
        physical_convergence_max = float("inf")
        if tcres is not None:
            if flxcnv_out is not None:
                # Fortran TCORR mode=3 updates FLXCNV only for J=2..NRHOX-1
                # (atlas12.for lines 716-719), leaving endpoints untouched.
                flxcnv_out = np.asarray(flxcnv_out, dtype=np.float64).copy()
                if flxcnv_out.size > 2:
                    flxcnv_out[1:-1] = np.asarray(tcres.cnvflx, dtype=np.float64)[1:-1]
            # Fortran TCORR (atlas12.for lines 951-991):
            #   1. Apply DRHOX correction: RHOX(J) += DRHOX(J)  [lines 951-952]
            #   2. Remap ALL state arrays from original TAUROS grid → TAUSTD via MAP1
            #      [lines 954-963]: RHOX, T, P, XNE, ABROSS, PRAD, VTURB, BMIN, PTURB, ACCRAD
            #   3. Copy remapped values back; set TAUROS = TAUSTD  [lines 969-991]
            # Python mirrors this exactly: first set corrected T+T1 and RHOX+DRHOX on the
            # original TAUROS grid, then remap every state variable to TAUSTD.
            atm.temperature[:] = tcres.temperature   # T + T1, on original TAUROS grid
            atm.rhox[:] = tcres.rhox                 # RHOX + DRHOX, on original TAUROS grid
            if tauros_out is not None:
                # atlas12.for lines 995-997: TAUSTD grid definition
                taustd = np.float64(10.0) ** (
                    -6.875 + np.arange(n_layers, dtype=np.float64) * 0.125
                )
                # atlas12.for lines 954-963, 969-991: remap all state arrays to TAUSTD.
                atm.rhox[:], _ = _josh_map1(tauros_out, atm.rhox, taustd)       # line 954, 970
                atm.temperature[:], _ = _josh_map1(tauros_out, atm.temperature, taustd)  # 955, 971
                state.p[:], _ = _josh_map1(tauros_out, state.p, taustd)         # line 956, 977
                state.xne[:], _ = _josh_map1(tauros_out, state.xne, taustd)     # line 957, 978
                if abross_out is not None:
                    abross_out, _ = _josh_map1(tauros_out, abross_out, taustd)  # line 958, 979
                if prad_out is not None:
                    prad_out, _ = _josh_map1(tauros_out, prad_out, taustd)      # line 959, 980
                    prad = prad_out.copy()                                        # line 981
                atm.vturb[:], _ = _josh_map1(tauros_out, atm.vturb, taustd)     # line 960, 982
                pturb[:], _ = _josh_map1(tauros_out, pturb, taustd)              # line 962, 983
                # Fortran behavior: MAP1 is called for ACCRAD (atlas12.for line 967),
                # but the remapped DUM10 is never copied back into ACCRAD before output
                # (lines 973-988), so ACCRAD remains on the pre-remap grid.
                if accrad_out is not None:
                    pass
                # atlas12.for line 991: TAUROS(J) = TAUSTD(J)
                tauros_out = taustd.copy()
            if abross_out is not None:
                atm.abross[:] = abross_out
            if accrad_out is not None:
                atm.accrad[:] = accrad_out
            # Fortran does NOT recompute XNATOM/RHO/CHARGESQ after TCORR.
            # These are recomputed by POPS at the start of the next iteration.
            flxerr_out = tcres.flxerr.copy()
            flxdrv_out = tcres.flxdrv.copy()
            dtflux_out = tcres.dtflux.copy()
            dtlamb_out = tcres.dtlamb.copy()
            t1_out = tcres.t1.copy()
            layer_idx = min(79, n_layers - 1)
            rel_dt = np.abs(
                (atm.temperature - temp_before_iter)
                / np.maximum(np.abs(temp_before_iter), 1e-300)
            )
            if tauros_out is not None:
                photo_mask = (tauros_out >= 1.0e-2) & (tauros_out <= 10.0)
            else:
                photo_mask = np.zeros(n_layers, dtype=bool)
            if np.any(photo_mask):
                photo_rel = rel_dt[photo_mask]
                photo_msg = (
                    " photo_median_dT=%.3e photo_max_dT=%.3e"
                    % (float(np.median(photo_rel)), float(np.max(photo_rel)))
                )
            else:
                photo_msg = ""
            logger.info(
                "ATLAS iteration %d/%d: T[layer%d]=%.3fK dTmax=%.3e "
                "T1[layer%d]=%.3e RHOX[layer%d]=%.3e TAUROS[layer%d]=%.3e%s",
                iter_no,
                numits,
                layer_idx + 1,
                float(atm.temperature[layer_idx]),
                float(np.max(rel_dt)),
                layer_idx + 1,
                float(t1_out[layer_idx]),
                layer_idx + 1,
                float(atm.rhox[layer_idx]),
                layer_idx + 1,
                float(tauros_out[layer_idx]) if tauros_out is not None else float("nan"),
                photo_msg,
            )
            convergence_after = {
                "RHOX": atm.rhox,
                "T": atm.temperature,
                "P": state.p,
                "XNE": state.xne,
                "ABROSS": atm.abross,
                "VTURB": atm.vturb,
                "ACCRAD": atm.accrad,
            }
            convergence_metrics = {
                name: _max_normalized_column_delta(
                    convergence_before[name],
                    value,
                    floor=1.0e-300,
                    symmetric=(name in {"VTURB", "ACCRAD"}),
                )
                for name, value in convergence_after.items()
            }
            physical_convergence_max = max(
                convergence_metrics[name]
                for name in ("RHOX", "T", "P", "XNE", "ABROSS", "VTURB")
                if np.isfinite(convergence_metrics[name])
            )
            convergence_max = max(
                value for value in convergence_metrics.values() if np.isfinite(value)
            )
            logger.info(
                "Convergence iteration %d/%d: physical_max=%.3e max_col=%.3e %s",
                iter_no,
                numits,
                physical_convergence_max,
                convergence_max,
                " ".join(
                    f"{name}={value:.3e}" for name, value in convergence_metrics.items()
                ),
            )
            converged_this_iter = (
                convergence_epsilon is not None
                and iter_no >= convergence_min_iterations
                and physical_convergence_max < convergence_epsilon
            )
            if converged_this_iter:
                convergence_consecutive_count += 1
            else:
                convergence_consecutive_count = 0
        # STATEQ MODE=3: solve H departure coefficients after frequency loop.
        if stateq_acc is not None:
            stateq_solve(
                stateq_acc,
                temperature_k=temp,
                xne=state.xne,
                tkev=atm.tkev,
                xnfp=state.xnfp,
                bhyd=state.bhyd,
                bmin=state.bmin,
            )
        if (
            convergence_epsilon is not None
            and convergence_consecutive_count >= convergence_consecutive_required
        ):
            logger.info(
                "Convergence early stop at iteration %d/%d: "
                "physical_max=%.3e epsilon=%.3e min_iterations=%d consecutive=%d",
                iter_no,
                numits,
                physical_convergence_max,
                convergence_epsilon,
                convergence_min_iterations,
                convergence_consecutive_required,
            )
            break

    out_metadata = atm.metadata.copy()
    out_metadata["begin_line"] = f"BEGIN                    ITERATION{completed_iterations:4d} COMPLETED"
    out_metadata["pradk_line"] = f"PRADK {float(radst.pradk0):.4E}"
    out_extra1 = np.asarray(flxcnv_out, dtype=np.float64) if flxcnv_out is not None else atm.extra1.copy()
    out_extra2 = np.asarray(vconv_out, dtype=np.float64) if vconv_out is not None else atm.extra2.copy()
    out = AtlasAtmosphere(
        rhox=atm.rhox.copy(),
        temperature=atm.temperature.copy(),
        gas_pressure=state.p.copy(),
        electron_density=state.xne.copy(),
        abross=abross_out,
        accrad=accrad_out,
        vturb=atm.vturb.copy(),
        # DECK6 trailing columns are FLXCNV and VCONV in ATLAS12 output.
        extra1=out_extra1,
        extra2=out_extra2,
        metadata=out_metadata,
        abundances=atm.abundances.copy(),
    )
    logger.info("Timing: post-loop (ROSS/CONVEC/TCORR/output) in %.3fs", time.perf_counter() - _t_stage)
    write_atm(out, cfg.outputs.output_atm_path)
    if cfg.outputs.debug_state_path is not None:
        _write_debug_state_npz(
            cfg.outputs.debug_state_path,
            atm=atm,
            state=state,
            iterations=completed_iterations,
            enable_molecules=ifmol,
            abross_out=abross_out,
            tauros_out=tauros_out,
            accrad_out=accrad_out,
            prad_out=prad_out,
            flxrad_out=flxrad_out,
            rjmins_out=rjmins_out,
            rdabh_out=rdabh_out,
            rdiagj_out=rdiagj_out,
            flxerr_out=flxerr_out,
            flxdrv_out=flxdrv_out,
            dtflux_out=dtflux_out,
            dtlamb_out=dtlamb_out,
            t1_out=t1_out,
            flxcnv_out=flxcnv_out,
            vconv_out=vconv_out,
            grdadb_out=grdadb_out,
            hscale_out=hscale_out,
        )
    logger.info("Timing: total atlas_py run in %.3fs", time.perf_counter() - _t_total)
    return out

