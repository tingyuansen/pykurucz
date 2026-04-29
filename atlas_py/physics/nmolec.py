"""Molecular equilibrium (NMOLEC/MOLEC) port from `atlas12.for`.

Fortran reference:
- `atlas12.for` lines 4038-4556 (`MOLEC`, `NMOLEC`)
- `atlas12.for` lines 23130-23231 (`PARTFNH2`, `EQUILH2`)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .pfsaha import pfsaha_depth
from .runtime import AtlasRuntimeState

from .nmolec_data import load_nmolec_tables as _load_nm_tables

_MAX_ITERS = 200
_TOL = 1.0e-4
_AMU_G = 1.660e-24
_K_BOLTZ = 1.38054e-16
_H_PLANCK = 6.6256e-27
_C_LIGHT = 2.997925e10

_NM_TABLES = _load_nm_tables()
_ATMASS = _NM_TABLES["_ATMASS"]
_H2_PF = _NM_TABLES["_H2_PF"]


@dataclass
class _NmolecContext:
    temperature_k: np.ndarray
    tk_erg: np.ndarray
    tlog: np.ndarray
    gas_pressure: np.ndarray
    state: AtlasRuntimeState
    mode: int
    nummol: int
    code_mol: np.ndarray
    equil: np.ndarray
    locj: np.ndarray
    kcomps: np.ndarray
    idequa: np.ndarray
    nequa: int
    xnmol: np.ndarray
    xnfpmol: np.ndarray
    xnz: np.ndarray
    xnsave: np.ndarray
    ifedns: int


_CTX: _NmolecContext | None = None


def set_nmolec_context(
    *,
    temperature_k: np.ndarray,
    tk_erg: np.ndarray,
    tlog: np.ndarray,
    gas_pressure: np.ndarray,
    state: AtlasRuntimeState,
    nummol: int,
    code_mol: np.ndarray,
    equil: np.ndarray,
    locj: np.ndarray,
    kcomps: np.ndarray,
    idequa: np.ndarray,
    nequa: int,
) -> None:
    """Bind current atmospheric state + READMOL data for NMOLEC/MOLEC calls."""
    global _CTX
    n_layers = temperature_k.size
    _CTX = _NmolecContext(
        temperature_k=temperature_k,
        tk_erg=tk_erg,
        tlog=tlog,
        gas_pressure=gas_pressure,
        state=state,
        mode=12,
        nummol=int(nummol),
        code_mol=np.asarray(code_mol, dtype=np.float64),
        equil=np.asarray(equil, dtype=np.float64),
        locj=np.asarray(locj, dtype=np.int32),
        kcomps=np.asarray(kcomps, dtype=np.int32),
        idequa=np.asarray(idequa, dtype=np.int32),
        nequa=int(nequa),
        xnmol=np.zeros((n_layers, int(nummol)), dtype=np.float64),
        xnfpmol=np.zeros((n_layers, int(nummol)), dtype=np.float64),
        xnz=np.zeros((n_layers, max(int(nequa), 1)), dtype=np.float64),
        xnsave=np.zeros((n_layers, max(int(nequa), 1)), dtype=np.float64),
        ifedns=0,
    )


def clear_nmolec_context() -> None:
    global _CTX
    _CTX = None


def get_nmolec_snapshot() -> dict[str, np.ndarray] | None:
    """Return current molecular-state arrays for debug dumping."""
    if _CTX is None:
        return None
    return {
        "nmolec_code_mol": np.asarray(_CTX.code_mol, dtype=np.float64).copy(),
        "nmolec_equil": np.asarray(_CTX.equil, dtype=np.float64).copy(),
        "nmolec_locj": np.asarray(_CTX.locj, dtype=np.int32).copy(),
        "nmolec_kcomps": np.asarray(_CTX.kcomps, dtype=np.int32).copy(),
        "nmolec_idequa": np.asarray(_CTX.idequa, dtype=np.int32).copy(),
        "nmolec_xnmol": np.asarray(_CTX.xnmol, dtype=np.float64).copy(),
        "nmolec_xnfpmol": np.asarray(_CTX.xnfpmol, dtype=np.float64).copy(),
        "nmolec_xnz": np.asarray(_CTX.xnz, dtype=np.float64).copy(),
    }


def _interp_h2_pf(t: float) -> float:
    n = int(t / 100.0)
    n = min(199, max(1, n))
    p0 = _H2_PF[n - 1]
    p1 = _H2_PF[n]
    return float(p0 + (p1 - p0) * (t - n * 100.0) / 100.0)


def _equilh2(t: float) -> float:
    pf = _interp_h2_pf(t)
    denom = (
        2.0
        * np.pi
        * 1.008
        * _AMU_G
        * _K_BOLTZ
        / (_H_PLANCK**2)
        * t
    ) ** 1.5
    expo = 36118.11 * _H_PLANCK * _C_LIGHT / _K_BOLTZ / max(t, 1e-30)
    return float(pf * (2.0**1.5) / 4.0 / max(denom, 1e-300) * np.exp(expo))


def _pfsaha_single(*, j: int, id_atomic: int, nion: int, mode: int, temp_override: float | None = None) -> np.ndarray:
    if _CTX is None:
        raise RuntimeError("NMOLEC context not initialized")
    t = float(_CTX.temperature_k[j]) if temp_override is None else float(temp_override)
    tk = t * _K_BOLTZ
    out = pfsaha_depth(
        temperature_k=t,
        electron_density_cm3=float(_CTX.state.xne[j]),
        xnatom_cm3=float(_CTX.state.xnatom[j]),
        xabund_linear=float(_CTX.state.xabund[j, id_atomic - 1]),
        atomic_number=int(id_atomic),
        nion=int(nion),
        mode=int(mode),
        chargesq_cm3=float(max(_CTX.state.chargesq[j], 1e-30)),
    )
    # Keep the same dimensionality assumptions as Fortran PFSAHA calls.
    if mode >= 10:
        return np.asarray(out, dtype=np.float64)
    # For scalar returns in mode < 10, wrap as length-1 vector.
    _ = tk
    return np.asarray(out, dtype=np.float64)


def _compute_equilj_for_depth(j: int, xn: np.ndarray) -> np.ndarray:
    if _CTX is None:
        raise RuntimeError("NMOLEC context not initialized")
    ctx = _CTX
    t = float(ctx.temperature_k[j])
    tkev = t / 11604.5
    tlog = float(np.log(max(t, 1e-300)))
    nequa = ctx.nequa
    equilj = np.zeros(ctx.nummol, dtype=np.float64)

    for jmol in range(ctx.nummol):
        loc1 = int(ctx.locj[jmol]) - 1
        loc2 = int(ctx.locj[jmol + 1]) - 1
        ncomp = loc2 - loc1
        e1 = float(ctx.equil[0, jmol])
        if e1 != 0.0:
            ion = int((ctx.code_mol[jmol] - float(int(ctx.code_mol[jmol]))) * 100.0 + 0.5)
            if abs(ctx.code_mol[jmol] - 101.0) < 0.005:
                if t > 20000.0:
                    continue
                equilj[jmol] = _equilh2(t)
            else:
                if t > 10000.0:
                    continue
                e2 = float(ctx.equil[1, jmol])
                e3 = float(ctx.equil[2, jmol])
                e4 = float(ctx.equil[3, jmol])
                e5 = float(ctx.equil[4, jmol])
                e6 = float(ctx.equil[5, jmol])
                poly = e3 + (-e4 + (e5 - e6 * t) * t) * t
                expo = e1 / max(tkev, 1e-30) - e2 + poly * t - 1.5 * (float(ncomp - ion - ion - 1)) * tlog
                equilj[jmol] = float(np.exp(expo))
            continue

        if ncomp <= 1:
            equilj[jmol] = 1.0
            continue

        id_atomic = int(ctx.code_mol[jmol])
        ion = ncomp - 1
        frac = _pfsaha_single(j=j, id_atomic=id_atomic, nion=ncomp, mode=12)
        if frac.size < ncomp or frac[0] <= 0.0:
            equilj[jmol] = 0.0
            continue
        equilj[jmol] = float(frac[ncomp - 1] / frac[0] * max(ctx.state.xne[j], 1e-300) ** ion)

    return equilj


def _solve_depth(j: int, xn_seed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if _CTX is None:
        raise RuntimeError("NMOLEC context not initialized")
    ctx = _CTX
    nequa = ctx.nequa
    nequa1 = nequa + 1
    neqn = nequa * nequa
    xn = xn_seed.copy()
    eqold = np.zeros(nequa, dtype=np.float64)
    xab = np.zeros(nequa, dtype=np.float64)

    for k in range(1, nequa):
        idk = int(ctx.idequa[k])
        if idk < 100 and idk > 0:
            xab[k] = max(float(ctx.state.xabund[j, idk - 1]), 1e-20)
    if int(ctx.idequa[nequa - 1]) == 100:
        xab[nequa - 1] = 0.0

    for _ in range(_MAX_ITERS):
        deq = np.zeros((nequa, nequa), dtype=np.float64)
        eq = np.zeros(nequa, dtype=np.float64)
        equilj = _compute_equilj_for_depth(j, xn)

        tk = float(ctx.temperature_k[j]) * _K_BOLTZ
        eq[0] = -float(ctx.gas_pressure[j] / max(tk, 1e-300))
        for k in range(1, nequa):
            eq[0] += xn[k]
            deq[0, k] = 1.0
            eq[k] = xn[k] - xab[k] * xn[0]
            deq[k, k] = 1.0
            deq[k, 0] = -xab[k]
        if int(ctx.idequa[nequa - 1]) == 100:
            eq[nequa - 1] = -xn[nequa - 1]
            deq[nequa - 1, nequa - 1] = -1.0

        for jmol in range(ctx.nummol):
            loc1 = int(ctx.locj[jmol]) - 1
            loc2 = int(ctx.locj[jmol + 1]) - 1
            ncomp = loc2 - loc1
            if ncomp <= 1:
                continue
            term = float(equilj[jmol])
            if term == 0.0:
                continue
            for loc in range(loc1, loc2):
                k = int(ctx.kcomps[loc])
                if k == nequa:
                    term = term / max(xn[nequa - 1], 1e-300)
                else:
                    term = term * xn[k]

            eq[0] += term
            for loc in range(loc1, loc2):
                kraw = int(ctx.kcomps[loc])
                if kraw == nequa:
                    k = nequa - 1
                    d = -term / max(xn[k], 1e-300)
                else:
                    k = kraw
                    d = term / max(xn[k], 1e-300)
                eq[k] += term
                deq[0, k] += d
                for locm in range(loc1, loc2):
                    mraw = int(ctx.kcomps[locm])
                    m = nequa - 1 if mraw == nequa else mraw
                    deq[m, k] += d

            k_last = int(ctx.kcomps[loc2 - 1])
            if k_last < nequa and int(ctx.idequa[k_last]) == 100:
                for loc in range(loc1, loc2):
                    k = int(ctx.kcomps[loc])
                    if k >= nequa:
                        k = nequa - 1
                    d = term / max(xn[k], 1e-300)
                    if k == nequa - 1:
                        eq[k] -= term + term
                    for locm in range(loc1, loc2):
                        mraw = int(ctx.kcomps[locm])
                        m = nequa - 1 if mraw == nequa else mraw
                        if m == nequa - 1:
                            deq[m, k] -= d + d

        try:
            delta = np.linalg.solve(deq, eq)
        except np.linalg.LinAlgError:
            delta, *_ = np.linalg.lstsq(deq, eq, rcond=None)
        if delta.size != nequa:
            raise RuntimeError("NMOLEC solver returned wrong vector size")

        iferr = False
        scale = 100.0
        for k in range(nequa):
            ratio = abs(float(delta[k])) / max(abs(float(xn[k])), 1e-300)
            if ratio > _TOL:
                iferr = True
            if eqold[k] * delta[k] < 0.0:
                delta[k] = delta[k] * 0.69
            xneq = xn[k] - delta[k]
            xn100 = xn[k] / 100.0
            if abs(xneq) >= xn100:
                xn[k] = abs(xneq)
            else:
                xn[k] = xn[k] / scale
                if eqold[k] * delta[k] < 0.0:
                    scale = np.sqrt(scale)
            eqold[k] = delta[k]

        if not iferr:
            return xn, equilj

    return xn, _compute_equilj_for_depth(j, xn)


def nmolec(*_args, **_kwargs):
    """Compute molecular equilibrium and update state arrays (ATLAS12 path)."""
    if _CTX is None:
        raise RuntimeError("NMOLEC context not initialized")

    ctx = _CTX
    n_layers = ctx.temperature_k.size
    nequa = ctx.nequa
    if nequa <= 0:
        return
    mode = int(_kwargs.get("mode", _args[0] if len(_args) > 0 else 1))

    xab = np.zeros(nequa, dtype=np.float64)
    for k in range(1, nequa):
        idk = int(ctx.idequa[k])
        if 0 < idk < 100:
            xab[k] = max(float(ctx.state.xabund[0, idk - 1]), 1e-20)
    if int(ctx.idequa[nequa - 1]) == 100:
        xab[nequa - 1] = 0.0

    xn = np.zeros(nequa, dtype=np.float64)
    xntot0 = float(ctx.gas_pressure[0] / max(float(ctx.temperature_k[0]) * _K_BOLTZ, 1e-300))
    xn[0] = xntot0 / 2.0
    if float(ctx.temperature_k[0]) < 4000.0:
        xn[0] = xntot0
    x = xn[0] / 10.0
    for k in range(1, nequa):
        xn[k] = x * xab[k]
    if int(ctx.idequa[nequa - 1]) == 100:
        xn[nequa - 1] = x
    ctx.state.xne[0] = x

    for j in range(n_layers):
        if j > 0:
            ratio = float(ctx.gas_pressure[j] / max(ctx.gas_pressure[j - 1], 1e-300))
            ctx.state.xne[j] = ctx.state.xne[j - 1] * ratio
            xn *= ratio
        if ctx.ifedns == 1 and np.any(ctx.xnsave[j] != 0.0):
            # Fortran IFEDNS path seeds XN from XNSAVE (atlas12.for line 4190+)
            # but still proceeds through the depth solution update.
            xn[:] = ctx.xnsave[j]
        xn, _ = _solve_depth(j, xn)

        ctx.xnz[j, :nequa] = xn[:nequa]
        ctx.state.xnatom[j] = xn[0]
        ctx.state.rho[j] = ctx.state.xnatom[j] * ctx.state.wtmole[j] * _AMU_G
        if int(ctx.idequa[nequa - 1]) == 100:
            ctx.state.xne[j] = xn[nequa - 1]

        equilj = _compute_equilj_for_depth(j, xn)
        for jmol in range(ctx.nummol):
            term = float(equilj[jmol])
            loc1 = int(ctx.locj[jmol]) - 1
            loc2 = int(ctx.locj[jmol + 1]) - 1
            for loc in range(loc1, loc2):
                k = int(ctx.kcomps[loc])
                if k == nequa:
                    term = term / max(xn[nequa - 1], 1e-300)
                else:
                    term = term * xn[k]
            ctx.xnmol[j, jmol] = term

    if ctx.ifedns == 0:
        ctx.xnsave[:, :nequa] = ctx.xnz[:, :nequa]

    if ctx.ifedns == 1:
        # Fortran NMOLEC jumps directly to label 160 when IFEDNS=1
        # (atlas12.for line 4351), skipping the XNFP/partition-function
        # conversion block below. CONVEC only needs XNMOL/RHO/EDENS samples.
        ctx.state.edens[:] = 1.5 * np.asarray(ctx.gas_pressure, dtype=np.float64) / np.maximum(
            ctx.state.rho, 1e-300
        )
        return

    if mode in (2, 12):
        return

    # Convert to NUMBER DENSITIES / PARTITION FUNCTIONS (`XNfpMOL`) as in atlas12.for.
    for k in range(1, nequa):
        idk = int(ctx.idequa[k])
        if idk <= 0:
            continue
        if idk == 100:
            t = np.asarray(ctx.temperature_k, dtype=np.float64)
            ctx.xnz[:, k] = ctx.xnz[:, k] / (2.0 * 2.4148e15 * t * np.sqrt(np.maximum(t, 1e-300)))
            continue
        amass = float(_ATMASS[idk - 1]) if 1 <= idk <= _ATMASS.size else float(idk)
        for j in range(n_layers):
            frac = _pfsaha_single(j=j, id_atomic=idk, nion=1, mode=3)
            pf = float(frac[0]) if frac.size else 1.0
            t = float(ctx.temperature_k[j])
            ctx.xnz[j, k] = ctx.xnz[j, k] / max(
                pf * 1.8786e20 * np.sqrt(max((amass * t) ** 3, 1e-300)),
                1e-300,
            )

    for jmol in range(ctx.nummol):
        e1 = float(ctx.equil[0, jmol])
        if e1 != 0.0:
            amass = 0.0
            loc1 = int(ctx.locj[jmol]) - 1
            loc2 = int(ctx.locj[jmol + 1]) - 1
            for loc in range(loc1, loc2):
                k = int(ctx.kcomps[loc])
                if k >= nequa:
                    continue
                idk = int(ctx.idequa[k])
                if 1 <= idk <= _ATMASS.size:
                    amass += float(_ATMASS[idk - 1])
            for j in range(n_layers):
                t = float(ctx.temperature_k[j])
                tkev = t / 11604.5
                val = np.exp(e1 / max(tkev, 1e-300))
                loc1j = int(ctx.locj[jmol]) - 1
                loc2j = int(ctx.locj[jmol + 1]) - 1
                for loc in range(loc1j, loc2j):
                    k = int(ctx.kcomps[loc])
                    if k == nequa:
                        val = val / max(ctx.xnz[j, nequa - 1], 1e-300)
                    else:
                        val = val * ctx.xnz[j, k]
                val = val * 1.8786e20 * np.sqrt(max((amass * t) ** 3, 1e-300))
                ctx.xnfpmol[j, jmol] = val
            continue

        id_atomic = int(ctx.code_mol[jmol])
        loc1 = int(ctx.locj[jmol]) - 1
        loc2 = int(ctx.locj[jmol + 1]) - 1
        ncomp = loc2 - loc1
        for j in range(n_layers):
            frac = _pfsaha_single(j=j, id_atomic=id_atomic, nion=max(ncomp, 1), mode=3)
            pf = float(frac[0]) if frac.size else 1.0
            ctx.xnfpmol[j, jmol] = ctx.xnmol[j, jmol] / max(pf, 1e-300)

    # EDENS baseline update matching Fortran NMOLEC label 160 (atlas12.for 4449+):
    # thermal base only; full molecular derivatives computed by compute_nmolec_edens().
    # Fortran label 160 base term: 1.5 * (P/TK) * TK / RHO == 1.5 * P / RHO.
    ctx.state.edens[:] = 1.5 * np.asarray(ctx.gas_pressure, dtype=np.float64) / np.maximum(
        ctx.state.rho, 1e-300
    )


def molec(*_args, **_kwargs):
    """Return requested molecule/ion populations from current molecular state."""
    if _CTX is None:
        raise RuntimeError("NMOLEC context not initialized")
    codout = float(_kwargs.get("codout", _args[0] if len(_args) > 0 else 0.0))
    mode = int(_kwargs.get("mode", _args[1] if len(_args) > 1 else 1))
    number = _kwargs.get("number", _args[2] if len(_args) > 2 else None)
    if number is None:
        raise ValueError("MOLEC requires output array `number`")

    ctx = _CTX
    n_layers = number.shape[0]
    number[:, :] = 0.0

    # Exact code lookup for molecular outputs (codout >= 100).
    # Atomic-like codes (<100) must follow the atlas12 ion-sequence logic below.
    if codout >= 100.0:
        for jmol in range(ctx.nummol):
            if abs(ctx.code_mol[jmol] - codout) < 1e-3:
                src = ctx.xnfpmol[:, jmol] if mode in (1, 11) else ctx.xnmol[:, jmol]
                number[:, 0] = src[:n_layers]
                return

    # If no molecular match and code is atomic-like, follow atlas12 MOLEC
    # ion-sequence lookup (atlas12.for lines 4096-4125): try C, C-0.01, ...
    # before falling back to PFSAHA.
    if codout < 100.0:
        c = float(codout)
        nn = 1
        if mode in (11, 12):
            nn = int((c - float(int(c))) * 100.0 + 1.5)
        fallback_to_pfsaha = False
        id_atomic = int(codout)
        for i in range(1, nn + 1):
            ion = nn - i + 1
            found_exact = False
            for jmol in range(ctx.nummol):
                if abs(ctx.code_mol[jmol] - c) < 1.0e-3:
                    src = ctx.xnfpmol[:, jmol] if mode in (1, 11) else ctx.xnmol[:, jmol]
                    number[:n_layers, ion - 1] = src[:n_layers]
                    found_exact = True
                    break
            if found_exact:
                c -= 0.01
                continue

            found_element_family = False
            for jmol in range(ctx.nummol):
                if int(ctx.code_mol[jmol]) == id_atomic:
                    found_element_family = True
                    break
            if found_element_family:
                number[:n_layers, ion - 1] = 0.0
                c -= 0.01
                continue

            fallback_to_pfsaha = True
            break

        if not fallback_to_pfsaha:
            return

        iz = id_atomic
        nion = int((codout - float(iz)) * 100.0 + 1.5)
        nret = nion if mode in (11, 12) else 1
        for j in range(n_layers):
            vals = pfsaha_depth(
                temperature_k=float(ctx.temperature_k[j]),
                electron_density_cm3=float(ctx.state.xne[j]),
                xnatom_cm3=float(ctx.state.xnatom[j]),
                xabund_linear=float(ctx.state.xabund[j, iz - 1]),
                atomic_number=iz,
                nion=nion,
                mode=mode,
            )
            ncopy = min(nret, vals.size, number.shape[1])
            number[j, :ncopy] = vals[:ncopy] * ctx.state.xnatom[j] * ctx.state.xabund[j, iz - 1]
        return

    raise ValueError(f"MOLEC: code {codout:.2f} not found in molecular table")


# ---------------------------------------------------------------------------
# CONVEC FD helpers: ifedns flag management and Fortran-matching EDENS
# ---------------------------------------------------------------------------

def set_nmolec_ifedns(val: int) -> None:
    """Set the nmolec ifedns flag (0=main iteration, 1=CONVEC FD warm-start)."""
    if _CTX is not None:
        _CTX.ifedns = int(val)


def save_nmolec_xnsave() -> "np.ndarray | None":
    """Return a copy of the current xnsave array for later restoration."""
    if _CTX is None:
        return None
    return _CTX.xnsave.copy()


def restore_nmolec_xnsave(saved: "np.ndarray | None") -> None:
    """Restore xnsave from a previously saved copy."""
    if _CTX is not None and saved is not None:
        _CTX.xnsave[:] = saved


def compute_nmolec_edens(
    *,
    temperature_k: "np.ndarray",
    tk_erg: "np.ndarray",
    gas_pressure: "np.ndarray",
    state: "AtlasRuntimeState",
) -> "np.ndarray":
    """Compute EDENS = (thermal + molecular) / rho matching Fortran label 160 in NMOLEC.

    Fortran reference: atlas12.for lines 4449-4554 (label 160 in SUBROUTINE NMOLEC).
    Called by _convec_fd_samples in place of compute_atomic_energy_density so that
    the CONVEC FD derivatives include molecular dissociation energy contributions
    (especially H2 at T<5000K).

    Returns energy density in erg g^-1 per depth layer.
    """
    ctx = _CTX
    n = int(temperature_k.size)
    edens = np.zeros(n, dtype=np.float64)
    log_path = os.getenv("ATLAS_NMOLEC_EDENS_LOG", "").strip()
    log_label = os.getenv("ATLAS_NMOLEC_EDENS_LABEL", "").strip()
    try:
        log_maxj = int(os.getenv("ATLAS_NMOLEC_EDENS_MAXJ", "4"))
    except ValueError:
        log_maxj = 4
    log_fh = None
    if log_path:
        out = Path(log_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        is_new = not out.exists()
        log_fh = out.open("a", encoding="utf-8")
        if is_new:
            log_fh.write("label,J,base_e,mol_true_sum,mol_pseudo_sum,e_total_before_div,rho,edens\n")

    try:
        for j in range(n):
            t = max(float(temperature_k[j]), 1.0)
            tk = float(tk_erg[j])
            tkev = t / 11604.5
            hckt = _H_PLANCK * _C_LIGHT / max(tk, 1e-300)
            xntot = float(gas_pressure[j]) / max(tk, 1e-300)
            e = 1.5 * xntot * tk
            base_e = e
            mol_true_sum = 0.0
            mol_pseudo_sum = 0.0
            max_pseudo_contrib = 0.0
            max_pseudo_code = 0.0
            max_pseudo_xnm = 0.0
            max_pseudo_eion = 0.0
            max_pseudo_pffrac = 0.0
            max_pseudo_pfp = 0.0
            max_pseudo_pfm = 0.0

            if ctx is not None:
                tplus = t * 1.001
                tminus = t * 0.999

                for jmol in range(ctx.nummol):
                    xnm = float(ctx.xnmol[j, jmol])
                    if xnm <= 0.0:
                        continue

                    e1 = float(ctx.equil[0, jmol])
                    code_val = float(ctx.code_mol[jmol])
                    loc1 = int(ctx.locj[jmol]) - 1
                    loc2 = int(ctx.locj[jmol + 1]) - 1
                    ncomp = loc2 - loc1

                    if e1 != 0.0:
                        if abs(code_val - 101.0) < 0.005:
                            pfplus = _interp_h2_pf(tplus) + 1e-30
                            pfminus = _interp_h2_pf(tminus) + 1e-30
                            ediss_per_kT = 36118.11 * hckt
                        else:
                            e2 = float(ctx.equil[1, jmol])
                            e3 = float(ctx.equil[2, jmol])
                            e4 = float(ctx.equil[3, jmol])
                            e5 = float(ctx.equil[4, jmol])
                            e6 = float(ctx.equil[5, jmol])
                            pfplus = np.exp(-e2 + (e3 + (-e4 + (e5 - e6 * tplus) * tplus) * tplus) * tplus) + 1e-30
                            pfminus = np.exp(-e2 + (e3 + (-e4 + (e5 - e6 * tminus) * tminus) * tminus) * tminus) + 1e-30
                            nequa = ctx.nequa
                            for loc in range(loc1, loc2):
                                k = int(ctx.kcomps[loc])
                                if k >= nequa:
                                    continue
                                idk = int(ctx.idequa[k])
                                if 0 < idk < 100:
                                    fp = pfsaha_depth(
                                        temperature_k=tplus,
                                        electron_density_cm3=float(state.xne[j]),
                                        xnatom_cm3=float(state.xnatom[j]),
                                        xabund_linear=float(state.xabund[j, idk - 1]),
                                        atomic_number=idk,
                                        nion=1,
                                        mode=3,
                                        chargesq_cm3=float(max(state.chargesq[j], 1e-30)),
                                    )
                                    pfplus *= float(fp.flat[0])
                                    fm = pfsaha_depth(
                                        temperature_k=tminus,
                                        electron_density_cm3=float(state.xne[j]),
                                        xnatom_cm3=float(state.xnatom[j]),
                                        xabund_linear=float(state.xabund[j, idk - 1]),
                                        atomic_number=idk,
                                        nion=1,
                                        mode=3,
                                        chargesq_cm3=float(max(state.chargesq[j], 1e-30)),
                                    )
                                    pfminus *= float(fm.flat[0])
                            ediss_per_kT = e1 / max(tkev, 1e-300)

                        pffrac = (pfplus - pfminus) / max(pfplus + pfminus, 1e-30) * 2.0 * 500.0
                        contrib = xnm * tk * (-ediss_per_kT + pffrac)
                        e += contrib
                        mol_true_sum += contrib
                    else:
                        id_atomic = int(code_val)
                        if id_atomic < 1 or id_atomic > 99:
                            continue
                        ion = max(ncomp, 1)
                        pfp = pfsaha_depth(
                            temperature_k=tplus,
                            electron_density_cm3=float(state.xne[j]),
                            xnatom_cm3=float(state.xnatom[j]),
                            xabund_linear=float(state.xabund[j, id_atomic - 1]),
                            atomic_number=id_atomic,
                            nion=ion,
                            mode=5,
                            chargesq_cm3=float(max(state.chargesq[j], 1e-30)),
                        )
                        pfm = pfsaha_depth(
                            temperature_k=tminus,
                            electron_density_cm3=float(state.xne[j]),
                            xnatom_cm3=float(state.xnatom[j]),
                            xabund_linear=float(state.xabund[j, id_atomic - 1]),
                            atomic_number=id_atomic,
                            nion=ion,
                            mode=5,
                            chargesq_cm3=float(max(state.chargesq[j], 1e-30)),
                        )
                        pfp_ion = float(pfp[ion - 1]) if pfp.size >= ion else 1.0
                        pfm_ion = float(pfm[ion - 1]) if pfm.size >= ion else 1.0
                        pfp_ion = max(pfp_ion, pfm_ion)
                        eion_idx = 30 + ion  # Fortran: EION(ION) == PFP(31+ION), 1-based
                        eion = float(pfp[eion_idx]) if pfp.size > eion_idx else 0.0
                        pffrac = (pfp_ion - pfm_ion) / max(pfp_ion + pfm_ion, 1e-30) * 2.0 * 500.0
                        contrib = xnm * tk * (eion / max(tkev, 1e-300) + pffrac)
                        e += contrib
                        mol_pseudo_sum += contrib
                        if abs(contrib) > abs(max_pseudo_contrib):
                            max_pseudo_contrib = contrib
                            max_pseudo_code = code_val
                            max_pseudo_xnm = xnm
                            max_pseudo_eion = eion
                            max_pseudo_pffrac = pffrac
                            max_pseudo_pfp = pfp_ion
                            max_pseudo_pfm = pfm_ion

            rhoj = max(float(state.rho[j]), 1e-300)
            edens[j] = e / rhoj
            if log_fh is not None and (j + 1) <= max(log_maxj, 0):
                log_fh.write(
                    f"{log_label},{j + 1},{base_e:.8e},{mol_true_sum:.8e},{mol_pseudo_sum:.8e},"
                    f"{e:.8e},{rhoj:.8e},{edens[j]:.8e}\n"
                )
                log_fh.write(
                    f"{log_label}_PSEUDO_MAX,{j + 1},{max_pseudo_code:.8e},{max_pseudo_xnm:.8e},"
                    f"{max_pseudo_eion:.8e},{max_pseudo_pffrac:.8e},{max_pseudo_pfp:.8e},"
                    f"{max_pseudo_pfm:.8e},{max_pseudo_contrib:.8e}\n"
                )
    finally:
        if log_fh is not None:
            log_fh.close()

    return edens

