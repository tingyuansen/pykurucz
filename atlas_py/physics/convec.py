"""ATLAS12 HIGH/CONVEC step.

Fortran references:
- `atlas12.for` lines 5093-5107 (`SUBROUTINE HIGH`)
- `atlas12.for` lines 4849-5092 (`SUBROUTINE CONVEC`)

Notes:
- When EDENS/RHO finite-difference samples are provided, this follows the
  Fortran CONVEC derivative path directly.
- A fallback ideal-gas derivative path is retained for safety when those
  samples are unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .josh_math import _deriv, _integ, _map1
from .tcorr import TcorrState, rosstab_eval


@dataclass
class ConvecResult:
    """Outputs corresponding to ATLAS12 `/CONV/` and `/HEIGHT/` blocks."""

    height: np.ndarray
    dltdlp: np.ndarray
    heatcp: np.ndarray
    dlrdlt: np.ndarray
    velsnd: np.ndarray
    grdadb: np.ndarray
    hscale: np.ndarray
    flxcnv: np.ndarray
    vconv: np.ndarray
    flxcnv0: np.ndarray
    flxcnv1: np.ndarray


def high_from_rhox(*, rhox: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Port of `HIGH`: integrate geometric height from `RHOX` and `RHO`."""
    rhox = np.asarray(rhox, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    rhoinv = 1.0e-5 / np.maximum(rho, 1e-300)
    return _integ(rhox, rhoinv, 0.0)


def convec(
    *,
    tcst: TcorrState,
    rhox: np.ndarray,
    tauros: np.ndarray,
    temperature_k: np.ndarray,
    gas_pressure: np.ndarray,
    mass_density: np.ndarray,
    abross: np.ndarray,
    vturb: np.ndarray,
    pradk: np.ndarray,
    ptotal: np.ndarray,
    gravity_cgs: float,
    flux: float,
    mixlth: float = 1.0,
    overwt: float = 1.0,
    ifconv: int = 1,
    nconv: int = 36,
    edens1: np.ndarray | None = None,
    edens2: np.ndarray | None = None,
    edens3: np.ndarray | None = None,
    edens4: np.ndarray | None = None,
    rho1: np.ndarray | None = None,
    rho2: np.ndarray | None = None,
    rho3: np.ndarray | None = None,
    rho4: np.ndarray | None = None,
    convec_log_path: str | None = None,
) -> ConvecResult:
    """Compute convection arrays for one iteration (ATLAS12 CONVEC structure)."""
    rhox = np.asarray(rhox, dtype=np.float64)
    tauros = np.asarray(tauros, dtype=np.float64)
    t = np.asarray(temperature_k, dtype=np.float64)
    p = np.asarray(gas_pressure, dtype=np.float64)
    rho = np.asarray(mass_density, dtype=np.float64)
    abross = np.asarray(abross, dtype=np.float64)
    vturb = np.asarray(vturb, dtype=np.float64)
    pradk = np.asarray(pradk, dtype=np.float64)
    ptotal = np.asarray(ptotal, dtype=np.float64)

    n = int(t.size)
    dtdrhx = _deriv(rhox, t)
    dilut = 1.0 - np.exp(-tauros)

    dltdlp = np.zeros(n, dtype=np.float64)
    heatcp = np.zeros(n, dtype=np.float64)
    dlrdlt = np.zeros(n, dtype=np.float64)
    velsnd = np.zeros(n, dtype=np.float64)
    grdadb = np.zeros(n, dtype=np.float64)
    hscale = np.zeros(n, dtype=np.float64)
    flxcnv = np.zeros(n, dtype=np.float64)
    vconv = np.zeros(n, dtype=np.float64)
    flxcnv0 = np.zeros(n, dtype=np.float64)
    flxcnv1 = np.zeros(n, dtype=np.float64)
    deltat = np.zeros(n, dtype=np.float64)
    rosst = np.zeros(n, dtype=np.float64)

    use_fd = all(
        x is not None for x in (edens1, edens2, edens3, edens4, rho1, rho2, rho3, rho4)
    )
    if use_fd:
        edens1 = np.asarray(edens1, dtype=np.float64)
        edens2 = np.asarray(edens2, dtype=np.float64)
        edens3 = np.asarray(edens3, dtype=np.float64)
        edens4 = np.asarray(edens4, dtype=np.float64)
        rho1 = np.asarray(rho1, dtype=np.float64)
        rho2 = np.asarray(rho2, dtype=np.float64)
        rho3 = np.asarray(rho3, dtype=np.float64)
        rho4 = np.asarray(rho4, dtype=np.float64)

    def _nz_signed(x: float, eps: float = 1e-300) -> float:
        if abs(x) >= eps:
            return x
        return eps if x >= 0.0 else -eps

    log_fh = None
    if convec_log_path:
        log_path = Path(convec_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = log_path.open("w", encoding="utf-8")
        log_fh.write(
            "J,EDENS1,EDENS2,EDENS3,EDENS4,RHO1,RHO2,RHO3,RHO4,"
            "DEDT,DRDT,DEDPG,DRDPG,DLTDLP,GRDADB,DEL,HEATCP,DLRDLT,FLXCNV,REASON\n"
        )

    try:
        for j in range(n):
            delt = 0.0
            if use_fd:
                dedt = (edens1[j] - edens2[j]) / np.maximum(t[j], 1e-300) * 500.0
                drdt = (rho1[j] - rho2[j]) / np.maximum(t[j], 1e-300) * 500.0
                dedpg = (edens3[j] - edens4[j]) / np.maximum(p[j], 1e-300) * 500.0
                drdpg = (rho3[j] - rho4[j]) / np.maximum(p[j], 1e-300) * 500.0
            else:
                # Fallback approximation when EOS finite-difference EDENS is unavailable.
                rspec = p[j] / np.maximum(rho[j] * t[j], 1e-300)
                dedt = 1.5 * rspec
                drdt = -rho[j] / np.maximum(t[j], 1e-300)
                dedpg = 0.0
                drdpg = rho[j] / np.maximum(p[j], 1e-300)

            dpdpg = 1.0
            dpdt = 4.0 * pradk[j] / np.maximum(t[j], 1e-300) * dilut[j]
            dltdlp[j] = ptotal[j] / np.maximum(t[j] * gravity_cgs, 1e-300) * dtdrhx[j]
            drdpg_safe = _nz_signed(float(drdpg))
            heatcv = dedt - dedpg * drdt / drdpg_safe
            heatcp[j] = (
                dedt
                - dedpg * dpdt / np.maximum(dpdpg, 1e-300)
                - ptotal[j]
                / np.maximum(rho[j] ** 2, 1e-300)
                * (drdt - drdpg * dpdt / np.maximum(dpdpg, 1e-300))
            )
            if heatcv > 0.0:
                velsnd[j] = np.sqrt(
                    max(heatcp[j] / heatcv * dpdpg / drdpg_safe, 0.0)
                )
            dlrdlt[j] = t[j] / np.maximum(rho[j], 1e-300) * (
                drdt - drdpg * dpdt / np.maximum(dpdpg, 1e-300)
            )
            if abs(heatcp[j]) > 1e-300:
                grdadb[j] = (
                    -ptotal[j]
                    / np.maximum(rho[j] * t[j], 1e-300)
                    * dlrdlt[j]
                    / heatcp[j]
                )
            hscale[j] = ptotal[j] / np.maximum(rho[j] * gravity_cgs, 1e-300)

            reason = "ACTIVE"
            if mixlth == 0.0:
                reason = "MIXLTH0"
            elif j < 3:
                reason = "JLT4"
            else:
                delt = dltdlp[j] - grdadb[j]
                if delt < 0.0:
                    reason = "DELLT0"
                else:
                    vco = 0.5 * mixlth * np.sqrt(
                        max(
                            -0.5
                            * ptotal[j]
                            / np.maximum(rho[j], 1e-300)
                            * dlrdlt[j],
                            0.0,
                        )
                    )
                    if vco == 0.0:
                        reason = "VCO0"
                    else:
                        fluxco = 0.5 * rho[j] * heatcp[j] * t[j] * mixlth / 12.5664
                        rosst[j] = rosstab_eval(tcst, float(t[j]), float(p[j]))
                        olddelt = 0.0
                        its30 = 30 if ifconv != 0 else 1
                        for _ in range(its30):
                            rosst_denom = _nz_signed(float(rosst[j]))
                            dplus = rosstab_eval(tcst, float(t[j] + deltat[j]), float(p[j])) / rosst_denom
                            dminus = rosstab_eval(tcst, float(t[j] - deltat[j]), float(p[j])) / rosst_denom
                            if dplus == 0.0 or dminus == 0.0:
                                abconv = 0.0
                            else:
                                abconv = 2.0 / (1.0 / dplus + 1.0 / dminus) * abross[j]
                            den1 = abconv * hscale[j] * rho[j]
                            den2 = fluxco * 12.5664
                            if den1 == 0.0 or den2 == 0.0 or vco == 0.0:
                                d = 0.0
                            else:
                                d = 8.0 * 5.6697e-5 * t[j] ** 4 / den1 / den2 / vco
                            taub = abconv * rho[j] * mixlth * hscale[j]
                            d = d * taub**2 / (2.0 + taub**2)
                            d = d**2 / 2.0
                            ddd = (delt / _nz_signed(float(d + delt))) ** 2
                            if ddd < 0.5:
                                delta = 0.5
                                term = 0.5
                                up = -1.0
                                down = 2.0
                                while term > 1.0e-6:
                                    up += 2.0
                                    down += 2.0
                                    term = up / down * ddd * term
                                    delta += term
                            else:
                                delta = (1.0 - np.sqrt(max(1.0 - ddd, 0.0))) / np.maximum(
                                    ddd, 1e-300
                                )
                            delta = delta * delt**2 / _nz_signed(float(d + delt))
                            vconv[j] = vco * np.sqrt(max(delta, 0.0))
                            flxcnv[j] = max(fluxco * vconv[j] * delta, 0.0)
                            deltat[j] = t[j] * mixlth * delta
                            deltat[j] = min(deltat[j], t[j] * 0.15)
                            deltat[j] = deltat[j] * 0.7 + olddelt * 0.3
                            if olddelt - 0.5 < deltat[j] < olddelt + 0.5:
                                break
                            olddelt = deltat[j]

            if log_fh is not None:
                e1 = float(edens1[j]) if use_fd else float("nan")
                e2 = float(edens2[j]) if use_fd else float("nan")
                e3 = float(edens3[j]) if use_fd else float("nan")
                e4 = float(edens4[j]) if use_fd else float("nan")
                r1v = float(rho1[j]) if use_fd else float("nan")
                r2v = float(rho2[j]) if use_fd else float("nan")
                r3v = float(rho3[j]) if use_fd else float("nan")
                r4v = float(rho4[j]) if use_fd else float("nan")
                log_fh.write(
                    f"{j + 1:d},{e1:.8e},{e2:.8e},{e3:.8e},{e4:.8e},"
                    f"{r1v:.8e},{r2v:.8e},{r3v:.8e},{r4v:.8e},"
                    f"{dedt:.8e},{drdt:.8e},{dedpg:.8e},{drdpg:.8e},"
                    f"{dltdlp[j]:.8e},{grdadb[j]:.8e},{delt:.8e},{heatcp[j]:.8e},"
                    f"{dlrdlt[j]:.8e},{flxcnv[j]:.8e},{reason}\n"
                )
    finally:
        pass

    flxcnv0[:] = flxcnv
    height = high_from_rhox(rhox=rhox, rho=rho)

    if overwt > 0.0:
        wtcnv = np.min([np.max(flxcnv / np.maximum(flux, 1e-300)), 1.0]) * overwt
        delhgt = np.minimum.reduce(
            [
                hscale * 0.5e-5 * wtcnv,
                np.maximum(height[-1] - height, 0.0),
                np.maximum(height - height[0], 0.0),
            ]
        )
        cnvint = _integ(height, flxcnv, 0.0)
        j0 = max(n // 2 - 1, 0)
        for j in range(j0, n - 1):
            if delhgt[j] == 0.0:
                continue
            cnv1, _ = _map1(height, cnvint, np.asarray([height[j] - delhgt[j]], dtype=np.float64))
            cnv2, _ = _map1(height, cnvint, np.asarray([height[j] + delhgt[j]], dtype=np.float64))
            flxcnv1[j] += (cnv2[0] - cnv1[0]) / delhgt[j] / 2.0
        if log_fh is not None:
            for j in range(max(34, 0), n):
                log_fh.write(
                    f"OVRPRE,{j + 1:d},{flxcnv0[j]:.8e},{flxcnv1[j]:.8e},{delhgt[j]:.8e},{wtcnv:.8e}\n"
                )
        flxcnv = np.maximum(flxcnv0, flxcnv1)

    # Fortran patch: zero top NCONV layers.
    k = int(max(min(nconv, n), 0))
    if k > 0:
        flxcnv[:k] = 0.0
    if log_fh is not None:
        for j in range(max(34, 0), n):
            log_fh.write(f"OVRFIN,{j + 1:d},{flxcnv[j]:.8e}\n")
        log_fh.close()

    return ConvecResult(
        height=height,
        dltdlp=dltdlp,
        heatcp=heatcp,
        dlrdlt=dlrdlt,
        velsnd=velsnd,
        grdadb=grdadb,
        hscale=hscale,
        flxcnv=flxcnv,
        vconv=vconv,
        flxcnv0=flxcnv0,
        flxcnv1=flxcnv1,
    )

