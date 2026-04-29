"""TCORR port from ATLAS12 (mode 1/2/3).

Fortran reference:
- `atlas12.for` lines 605-701 (`SUBROUTINE TCORR`)
- `atlas12.for` lines 16024-16069 (`FUNCTION EXPI`)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
try:
    import numba

    _NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - optional acceleration
    numba = None
    _NUMBA_AVAILABLE = False

from .josh_math import _deriv, _integ, _map1


@dataclass
class TcorrState:
    """Frequency-integrated accumulators used by TCORR."""

    rjmins: np.ndarray
    rdabh: np.ndarray
    rdiagj: np.ndarray
    flxrad: np.ndarray
    oldt1: np.ndarray
    ross_tabt: np.ndarray
    ross_tabp: np.ndarray
    ross_logk: np.ndarray
    nross: int
    zerot: float
    zerop: float
    slopet: float
    slopep: float


@dataclass
class TcorrMode3Result:
    """Mode-3 outputs for diagnostics and driver updates.

    Returns corrected T (T+T1) and RHOX (RHOX+DRHOX) on the original TAUROS
    grid.  The driver (atlas_py/engine/driver.py) is responsible for the full
    Fortran TCORR (atlas12.for lines 951-991) state remap: after applying these
    corrections, it MAP1-remaps ALL state arrays (RHOX, T, P, XNE, ABROSS, PRAD,
    ACCRAD, VTURB, PTURB) from TAUROS to TAUSTD and sets TAUROS=TAUSTD.
    """

    temperature: np.ndarray
    flxerr: np.ndarray
    flxdrv: np.ndarray
    dtflux: np.ndarray
    dtlamb: np.ndarray
    dtsurf: np.ndarray
    t1: np.ndarray
    hratio: np.ndarray
    cnvflx: np.ndarray
    rhox: np.ndarray
    drhox: np.ndarray


def _expi(n: int, x: float) -> float:
    """ATLAS12 EXPI(N,X) approximation for x > 0."""
    a0, a1, a2, a3, a4, a5 = (
        -44178.5471728217,
        57721.7247139444,
        9938.31388962037,
        1842.11088668,
        101.093806161906,
        5.03416184097568,
    )
    b0, b1, b2, b3, b4 = (
        76537.3323337614,
        32597.1881290275,
        6106.10794245759,
        635.419418378382,
        37.2298352833327,
    )
    c0, c1, c2, c3, c4, c5, c6 = (
        4.65627107975096e-7,
        0.999979577051595,
        9.04161556946329,
        24.3784088791317,
        23.0192559391333,
        6.90522522784444,
        0.430967839469389,
    )
    d1, d2, d3, d4, d5, d6 = (
        10.0411643829054,
        32.4264210695138,
        41.2807841891424,
        20.4494785013794,
        3.31909213593302,
        0.103400130404874,
    )
    e0, e1, e2, e3, e4, e5, e6 = (
        -0.999999999998447,
        -26.6271060431811,
        -241.055827097015,
        -895.927957772937,
        -1298.85688746484,
        -545.374158883133,
        -5.66575206533869,
    )
    f1, f2, f3, f4, f5, f6 = (
        28.6271060422192,
        292.310039388533,
        1332.78537748257,
        2777.61949509163,
        2404.01713225909,
        631.6574832808,
    )
    if x <= 0.0:
        ex1 = 0.0
    else:
        ex = np.exp(-x)
        if x > 4.0:
            ex1 = (
                ex
                + ex
                * (e0 + (e1 + (e2 + (e3 + (e4 + (e5 + e6 / x) / x) / x) / x) / x) / x)
                / (x + f1 + (f2 + (f3 + (f4 + (f5 + f6 / x) / x) / x) / x) / x)
            ) / x
        elif x > 1.0:
            ex1 = ex * (c6 + (c5 + (c4 + (c3 + (c2 + (c1 + c0 * x) * x) * x) * x) * x) * x) / (
                d6 + (d5 + (d4 + (d3 + (d2 + (d1 + x) * x) * x) * x) * x) * x
            )
        else:
            ex1 = (a0 + (a1 + (a2 + (a3 + (a4 + a5 * x) * x) * x) * x) * x) / (
                b0 + (b1 + (b2 + (b3 + (b4 + x) * x) * x) * x) * x
            ) - np.log(x)
    out = ex1
    for i in range(1, max(n, 1)):
        out = (np.exp(-x) - x * out) / float(i)
    return float(out)


if _NUMBA_AVAILABLE:
    @numba.njit(cache=True)
    def _expi_nb(n: int, x: float) -> float:
        a0, a1, a2, a3, a4, a5 = (
            -44178.5471728217,
            57721.7247139444,
            9938.31388962037,
            1842.11088668,
            101.093806161906,
            5.03416184097568,
        )
        b0, b1, b2, b3, b4 = (
            76537.3323337614,
            32597.1881290275,
            6106.10794245759,
            635.419418378382,
            37.2298352833327,
        )
        c0, c1, c2, c3, c4, c5, c6 = (
            4.65627107975096e-7,
            0.999979577051595,
            9.04161556946329,
            24.3784088791317,
            23.0192559391333,
            6.90522522784444,
            0.430967839469389,
        )
        d1, d2, d3, d4, d5, d6 = (
            10.0411643829054,
            32.4264210695138,
            41.2807841891424,
            20.4494785013794,
            3.31909213593302,
            0.103400130404874,
        )
        e0, e1, e2, e3, e4, e5, e6 = (
            -0.999999999998447,
            -26.6271060431811,
            -241.055827097015,
            -895.927957772937,
            -1298.85688746484,
            -545.374158883133,
            -5.66575206533869,
        )
        f1, f2, f3, f4, f5, f6 = (
            28.6271060422192,
            292.310039388533,
            1332.78537748257,
            2777.61949509163,
            2404.01713225909,
            631.6574832808,
        )
        if x <= 0.0:
            ex1 = 0.0
        else:
            ex = np.exp(-x)
            if x > 4.0:
                ex1 = (
                    ex
                    + ex
                    * (e0 + (e1 + (e2 + (e3 + (e4 + (e5 + e6 / x) / x) / x) / x) / x) / x)
                    / (x + f1 + (f2 + (f3 + (f4 + (f5 + f6 / x) / x) / x) / x) / x)
                ) / x
            elif x > 1.0:
                ex1 = ex * (c6 + (c5 + (c4 + (c3 + (c2 + (c1 + c0 * x) * x) * x) * x) * x) * x) / (
                    d6 + (d5 + (d4 + (d3 + (d2 + (d1 + x) * x) * x) * x) * x) * x
                )
            else:
                ex1 = (a0 + (a1 + (a2 + (a3 + (a4 + a5 * x) * x) * x) * x) * x) / (
                    b0 + (b1 + (b2 + (b3 + (b4 + x) * x) * x) * x) * x
                ) - np.log(x)
        out = ex1
        for i in range(1, max(n, 1)):
            out = (np.exp(-x) - x * out) / float(i)
        return float(out)


    @numba.njit(cache=True)
    def _tcorr_mode2_nb(
        rjmins: np.ndarray,
        rdabh: np.ndarray,
        rdiagj: np.ndarray,
        flxrad: np.ndarray,
        rcowt: float,
        rhox: np.ndarray,
        abtot: np.ndarray,
        hnu: np.ndarray,
        jmins: np.ndarray,
        taunu: np.ndarray,
        bnu: np.ndarray,
        freq_hz: float,
        hkt: np.ndarray,
        temperature_k: np.ndarray,
        stim: np.ndarray,
        alpha: np.ndarray,
        flux: float,
        teff: float,
        numnu: int,
    ) -> None:
        dabtot = _deriv(rhox, abtot)
        n = temperature_k.size
        for j in range(n):
            den_ab = abtot[j] if abtot[j] >= 1.0e-300 else 1.0e-300
            rdabh[j] += dabtot[j] / den_ab * hnu[j] * rcowt
            rjmins[j] += abtot[j] * jmins[j] * rcowt
            flxrad[j] += hnu[j] * rcowt

        term2 = 0.0
        for j in range(n):
            term1 = term2
            d = 1.0e-10
            if j != n - 1:
                d = taunu[j + 1] - taunu[j]
            if d < 1.0e-10:
                d = 1.0e-10
            if d <= 0.01:
                term2 = (0.922784335098467 - np.log(d)) * d / 4.0 + d * d / 12.0 - d ** 3 / 96.0 + d ** 4 / 720.0
            else:
                ex = 0.0
                if d < 10.0:
                    ex = _expi_nb(3, d)
                if teff <= 4250.0 and d > 0.005 and d < 0.02:
                    ex = 0.0
                term2 = 0.5 * (d + ex - 0.5) / d
            diagj = term1 + term2
            den = temperature_k[j] * stim[j]
            if den < 1.0e-300:
                den = 1.0e-300
            dbdt = bnu[j] * freq_hz * hkt[j] / den
            if numnu == 1:
                temp_den = temperature_k[j] if temperature_k[j] >= 1.0e-300 else 1.0e-300
                dbdt = flux * 16.0 / temp_den
            den_diag = 1.0 - alpha[j] * diagj
            if den_diag < 1.0e-300:
                den_diag = 1.0e-300
            rdiagj[j] += abtot[j] * (diagj - 1.0) / den_diag * (1.0 - alpha[j]) * dbdt * rcowt


def init_tcorr(n_layers: int) -> TcorrState:
    z = np.zeros(n_layers, dtype=np.float64)
    nmax = n_layers * 60
    return TcorrState(
        rjmins=z.copy(),
        rdabh=z.copy(),
        rdiagj=z.copy(),
        flxrad=z.copy(),
        oldt1=z.copy(),
        ross_tabt=np.zeros(nmax, dtype=np.float64),
        ross_tabp=np.zeros(nmax, dtype=np.float64),
        ross_logk=np.zeros(nmax, dtype=np.float64),
        nross=0,
        zerot=0.0,
        zerop=0.0,
        slopet=1.0,
        slopep=1.0,
    )


def _map1_scalar(xold: np.ndarray, fold: np.ndarray, xnew: float) -> float:
    arr = np.asarray([xnew], dtype=np.float64)
    out, _ = _map1(np.asarray(xold, dtype=np.float64), np.asarray(fold, dtype=np.float64), arr)
    return float(out[0])


def rosstab_ingest(st: TcorrState, t: np.ndarray, p: np.ndarray, abross: np.ndarray) -> None:
    """Mimic `ROSSTAB(0,0,0)` table accumulation from ATLAS12."""
    n = int(t.size)
    if st.nross == 0:
        st.zerot = np.log10(max(float(t[0]), 1e-300))
        st.zerop = np.log10(max(float(p[0]), 1e-300))
        st.slopet = np.log10(max(float(t[-1]), 1e-300)) - st.zerot
        st.slopep = np.log10(max(float(p[-1]), 1e-300)) - st.zerop
        if abs(st.slopet) < 1e-300:
            st.slopet = 1.0
        if abs(st.slopep) < 1e-300:
            st.slopep = 1.0
    for j in range(n):
        if st.nross >= st.ross_tabt.size:
            break
        st.ross_tabt[st.nross] = (np.log10(max(float(t[j]), 1e-300)) - st.zerot) / st.slopet
        st.ross_tabp[st.nross] = (np.log10(max(float(p[j]), 1e-300)) - st.zerop) / st.slopep
        st.ross_logk[st.nross] = np.log10(max(float(abross[j]), 1e-300))
        st.nross += 1


def rosstab_eval(st: TcorrState, temp: float, pressure: float) -> float:
    """Evaluate ROSSTAB(T,P,V) interpolation from cached table.

    Fortran reference: atlas12.for lines 1423-1549 (FUNCTION ROSSTAB).
    When the table is empty (NROSS=0), Fortran's DO 21 loop doesn't execute,
    all INDEX* remain 0, and label 30 runs with ROSS(MAX(1,0))=ROSS(1).
    Because NROSS has a DATA statement, gfortran places all locals in static
    storage (implicit SAVE), so ROSS(1) is zero-initialized → R=0 →
    ROSSTAB = 10.**0 = 1.0.
    """
    if st.nross <= 0:
        return 1.0
    templog = (np.log10(max(temp, 1e-300)) - st.zerot) / st.slopet
    presslog = (np.log10(max(pressure, 1e-300)) - st.zerop) / st.slopep

    rpp = rpm = rmp = rmm = 1.0e30
    i_pp = i_pm = i_mp = i_mm = -1
    v_pp = v_pm = v_mp = v_mm = 0.0
    for i in range(st.nross):
        dp = st.ross_tabp[i] - presslog
        dt = st.ross_tabt[i] - templog
        radius2 = dt * dt + dp * dp
        if dt >= 0.0 and dp >= 0.0:
            if radius2 < rpp:
                rpp = radius2
                i_pp = i
                v_pp = st.ross_logk[i]
        elif dt >= 0.0 and dp < 0.0:
            if radius2 < rpm:
                rpm = radius2
                i_pm = i
                v_pm = st.ross_logk[i]
        elif dt < 0.0 and dp >= 0.0:
            if radius2 < rmp:
                rmp = radius2
                i_mp = i
                v_mp = st.ross_logk[i]
        else:
            if radius2 < rmm:
                rmm = radius2
                i_mm = i
                v_mm = st.ross_logk[i]

    if i_pp >= 0 and i_pm >= 0 and i_mp >= 0 and i_mm >= 0:
        tpp, ppp, rvpp = st.ross_tabt[i_pp], st.ross_tabp[i_pp], v_pp
        tpm, ppm, rvpm = st.ross_tabt[i_pm], st.ross_tabp[i_pm], v_pm
        tmp, pmp, rvmp = st.ross_tabt[i_mp], st.ross_tabp[i_mp], v_mp
        tmm, pmm, rvmm = st.ross_tabt[i_mm], st.ross_tabp[i_mm], v_mm
        den_tp = max(tpp - tmp, 1e-300)
        den_tm = max(tpm - tmm, 1e-300)
        rppmp = ((templog - tmp) * rvpp + (tpp - templog) * rvmp) / den_tp
        rpmmm = ((templog - tmm) * rvpm + (tpm - templog) * rvmm) / den_tm
        pppmp = ((templog - tmp) * ppp + (tpp - templog) * pmp) / den_tp
        ppmmm = ((templog - tmm) * ppm + (tpm - templog) * pmm) / den_tm
        r = ((presslog - ppmmm) * rppmp + (pppmp - presslog) * rpmmm) / max(pppmp - ppmmm, 1e-300)
        return float(10.0**r)

    w_pp = 1.0 / (np.sqrt(rpp) + 1.0e-5)
    w_pm = 1.0 / (np.sqrt(rpm) + 1.0e-5)
    w_mp = 1.0 / (np.sqrt(rmp) + 1.0e-5)
    w_mm = 1.0 / (np.sqrt(rmm) + 1.0e-5)
    rwt = w_pp + w_pm + w_mp + w_mm
    i_pp = max(i_pp, 0)
    i_pm = max(i_pm, 0)
    i_mp = max(i_mp, 0)
    i_mm = max(i_mm, 0)
    r = (
        st.ross_logk[i_pp] * w_pp
        + st.ross_logk[i_pm] * w_pm
        + st.ross_logk[i_mp] * w_mp
        + st.ross_logk[i_mm] * w_mm
    ) / max(rwt, 1e-300)
    return float(10.0**r)


def _ttaup(
    *,
    st: TcorrState,
    t: np.ndarray,
    tau: np.ndarray,
    prad: np.ndarray,
    pturb: np.ndarray,
    gravity_cgs: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Port of `SUBROUTINE TTAUP` from ATLAS12."""
    n = int(t.size)
    abstd = np.zeros(n, dtype=np.float64)
    ptotal = np.zeros(n, dtype=np.float64)
    pgas = np.zeros(n, dtype=np.float64)
    dlg_tau = np.log(max(float(tau[1] / np.maximum(tau[0], 1e-300)), 1e-300)) if n > 1 else 0.0
    plog3 = plog2 = plog1 = 0.0
    dplog2 = dplog1 = 0.0

    abstd[0] = 0.1
    if prad[0] > 0.0:
        abstd[0] = min(0.1, gravity_cgs * tau[0] / np.maximum(prad[0], 1e-300) / 2.0)

    for j in range(n):
        if j == 0:
            plog = np.log(max(gravity_cgs / np.maximum(abstd[0], 1e-300) * tau[0], 1e-300))
        elif j <= 3:
            plog = plog1 + dplog1
        else:
            plog = (3.0 * plog4 + 8.0 * dplog1 - 4.0 * dplog2 + 8.0 * dplog3) / 3.0
        # Fortran sets ERROR=1, N=1, then GO TO 21 — i.e. it evaluates PTOTAL/ROSSTAB/DPLOG
        # BEFORE computing PNEW and before any convergence check.  Python must match this
        # "label-21-first" flow so that abstd is updated before the first convergence test.
        error = 1.0
        dplog = 0.0
        itn = 1
        while True:
            # FORTRAN LABEL 21: evaluate ptotal, rosstab, dplog first.
            plog = min(plog, 709.78)
            ptotal[j] = np.exp(plog)
            pgas[j] = ptotal[j] + (prad[0] - prad[j]) - pturb[j]
            if pgas[j] <= 0.0:
                pgas[j] = 1e-30
                abstd[j] = 0.1
                break
            abstd[j] = rosstab_eval(st, float(t[j]), float(pgas[j]))
            dplog = gravity_cgs / np.maximum(abstd[j], 1e-300) * tau[j] / np.maximum(ptotal[j], 1e-300) * dlg_tau
            itn += 1
            if itn > 1000 or error <= 5.0e-5:
                break
            # FORTRAN LABEL 20: compute pnew using the just-updated abstd/dplog.
            if j == 0:
                pnew = np.log(max(gravity_cgs / np.maximum(abstd[j], 1e-300) * tau[j], 1e-300))
            elif j <= 3:
                pnew = (plog + 2.0 * plog1 + dplog + dplog1) / 3.0
            else:
                pnew = (
                    126.0 * plog1
                    - 14.0 * plog3
                    + 9.0 * plog4
                    + 42.0 * dplog
                    + 108.0 * dplog1
                    - 54.0 * dplog2
                    + 24.0 * dplog3
                ) / 121.0
            error = abs(pnew - plog)
            plog = 0.5 * (pnew + plog)
        plog4 = plog3
        plog3 = plog2
        plog2 = plog1
        plog1 = plog
        dplog3 = dplog2
        dplog2 = dplog1
        dplog1 = dplog

    return abstd, ptotal, pgas


def tcorr_step(
    st: TcorrState,
    *,
    mode: int,
    rcowt: float,
    rhox: np.ndarray,
    abtot: np.ndarray,
    hnu: np.ndarray,
    jmins: np.ndarray,
    taunu: np.ndarray,
    bnu: np.ndarray,
    freq_hz: float,
    hkt: np.ndarray,
    temperature_k: np.ndarray,
    stim: np.ndarray,
    alpha: np.ndarray,
    flux: float,
    teff: float,
    numnu: int,
    tauros: np.ndarray | None = None,
    abross: np.ndarray | None = None,
    iter_index: int = 1,
    ifconv: int = 0,
    flxcnv: np.ndarray | None = None,
    flxcnv0: np.ndarray | None = None,
    dltdlp: np.ndarray | None = None,
    grdadb: np.ndarray | None = None,
    hscale: np.ndarray | None = None,
    ptotal: np.ndarray | None = None,
    rho: np.ndarray | None = None,
    dlrdlt: np.ndarray | None = None,
    heatcp: np.ndarray | None = None,
    mixlth: float = 1.0,
    j1smooth: int = 0,
    j2smooth: int = 0,
    wtjm1: float = 0.3,
    wtj: float = 0.4,
    wtjp1: float = 0.3,
    prad: np.ndarray | None = None,
    pturb: np.ndarray | None = None,
    gravity_cgs: float = 1.0e4,
    steplg: float = 0.125,
    tau1lg: float = -6.875,
) -> TcorrMode3Result | None:
    """Apply TCORR mode updates.

    - Mode 1: zero frequency integrals
    - Mode 2: accumulate frequency integrals
    - Mode 3: apply temperature correction (ATLAS12 lines ~703-947)
    """

    if mode == 1:
        st.rjmins[:] = 0.0
        st.rdabh[:] = 0.0
        st.rdiagj[:] = 0.0
        st.flxrad[:] = 0.0
        return None
    if mode == 2:
        if _NUMBA_AVAILABLE:
            _tcorr_mode2_nb(
                st.rjmins,
                st.rdabh,
                st.rdiagj,
                st.flxrad,
                rcowt,
                np.asarray(rhox, dtype=np.float64),
                np.asarray(abtot, dtype=np.float64),
                np.asarray(hnu, dtype=np.float64),
                np.asarray(jmins, dtype=np.float64),
                np.asarray(taunu, dtype=np.float64),
                np.asarray(bnu, dtype=np.float64),
                freq_hz,
                np.asarray(hkt, dtype=np.float64),
                np.asarray(temperature_k, dtype=np.float64),
                np.asarray(stim, dtype=np.float64),
                np.asarray(alpha, dtype=np.float64),
                flux,
                teff,
                numnu,
            )
            return None
        dabtot = _deriv(np.asarray(rhox, dtype=np.float64), np.asarray(abtot, dtype=np.float64))
        st.rdabh += dabtot / np.maximum(abtot, 1e-300) * hnu * rcowt
        st.rjmins += abtot * jmins * rcowt
        st.flxrad += hnu * rcowt
        term2 = 0.0
        n = int(temperature_k.size)
        for j in range(n):
            term1 = term2
            d = 1e-10
            if j != n - 1:
                d = taunu[j + 1] - taunu[j]
            d = max(1e-10, float(d))
            if d <= 0.01:
                term2 = (0.922784335098467 - np.log(d)) * d / 4.0 + d * d / 12.0 - d**3 / 96.0 + d**4 / 720.0
            else:
                ex = 0.0
                if d < 10.0:
                    ex = _expi(3, d)
                if teff <= 4250.0 and d > 0.005 and d < 0.02:
                    ex = 0.0
                term2 = 0.5 * (d + ex - 0.5) / d
            diagj = term1 + term2
            dbdt = bnu[j] * freq_hz * hkt[j] / np.maximum(temperature_k[j] * stim[j], 1e-300)
            if numnu == 1:
                dbdt = flux * 16.0 / np.maximum(temperature_k[j], 1e-300)
            st.rdiagj[j] += (
                abtot[j]
                * (diagj - 1.0)
                / np.maximum(1.0 - alpha[j] * diagj, 1e-300)
                * (1.0 - alpha[j])
                * dbdt
                * rcowt
            )
        return None
    if mode != 3:
        raise ValueError(f"Unsupported TCORR mode: {mode}")

    if tauros is None or abross is None:
        raise ValueError("TCORR mode 3 requires tauros and abross arrays")

    t = np.asarray(temperature_k, dtype=np.float64).copy()
    rhox = np.asarray(rhox, dtype=np.float64)
    tauros = np.asarray(tauros, dtype=np.float64)
    abross = np.asarray(abross, dtype=np.float64)
    n = int(t.size)

    # Convection-related arrays default to ATLAS block-data zero state.
    if flxcnv is None:
        flxcnv = np.zeros(n, dtype=np.float64)
    if flxcnv0 is None:
        flxcnv0 = np.zeros(n, dtype=np.float64)
    if dltdlp is None:
        dltdlp = np.zeros(n, dtype=np.float64)
    if grdadb is None:
        grdadb = np.zeros(n, dtype=np.float64)
    if hscale is None:
        hscale = np.ones(n, dtype=np.float64)
    if ptotal is None:
        ptotal = np.ones(n, dtype=np.float64)
    if rho is None:
        rho = np.ones(n, dtype=np.float64)
    if dlrdlt is None:
        dlrdlt = np.zeros(n, dtype=np.float64)
    if heatcp is None:
        heatcp = np.zeros(n, dtype=np.float64)
    flxcnv = np.asarray(flxcnv, dtype=np.float64).copy()
    flxcnv0 = np.asarray(flxcnv0, dtype=np.float64)
    dltdlp = np.asarray(dltdlp, dtype=np.float64)
    grdadb = np.asarray(grdadb, dtype=np.float64)
    hscale = np.asarray(hscale, dtype=np.float64)
    ptotal = np.asarray(ptotal, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    dlrdlt = np.asarray(dlrdlt, dtype=np.float64)
    heatcp = np.asarray(heatcp, dtype=np.float64)

    def _nz_signed(x: float, eps: float = 1e-300) -> float:
        if abs(x) >= eps:
            return x
        return eps if x >= 0.0 else -eps

    dtdrhx = _deriv(rhox, t)
    ddlt = _deriv(rhox, dltdlp)
    dabros = _deriv(rhox, abross)

    cnvflx = np.zeros(n, dtype=np.float64)
    if ifconv == 1:
        cnvflx[:] = flxcnv
    if n >= 1:
        cnvflx[0] = 0.0
    if n >= 2:
        cnvflx[1] = 0.0
    if n >= 3:
        ccc = cnvflx.copy()
        for j in range(1, n - 1):
            ccc[j] = 0.25 * cnvflx[j - 1] + 0.5 * cnvflx[j] + 0.25 * cnvflx[j + 1]
        ccc[-1] = 0.25 * cnvflx[-3] + 0.25 * cnvflx[-2] + 0.5 * cnvflx[-1]
        for j in range(1, n - 1):
            cnvflx[j] = ccc[j]
        cnvflx[-1] = ccc[-1]

    rdabh = st.rdabh - st.flxrad * dabros / np.maximum(abross, 1e-300)
    codrhx = np.zeros(n, dtype=np.float64)
    ddel = np.zeros(n, dtype=np.float64)
    for j in range(n):
        delv = 1.0
        d = 0.0
        if cnvflx[j] > 0.0 and flxcnv0[j] > 0.0:
            delv = dltdlp[j] - grdadb[j]
            vco = 0.5 * mixlth * np.sqrt(max(-0.5 * ptotal[j] / max(rho[j], 1e-300) * dlrdlt[j], 0.0))
            fluxco = 0.5 * rho[j] * heatcp[j] * t[j] * mixlth / 12.5664
            if mixlth > 0.0 and vco > 0.0:
                d = (
                    8.0
                    * 5.6697e-5
                    * t[j] ** 4
                    / np.maximum(abross[j] * hscale[j] * rho[j], 1e-300)
                    / np.maximum(fluxco * 12.5664, 1e-300)
                    / vco
                )
            taub = abross[j] * rho[j] * mixlth * hscale[j]
            d = d * taub * taub / (2.0 + taub * taub)
            d = d * d / 2.0
            den_deld = _nz_signed(float(d + delv))
            del_safe = _nz_signed(float(delv))
            ddel[j] = (1.0 + d / den_deld) / del_safe
        cnvfl = 0.0
        if st.flxrad[j] > 0.0:
            if cnvflx[j] / st.flxrad[j] > 1.0e-3 and flxcnv0[j] / st.flxrad[j] > 1.0e-3:
                cnvfl = cnvflx[j]
        den_deld = _nz_signed(float(d + delv))
        del_safe = _nz_signed(float(delv))
        num = rdabh[j] + cnvfl * (
            dtdrhx[j] / np.maximum(t[j], 1e-300) * (1.0 - 9.0 * d / den_deld)
            + 1.5 * ddlt[j] / del_safe * (1.0 + d / den_deld)
        )
        den = st.flxrad[j] + cnvflx[j] * 1.5 * dltdlp[j] * ddel[j]
        codrhx[j] = num / _nz_signed(float(den))
    if n >= 1:
        codrhx[0] = 0.0
    if n >= 2:
        codrhx[1] = 0.0

    g = np.exp(_integ(rhox, codrhx, 0.0))
    gfden = st.flxrad + cnvflx * 1.5 * dltdlp * ddel
    gfden_safe = np.where(np.abs(gfden) >= 1e-300, gfden, np.where(gfden >= 0.0, 1e-300, -1e-300))
    gflux = g * (st.flxrad + cnvflx - flux) / gfden_safe
    dtau = _integ(tauros, gflux, 0.0) / np.maximum(g, 1e-300)
    dtau = np.maximum(-tauros / 3.0, np.minimum(tauros / 3.0, dtau))
    dtflux = -dtau * dtdrhx / np.maximum(abross, 1e-300)

    flxerr = (st.flxrad + cnvflx - flux) / np.maximum(flux, 1e-300) * 100.0
    flxdrv = _deriv(tauros, flxerr)
    dtlamb = np.zeros(n, dtype=np.float64)
    teff25 = teff / 25.0
    for j in range(n):
        ratio = cnvflx[j] / np.maximum(st.flxrad[j], 1e-300)
        if ratio < 1.0e-5:
            flxdrv[j] = st.rjmins[j] / np.maximum(abross[j], 1e-300) / np.maximum(flux, 1e-300) * 100.0
        dtlamb[j] = (
            -flxdrv[j]
            * flux
            / 100.0
            / (st.rdiagj[j] if abs(st.rdiagj[j]) > 1e-300 else np.sign(st.rdiagj[j]) * 1e-300)
            * abross[j]
        )
        if not (ratio < 1.0e-5 and tauros[j] < 1.0):
            dtlamb[j] = 0.0
            for k in range(1, 6):
                jj = j - k
                if jj >= 0:
                    dtlamb[jj] *= 0.5
        dtlamb[j] = float(np.clip(dtlamb[j], -teff25, teff25))

    dtsur = (flux - st.flxrad[0]) / np.maximum(flux, 1e-300) * 0.25 * t[0]
    dtsur = float(np.clip(dtsur, -teff25, teff25))
    dum = dtflux + dtlamb
    tinteg = _integ(tauros, dum, 0.0)
    tone = _map1_scalar(tauros, tinteg, 0.1)
    ttwo = _map1_scalar(tauros, tinteg, 2.0)
    tav = (ttwo - tone) / 2.0
    if dtsur * tav <= 0.0:
        tav = 0.0
    if abs(tav) > abs(dtsur):
        tav = dtsur
    dtsur = dtsur - tav

    dtsurf = np.full(n, dtsur, dtype=np.float64)
    hratio = cnvflx / np.maximum(cnvflx + st.flxrad, 1e-300)
    t1 = dtflux + dtlamb + dtsurf

    for j in range(n):
        skip_damp = False
        if ifconv == 1 and hratio[j] > 0.0:
            skip_damp = True
        if ifconv == 1 and (j + 1) >= (n / 3.0):
            skip_damp = True
        if iter_index == 1:
            skip_damp = True
        if not skip_damp:
            if st.oldt1[j] * t1[j] > 0.0 and abs(st.oldt1[j]) > abs(t1[j]):
                t1[j] *= 1.25
            if st.oldt1[j] * t1[j] < 0.0:
                t1[j] *= 0.5
        st.oldt1[j] = t1[j]

    tnew = t + t1
    if j1smooth > 0:
        jlo = max(j1smooth - 1, 1)
        jhi = min(j2smooth - 1, n - 2)
        if jhi >= jlo:
            tsmooth = tnew.copy()
            for j in range(jlo, jhi + 1):
                tsmooth[j] = wtjm1 * tnew[j - 1] + wtj * tnew[j] + wtjp1 * tnew[j + 1]
            for j in range(jlo, jhi + 1):
                tnew[j] = tsmooth[j]

    for i in range(1, n):
        j = n - 1 - i
        tnew[j] = min(tnew[j], tnew[j + 1] - 1.0)

    if prad is None:
        prad = np.zeros(n, dtype=np.float64)
    if pturb is None:
        pturb = np.zeros(n, dtype=np.float64)
    prad = np.asarray(prad, dtype=np.float64)
    pturb = np.asarray(pturb, dtype=np.float64)

    # atlas12.for lines 856, 865-877: compute DRHOX via TTAUP on TAUSTD grid.
    # Note: Fortran subsequently remaps ALL state arrays (RHOX, T, P, XNE, ABROSS,
    # PRAD, ACCRAD, VTURB, etc.) from TAUROS to TAUSTD (atlas12.for lines 951-991).
    # Python returns only the corrected T and RHOX; the driver handles PRAD/ACCRAD
    # remapping to TAUSTD explicitly after calling this function.
    taustd = 10.0 ** (tau1lg + np.arange(n, dtype=np.float64) * steplg)
    tplus = t + t1

    tnew1, _ = _map1(tauros, t, taustd)
    prdnew, _ = _map1(tauros, prad, taustd)
    _ab1, ptot1, _p1 = _ttaup(
        st=st,
        t=tnew1,
        tau=taustd,
        prad=prdnew,
        pturb=pturb,
        gravity_cgs=gravity_cgs,
    )
    tnew2, _ = _map1(tauros, tplus, taustd)
    _ab2, ptot2, _p2 = _ttaup(
        st=st,
        t=tnew2,
        tau=taustd,
        prad=prdnew,
        pturb=pturb,
        gravity_cgs=gravity_cgs,
    )
    _ = _ab1, _ab2, _p1, _p2
    ppp = (ptot2 - ptot1) / np.maximum(ptot1, 1e-300)
    rrr, _ = _map1(taustd, ppp, tauros)
    drhox = rrr * rhox

    # Fortran applies corrections in-place on the original TAUROS grid
    # (atlas12.for lines 888, 952).  Do NOT remap to TAUSTD — that would
    # introduce interpolation artefacts into P, XNE, ABROSS, etc.
    rhox_new = rhox + drhox

    return TcorrMode3Result(
        temperature=tnew,
        flxerr=flxerr,
        flxdrv=flxdrv,
        dtflux=dtflux,
        dtlamb=dtlamb,
        dtsurf=dtsurf,
        t1=t1,
        hratio=hratio,
        cnvflx=cnvflx,
        rhox=rhox_new,
        drhox=drhox,
    )

