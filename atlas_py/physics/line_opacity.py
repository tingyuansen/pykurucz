"""Line-opacity kernels for atlas12 SELECTLINES/LINOP1/XLINOP."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os

import numpy as np

try:
    import numba
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

from .hydrogen_profile import (
    HydrogenProfileEvaluator,
    _C_LIGHT_A,
    _doppler_profile_nb,
    _lorentz_profile_nb,
    _stark_profile_nb,
    compute_xnh2,
    load_hydrogen_profile_tables_from_atlas12,
)
from .line_opacity_data import load_line_opacity_tables as _load_lo_tables
from .line_selection import SelectedLineRecords, XLineRecords
from .trace_runtime import trace_emit, trace_enabled, trace_in_focus

@dataclass
class LineOpacityState:
    """Runtime line-opacity container mirroring `COMMON /XLINES/` semantics."""

    xlines: np.ndarray  # shape (layers, n_freq)
    lineused: int = 0


def _subset_xline_records(records: XLineRecords, mask: np.ndarray) -> XLineRecords:
    """Return a field-wise subset of decoded `fort.19` records."""

    return XLineRecords(
        wlvac=records.wlvac[mask],
        elo=records.elo[mask],
        gf=records.gf[mask],
        nblo=records.nblo[mask],
        nbup=records.nbup[mask],
        nelion=records.nelion[mask],
        type_code=records.type_code[mask],
        ncon=records.ncon[mask],
        nelionx=records.nelionx[mask],
        gammar=records.gammar[mask],
        gammas=records.gammas[mask],
        gammaw=records.gammaw[mask],
        iwl=records.iwl[mask],
        lim=records.lim[mask],
    )


_RATIOLG = np.log(1.0 + 1.0 / 2_000_000.0)
_CGF_SCALE = 0.026538 / 1.77245 / 2.99792458e17
_GAMMA_SCALE = 1.0 / 12.5664 / 2.99792458e17

_LO_TABLES = _load_lo_tables()
_TABVI = _LO_TABLES["_TABVI"]
_TABH1 = _LO_TABLES["_TABH1"]
_C_NM = 2.99792458e17


@lru_cache(maxsize=1)
def _build_contx() -> np.ndarray:
    """CONTX(25,16) used by XLINOP normal/hydrogen branches."""

    c = np.zeros((25, 16), dtype=np.float64)
    # NELIONX = 1.00
    c[:10, 0] = np.array(
        [109678.764, 27419.659, 12186.462, 6854.871, 4387.113, 3046.604, 2238.320, 1713.711, 1354.044, 1096.776],
        dtype=np.float64,
    )
    # NELIONX = 2.00
    c[:10, 1] = np.array([198310.760, 38454.691, 32033.214, 29223.753, 27175.760, 15073.868, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    # NELIONX = 2.01
    c[:10, 2] = np.array([438908.850, 109726.529, 48766.491, 27430.925, 17555.715, 12191.437, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    # NELIONX = 6.00
    c[:10, 3] = np.array([90883.840, 90867.420, 90840.420, 90820.420, 90804.000, 90777.000, 80691.180, 80627.760, 69235.820, 69172.400], dtype=np.float64)
    # NELIONX = 12.00
    c[:10, 5] = np.array([61671.020, 39820.615, 39800.556, 39759.842, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    # NELIONX = 13.00
    c[:10, 7] = np.array([48278.370, 48166.309, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    # NELIONX = 14.00
    c[:10, 9] = np.array([66035.000, 65957.885, 65811.843, 65747.550, 65670.435, 65524.393, 59736.150, 59448.700, 50640.630, 50553.180], dtype=np.float64)
    return c


@lru_cache(maxsize=1)
def _build_tablog() -> np.ndarray:
    i = np.arange(1, 32769, dtype=np.float64)
    return 10.0 ** ((i - 16384.0) * 0.001)


@lru_cache(maxsize=1)
def _build_exptab() -> tuple[np.ndarray, np.ndarray]:
    i = np.arange(1001, dtype=np.float64)
    return np.exp(-i), np.exp(-i * 0.001)


def _map4(xold: np.ndarray, fold: np.ndarray, xnew: np.ndarray) -> np.ndarray:
    nold = int(xold.size)
    fnew = np.zeros_like(xnew, dtype=np.float64)
    l = 1
    ll = -1
    cfor = bfor = afor = 0.0
    cbac = bbac = abac = 0.0
    for k, xk in enumerate(xnew):
        while l < nold and xk >= xold[l]:
            l += 1
        if l == ll:
            fnew[k] = a + (b + c * xk) * xk
            continue
        if l > 1:
            if l == 2:
                pass
            else:
                l1 = l - 1
                if l <= ll + 1 and l != 3:
                    cbac = cfor
                    bbac = bfor
                    abac = afor
                    if l == nold:
                        c = cbac
                        b = bbac
                        a = abac
                        ll = l
                        fnew[k] = a + (b + c * xk) * xk
                        continue
                else:
                    l2 = l - 2
                    d = (fold[l1] - fold[l2]) / (xold[l1] - xold[l2])
                    cbac = (
                        fold[l]
                        / ((xold[l] - xold[l1]) * (xold[l] - xold[l2]))
                        + (
                            fold[l2] / (xold[l] - xold[l2])
                            - fold[l1] / (xold[l] - xold[l1])
                        )
                        / (xold[l1] - xold[l2])
                    )
                    bbac = d - (xold[l1] + xold[l2]) * cbac
                    abac = fold[l2] - xold[l2] * d + xold[l1] * xold[l2] * cbac
                    if l >= nold:
                        c = cbac
                        b = bbac
                        a = abac
                        ll = l
                        fnew[k] = a + (b + c * xk) * xk
                        continue
                d = (fold[l] - fold[l1]) / (xold[l] - xold[l1])
                cfor = (
                    fold[l + 1]
                    / ((xold[l + 1] - xold[l]) * (xold[l + 1] - xold[l1]))
                    + (
                        fold[l1] / (xold[l + 1] - xold[l1])
                        - fold[l] / (xold[l + 1] - xold[l])
                    )
                    / (xold[l] - xold[l1])
                )
                bfor = d - (xold[l] + xold[l1]) * cfor
                afor = fold[l1] - xold[l1] * d + xold[l] * xold[l1] * cfor
                wt = 0.0
                if abs(cfor) != 0.0:
                    wt = abs(cfor) / (abs(cfor) + abs(cbac))
                a = afor + wt * (abac - afor)
                b = bfor + wt * (bbac - bfor)
                c = cfor + wt * (cbac - cfor)
                ll = l
                fnew[k] = a + (b + c * xk) * xk
                continue
        l = min(nold - 1, l)
        c = 0.0
        b = (fold[l] - fold[l - 1]) / (xold[l] - xold[l - 1])
        a = fold[l] - xold[l] * b
        ll = l
        fnew[k] = a + (b + c * xk) * xk
    return fnew


@lru_cache(maxsize=1)
def _build_h_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vsteps = 200.0
    h0lin = np.arange(2001, dtype=np.float64) / vsteps
    h1 = _map4(_TABVI, _TABH1, h0lin)
    vv = h0lin * h0lin
    h0 = np.exp(-vv)
    h2 = h0 - 2.0 * vv * h0
    return h0, h1, h2


def _fastex(x: float, extab: np.ndarray, extabf: np.ndarray) -> float:
    if x < 0.0 or x >= 1001.0:
        return 0.0
    i = int(x)
    j = int((x - float(i)) * 1000.0 + 1.5)
    if j < 1:
        j = 1
    if j > 1001:
        j = 1001
    return float(extab[i] * extabf[j - 1])


def _voigt(v: float, a: float, h0: np.ndarray, h1: np.ndarray, h2: np.ndarray) -> float:
    iv = int(v * 200.0 + 1.5)
    if iv < 1:
        iv = 1
    if iv > 2001:
        iv = 2001
    i = iv - 1
    if a >= 0.2:
        if a > 1.4 or a + v > 3.2:
            aa = a * a
            vv = v * v
            u = (aa + vv) * 1.4142
            out = a * 0.79788 / u
            if a > 100.0:
                return out
            aau = aa / u
            vvu = vv / u
            uu = u * u
            out = ((((aau - 10.0 * vvu) * aau * 3.0 + 15.0 * vvu * vvu) + 3.0 * vv - aa) / uu + 1.0) * out
            return out
        vv = v * v
        hh1 = h1[i] + h0[i] * 1.12838
        hh2 = h2[i] + hh1 * 1.12838 - h0[i]
        hh3 = (1.0 - h2[i]) * 0.37613 - hh1 * 0.66667 * vv + hh2 * 1.12838
        hh4 = (3.0 * hh3 - hh1) * 0.37613 + h0[i] * 0.66667 * vv * vv
        return (
            ((((hh4 * a + hh3) * a + hh2) * a + hh1) * a + h0[i])
            * (((-0.122727278 * a + 0.532770573) * a - 0.96284325) * a + 0.979895032)
        )
    if v > 10.0:
        return 0.5642 * a / (v * v)
    return (h2[i] * a + h1[i]) * a + h0[i]


def _accumulate_wings(
    *,
    xlines: np.ndarray,
    j0: int,
    nu0: int,
    wlvac: float,
    center: float,
    adamp: float,
    dopwave: float,
    tabcont_ref: float,
    waveset: np.ndarray,
    h0tab: np.ndarray,
    h1tab: np.ndarray,
    h2tab: np.ndarray,
    wing_steps: int,
    blue_cutoff: float | None = None,
    debug_stats: dict[str, int | float] | None = None,
) -> None:
    numnu = waveset.size
    if dopwave <= 0.0:
        return
    ired_max = int(wing_steps)
    ired_hi = min(nu0 + ired_max + 1, numnu)
    if adamp <= 0.2:
        # RED WING
        for iw in range(nu0, ired_hi):
            if debug_stats is not None:
                debug_stats["red_steps"] = int(debug_stats.get("red_steps", 0)) + 1
            vvoigt = float(waveset[iw] - wlvac) / dopwave
            if vvoigt > 10.0:
                cv = center * 0.5642 * adamp / (vvoigt * vvoigt)
            else:
                iv = int(vvoigt * 200.0 + 1.5)
                if iv < 1:
                    iv = 1
                if iv > 2001:
                    iv = 2001
                i = iv - 1
                cv = center * ((h2tab[i] * adamp + h1tab[i]) * adamp + h0tab[i])
            xlines[j0, iw] += cv
            if cv < tabcont_ref:
                if debug_stats is not None:
                    debug_stats["red_break_tabcont"] = int(debug_stats.get("red_break_tabcont", 0)) + 1
                break
        # BLUE WING
        for ired in range(1, ired_max + 1):
            iw = nu0 - ired
            if iw < 0:
                break
            if blue_cutoff is not None and float(waveset[iw]) < blue_cutoff:
                if debug_stats is not None:
                    debug_stats["blue_break_cutoff"] = int(debug_stats.get("blue_break_cutoff", 0)) + 1
                break
            if debug_stats is not None:
                debug_stats["blue_steps"] = int(debug_stats.get("blue_steps", 0)) + 1
            vvoigt = float(wlvac - waveset[iw]) / dopwave
            if vvoigt > 10.0:
                cv = center * 0.5642 * adamp / (vvoigt * vvoigt)
            else:
                iv = int(vvoigt * 200.0 + 1.5)
                if iv < 1:
                    iv = 1
                if iv > 2001:
                    iv = 2001
                i = iv - 1
                cv = center * ((h2tab[i] * adamp + h1tab[i]) * adamp + h0tab[i])
            xlines[j0, iw] += cv
            if cv < tabcont_ref:
                if debug_stats is not None:
                    debug_stats["blue_break_tabcont"] = int(debug_stats.get("blue_break_tabcont", 0)) + 1
                break
        return
    # RED WING
    for iw in range(nu0, ired_hi):
        if debug_stats is not None:
            debug_stats["red_steps"] = int(debug_stats.get("red_steps", 0)) + 1
        cv = center * _voigt(float(waveset[iw] - wlvac) / dopwave, adamp, h0tab, h1tab, h2tab)
        xlines[j0, iw] += cv
        if cv < tabcont_ref:
            if debug_stats is not None:
                debug_stats["red_break_tabcont"] = int(debug_stats.get("red_break_tabcont", 0)) + 1
            break
    # BLUE WING
    for ired in range(1, ired_max + 1):
        iw = nu0 - ired
        if iw < 0:
            break
        if blue_cutoff is not None and float(waveset[iw]) < blue_cutoff:
            if debug_stats is not None:
                debug_stats["blue_break_cutoff"] = int(debug_stats.get("blue_break_cutoff", 0)) + 1
            break
        if debug_stats is not None:
            debug_stats["blue_steps"] = int(debug_stats.get("blue_steps", 0)) + 1
        cv = center * _voigt(float(wlvac - waveset[iw]) / dopwave, adamp, h0tab, h1tab, h2tab)
        xlines[j0, iw] += cv
        if cv < tabcont_ref:
            if debug_stats is not None:
                debug_stats["blue_break_tabcont"] = int(debug_stats.get("blue_break_tabcont", 0)) + 1
            break





# ---------------------------------------------------------------------------
# Numba-accelerated LINOP1 kernel (JIT-compiled to near-Fortran speed).
# When numba is unavailable the code falls back to the pure-Python path.
# ---------------------------------------------------------------------------

if _NUMBA_AVAILABLE:
    _njit = numba.njit(cache=True)
    _njit_parallel = numba.njit(cache=True, parallel=True)

    @_njit
    def _fastex_nb(x, extab, extabf):
        if x < 0.0 or x >= 1001.0:
            return 0.0
        i = int(x)
        j = int((x - float(i)) * 1000.0 + 1.5)
        if j < 1:
            j = 1
        if j > 1001:
            j = 1001
        return extab[i] * extabf[j - 1]

    @_njit
    def _voigt_nb(v, a, h0, h1, h2):
        iv = int(v * 200.0 + 1.5)
        if iv < 1:
            iv = 1
        if iv > 2001:
            iv = 2001
        i = iv - 1
        if a >= 0.2:
            if a > 1.4 or a + v > 3.2:
                aa = a * a
                vv = v * v
                u = (aa + vv) * 1.4142
                out = a * 0.79788 / u
                if a > 100.0:
                    return out
                aau = aa / u
                vvu = vv / u
                uu = u * u
                out = ((((aau - 10.0 * vvu) * aau * 3.0 + 15.0 * vvu * vvu) + 3.0 * vv - aa) / uu + 1.0) * out
                return out
            vv = v * v
            hh1 = h1[i] + h0[i] * 1.12838
            hh2 = h2[i] + hh1 * 1.12838 - h0[i]
            hh3 = (1.0 - h2[i]) * 0.37613 - hh1 * 0.66667 * vv + hh2 * 1.12838
            hh4 = (3.0 * hh3 - hh1) * 0.37613 + h0[i] * 0.66667 * vv * vv
            return (
                ((((hh4 * a + hh3) * a + hh2) * a + hh1) * a + h0[i])
                * (((-0.122727278 * a + 0.532770573) * a - 0.96284325) * a + 0.979895032)
            )
        if v > 10.0:
            return 0.5642 * a / (v * v)
        return (h2[i] * a + h1[i]) * a + h0[i]

    @_njit
    def _accwings_nb(xlines, j0, nu0, wlvac, center, adamp, dopwave, tabcont_ref, waveset, h0tab, h1tab, h2tab):
        """_accumulate_wings without blue_cutoff (linop1 never sets it).

        Fortran LINOP1 uses IMPLICIT REAL*4 — all intermediates (VVOIGT, CV)
        are float32.  VVOIGT = SNGL(WAVESET-WLVAC)/DOPWAVE.
        """
        numnu = waveset.shape[0]
        if dopwave <= 0.0:
            return
        ired_max = 100
        ired_hi = min(nu0 + ired_max + 1, numnu)
        f32_5642 = np.float32(0.5642)
        if adamp <= 0.2:
            for iw in range(nu0, ired_hi):
                vvoigt = np.float32(waveset[iw] - wlvac) / dopwave
                if vvoigt > np.float32(10.0):
                    cv = center * f32_5642 * adamp / (vvoigt * vvoigt)
                else:
                    iv = int(vvoigt * np.float32(200.0) + np.float32(1.5))
                    if iv < 1:
                        iv = 1
                    if iv > 2001:
                        iv = 2001
                    ii = iv - 1
                    cv = center * ((h2tab[ii] * adamp + h1tab[ii]) * adamp + h0tab[ii])
                xlines[j0, iw] += cv
                if cv < tabcont_ref:
                    break
            for ired in range(1, ired_max + 1):
                iw = nu0 - ired
                if iw < 0:
                    break
                vvoigt = np.float32(wlvac - waveset[iw]) / dopwave
                if vvoigt > np.float32(10.0):
                    cv = center * f32_5642 * adamp / (vvoigt * vvoigt)
                else:
                    iv = int(vvoigt * np.float32(200.0) + np.float32(1.5))
                    if iv < 1:
                        iv = 1
                    if iv > 2001:
                        iv = 2001
                    ii = iv - 1
                    cv = center * ((h2tab[ii] * adamp + h1tab[ii]) * adamp + h0tab[ii])
                xlines[j0, iw] += cv
                if cv < tabcont_ref:
                    break
            return
        for iw in range(nu0, ired_hi):
            cv = np.float32(center * _voigt_nb(np.float32(waveset[iw] - wlvac) / dopwave, adamp, h0tab, h1tab, h2tab))
            xlines[j0, iw] += cv
            if cv < tabcont_ref:
                break
        for ired in range(1, ired_max + 1):
            iw = nu0 - ired
            if iw < 0:
                break
            cv = np.float32(center * _voigt_nb(np.float32(wlvac - waveset[iw]) / dopwave, adamp, h0tab, h1tab, h2tab))
            xlines[j0, iw] += cv
            if cv < tabcont_ref:
                break

    @_njit
    def _accwings_cutoff_nb(
        xlines, j0, nu0, wlvac, center, adamp, dopwave, tabcont_ref,
        waveset, h0tab, h1tab, h2tab, wing_steps, blue_cutoff,
    ):
        numnu = waveset.shape[0]
        if dopwave <= 0.0:
            return
        ired_max = int(wing_steps)
        ired_hi = min(nu0 + ired_max + 1, numnu)
        if adamp <= 0.2:
            for iw in range(nu0, ired_hi):
                vvoigt = float(waveset[iw] - wlvac) / dopwave
                if vvoigt > 10.0:
                    cv = center * 0.5642 * adamp / (vvoigt * vvoigt)
                else:
                    iv = int(vvoigt * 200.0 + 1.5)
                    if iv < 1:
                        iv = 1
                    if iv > 2001:
                        iv = 2001
                    ii = iv - 1
                    cv = center * ((h2tab[ii] * adamp + h1tab[ii]) * adamp + h0tab[ii])
                xlines[j0, iw] += cv
                if cv < tabcont_ref:
                    break
            for ired in range(1, ired_max + 1):
                iw = nu0 - ired
                if iw < 0:
                    break
                if blue_cutoff > 0.0 and waveset[iw] < blue_cutoff:
                    break
                vvoigt = float(wlvac - waveset[iw]) / dopwave
                if vvoigt > 10.0:
                    cv = center * 0.5642 * adamp / (vvoigt * vvoigt)
                else:
                    iv = int(vvoigt * 200.0 + 1.5)
                    if iv < 1:
                        iv = 1
                    if iv > 2001:
                        iv = 2001
                    ii = iv - 1
                    cv = center * ((h2tab[ii] * adamp + h1tab[ii]) * adamp + h0tab[ii])
                xlines[j0, iw] += cv
                if cv < tabcont_ref:
                    break
            return
        for iw in range(nu0, ired_hi):
            cv = center * _voigt_nb(float(waveset[iw] - wlvac) / dopwave, adamp, h0tab, h1tab, h2tab)
            xlines[j0, iw] += cv
            if cv < tabcont_ref:
                break
        for ired in range(1, ired_max + 1):
            iw = nu0 - ired
            if iw < 0:
                break
            if blue_cutoff > 0.0 and waveset[iw] < blue_cutoff:
                break
            cv = center * _voigt_nb(float(wlvac - waveset[iw]) / dopwave, adamp, h0tab, h1tab, h2tab)
            xlines[j0, iw] += cv
            if cv < tabcont_ref:
                break

    @_njit
    def _linop1_kernel_nb(
        n_records,
        iwl_arr, ielion_arr, ielo_arr, igflog_arr, igr_arr, igs_arr, igw_arr,
        waveset, iwavetab, tab,
        hckt_arr, xne_arr, xnfdop_arr, dopple_arr, txnxn,
        extab, extabf, tablog,
        h0tab, h1tab, h2tab,
        nrhox, numnu, nuhi_eff, nulo,
        start, stop,
        ratiolg, cgf_scale, gamma_scale,
    ):
        xlines = np.zeros((nrhox, numnu), dtype=np.float32)
        ifj = np.zeros(nrhox + 2, dtype=np.int32)

        nucont0 = 0
        nu0 = max(0, nulo - 1)
        iwlold = 0
        lineused = 0

        for iline in range(n_records):
            iwl = int(iwl_arr[iline])
            if iwl < iwlold:
                nucont0 = 0
                nu0 = max(0, nulo - 1)

            while nucont0 < iwavetab.shape[0] and iwl >= int(iwavetab[nucont0]):
                nucont0 += 1
            if nucont0 >= tab.shape[1]:
                iwlold = iwl
                continue

            nelion = abs(int(ielion_arr[iline])) // 10
            if nelion < 1 or nelion > xnfdop_arr.shape[1]:
                iwlold = iwl
                continue

            wlvac = np.exp(float(iwl) * ratiolg)
            if wlvac < start or wlvac > stop:
                iwlold = iwl
                continue
            wlvac4 = np.float32(wlvac)

            while nu0 < numnu and wlvac >= waveset[nu0]:
                nu0 += 1
            if nu0 >= numnu:
                iwlold = iwl
                continue

            igflog = int(igflog_arr[iline])
            ielo = int(ielo_arr[iline])
            igr_v = int(igr_arr[iline])
            igs_v = int(igs_arr[iline])
            igw_v = int(igw_arr[iline])
            if igflog < 1 or ielo < 1 or igr_v < 1 or igs_v < 1 or igw_v < 1:
                iwlold = iwl
                continue
            tl = tablog.shape[0]
            if igflog > tl or ielo > tl or igr_v > tl or igs_v > tl or igw_v > tl:
                iwlold = iwl
                continue

            cgf = cgf_scale * wlvac4 * tablog[igflog - 1]
            elo_val = tablog[ielo - 1]
            ifline = 0
            adamp_seed = 0.0
            gammar = 0.0
            gammas = 0.0
            gammaw = 0.0

            for j1 in range(8, nrhox + 1, 8):
                ifj[j1 + 1] = 0
                j0 = j1 - 1
                center = cgf * xnfdop_arr[j0, nelion - 1]
                if center < tab[j0, nucont0]:
                    continue
                center = center * _fastex_nb(elo_val * hckt_arr[j0], extab, extabf)
                if center < tab[j0, nucont0]:
                    continue
                ifj[j1 + 1] = 1
                ifline = 1
                if adamp_seed == 0.0:
                    gammar = tablog[igr_v - 1] * wlvac4 * gamma_scale
                    gammas = tablog[igs_v - 1] * wlvac4 * gamma_scale
                    gammaw = tablog[igw_v - 1] * wlvac4 * gamma_scale
                    adamp_seed = 1.0
                dop = dopple_arr[j0, nelion - 1]
                if dop <= 0.0:
                    continue
                adamp = (gammar + gammas * xne_arr[j0] + gammaw * txnxn[j0]) / dop
                dopwave = dop * wlvac4
                _accwings_nb(
                    xlines, j0, nu0, wlvac, center, adamp, dopwave,
                    tab[j0, nucont0], waveset, h0tab, h1tab, h2tab,
                )

            for k1 in range(8, nrhox + 1, 8):
                if ifj[k1 - 7] + ifj[k1 + 1] == 0:
                    continue
                for j1 in range(k1 - 7, k1):
                    j0 = j1 - 1
                    center = cgf * xnfdop_arr[j0, nelion - 1]
                    if center < tab[j0, nucont0]:
                        continue
                    center = center * _fastex_nb(elo_val * hckt_arr[j0], extab, extabf)
                    if center < tab[j0, nucont0]:
                        continue
                    dop = dopple_arr[j0, nelion - 1]
                    if dop <= 0.0:
                        continue
                    adamp = (gammar + gammas * xne_arr[j0] + gammaw * txnxn[j0]) / dop
                    dopwave = dop * wlvac4
                    _accwings_nb(
                        xlines, j0, nu0, wlvac, center, adamp, dopwave,
                        tab[j0, nucont0], waveset, h0tab, h1tab, h2tab,
                    )

            if ifline == 1:
                lineused += 1
            iwlold = iwl

        return xlines, lineused

    @_njit
    def _xlinop_type0_kernel_nb(
        xlines,
        n_records,
        wlvac_arr, gf_arr, elo_arr, gammar_arr, gammas_arr, gammaw_arr,
        nelion_arr, type_arr, ncon_arr, nelionx_arr, iwl_arr,
        waveset, iwavetab, tab,
        hckt_arr, xne_arr, xnfdop_arr, dopple_arr, txnxn,
        emerge, contx,
        extab, extabf, h0tab, h1tab, h2tab,
        nrhox, numnu, nuhi_eff, nulo,
    ):
        ifj = np.zeros(nrhox + 2, dtype=np.int32)
        nucont0 = 0
        nu0 = max(0, nulo - 1)
        lineused = 0
        for iline in range(n_records):
            typ = int(type_arr[iline])
            if typ != 0 and typ != 3:
                continue
            wlvac = float(wlvac_arr[iline])
            if wlvac > waveset[min(nuhi_eff, numnu) - 1]:
                break
            iwl = int(iwl_arr[iline])
            while nucont0 < iwavetab.shape[0] and iwl >= int(iwavetab[nucont0]):
                nucont0 += 1
            if nucont0 >= tab.shape[1]:
                continue
            while nu0 < numnu and wlvac >= waveset[nu0]:
                nu0 += 1
            if nu0 >= numnu:
                break

            nelion = int(nelion_arr[iline])
            if nelion < 1 or nelion > xnfdop_arr.shape[1]:
                lineused += 1
                continue
            cgf = float(gf_arr[iline])
            elo = float(elo_arr[iline])
            gammar = float(gammar_arr[iline])
            gammas = float(gammas_arr[iline])
            gammaw = float(gammaw_arr[iline])
            ncon_eff = int(ncon_arr[iline])
            if ncon_eff > 10:
                ncon_eff = 0
            nelionx = int(nelionx_arr[iline])
            for j1 in range(8, nrhox + 1, 8):
                ifj[j1 + 1] = 0
                j0 = j1 - 1
                center = cgf * float(xnfdop_arr[j0, nelion - 1])
                if center < float(tab[j0, nucont0]):
                    continue
                center *= _fastex_nb(elo * float(hckt_arr[j0]), extab, extabf)
                if center < float(tab[j0, nucont0]):
                    continue
                dop = float(dopple_arr[j0, nelion - 1])
                if dop <= 0.0:
                    continue
                ifj[j1 + 1] = 1
                adamp = (gammar + gammas * float(xne_arr[j0]) + gammaw * float(txnxn[j0])) / dop
                dopwave = dop * wlvac
                blue_cutoff = -1.0
                if ncon_eff > 0 and 1 <= nelionx <= contx.shape[1]:
                    den = float(contx[ncon_eff - 1, nelionx - 1]) - float(emerge[j0])
                    if den != 0.0:
                        blue_cutoff = 1.0e7 / den
                if blue_cutoff > 0.0 and wlvac < blue_cutoff:
                    continue
                _accwings_cutoff_nb(
                    xlines, j0, nu0, wlvac, center, adamp, dopwave,
                    float(tab[j0, nucont0]), waveset, h0tab, h1tab, h2tab,
                    2000, blue_cutoff,
                )

            for k1 in range(8, nrhox + 1, 8):
                if ifj[k1 - 7] + ifj[k1 + 1] == 0:
                    continue
                for j1 in range(k1 - 7, k1):
                    j0 = j1 - 1
                    center = cgf * float(xnfdop_arr[j0, nelion - 1])
                    if center < float(tab[j0, nucont0]):
                        continue
                    center *= _fastex_nb(elo * float(hckt_arr[j0]), extab, extabf)
                    if center < float(tab[j0, nucont0]):
                        continue
                    dop = float(dopple_arr[j0, nelion - 1])
                    if dop <= 0.0:
                        continue
                    adamp = (gammar + gammas * float(xne_arr[j0]) + gammaw * float(txnxn[j0])) / dop
                    dopwave = dop * wlvac
                    blue_cutoff = -1.0
                    if ncon_eff > 0 and 1 <= nelionx <= contx.shape[1]:
                        den = float(contx[ncon_eff - 1, nelionx - 1]) - float(emerge[j0])
                        if den != 0.0:
                            blue_cutoff = 1.0e7 / den
                    if blue_cutoff > 0.0 and wlvac < blue_cutoff:
                        continue
                    _accwings_cutoff_nb(
                        xlines, j0, nu0, wlvac, center, adamp, dopwave,
                        float(tab[j0, nucont0]), waveset, h0tab, h1tab, h2tab,
                        2000, blue_cutoff,
                    )
            lineused += 1
        return xlines, lineused

    @_njit
    def _profile_for_setup_precomputed_nb(
        n: int,
        m: int,
        freqnm: float,
        wavenm: float,
        dbeta: float,
        c1con: float,
        c2con: float,
        radamp: float,
        resont: float,
        vdw: float,
        y1num: float,
        y1wht: float,
        finest: np.ndarray,
        finswt: np.ndarray,
        delw_nm: float,
        dopple: float,
        dop: float,
        hwres: float,
        hwvdw: float,
        hwrad: float,
        hwlor: float,
        hfwid: float,
        nwid: int,
        fo_j: float,
        xne_j: float,
        y1s_j: float,
        y1b_j: float,
        c1d_j: float,
        c2d_j: float,
        gcon1_j: float,
        gcon2_j: float,
        pp_h_j: float,
        xnf_h2_j: float,
        xnfp_h1_j: float,
        extab: np.ndarray,
        extabf: np.ndarray,
        faste1_tbl: np.ndarray,
        propbm: np.ndarray,
        c_arr: np.ndarray,
        d_arr: np.ndarray,
        pp_tab: np.ndarray,
        beta_arr: np.ndarray,
        cutoff_h2: np.ndarray,
        cutoff_h2_plus: np.ndarray,
    ) -> float:
        wl = wavenm + delw_nm * 10.0
        if wl <= 0.0 or dopple <= 0.0 or dop <= 0.0:
            return 0.0
        freq4 = _C_LIGHT_A / wl
        delt = abs(freq4 - freqnm)
        delstark = -10.0 * delw_nm / wavenm * freqnm
        ifcore = abs(delt) <= hfwid
        out = 0.0
        if ifcore:
            if nwid == 1:
                out = _doppler_profile_nb(finest, finswt, freqnm, freq4, dop, extab, extabf)
            elif nwid == 2:
                out = _lorentz_profile_nb(
                    n, m, freqnm, wavenm, radamp, resont, vdw, freq4, delt, dop,
                    hwres, hwvdw, hwrad, xnfp_h1_j, cutoff_h2,
                )
            else:
                out = _stark_profile_nb(
                    n, m, dbeta, c1con, c2con, y1num, y1wht, freq4, delstark, dop,
                    fo_j, xne_j, y1s_j, y1b_j, c1d_j, c2d_j, gcon1_j, gcon2_j,
                    pp_h_j, xnf_h2_j, extab, extabf, faste1_tbl, propbm, c_arr,
                    d_arr, pp_tab, beta_arr, cutoff_h2_plus,
                )
        else:
            out = (
                _doppler_profile_nb(finest, finswt, freqnm, freq4, dop, extab, extabf)
                + _lorentz_profile_nb(
                    n, m, freqnm, wavenm, radamp, resont, vdw, freq4, delt, dop,
                    hwres, hwvdw, hwrad, xnfp_h1_j, cutoff_h2,
                )
                + _stark_profile_nb(
                    n, m, dbeta, c1con, c2con, y1num, y1wht, freq4, delstark, dop,
                    fo_j, xne_j, y1s_j, y1b_j, c1d_j, c2d_j, gcon1_j, gcon2_j,
                    pp_h_j, xnf_h2_j, extab, extabf, faste1_tbl, propbm, c_arr,
                    d_arr, pp_tab, beta_arr, cutoff_h2_plus,
                )
            )
        if out < 0.0:
            return 0.0
        return out

    @_njit_parallel
    def _xlinop_type_minus1_kernel_nb(
        xlines,
        n_records,
        wlvac_arr,
        gf_arr,
        nblo_arr,
        type_arr,
        ncon_arr,
        iwl_arr,
        setup_valid,
        setup_n,
        setup_m,
        setup_freqnm,
        setup_wavenm,
        setup_dbeta,
        setup_c1con,
        setup_c2con,
        setup_radamp,
        setup_resont,
        setup_vdw,
        setup_stark,
        setup_y1num,
        setup_y1wht,
        setup_finest,
        setup_finswt,
        waveset,
        iwavetab,
        tab,
        bolth,
        contx,
        emerge,
        xne_arr,
        xnf_h1,
        xnf_h2,
        xnfp_h1,
        dopple_h1,
        fo_arr,
        pp_arr_h,
        y1s_arr,
        y1b_arr,
        t3nhe_arr,
        t3nh2_arr,
        c1d_arr,
        c2d_arr,
        gcon1_arr,
        gcon2_arr,
        extab_h,
        extabf_h,
        faste1_tbl,
        propbm,
        c_arr,
        d_arr,
        pp_tab,
        beta_arr,
        cutoff_h2,
        cutoff_h2_plus,
        nrhox,
        numnu,
        nuhi_eff,
        nulo,
    ):
        nucont0 = 0
        nu0 = max(0, nulo - 1)
        lineused = 0
        for iline in range(n_records):
            typ = int(type_arr[iline])
            if typ == 2:
                continue
            if typ != -1:
                continue
            wlvac = float(wlvac_arr[iline])
            if wlvac > waveset[min(nuhi_eff, numnu) - 1]:
                break
            iwl = int(iwl_arr[iline])
            while nucont0 < iwavetab.shape[0] and iwl >= int(iwavetab[nucont0]):
                nucont0 += 1
            if nucont0 >= tab.shape[1]:
                continue
            while nu0 < numnu and wlvac >= waveset[nu0]:
                nu0 += 1
            if nu0 >= numnu:
                break

            nblo = int(nblo_arr[iline])
            ncon_eff = int(ncon_arr[iline])
            if nblo < 1 or nblo > bolth.shape[1]:
                continue
            if ncon_eff == 0:
                continue
            if ncon_eff < 1 or ncon_eff > contx.shape[0]:
                continue
            if int(setup_valid[iline]) == 0:
                continue

            cgf = np.float32(float(gf_arr[iline]))
            for j0 in numba.prange(nrhox):
                center = np.float32(cgf * float(bolth[j0, nblo - 1]))
                if center < float(tab[j0, nucont0]):
                    continue
                den = float(contx[ncon_eff - 1, 0]) - float(emerge[j0])
                if den == 0.0:
                    continue
                wcon = 1.0e7 / den
                dopple = float(dopple_h1[j0])
                if dopple <= 0.0:
                    continue
                setup_freq = float(setup_freqnm[iline])
                setup_vdw_i = float(setup_vdw[iline])
                setup_resont_i = float(setup_resont[iline])
                setup_radamp_i = float(setup_radamp[iline])
                setup_stark_i = float(setup_stark[iline])
                fo_j = float(fo_arr[j0])
                hwstk = setup_stark_i * fo_j
                hwvdw = setup_vdw_i * float(t3nhe_arr[j0]) + 2.0 * setup_vdw_i * float(t3nh2_arr[j0])
                hwrad = setup_radamp_i
                hwres = setup_resont_i * float(xnf_h1[j0])
                hwlor = hwres + hwvdw + hwrad
                nwid = 1
                if not (dopple >= hwstk and dopple >= hwlor):
                    nwid = 2
                    if hwlor < hwstk:
                        nwid = 3
                hfwid = setup_freq * max(dopple, hwlor, hwstk)
                dop = setup_freq * dopple
                if dop <= 0.0:
                    continue

                ired_hi = min(nu0 + 2001, numnu)
                for iw in range(nu0, ired_hi):
                    if float(waveset[iw]) < wcon:
                        continue
                    delw = np.float32(float(waveset[iw]) - wlvac)
                    hprof = np.float32(
                        _profile_for_setup_precomputed_nb(
                            int(setup_n[iline]),
                            int(setup_m[iline]),
                            setup_freq,
                            float(setup_wavenm[iline]),
                            float(setup_dbeta[iline]),
                            float(setup_c1con[iline]),
                            float(setup_c2con[iline]),
                            setup_radamp_i,
                            setup_resont_i,
                            setup_vdw_i,
                            float(setup_y1num[iline]),
                            float(setup_y1wht[iline]),
                            setup_finest[iline],
                            setup_finswt[iline],
                            float(delw),
                            dopple,
                            dop,
                            hwres,
                            hwvdw,
                            hwrad,
                            hwlor,
                            hfwid,
                            nwid,
                            fo_j,
                            float(xne_arr[j0]),
                            float(y1s_arr[j0]),
                            float(y1b_arr[j0]),
                            float(c1d_arr[j0]),
                            float(c2d_arr[j0]),
                            float(gcon1_arr[j0]),
                            float(gcon2_arr[j0]),
                            float(pp_arr_h[j0]),
                            float(xnf_h2[j0]),
                            float(xnfp_h1[j0]),
                            extab_h,
                            extabf_h,
                            faste1_tbl,
                            propbm,
                            c_arr,
                            d_arr,
                            pp_tab,
                            beta_arr,
                            cutoff_h2,
                            cutoff_h2_plus,
                        )
                    )
                    cv = np.float32(center * hprof)
                    xlines[j0, iw] += cv
                    if cv < float(tab[j0, nucont0]):
                        break

                for ired in range(1, 2001):
                    iw = nu0 - ired
                    if iw < 0:
                        break
                    if float(waveset[iw]) < wcon:
                        break
                    delw = np.float32(float(waveset[iw]) - wlvac)
                    hprof = np.float32(
                        _profile_for_setup_precomputed_nb(
                            int(setup_n[iline]),
                            int(setup_m[iline]),
                            setup_freq,
                            float(setup_wavenm[iline]),
                            float(setup_dbeta[iline]),
                            float(setup_c1con[iline]),
                            float(setup_c2con[iline]),
                            setup_radamp_i,
                            setup_resont_i,
                            setup_vdw_i,
                            float(setup_y1num[iline]),
                            float(setup_y1wht[iline]),
                            setup_finest[iline],
                            setup_finswt[iline],
                            float(delw),
                            dopple,
                            dop,
                            hwres,
                            hwvdw,
                            hwrad,
                            hwlor,
                            hfwid,
                            nwid,
                            fo_j,
                            float(xne_arr[j0]),
                            float(y1s_arr[j0]),
                            float(y1b_arr[j0]),
                            float(c1d_arr[j0]),
                            float(c2d_arr[j0]),
                            float(gcon1_arr[j0]),
                            float(gcon2_arr[j0]),
                            float(pp_arr_h[j0]),
                            float(xnf_h2[j0]),
                            float(xnfp_h1[j0]),
                            extab_h,
                            extabf_h,
                            faste1_tbl,
                            propbm,
                            c_arr,
                            d_arr,
                            pp_tab,
                            beta_arr,
                            cutoff_h2,
                            cutoff_h2_plus,
                        )
                    )
                    cv = np.float32(center * hprof)
                    xlines[j0, iw] += cv
                    if cv < float(tab[j0, nucont0]):
                        break
            lineused += 1
        return xlines, lineused


def selectlines(*, layers: int, n_freq: int) -> LineOpacityState:  # noqa: E302
    """Allocate `XLINES` workspace (selection itself is done by Fortran pass-1)."""
    return LineOpacityState(xlines=np.zeros((layers, n_freq), dtype=np.float64), lineused=0)


def _hydrogen_setup_arrays(records: XLineRecords, hprof_eval: HydrogenProfileEvaluator) -> dict[str, np.ndarray]:
    nrec = int(records.size)
    valid = np.zeros(nrec, dtype=np.int32)
    n_arr = np.zeros(nrec, dtype=np.int32)
    m_arr = np.zeros(nrec, dtype=np.int32)
    freqnm = np.zeros(nrec, dtype=np.float64)
    wavenm = np.zeros(nrec, dtype=np.float64)
    dbeta = np.zeros(nrec, dtype=np.float64)
    c1con = np.zeros(nrec, dtype=np.float64)
    c2con = np.zeros(nrec, dtype=np.float64)
    radamp = np.zeros(nrec, dtype=np.float64)
    resont = np.zeros(nrec, dtype=np.float64)
    vdw = np.zeros(nrec, dtype=np.float64)
    stark = np.zeros(nrec, dtype=np.float64)
    y1num = np.zeros(nrec, dtype=np.float64)
    y1wht = np.zeros(nrec, dtype=np.float64)
    max_comp = 1
    setups: list[object] = [None] * nrec
    for iline in range(nrec):
        if int(records.type_code[iline]) != -1:
            continue
        setup = hprof_eval._line_setup(int(records.nblo[iline]), int(records.nbup[iline]))
        if setup is None:
            continue
        setups[iline] = setup
        max_comp = max(max_comp, int(setup.finest.size))
    finest = np.zeros((nrec, max_comp), dtype=np.float64)
    finswt = np.zeros((nrec, max_comp), dtype=np.float64)
    for iline, setup_obj in enumerate(setups):
        if setup_obj is None:
            continue
        setup = setup_obj
        valid[iline] = 1
        n_arr[iline] = int(setup.n)
        m_arr[iline] = int(setup.m)
        freqnm[iline] = float(setup.freqnm)
        wavenm[iline] = float(setup.wavenm)
        dbeta[iline] = float(setup.dbeta)
        c1con[iline] = float(setup.c1con)
        c2con[iline] = float(setup.c2con)
        radamp[iline] = float(setup.radamp)
        resont[iline] = float(setup.resont)
        vdw[iline] = float(setup.vdw)
        stark[iline] = float(setup.stark)
        y1num[iline] = float(setup.y1num)
        y1wht[iline] = float(setup.y1wht)
        ncomp = int(setup.finest.size)
        finest[iline, :ncomp] = np.asarray(setup.finest, dtype=np.float64)
        finswt[iline, :ncomp] = np.asarray(setup.finswt, dtype=np.float64)
    return {
        "valid": valid,
        "n": n_arr,
        "m": m_arr,
        "freqnm": freqnm,
        "wavenm": wavenm,
        "dbeta": dbeta,
        "c1con": c1con,
        "c2con": c2con,
        "radamp": radamp,
        "resont": resont,
        "vdw": vdw,
        "stark": stark,
        "y1num": y1num,
        "y1wht": y1wht,
        "finest": finest,
        "finswt": finswt,
    }


def linop1(
    *,
    records: SelectedLineRecords,
    wave_set_nm: np.ndarray,
    i_wavetab: np.ndarray,
    tabcont: np.ndarray,
    temperature_k: np.ndarray,
    hckt: np.ndarray,
    xne: np.ndarray,
    xnf: np.ndarray,
    xnfdop: np.ndarray,
    dopple: np.ndarray,
    nulo: int = 1,
    nuhi: int | None = None,
) -> LineOpacityState:
    """Fortran-faithful LINOP1 accumulation from preselected `fort.12` records.

    When numba is available the hot kernel is JIT-compiled to near-Fortran
    speed; otherwise falls back to a pure-Python loop.
    """

    waveset = np.asarray(wave_set_nm, dtype=np.float64)
    iwavetab = np.asarray(i_wavetab, dtype=np.int64)
    # Fortran LINOP1 uses IMPLICIT REAL*4: TABCONT, XNFDOP, DOPPLE, HCKT4,
    # XNE4 are all REAL*4.  Keep WAVESET as float64 (Fortran REAL*8).
    tab = np.asarray(tabcont, dtype=np.float32)
    t = np.asarray(temperature_k, dtype=np.float64)
    hckt_arr = np.asarray(hckt, dtype=np.float32)
    xne_arr = np.asarray(xne, dtype=np.float32)
    xnf_arr = np.asarray(xnf, dtype=np.float64)
    xnfdop_arr = np.asarray(xnfdop, dtype=np.float32)
    dopple_arr = np.asarray(dopple, dtype=np.float32)

    nrhox = int(t.size)
    numnu = int(waveset.size)
    nuhi_eff = numnu if nuhi is None else int(nuhi)
    if nrhox != 80:
        raise ValueError(f"LINOP1 expects 80 depth layers, got {nrhox}")

    extab, extabf = _build_exptab()
    extab = np.asarray(extab, dtype=np.float32)
    extabf = np.asarray(extabf, dtype=np.float32)
    tablog = np.asarray(_build_tablog(), dtype=np.float32)
    h0tab, h1tab, h2tab = _build_h_tables()
    h0tab = np.asarray(h0tab, dtype=np.float32)
    h1tab = np.asarray(h1tab, dtype=np.float32)
    h2tab = np.asarray(h2tab, dtype=np.float32)

    # Fortran: TXNXN is REAL*4 (line 9950), computed from REAL*8 then truncated
    txnxn = np.asarray(
        (xnf_arr[:, 0] + 0.42 * xnf_arr[:, 2] + 0.85 * xnf_arr[:, 840]) * (t / 10000.0) ** 0.3,
        dtype=np.float32,
    )
    start = float(waveset[max(0, int(nulo) - 1)] - 1.0)
    stop = float(waveset[min(nuhi_eff, numnu) - 1] + 1.0)

    force_py_linop1 = os.environ.get("ATLAS_TRACE_FORCE_PY_LINOP1", "0") == "1"
    if _NUMBA_AVAILABLE and not force_py_linop1:
        xlines, lineused = _linop1_kernel_nb(
            records.size,
            np.asarray(records.iwl, dtype=np.int32),
            np.asarray(records.ielion, dtype=np.int16),
            np.asarray(records.ielo, dtype=np.int16),
            np.asarray(records.igflog, dtype=np.int16),
            np.asarray(records.igr, dtype=np.int16),
            np.asarray(records.igs, dtype=np.int16),
            np.asarray(records.igw, dtype=np.int16),
            waveset, iwavetab, tab,
            hckt_arr, xne_arr, xnfdop_arr, dopple_arr, txnxn,
            extab, extabf, tablog,
            h0tab, h1tab, h2tab,
            nrhox, numnu, nuhi_eff, nulo,
            start, stop,
            float(_RATIOLG), np.float32(_CGF_SCALE), np.float32(_GAMMA_SCALE),
        )
        return LineOpacityState(xlines=xlines, lineused=int(lineused))

    # --- pure-Python fallback (slow, only used when numba is absent) ---
    xlines = np.zeros((nrhox, numnu), dtype=np.float32)
    ifj = np.zeros(nrhox + 2, dtype=np.int32)  # 1-based helper, includes +1 access.

    nucont0 = 0
    nu0 = max(0, int(nulo) - 1)
    iwlold = 0
    lineused = 0

    for iline in range(records.size):
        iwl = int(records.iwl[iline])
        if iwl < iwlold:
            nucont0 = 0
            nu0 = max(0, int(nulo) - 1)

        while nucont0 < iwavetab.size and iwl >= int(iwavetab[nucont0]):
            nucont0 += 1
        if nucont0 >= tab.shape[1]:
            iwlold = iwl
            continue

        nelion = abs(int(records.ielion[iline])) // 10
        if nelion < 1 or nelion > xnfdop_arr.shape[1]:
            iwlold = iwl
            continue

        wlvac = float(np.exp(iwl * _RATIOLG))
        if wlvac < start or wlvac > stop:
            iwlold = iwl
            continue

        while nu0 < numnu and wlvac >= float(waveset[nu0]):
            nu0 += 1
        if nu0 >= numnu:
            iwlold = iwl
            continue

        igflog = int(records.igflog[iline])
        ielo = int(records.ielo[iline])
        igr = int(records.igr[iline])
        igs = int(records.igs[iline])
        igw = int(records.igw[iline])
        if igflog < 1 or ielo < 1 or igr < 1 or igs < 1 or igw < 1:
            iwlold = iwl
            continue
        if igflog > tablog.size or ielo > tablog.size or igr > tablog.size or igs > tablog.size or igw > tablog.size:
            iwlold = iwl
            continue

        cgf = _CGF_SCALE * wlvac * float(tablog[igflog - 1])
        elo = float(tablog[ielo - 1])
        ifline = 0
        adamp_seed = 0.0
        gammar = gammas = gammaw = 0.0

        for j1 in range(8, nrhox + 1, 8):
            ifj[j1 + 1] = 0
            j0 = j1 - 1
            center = cgf * float(xnfdop_arr[j0, nelion - 1])
            if center < float(tab[j0, nucont0]):
                continue
            center *= _fastex(elo * float(hckt_arr[j0]), extab, extabf)
            if center < float(tab[j0, nucont0]):
                continue
            ifj[j1 + 1] = 1
            ifline = 1
            if adamp_seed == 0.0:
                gammar = float(tablog[igr - 1]) * wlvac * _GAMMA_SCALE
                gammas = float(tablog[igs - 1]) * wlvac * _GAMMA_SCALE
                gammaw = float(tablog[igw - 1]) * wlvac * _GAMMA_SCALE
                adamp_seed = 1.0
            dop = float(dopple_arr[j0, nelion - 1])
            if dop <= 0.0:
                continue
            adamp = (gammar + gammas * float(xne_arr[j0]) + gammaw * float(txnxn[j0])) / dop
            dopwave = dop * wlvac
            if trace_in_focus(wlvac_nm=wlvac, j0=j0):
                trace_emit(
                    event="center_pass",
                    iter_num=1,
                    line_num_1b=iline + 1,
                    depth_1b=j0 + 1,
                    nu_1b=nu0 + 1,
                    type_code=0,
                    wlvac_nm=wlvac,
                    center=center,
                    adamp=adamp,
                    cv=0.0,
                    tabcont=float(tab[j0, nucont0]),
                    branch="linop1",
                    reason="center_gate",
                )
            _accumulate_wings(
                xlines=xlines,
                j0=j0,
                nu0=nu0,
                wlvac=wlvac,
                center=center,
                adamp=adamp,
                dopwave=dopwave,
                tabcont_ref=float(tab[j0, nucont0]),
                waveset=waveset,
                h0tab=h0tab,
                h1tab=h1tab,
                h2tab=h2tab,
                wing_steps=100,
                blue_cutoff=None,
            )

        for k1 in range(8, nrhox + 1, 8):
            if ifj[k1 - 7] + ifj[k1 + 1] == 0:
                continue
            for j1 in range(k1 - 7, k1):
                j0 = j1 - 1
                center = cgf * float(xnfdop_arr[j0, nelion - 1])
                if center < float(tab[j0, nucont0]):
                    continue
                center *= _fastex(elo * float(hckt_arr[j0]), extab, extabf)
                if center < float(tab[j0, nucont0]):
                    continue
                dop = float(dopple_arr[j0, nelion - 1])
                if dop <= 0.0:
                    continue
                adamp = (gammar + gammas * float(xne_arr[j0]) + gammaw * float(txnxn[j0])) / dop
                dopwave = dop * wlvac
                if trace_focus and trace_in_focus(wlvac_nm=wlvac, j0=j0):
                    trace_emit(
                        event="center_pass_inner",
                        iter_num=1,
                        line_num_1b=iline + 1,
                        depth_1b=j0 + 1,
                        nu_1b=nu0 + 1,
                        type_code=0,
                        wlvac_nm=wlvac,
                        center=center,
                        adamp=adamp,
                        cv=0.0,
                        tabcont=float(tab[j0, nucont0]),
                        branch="linop1",
                        reason="ifj_fill",
                    )
                _accumulate_wings(
                    xlines=xlines,
                    j0=j0,
                    nu0=nu0,
                    wlvac=wlvac,
                    center=center,
                    adamp=adamp,
                    dopwave=dopwave,
                    tabcont_ref=float(tab[j0, nucont0]),
                    waveset=waveset,
                    h0tab=h0tab,
                    h1tab=h1tab,
                    h2tab=h2tab,
                    wing_steps=100,
                    blue_cutoff=None,
                )

        if ifline == 1:
            lineused += 1
        iwlold = iwl

    return LineOpacityState(xlines=xlines, lineused=lineused)


def xlinop(
    *,
    records: XLineRecords,
    wave_set_nm: np.ndarray,
    i_wavetab: np.ndarray,
    tabcont: np.ndarray,
    temperature_k: np.ndarray,
    hckt: np.ndarray,
    xne: np.ndarray,
    xnf: np.ndarray,
    xnfp: np.ndarray,
    rho: np.ndarray,
    xnfdop: np.ndarray,
    dopple: np.ndarray,
    ifop15_enabled: bool,
    base_xlines: np.ndarray | None = None,
    nulo: int = 1,
    nuhi: int | None = None,
) -> LineOpacityState:
    """Fortran-faithful XLINOP accumulation from `fort.19` records."""

    waveset = np.asarray(wave_set_nm, dtype=np.float64)
    iwavetab = np.asarray(i_wavetab, dtype=np.int64)
    tab = np.asarray(tabcont, dtype=np.float64)
    t = np.asarray(temperature_k, dtype=np.float64)
    hckt_arr = np.asarray(hckt, dtype=np.float64)
    xne_arr = np.asarray(xne, dtype=np.float64)
    xnf_arr = np.asarray(xnf, dtype=np.float64)
    xnfp_arr = np.asarray(xnfp, dtype=np.float64)
    rho_arr = np.asarray(rho, dtype=np.float64)
    xnfdop_arr = np.asarray(xnfdop, dtype=np.float64)
    dopple_arr = np.asarray(dopple, dtype=np.float64)

    nrhox = int(t.size)
    numnu = int(waveset.size)
    nuhi_eff = numnu if nuhi is None else int(nuhi)
    if nrhox != 80:
        raise ValueError(f"XLINOP expects 80 depth layers, got {nrhox}")

    extab, extabf = _build_exptab()
    h0tab, h1tab, h2tab = _build_h_tables()
    contx = _build_contx()
    txnxn = (xnf_arr[:, 0] + 0.42 * xnf_arr[:, 2] + 0.85 * xnf_arr[:, 840]) * (t / 10000.0) ** 0.3
    nstark = 1600.0 / np.maximum(xne_arr, 1e-300) ** (2.0 / 15.0)
    emerge = 109737.312 / np.maximum(nstark * nstark, 1e-300)
    eh = np.zeros(100, dtype=np.float64)
    eh[0] = 0.0
    eh[1] = 82259.105
    eh[2] = 97492.302
    eh[3] = 102823.893
    eh[4] = 105291.651
    eh[5] = 106632.160
    eh[6] = 107440.444
    eh[7] = 107965.051
    eh[8] = 108324.720
    eh[9] = 108581.988
    for n in range(11, 101):
        eh[n - 1] = 109678.764 - 109677.576 / float(n * n)

    if ifop15_enabled and base_xlines is not None:
        xlines = np.asarray(base_xlines, dtype=np.float32).copy()
    else:
        xlines = np.zeros((nrhox, numnu), dtype=np.float32)

    nucont0 = 0
    nu0 = max(0, int(nulo) - 1)
    ifj = np.zeros(nrhox + 2, dtype=np.int32)
    lineused = 0
    trace_focus = trace_enabled()
    focus_wave_mask = trace_focus & (waveset >= 381.0) & (waveset <= 410.0)
    focus_layer_lo = 57
    focus_layer_hi = 63
    focus_base_sum = float(np.sum(np.asarray(xlines[focus_layer_lo:focus_layer_hi, :], dtype=np.float64)[:, focus_wave_mask]))
    focus_band_records_total = 0
    focus_band_records_typ0 = 0
    focus_band_records_typm1 = 0
    focus_adamp_le_02 = 0
    focus_adamp_gt_02 = 0
    focus_wcon_skip = 0
    focus_center_skip = 0
    focus_typ0_added = 0.0
    focus_typ0_call_count = 0
    focus_typ0_max_call_added = -1.0
    focus_typ0_max_call_meta: dict[str, float | int] = {}
    focus_red_steps = 0
    focus_blue_steps = 0
    focus_red_break_tabcont = 0
    focus_blue_break_tabcont = 0
    focus_blue_break_cutoff = 0
    focus_typ0_inband_added = 0.0
    focus_typ0_outband_added = 0.0
    focus_typm1_inband_added = 0.0
    focus_typm1_outband_added = 0.0
    focus_typ1_inband_added = 0.0
    focus_typ1_outband_added = 0.0
    focus_merged_inband_added = 0.0
    focus_merged_outband_added = 0.0
    focus_line_max_added = -1.0
    focus_line_max_iline = -1
    focus_line_max_typ = 0
    focus_line_max_wlvac = -1.0
    focus_line_max_inband = False
    focus_typm1_line_summaries: list[str] = []
    focus_typm1_top_cell_cv = -1.0
    focus_typm1_top_cell_meta: dict[str, float | int] = {}
    focus_typm1_ifcore_count = 0
    focus_typm1_noncore_count = 0
    focus_typm1_nwid1_count = 0
    focus_typm1_nwid2_count = 0
    focus_typm1_nwid3_count = 0
    focus_typm1_doppler_sum = 0.0
    focus_typm1_lorentz_sum = 0.0
    focus_typm1_stark_sum = 0.0
    focus_typm1_hprof_sum = 0.0
    focus_typm1_min_beta = 1.0e300
    focus_typm1_max_beta = -1.0
    focus_typm1_top_cell_hprof_parts: dict[str, float | int | bool] = {}
    focus_iw = np.where(focus_wave_mask)[0]
    focus_iw_lo = int(focus_iw[0]) if focus_iw.size > 0 else 0
    focus_iw_hi = int(focus_iw[-1]) if focus_iw.size > 0 else -1

    if _NUMBA_AVAILABLE and not trace_focus and records.size > 0:
        normal_mask = (records.type_code == 0) | (records.type_code == 3)
        if np.any(normal_mask):
            xlines, lineused_normal = _xlinop_type0_kernel_nb(
                xlines,
                records.size,
                np.asarray(records.wlvac, dtype=np.float64),
                np.asarray(records.gf, dtype=np.float64),
                np.asarray(records.elo, dtype=np.float64),
                np.asarray(records.gammar, dtype=np.float64),
                np.asarray(records.gammas, dtype=np.float64),
                np.asarray(records.gammaw, dtype=np.float64),
                np.asarray(records.nelion, dtype=np.int64),
                np.asarray(records.type_code, dtype=np.int64),
                np.asarray(records.ncon, dtype=np.int64),
                np.asarray(records.nelionx, dtype=np.int64),
                np.asarray(records.iwl, dtype=np.int64),
                waveset,
                iwavetab,
                tab,
                hckt_arr,
                xne_arr,
                xnfdop_arr,
                dopple_arr,
                txnxn,
                emerge,
                contx,
                extab,
                extabf,
                h0tab,
                h1tab,
                h2tab,
                nrhox,
                numnu,
                nuhi_eff,
                nulo,
            )
            special_mask = ~normal_mask
            if not np.any(special_mask):
                return LineOpacityState(xlines=xlines, lineused=int(lineused_normal))
            lineused_total = int(lineused_normal)
            hyd_mask = special_mask & ((records.type_code == -1) | (records.type_code == 2))
            if np.any(hyd_mask):
                hyd_state = xlinop(
                    records=_subset_xline_records(records, hyd_mask),
                    wave_set_nm=waveset,
                    i_wavetab=iwavetab,
                    tabcont=tab,
                    temperature_k=t,
                    hckt=hckt_arr,
                    xne=xne_arr,
                    xnf=xnf_arr,
                    xnfp=xnfp_arr,
                    rho=rho_arr,
                    xnfdop=xnfdop_arr,
                    dopple=dopple_arr,
                    ifop15_enabled=True,
                    base_xlines=xlines,
                    nulo=nulo,
                    nuhi=nuhi,
                )
                xlines = hyd_state.xlines
                lineused_total += int(hyd_state.lineused)
            rare_mask = special_mask & ~hyd_mask
            if np.any(rare_mask):
                rare_state = xlinop(
                    records=_subset_xline_records(records, rare_mask),
                    wave_set_nm=waveset,
                    i_wavetab=iwavetab,
                    tabcont=tab,
                    temperature_k=t,
                    hckt=hckt_arr,
                    xne=xne_arr,
                    xnf=xnf_arr,
                    xnfp=xnfp_arr,
                    rho=rho_arr,
                    xnfdop=xnfdop_arr,
                    dopple=dopple_arr,
                    ifop15_enabled=True,
                    base_xlines=xlines,
                    nulo=nulo,
                    nuhi=nuhi,
                )
                rare_state.lineused += lineused_total
                return rare_state
            return LineOpacityState(xlines=xlines, lineused=lineused_total)

    bolth = np.exp(-np.outer(hckt_arr, eh)) * xnfdop_arr[:, 0:1]
    hyd_tables = load_hydrogen_profile_tables_from_atlas12()
    xnh2 = compute_xnh2(
        temperature_k=t,
        xnfp_h1=xnfp_arr[:, 0],
        bhyd1=np.ones(nrhox, dtype=np.float64),
        tables=hyd_tables,
    )
    hprof_eval = HydrogenProfileEvaluator(
        temperature_k=t,
        xne=xne_arr,
        xnf_h1=xnf_arr[:, 0],
        xnf_h2=xnf_arr[:, 1],
        xnfp_h1=xnfp_arr[:, 0],
        xnfp_he1=xnfp_arr[:, 2],
        dopple_h1=dopple_arr[:, 0],
        xnh2=xnh2,
        tables=hyd_tables,
    )

    if _NUMBA_AVAILABLE and not trace_focus and records.size > 0:
        typ_arr = np.asarray(records.type_code, dtype=np.int64)
        if np.all((typ_arr == -1) | (typ_arr == 2)):
            for iline in range(records.size):
                if int(records.type_code[iline]) != -1:
                    continue
                nblo_v = int(records.nblo[iline])
                nbup_v = int(records.nbup[iline])
                ncon_v = int(records.ncon[iline])
                if nblo_v < 1 or nblo_v > 100:
                    raise ValueError(f"XLINOP hydrogen line has NBLO={nblo_v}, expected 1..100")
                if nbup_v < 1 or nbup_v > 100:
                    raise ValueError(f"XLINOP hydrogen line has NBUP={nbup_v}, expected 1..100")
                if ncon_v != 0 and (ncon_v < 1 or ncon_v > contx.shape[0]):
                    raise ValueError(f"XLINOP hydrogen line has NCON={ncon_v}, expected 1..{contx.shape[0]}")
            setup_arrays = _hydrogen_setup_arrays(records, hprof_eval)
            xlines, lineused_h = _xlinop_type_minus1_kernel_nb(
                xlines,
                records.size,
                np.asarray(records.wlvac, dtype=np.float64),
                np.asarray(records.gf, dtype=np.float64),
                np.asarray(records.nblo, dtype=np.int64),
                typ_arr,
                np.asarray(records.ncon, dtype=np.int64),
                np.asarray(records.iwl, dtype=np.int64),
                setup_arrays["valid"],
                setup_arrays["n"],
                setup_arrays["m"],
                setup_arrays["freqnm"],
                setup_arrays["wavenm"],
                setup_arrays["dbeta"],
                setup_arrays["c1con"],
                setup_arrays["c2con"],
                setup_arrays["radamp"],
                setup_arrays["resont"],
                setup_arrays["vdw"],
                setup_arrays["stark"],
                setup_arrays["y1num"],
                setup_arrays["y1wht"],
                setup_arrays["finest"],
                setup_arrays["finswt"],
                waveset,
                iwavetab,
                tab,
                bolth,
                contx,
                emerge,
                xne_arr,
                xnf_arr[:, 0],
                xnf_arr[:, 1],
                xnfp_arr[:, 0],
                dopple_arr[:, 0],
                hprof_eval.fo,
                hprof_eval.pp,
                hprof_eval.y1s,
                hprof_eval.y1b,
                hprof_eval.t3nhe,
                hprof_eval.t3nh2,
                hprof_eval.c1d,
                hprof_eval.c2d,
                hprof_eval.gcon1,
                hprof_eval.gcon2,
                hprof_eval.extab,
                hprof_eval.extabf,
                hprof_eval.faste1_tbl,
                hyd_tables.propbm,
                hyd_tables.c,
                hyd_tables.d,
                hyd_tables.pp,
                hyd_tables.beta,
                hyd_tables.cutoff_h2,
                hyd_tables.cutoff_h2_plus,
                nrhox,
                numnu,
                nuhi_eff,
                nulo,
            )
            return LineOpacityState(xlines=xlines, lineused=int(lineused_h))

    for iline in range(records.size):
        wlvac = float(records.wlvac[iline])
        in_focus_band = trace_focus and 381.0 <= wlvac <= 410.0
        if wlvac > float(waveset[min(nuhi_eff, numnu) - 1]):
            break
        iwl = int(records.iwl[iline])
        while nucont0 < iwavetab.size and iwl >= int(iwavetab[nucont0]):
            nucont0 += 1
        if nucont0 >= tab.shape[1]:
            continue
        while nu0 < numnu and wlvac >= float(waveset[nu0]):
            nu0 += 1
        if nu0 >= numnu:
            break
        line_may_hit_focus = bool(
            trace_focus
            and focus_iw_hi >= 0
            and (nu0 - 2000) <= focus_iw_hi
            and (nu0 + 2000) >= focus_iw_lo
        )
        line_focus_before = 0.0
        if line_may_hit_focus:
            line_focus_before = float(np.sum(np.asarray(xlines[focus_layer_lo:focus_layer_hi, :], dtype=np.float64)[:, focus_wave_mask]))

        nblo = int(records.nblo[iline])
        nbup = int(records.nbup[iline])
        nelion = int(records.nelion[iline])
        typ = int(records.type_code[iline])
        if in_focus_band and trace_enabled():
            trace_emit(
                event="type_dispatch",
                iter_num=1,
                line_num_1b=iline + 1,
                depth_1b=0,
                nu_1b=nu0 + 1,
                type_code=typ,
                wlvac_nm=wlvac,
                center=0.0,
                adamp=0.0,
                cv=0.0,
                tabcont=0.0,
                branch="xlinop",
                reason="dispatch",
            )
        ncon = int(records.ncon[iline])
        nelionx = int(records.nelionx[iline])
        elo = float(records.elo[iline])
        gf = float(records.gf[iline])
        gammar = float(records.gammar[iline])
        gammas = float(records.gammas[iline])
        gammaw = float(records.gammaw[iline])

        if typ == 2:
            continue
        if typ == -1:
            if in_focus_band:
                focus_band_records_total += 1
                focus_band_records_typm1 += 1
            line_typm1_before = 0.0
            line_typm1_red_steps = 0
            line_typm1_blue_steps = 0
            line_typm1_red_break_tab = 0
            line_typm1_blue_break_tab = 0
            line_typm1_blue_break_wcon = 0
            line_typm1_max_cv = -1.0
            line_typm1_max_hprof = -1.0
            line_typm1_max_center = -1.0
            line_typm1_min_tabcont = 1.0e300
            line_typm1_max_tabcont = -1.0
            if in_focus_band:
                line_typm1_before = float(np.sum(np.asarray(xlines[focus_layer_lo:focus_layer_hi, :], dtype=np.float64)[:, focus_wave_mask]))
            # HYDROGEN LINE (HPROF4)
            if nblo < 1 or nblo > 100:
                raise ValueError(f"XLINOP hydrogen line has NBLO={nblo}, expected 1..100")
            if nbup < 1 or nbup > 100:
                raise ValueError(f"XLINOP hydrogen line has NBUP={nbup}, expected 1..100")
            # NCON=0 means ISO2=0 in rpunchbin; Fortran would read CONTX(0,1) out-of-bounds
            # but these lines are always filtered by CENTER<TABCONT so they effectively skip.
            if ncon == 0:
                continue
            if ncon < 1 or ncon > contx.shape[0]:
                raise ValueError(f"XLINOP hydrogen line has NCON={ncon}, expected 1..{contx.shape[0]}")
            cgf = np.float32(gf)
            h_setup = hprof_eval._line_setup(nblo, nbup)
            if h_setup is None:
                continue
            for j0 in range(nrhox):
                center = np.float32(cgf * float(bolth[j0, nblo - 1]))
                if center < float(tab[j0, nucont0]):
                    continue
                if trace_focus and trace_in_focus(wlvac_nm=wlvac, j0=j0):
                    trace_emit(
                        event="h_center_pass",
                        iter_num=1,
                        line_num_1b=iline + 1,
                        depth_1b=j0 + 1,
                        nu_1b=nu0 + 1,
                        type_code=-1,
                        wlvac_nm=wlvac,
                        center=float(center),
                        adamp=0.0,
                        cv=0.0,
                        tabcont=float(tab[j0, nucont0]),
                        branch="xlinop_h",
                        reason="center_gate",
                    )
                if focus_layer_lo <= j0 < focus_layer_hi:
                    line_typm1_max_center = max(line_typm1_max_center, float(center))
                    tval = float(tab[j0, nucont0])
                    line_typm1_min_tabcont = min(line_typm1_min_tabcont, tval)
                    line_typm1_max_tabcont = max(line_typm1_max_tabcont, tval)
                # Fortran XLINOP hardcodes CONTX(NCON,1) for TYPE=-1 hydrogen lines.
                den = float(contx[ncon - 1, 0]) - float(emerge[j0])
                if den == 0.0:
                    continue
                wcon = 1.0e7 / den
                for iw in range(nu0, min(nu0 + 2001, numnu)):
                    if float(waveset[iw]) < wcon:
                        continue
                    in_focus_cell = bool(
                        trace_focus
                        and focus_layer_lo <= j0 < focus_layer_hi
                        and focus_iw_lo <= iw <= focus_iw_hi
                    )
                    if in_focus_cell:
                        line_typm1_red_steps += 1
                    delw = np.float32(float(waveset[iw]) - wlvac)
                    hprof = np.float32(
                        hprof_eval.profile_for_setup(
                            h_setup,
                            j0,
                            float(delw),
                            trace_line_1b=iline + 1,
                            trace_nu_1b=iw + 1,
                        )
                    )
                    cv = np.float32(center * hprof)
                    if trace_focus and iw == nu0 and trace_in_focus(wlvac_nm=wlvac, j0=j0):
                        trace_emit(
                            event="h_red_nu",
                            iter_num=1,
                            line_num_1b=iline + 1,
                            depth_1b=j0 + 1,
                            nu_1b=iw + 1,
                            type_code=-1,
                            wlvac_nm=wlvac,
                            center=float(center),
                            adamp=0.0,
                            cv=float(cv),
                            tabcont=float(tab[j0, nucont0]),
                            branch="xlinop_h",
                            reason="nu_anchor",
                        )
                    if in_focus_cell:
                        # #region agent log
                        setup = hprof_eval._line_setup(nblo, nbup)
                        if setup is not None:
                            wl_dbg = setup.wavenm + float(delw) * 10.0
                            if wl_dbg > 0.0:
                                freq4_dbg = _C_NM / wl_dbg
                                delt_dbg = abs(freq4_dbg - setup.freqnm)
                                dopple_dbg = float(dopple_arr[j0, 0])
                                if dopple_dbg > 0.0:
                                    hwstk_dbg = setup.stark * float(hprof_eval.fo[j0])
                                    hwvdw_dbg = setup.vdw * float(hprof_eval.t3nhe[j0]) + 2.0 * setup.vdw * float(hprof_eval.t3nh2[j0])
                                    hwrad_dbg = setup.radamp
                                    hwres_dbg = setup.resont * float(hprof_eval.xnf_h1[j0])
                                    hwlor_dbg = hwres_dbg + hwvdw_dbg + hwrad_dbg
                                    nwid_dbg = 1
                                    if not (dopple_dbg >= hwstk_dbg and dopple_dbg >= hwlor_dbg):
                                        nwid_dbg = 2
                                        if hwlor_dbg < hwstk_dbg:
                                            nwid_dbg = 3
                                    hfwid_dbg = setup.freqnm * max(dopple_dbg, hwlor_dbg, hwstk_dbg)
                                    ifcore_dbg = abs(delt_dbg) <= hfwid_dbg
                                    dop_dbg = setup.freqnm * dopple_dbg
                                    if dop_dbg > 0.0:
                                        delstark_dbg = -10.0 * float(delw) / setup.wavenm * setup.freqnm
                                        beta_dbg = abs(delstark_dbg) / max(float(hprof_eval.fo[j0]), 1.0e-300) * setup.dbeta
                                        doppler_dbg = float(hprof_eval._doppler_profile(setup, freq4_dbg, dop_dbg))
                                        lorentz_dbg = float(
                                            hprof_eval._lorentz_profile(
                                                setup,
                                                j0,
                                                freq4_dbg,
                                                delt_dbg,
                                                dop_dbg,
                                                hwres_dbg,
                                                hwvdw_dbg,
                                                hwrad_dbg,
                                            )
                                        )
                                        stark_dbg = float(hprof_eval._stark_profile(setup, j0, freq4_dbg, delstark_dbg, dop_dbg))
                                        focus_typm1_doppler_sum += doppler_dbg
                                        focus_typm1_lorentz_sum += lorentz_dbg
                                        focus_typm1_stark_sum += stark_dbg
                                        focus_typm1_hprof_sum += float(hprof)
                                        focus_typm1_min_beta = min(focus_typm1_min_beta, float(beta_dbg))
                                        focus_typm1_max_beta = max(focus_typm1_max_beta, float(beta_dbg))
                                        if ifcore_dbg:
                                            focus_typm1_ifcore_count += 1
                                        else:
                                            focus_typm1_noncore_count += 1
                                        if nwid_dbg == 1:
                                            focus_typm1_nwid1_count += 1
                                        elif nwid_dbg == 2:
                                            focus_typm1_nwid2_count += 1
                                        else:
                                            focus_typm1_nwid3_count += 1
                                        if float(cv) > focus_typm1_top_cell_cv:
                                            focus_typm1_top_cell_hprof_parts = {
                                                "ifcore": bool(ifcore_dbg),
                                                "nwid": int(nwid_dbg),
                                                "beta": float(beta_dbg),
                                                "doppler": float(doppler_dbg),
                                                "lorentz": float(lorentz_dbg),
                                                "stark": float(stark_dbg),
                                                "hprof": float(hprof),
                                                "hfwid": float(hfwid_dbg),
                                                "delt": float(delt_dbg),
                                            }
                        # #endregion
                        if float(cv) > focus_typm1_top_cell_cv:
                            focus_typm1_top_cell_cv = float(cv)
                            focus_typm1_top_cell_meta = {
                                "iline": int(iline),
                                "j0": int(j0),
                                "iw": int(iw),
                                "wlvac_nm": float(wlvac),
                                "waveset_nm": float(waveset[iw]),
                                "center": float(center),
                                "hprof": float(hprof),
                                "cv": float(cv),
                            }
                        # #endregion
                    xlines[j0, iw] += cv
                    if in_focus_cell and float(hprof) > line_typm1_max_hprof:
                        line_typm1_max_hprof = float(hprof)
                    if in_focus_cell and float(cv) > line_typm1_max_cv:
                        line_typm1_max_cv = float(cv)
                    if cv < float(tab[j0, nucont0]):
                        if trace_focus and trace_in_focus(wlvac_nm=float(waveset[iw]), j0=j0):
                            trace_emit(
                                event="h_red_break",
                                iter_num=1,
                                line_num_1b=iline + 1,
                                depth_1b=j0 + 1,
                                nu_1b=iw + 1,
                                type_code=-1,
                                wlvac_nm=float(waveset[iw]),
                                center=float(center),
                                adamp=float(nucont0 + 1),
                                cv=float(cv),
                                tabcont=float(tab[j0, nucont0]),
                                branch="xlinop_h",
                                reason="cv_lt_tab",
                            )
                        if in_focus_cell:
                            line_typm1_red_break_tab += 1
                        break
                for ired in range(1, 2001):
                    iw = nu0 - ired
                    if iw < 0:
                        break
                    if float(waveset[iw]) < wcon:
                        if focus_layer_lo <= j0 < focus_layer_hi:
                            line_typm1_blue_break_wcon += 1
                        break
                    in_focus_cell = bool(
                        trace_focus
                        and focus_layer_lo <= j0 < focus_layer_hi
                        and focus_iw_lo <= iw <= focus_iw_hi
                    )
                    if in_focus_cell:
                        line_typm1_blue_steps += 1
                    delw = np.float32(float(waveset[iw]) - wlvac)
                    hprof = np.float32(
                        hprof_eval.profile_for_setup(
                            h_setup,
                            j0,
                            float(delw),
                            trace_line_1b=iline + 1,
                            trace_nu_1b=iw + 1,
                        )
                    )
                    cv = np.float32(center * hprof)
                    if trace_focus and ired == 1 and trace_in_focus(wlvac_nm=wlvac, j0=j0):
                        trace_emit(
                            event="h_blue_nu1",
                            iter_num=1,
                            line_num_1b=iline + 1,
                            depth_1b=j0 + 1,
                            nu_1b=iw + 1,
                            type_code=-1,
                            wlvac_nm=wlvac,
                            center=float(center),
                            adamp=0.0,
                            cv=float(cv),
                            tabcont=float(tab[j0, nucont0]),
                            branch="xlinop_h",
                            reason="blue_first",
                        )
                    if in_focus_cell:
                        # #region agent log
                        setup = hprof_eval._line_setup(nblo, nbup)
                        if setup is not None:
                            wl_dbg = setup.wavenm + float(delw) * 10.0
                            if wl_dbg > 0.0:
                                freq4_dbg = _C_NM / wl_dbg
                                delt_dbg = abs(freq4_dbg - setup.freqnm)
                                dopple_dbg = float(dopple_arr[j0, 0])
                                if dopple_dbg > 0.0:
                                    hwstk_dbg = setup.stark * float(hprof_eval.fo[j0])
                                    hwvdw_dbg = setup.vdw * float(hprof_eval.t3nhe[j0]) + 2.0 * setup.vdw * float(hprof_eval.t3nh2[j0])
                                    hwrad_dbg = setup.radamp
                                    hwres_dbg = setup.resont * float(hprof_eval.xnf_h1[j0])
                                    hwlor_dbg = hwres_dbg + hwvdw_dbg + hwrad_dbg
                                    nwid_dbg = 1
                                    if not (dopple_dbg >= hwstk_dbg and dopple_dbg >= hwlor_dbg):
                                        nwid_dbg = 2
                                        if hwlor_dbg < hwstk_dbg:
                                            nwid_dbg = 3
                                    hfwid_dbg = setup.freqnm * max(dopple_dbg, hwlor_dbg, hwstk_dbg)
                                    ifcore_dbg = abs(delt_dbg) <= hfwid_dbg
                                    dop_dbg = setup.freqnm * dopple_dbg
                                    if dop_dbg > 0.0:
                                        delstark_dbg = -10.0 * float(delw) / setup.wavenm * setup.freqnm
                                        beta_dbg = abs(delstark_dbg) / max(float(hprof_eval.fo[j0]), 1.0e-300) * setup.dbeta
                                        doppler_dbg = float(hprof_eval._doppler_profile(setup, freq4_dbg, dop_dbg))
                                        lorentz_dbg = float(
                                            hprof_eval._lorentz_profile(
                                                setup,
                                                j0,
                                                freq4_dbg,
                                                delt_dbg,
                                                dop_dbg,
                                                hwres_dbg,
                                                hwvdw_dbg,
                                                hwrad_dbg,
                                            )
                                        )
                                        stark_dbg = float(hprof_eval._stark_profile(setup, j0, freq4_dbg, delstark_dbg, dop_dbg))
                                        focus_typm1_doppler_sum += doppler_dbg
                                        focus_typm1_lorentz_sum += lorentz_dbg
                                        focus_typm1_stark_sum += stark_dbg
                                        focus_typm1_hprof_sum += float(hprof)
                                        focus_typm1_min_beta = min(focus_typm1_min_beta, float(beta_dbg))
                                        focus_typm1_max_beta = max(focus_typm1_max_beta, float(beta_dbg))
                                        if ifcore_dbg:
                                            focus_typm1_ifcore_count += 1
                                        else:
                                            focus_typm1_noncore_count += 1
                                        if nwid_dbg == 1:
                                            focus_typm1_nwid1_count += 1
                                        elif nwid_dbg == 2:
                                            focus_typm1_nwid2_count += 1
                                        else:
                                            focus_typm1_nwid3_count += 1
                                        if float(cv) > focus_typm1_top_cell_cv:
                                            focus_typm1_top_cell_hprof_parts = {
                                                "ifcore": bool(ifcore_dbg),
                                                "nwid": int(nwid_dbg),
                                                "beta": float(beta_dbg),
                                                "doppler": float(doppler_dbg),
                                                "lorentz": float(lorentz_dbg),
                                                "stark": float(stark_dbg),
                                                "hprof": float(hprof),
                                                "hfwid": float(hfwid_dbg),
                                                "delt": float(delt_dbg),
                                            }
                        # #endregion
                        if float(cv) > focus_typm1_top_cell_cv:
                            focus_typm1_top_cell_cv = float(cv)
                            focus_typm1_top_cell_meta = {
                                "iline": int(iline),
                                "j0": int(j0),
                                "iw": int(iw),
                                "wlvac_nm": float(wlvac),
                                "waveset_nm": float(waveset[iw]),
                                "center": float(center),
                                "hprof": float(hprof),
                                "cv": float(cv),
                            }
                        # #endregion
                    xlines[j0, iw] += cv
                    if in_focus_cell and float(hprof) > line_typm1_max_hprof:
                        line_typm1_max_hprof = float(hprof)
                    if in_focus_cell and float(cv) > line_typm1_max_cv:
                        line_typm1_max_cv = float(cv)
                    if cv < float(tab[j0, nucont0]):
                        if trace_focus and trace_in_focus(wlvac_nm=float(waveset[iw]), j0=j0):
                            trace_emit(
                                event="h_blue_break",
                                iter_num=1,
                                line_num_1b=iline + 1,
                                depth_1b=j0 + 1,
                                nu_1b=iw + 1,
                                type_code=-1,
                                wlvac_nm=float(waveset[iw]),
                                center=float(center),
                                adamp=float(nucont0 + 1),
                                cv=float(cv),
                                tabcont=float(tab[j0, nucont0]),
                                branch="xlinop_h",
                                reason="cv_lt_tab",
                            )
                        if in_focus_cell:
                            line_typm1_blue_break_tab += 1
                        break
            lineused += 1
            if line_may_hit_focus:
                line_focus_after = float(np.sum(np.asarray(xlines[focus_layer_lo:focus_layer_hi, :], dtype=np.float64)[:, focus_wave_mask]))
                line_added = line_focus_after - line_focus_before
                if in_focus_band:
                    focus_typm1_inband_added += line_added
                else:
                    focus_typm1_outband_added += line_added
                if line_added > focus_line_max_added:
                    focus_line_max_added = line_added
                    focus_line_max_iline = int(iline)
                    focus_line_max_typ = int(typ)
                    focus_line_max_wlvac = float(wlvac)
                    focus_line_max_inband = bool(in_focus_band)
            if in_focus_band:
                line_typm1_after = float(np.sum(np.asarray(xlines[focus_layer_lo:focus_layer_hi, :], dtype=np.float64)[:, focus_wave_mask]))
                line_typm1_added = line_typm1_after - line_typm1_before
                focus_typm1_line_summaries.append(
                    f"iline={int(iline)} wlvac={wlvac:.6f} nb={int(nblo)}->{int(nbup)} added={line_typm1_added:.3f} "
                    f"red_steps={int(line_typm1_red_steps)} blue_steps={int(line_typm1_blue_steps)} "
                    f"red_break_tab={int(line_typm1_red_break_tab)} blue_break_tab={int(line_typm1_blue_break_tab)} "
                    f"blue_break_wcon={int(line_typm1_blue_break_wcon)} max_cv={float(line_typm1_max_cv):.6e} "
                    f"max_hprof={float(line_typm1_max_hprof):.6e} max_center={float(line_typm1_max_center):.6e} "
                    f"min_tab={float(line_typm1_min_tabcont):.6e} max_tab={float(line_typm1_max_tabcont):.6e}"
                )
            continue

        if typ == 1:
            # AUTOIONIZING LINE
            if nelion < 1 or nelion > xnfp_arr.shape[1]:
                continue
            frelin = _C_NM / max(wlvac, 1e-300)
            ashore = gammas
            bshore = gammaw
            g = gf
            if bshore == 0.0 or gammar == 0.0:
                continue
            for j0 in range(nrhox):
                center = bshore * g * float(xnfp_arr[j0, nelion - 1]) / max(float(rho_arr[j0]), 1e-300)
                if center < float(tab[j0, nucont0]):
                    continue
                center *= _fastex(elo * float(hckt_arr[j0]), extab, extabf)
                if center < float(tab[j0, nucont0]):
                    continue
                for iw in range(nu0, min(nu0 + 2001, numnu)):
                    epsil = 2.0 * (_C_NM / max(float(waveset[iw]), 1e-300) - frelin) / gammar
                    cv = center * (ashore * epsil + bshore) / (epsil * epsil + 1.0) / bshore
                    xlines[j0, iw] += cv
                    if cv < float(tab[j0, nucont0]):
                        break
                for ired in range(1, 2001):
                    iw = max(nu0 - ired, 0)
                    epsil = 2.0 * (_C_NM / max(float(waveset[iw]), 1e-300) - frelin) / gammar
                    cv = center * (ashore * epsil + bshore) / (epsil * epsil + 1.0) / bshore
                    xlines[j0, iw] += cv
                    if cv < float(tab[j0, nucont0]):
                        break
            lineused += 1
            if line_may_hit_focus:
                line_focus_after = float(np.sum(np.asarray(xlines[focus_layer_lo:focus_layer_hi, :], dtype=np.float64)[:, focus_wave_mask]))
                line_added = line_focus_after - line_focus_before
                if in_focus_band:
                    focus_typ1_inband_added += line_added
                else:
                    focus_typ1_outband_added += line_added
                if line_added > focus_line_max_added:
                    focus_line_max_added = line_added
                    focus_line_max_iline = int(iline)
                    focus_line_max_typ = int(typ)
                    focus_line_max_wlvac = float(wlvac)
                    focus_line_max_inband = bool(in_focus_band)
            continue

        if typ != 0 and typ != 3:
            # MERGED CONTINUUM
            if nelion < 1 or nelion > xnfp_arr.shape[1]:
                continue
            z = 2.0 if nelion == 4 else 1.0
            nlast = float(typ)
            if nlast == 0.0:
                continue
            wshift = 1.0e7 / (1.0e7 / wlvac - 109737.312 * z * z / (nlast * nlast))
            xsectg = gf
            for j0 in range(nrhox):
                wmerge = 1.0e7 / (1.0e7 / wlvac - float(emerge[j0]) * z * z)
                wmax = max(wmerge, wshift)
                con = xsectg * float(xnfp_arr[j0, nelion - 1]) * _fastex(elo * float(hckt_arr[j0]), extab, extabf) / max(float(rho_arr[j0]), 1e-300)
                for iw in range(nu0, min(nu0 + 1001, numnu)):
                    if wmax < float(waveset[iw]):
                        break
                    xlines[j0, iw] += con
            lineused += 1
            if line_may_hit_focus:
                line_focus_after = float(np.sum(np.asarray(xlines[focus_layer_lo:focus_layer_hi, :], dtype=np.float64)[:, focus_wave_mask]))
                line_added = line_focus_after - line_focus_before
                if in_focus_band:
                    focus_merged_inband_added += line_added
                else:
                    focus_merged_outband_added += line_added
                if line_added > focus_line_max_added:
                    focus_line_max_added = line_added
                    focus_line_max_iline = int(iline)
                    focus_line_max_typ = int(typ)
                    focus_line_max_wlvac = float(wlvac)
                    focus_line_max_inband = bool(in_focus_band)
            continue

        # NORMAL LINE (TYPE 0) and PRD LINE (TYPE 3 -> same)
        if in_focus_band:
            focus_band_records_total += 1
            focus_band_records_typ0 += 1
        cgf = gf
        ncon_eff = 0 if ncon > 10 else ncon
        for j1 in range(8, nrhox + 1, 8):
            ifj[j1 + 1] = 0
            j0 = j1 - 1
            if nelion < 1 or nelion > xnfdop_arr.shape[1]:
                continue
            center = cgf * float(xnfdop_arr[j0, nelion - 1])
            if center < float(tab[j0, nucont0]):
                if in_focus_band and focus_layer_lo <= j0 < focus_layer_hi:
                    focus_center_skip += 1
                continue
            center *= _fastex(elo * float(hckt_arr[j0]), extab, extabf)
            if center < float(tab[j0, nucont0]):
                if in_focus_band and focus_layer_lo <= j0 < focus_layer_hi:
                    focus_center_skip += 1
                continue
            dop = float(dopple_arr[j0, nelion - 1])
            if dop <= 0.0:
                continue
            ifj[j1 + 1] = 1
            adamp = (gammar + gammas * float(xne_arr[j0]) + gammaw * float(txnxn[j0])) / dop
            dopwave = dop * wlvac
            if trace_in_focus(wlvac_nm=wlvac, j0=j0):
                trace_emit(
                    event="center_pass",
                    iter_num=1,
                    line_num_1b=iline + 1,
                    depth_1b=j0 + 1,
                    nu_1b=nu0 + 1,
                    type_code=0 if typ == 0 else typ,
                    wlvac_nm=wlvac,
                    center=center,
                    adamp=adamp,
                    cv=0.0,
                    tabcont=float(tab[j0, nucont0]),
                    branch="xlinop_typ0",
                    reason="center_gate",
                )
            wcon = None
            if ncon_eff > 0 and 1 <= nelionx <= contx.shape[1]:
                den = float(contx[ncon_eff - 1, nelionx - 1]) - float(emerge[j0])
                if den != 0.0:
                    wcon = 1.0e7 / den
            if wcon is not None and wlvac < wcon:
                if in_focus_band and focus_layer_lo <= j0 < focus_layer_hi:
                    focus_wcon_skip += 1
                continue
            debug_stats_call: dict[str, int | float] | None = None
            before_focus = 0.0
            if in_focus_band and focus_layer_lo <= j0 < focus_layer_hi:
                before_focus = float(np.sum(np.asarray(xlines[j0, :], dtype=np.float64)[focus_wave_mask]))
                if adamp <= 0.2:
                    focus_adamp_le_02 += 1
                else:
                    focus_adamp_gt_02 += 1
                debug_stats_call = {}
            _accumulate_wings(
                xlines=xlines,
                j0=j0,
                nu0=nu0,
                wlvac=wlvac,
                center=center,
                adamp=adamp,
                dopwave=dopwave,
                tabcont_ref=float(tab[j0, nucont0]),
                waveset=waveset,
                h0tab=h0tab,
                h1tab=h1tab,
                h2tab=h2tab,
                wing_steps=2000,
                blue_cutoff=wcon,
                debug_stats=debug_stats_call,
            )
            if debug_stats_call is not None:
                after_focus = float(np.sum(np.asarray(xlines[j0, :], dtype=np.float64)[focus_wave_mask]))
                call_added = after_focus - before_focus
                focus_typ0_added += call_added
                focus_typ0_call_count += 1
                if call_added > focus_typ0_max_call_added:
                    focus_typ0_max_call_added = call_added
                    focus_typ0_max_call_meta = {
                        "iline": int(iline),
                        "j0": int(j0),
                        "wlvac_nm": float(wlvac),
                        "adamp": float(adamp),
                        "center": float(center),
                    }
                focus_red_steps += int(debug_stats_call.get("red_steps", 0))
                focus_blue_steps += int(debug_stats_call.get("blue_steps", 0))
                focus_red_break_tabcont += int(debug_stats_call.get("red_break_tabcont", 0))
                focus_blue_break_tabcont += int(debug_stats_call.get("blue_break_tabcont", 0))
                focus_blue_break_cutoff += int(debug_stats_call.get("blue_break_cutoff", 0))
        for k1 in range(8, nrhox + 1, 8):
            if ifj[k1 - 7] + ifj[k1 + 1] == 0:
                continue
            for j1 in range(k1 - 7, k1):
                j0 = j1 - 1
                if nelion < 1 or nelion > xnfdop_arr.shape[1]:
                    continue
                center = cgf * float(xnfdop_arr[j0, nelion - 1])
                if center < float(tab[j0, nucont0]):
                    if in_focus_band and focus_layer_lo <= j0 < focus_layer_hi:
                        focus_center_skip += 1
                    continue
                center *= _fastex(elo * float(hckt_arr[j0]), extab, extabf)
                if center < float(tab[j0, nucont0]):
                    if in_focus_band and focus_layer_lo <= j0 < focus_layer_hi:
                        focus_center_skip += 1
                    continue
                dop = float(dopple_arr[j0, nelion - 1])
                if dop <= 0.0:
                    continue
                adamp = (gammar + gammas * float(xne_arr[j0]) + gammaw * float(txnxn[j0])) / dop
                dopwave = dop * wlvac
                if trace_in_focus(wlvac_nm=wlvac, j0=j0):
                    trace_emit(
                        event="center_pass_inner",
                        iter_num=1,
                        line_num_1b=iline + 1,
                        depth_1b=j0 + 1,
                        nu_1b=nu0 + 1,
                        type_code=0 if typ == 0 else typ,
                        wlvac_nm=wlvac,
                        center=center,
                        adamp=adamp,
                        cv=0.0,
                        tabcont=float(tab[j0, nucont0]),
                        branch="xlinop_typ0",
                        reason="ifj_fill",
                    )
                wcon = None
                if ncon_eff > 0 and 1 <= nelionx <= contx.shape[1]:
                    den = float(contx[ncon_eff - 1, nelionx - 1]) - float(emerge[j0])
                    if den != 0.0:
                        wcon = 1.0e7 / den
                if wcon is not None and wlvac < wcon:
                    if in_focus_band and focus_layer_lo <= j0 < focus_layer_hi:
                        focus_wcon_skip += 1
                    continue
                debug_stats_call: dict[str, int | float] | None = None
                before_focus = 0.0
                if in_focus_band and focus_layer_lo <= j0 < focus_layer_hi:
                    before_focus = float(np.sum(np.asarray(xlines[j0, :], dtype=np.float64)[focus_wave_mask]))
                    if adamp <= 0.2:
                        focus_adamp_le_02 += 1
                    else:
                        focus_adamp_gt_02 += 1
                    debug_stats_call = {}
                _accumulate_wings(
                    xlines=xlines,
                    j0=j0,
                    nu0=nu0,
                    wlvac=wlvac,
                    center=center,
                    adamp=adamp,
                    dopwave=dopwave,
                    tabcont_ref=float(tab[j0, nucont0]),
                    waveset=waveset,
                    h0tab=h0tab,
                    h1tab=h1tab,
                    h2tab=h2tab,
                    wing_steps=2000,
                    blue_cutoff=wcon,
                    debug_stats=debug_stats_call,
                )
                if debug_stats_call is not None:
                    after_focus = float(np.sum(np.asarray(xlines[j0, :], dtype=np.float64)[focus_wave_mask]))
                    call_added = after_focus - before_focus
                    focus_typ0_added += call_added
                    focus_typ0_call_count += 1
                    if call_added > focus_typ0_max_call_added:
                        focus_typ0_max_call_added = call_added
                        focus_typ0_max_call_meta = {
                            "iline": int(iline),
                            "j0": int(j0),
                            "wlvac_nm": float(wlvac),
                            "adamp": float(adamp),
                            "center": float(center),
                        }
                    focus_red_steps += int(debug_stats_call.get("red_steps", 0))
                    focus_blue_steps += int(debug_stats_call.get("blue_steps", 0))
                    focus_red_break_tabcont += int(debug_stats_call.get("red_break_tabcont", 0))
                    focus_blue_break_tabcont += int(debug_stats_call.get("blue_break_tabcont", 0))
                    focus_blue_break_cutoff += int(debug_stats_call.get("blue_break_cutoff", 0))
        lineused += 1
        if line_may_hit_focus:
            line_focus_after = float(np.sum(np.asarray(xlines[focus_layer_lo:focus_layer_hi, :], dtype=np.float64)[:, focus_wave_mask]))
            line_added = line_focus_after - line_focus_before
            if in_focus_band:
                focus_typ0_inband_added += line_added
            else:
                focus_typ0_outband_added += line_added
            if line_added > focus_line_max_added:
                focus_line_max_added = line_added
                focus_line_max_iline = int(iline)
                focus_line_max_typ = int(typ)
                focus_line_max_wlvac = float(wlvac)
                focus_line_max_inband = bool(in_focus_band)

    return LineOpacityState(xlines=xlines, lineused=lineused)

