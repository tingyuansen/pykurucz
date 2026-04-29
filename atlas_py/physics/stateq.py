"""NLTE H-minus statistical equilibrium (STATEQ) ported from ``atlas12.for``.

Fortran reference:
- ``STATEQ`` (atlas12.for lines 994-1151): three-mode H-level departure
  coefficient solver.
- ``COULX`` (atlas12.for lines 5766-5784): H photoionization cross-section.
- ``COULBF1S`` (atlas12.for lines 5785-5814): H 1s Gaunt factor table.
- ``EXPI`` (atlas12.for lines 16024-16068): exponential integral E_n(x).
- ``SOLVIT`` (atlas12.for lines 1357-1413): Gaussian elimination solver.

Only called when NLTEON=1 (deck card ``NLTE``).  In the default LTE path
NLTEON=0 so these functions are never invoked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# COULBF1S Gaunt-factor table (atlas12.for DATA GAUNT1S, lines 5789-5806)
# 151 entries, index step Δlog₁₀(ν/ν_thr) = 0.02
# ---------------------------------------------------------------------------
_GAUNT1S = np.array(
    [
        0.7973, 0.8094, 0.8212, 0.8328, 0.8439, 0.8548, 0.8653, 0.8754, 0.8852,
        0.8946, 0.9035, 0.9120, 0.9201, 0.9278, 0.9351, 0.9420, 0.9484, 0.9544,
        0.9601, 0.9653, 0.9702, 0.9745, 0.9785, 0.9820, 0.9852, 0.9879, 0.9903,
        0.9922, 0.9938, 0.9949, 0.9957, 0.9960, 0.9960, 0.9957, 0.9949, 0.9938,
        0.9923, 0.9905, 0.9884, 0.9859, 0.9832, 0.9801, 0.9767, 0.9730, 0.9688,
        0.9645, 0.9598, 0.9550, 0.9499, 0.9445, 0.9389, 0.9330, 0.9269, 0.9206,
        0.9140, 0.9071, 0.9001, 0.8930, 0.8856, 0.8781, 0.8705, 0.8627, 0.8546,
        0.8464, 0.8381, 0.8298, 0.8213, 0.8128, 0.8042, 0.7954, 0.7866, 0.7777,
        0.7685, 0.7593, 0.7502, 0.7410, 0.7318, 0.7226, 0.7134, 0.7042, 0.6951,
        0.6859, 0.6767, 0.6675, 0.6584, 0.6492, 0.6401, 0.6310, 0.6219, 0.6129,
        0.6039, 0.5948, 0.5859, 0.5769, 0.5680, 0.5590, 0.5502, 0.5413, 0.5324,
        0.5236, 0.5148, 0.5063, 0.4979, 0.4896, 0.4814, 0.4733, 0.4652, 0.4572,
        0.4493, 0.4415, 0.4337, 0.4261, 0.4185, 0.4110, 0.4035, 0.3962, 0.3889,
        0.3818, 0.3749, 0.3680, 0.3611, 0.3544, 0.3478, 0.3413, 0.3348, 0.3285,
        0.3222, 0.3160, 0.3099, 0.3039, 0.2980, 0.2923, 0.2866, 0.2810, 0.2755,
        0.2701, 0.2648, 0.2595, 0.2544, 0.2493, 0.2443, 0.2394, 0.2345, 0.2298,
        0.2251, 0.2205, 0.2160, 0.2115, 0.2072, 0.2029, 0.1987,
    ],
    dtype=np.float64,
)

# ---------------------------------------------------------------------------
# H oscillator strengths F(I,K) for I=1..8, K=1..8 (upper triangle only)
# (atlas12.for DATA F lines 1026-1029, column-major in Fortran)
# Here stored as a Python (8,8) row-major array; F[i,k] for i<k (0-based).
# ---------------------------------------------------------------------------
_F_OSC = np.array(
    [
        # i=0 (level 1)
        [0., 0.4162, 0.07910, 0.02899, 0.01394, 0.007800, 0.004814, 0.003184],
        # i=1 (level 2)
        [0., 0., 0.6408, 0.1193, 0.04467, 0.02209, 0.01271, 0.008037],
        # i=2 (level 3)
        [0., 0., 0., 0.8420, 0.1506, 0.05585, 0.02768, 0.01604],
        # i=3 (level 4)
        [0., 0., 0., 0., 1.038, 0.1794, 0.06551, 0.03229],
        # i=4 (level 5)
        [0., 0., 0., 0., 0., 1.231, 0.2070, 0.07455],
        # i=5 (level 6)
        [0., 0., 0., 0., 0., 0., 1.425, 0.2340],
        # i=6 (level 7)
        [0., 0., 0., 0., 0., 0., 0., 1.615],
        # i=7 (level 8)
        [0., 0., 0., 0., 0., 0., 0., 0.],
    ],
    dtype=np.float64,
)

# ---------------------------------------------------------------------------
# COULX polynomial coefficients (atlas12.for DATA A/B/C lines 5770-5772)
# Index 0 = level N=1, ..., 5 = level N=6
# ---------------------------------------------------------------------------
_COULX_A = np.array([0.9916, 1.105, 1.101, 1.101, 1.102, 1.0986], dtype=np.float64)
_COULX_B = np.array([2.719e13, -2.375e14, -9.863e13, -5.765e13, -3.909e13, -2.704e13],
                    dtype=np.float64)
_COULX_C = np.array([-2.268e30, 4.077e28, 1.035e28, 4.593e27, 2.371e27, 1.229e27],
                    dtype=np.float64)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def coulbf1s(freq: float, z: float) -> float:
    """Bound-free Gaunt factor for H 1s level (atlas12.for lines 5785-5814).

    Returns 0 below the threshold frequency.
    """
    threshold = 3.28805e15 * z * z
    if freq < threshold:
        return 0.0
    elog = math.log10(freq / z**2 / 3.28805e15)
    i = int(elog / 0.02)
    i = max(0, min(i, 149))  # 0-based; table has 151 entries (0..150)
    g = _GAUNT1S[i] + (_GAUNT1S[i + 1] - _GAUNT1S[i]) / 0.02 * (elog - i * 0.02)
    return float(g)


def coulx(n: int, freq: float, z: float) -> float:
    """Hydrogen photoionization cross-section (atlas12.for lines 5766-5784).

    Parameters
    ----------
    n:   Principal quantum number (1..6).
    freq: Frequency (Hz).
    z:   Charge.
    """
    threshold = z * z * 3.28805e15 / float(n * n)
    if freq < threshold:
        return 0.0
    cx = 2.815e29 / (freq**3) / float(n**5) * z**4
    if n > 6:
        return cx
    if n == 1:
        return cx * coulbf1s(freq, z)
    idx = n - 1  # 0-based
    ratio = z * z / freq
    return cx * (_COULX_A[idx] + (_COULX_B[idx] + _COULX_C[idx] * ratio) * ratio)


def expi(n: int, x: float) -> float:
    """Exponential integral E_n(x) (atlas12.for lines 16024-16068).

    Uses rational approximations after Cody & Thacher (1968).
    Only handles positive x. Returns 0 for x <= 0.
    """
    if x <= 0.0:
        return 0.0
    ex = math.exp(-x)
    # E1(x) approximation
    if x > 4.0:
        e0 = -0.999999999998447
        e1 = -26.6271060431811
        e2 = -241.055827097015
        e3 = -895.927957772937
        e4 = -1298.85688746484
        e5 = -545.374158883133
        e6 = -5.66575206533869
        f1 = 28.6271060422192
        f2 = 292.310039388533
        f3 = 1332.78537748257
        f4 = 2777.61949509163
        f5 = 2404.01713225909
        f6 = 631.657483280800
        ex1 = (
            ex
            + ex
            * (e0 + (e1 + (e2 + (e3 + (e4 + (e5 + e6 / x) / x) / x) / x) / x) / x)
            / (x + f1 + (f2 + (f3 + (f4 + (f5 + f6 / x) / x) / x) / x) / x)
        ) / x
    elif x > 1.0:
        c0 = 4.65627107975096e-7
        c1 = 0.999979577051595
        c2 = 9.04161556946329
        c3 = 24.3784088791317
        c4 = 23.0192559391333
        c5 = 6.90522522784444
        c6 = 0.430967839469389
        d1 = 10.0411643829054
        d2 = 32.4264210695138
        d3 = 41.2807841891424
        d4 = 20.4494785013794
        d5 = 3.31909213593302
        d6 = 0.103400130404874
        ex1 = ex * (c6 + (c5 + (c4 + (c3 + (c2 + (c1 + c0 * x) * x) * x) * x) * x) * x) / (
            d6 + (d5 + (d4 + (d3 + (d2 + (d1 + x) * x) * x) * x) * x) * x)
    else:
        a0 = -44178.5471728217
        a1 = 57721.7247139444
        a2 = 9938.31388962037
        a3 = 1842.11088668000
        a4 = 101.093806161906
        a5 = 5.03416184097568
        b0 = 76537.3323337614
        b1 = 32597.1881290275
        b2 = 6106.10794245759
        b3 = 635.419418378382
        b4 = 37.2298352833327
        ex1 = (
            (a0 + (a1 + (a2 + (a3 + (a4 + a5 * x) * x) * x) * x) * x)
            / (b0 + (b1 + (b2 + (b3 + (b4 + x) * x) * x) * x) * x)
            - math.log(x)
        )
    # Recurrence E_n(x) = (e^-x - x * E_{n-1}(x)) / (n-1) for n >= 2
    if n == 1:
        return ex1
    for i in range(1, n):
        ex1 = (ex - x * ex1) / float(i)
    return ex1


def _solvit(a: np.ndarray, n: int, b: np.ndarray) -> np.ndarray:
    """Gaussian elimination with partial pivoting (atlas12.for SOLVIT 1357-1413).

    Parameters
    ----------
    a:   (n, n) coefficient matrix (modified in place).
    n:   System size.
    b:   Right-hand side vector (modified in place → solution on return).

    Returns
    -------
    b:   Solution vector (same array).
    """
    a = a.copy()
    b = b.copy()
    ipivot = np.zeros(n, dtype=np.int64)
    for i in range(n - 1):
        # Find pivot.
        m = i
        for k in range(i + 1, n):
            if abs(a[k, i]) > abs(a[m, i]):
                m = k
        ipivot[i] = m
        if m != i:
            for k in range(i + 1, n):
                a[i, k], a[m, k] = a[m, k], a[i, k]
        pivot = 1.0 / a[m, i]
        a[m, i] = a[i, i]
        a[i, i] = pivot
        for k in range(i + 1, n):
            a[k, i] *= pivot
        for j in range(i + 1, n):
            c = a[i, j]
            if c == 0.0:
                continue
            for k in range(i + 1, n):
                a[k, j] -= a[k, i] * c
    a[n - 1, n - 1] = 1.0 / a[n - 1, n - 1]
    for i in range(n - 1):
        m = ipivot[i]
        if m != i:
            b[m], b[i] = b[i], b[m]
        c = b[i]
        for k in range(i + 1, n):
            b[k] -= a[k, i] * c
    j1 = n - 1
    for i in range(n - 1):
        j = j1
        j1 -= 1
        b[j] *= a[j, j]
        c = b[j]
        for k in range(j1 + 1):
            b[k] -= a[k, j] * c
    b[0] *= a[0, 0]
    return b


# ---------------------------------------------------------------------------
# Accumulator state for STATEQ (mirrors Fortran arrays local to STATEQ)
# ---------------------------------------------------------------------------

@dataclass
class StateqAccumulator:
    """Per-layer STATEQ frequency-integral accumulators.

    Shape of each array: ``(nrhox,)`` or ``(nrhox, 6)`` as noted.
    """

    nrhox: int
    qradik: np.ndarray = field(default_factory=lambda: np.array([]))   # (nrhox, 6)
    qradki: np.ndarray = field(default_factory=lambda: np.array([]))   # (nrhox, 6)
    dqrad:  np.ndarray = field(default_factory=lambda: np.array([]))   # (nrhox, 6)
    qrdhmk: np.ndarray = field(default_factory=lambda: np.array([]))   # (nrhox,)
    qrdkhm: np.ndarray = field(default_factory=lambda: np.array([]))   # (nrhox,)
    dqrd:   np.ndarray = field(default_factory=lambda: np.array([]))   # (nrhox,)
    told:   np.ndarray = field(default_factory=lambda: np.array([]))   # (nrhox,)

    def __post_init__(self) -> None:
        n = self.nrhox
        self.qradik  = np.zeros((n, 6), dtype=np.float64)
        self.qradki  = np.zeros((n, 6), dtype=np.float64)
        self.dqrad   = np.zeros((n, 6), dtype=np.float64)
        self.qrdhmk  = np.zeros(n, dtype=np.float64)
        self.qrdkhm  = np.zeros(n, dtype=np.float64)
        self.dqrd    = np.zeros(n, dtype=np.float64)
        self.told    = np.zeros(n, dtype=np.float64)


def stateq_init(acc: StateqAccumulator, temperature_k: np.ndarray) -> None:
    """MODE=1: Erase frequency integrals and record current T (atlas12.for 1031-1041)."""
    n = acc.nrhox
    acc.qradik[:] = 0.0
    acc.qradki[:] = 0.0
    acc.dqrad[:]  = 0.0
    acc.qrdhmk[:] = 0.0
    acc.qrdkhm[:] = 0.0
    acc.dqrd[:]   = 0.0
    acc.told[:n]  = temperature_k[:n]


def stateq_accumulate(
    acc: StateqAccumulator,
    *,
    freq: float,
    rcowt: float,
    jnu: np.ndarray,
    hkt: np.ndarray,
    temperature_k: np.ndarray,
) -> None:
    """MODE=2: Accumulate radiative rates at one frequency (atlas12.for 1042-1063).

    Parameters
    ----------
    freq:  Frequency in Hz.
    rcowt: Frequency quadrature weight.
    jnu:   Mean intensity J_nu at each depth layer, shape ``(nrhox,)``.
    hkt:   h*nu/(k*T) at each layer, shape ``(nrhox,)``.
    temperature_k:  Temperature array (K), shape ``(nrhox,)``.
    """
    rfrwt = 12.5664 / 6.6256e-27 * rcowt / freq
    hvc = 2.0 * 6.6256e-27 * freq * (freq / 2.99792458e10) ** 2

    # H bound-free cross sections for levels 2..6 (N=2 to 6, 0-based idx 0..4).
    hcont = np.zeros(6, dtype=np.float64)  # index 0 unused, 1..5 for N=2..6
    for n in range(2, 7):
        hcont[n - 1] = coulx(n, freq, 1.0)

    # H-minus bound-free cross-section (Doughty & Fraser approximation).
    freq_hz = freq
    hminbf = 0.0
    if 1.8259e14 < freq_hz < 2.111e14:
        hminbf = 3.695e-16 + (-1.251e-1 + 1.052e13 / freq_hz) / freq_hz
    elif freq_hz >= 2.111e14:
        hminbf = 6.801e-20 + (
            5.358e-3 + (1.481e13 + (-5.519e27 + 4.808e41 / freq_hz) / freq_hz) / freq_hz
        ) / freq_hz

    n = acc.nrhox
    ehvkt = np.exp(-hkt[:n] * freq / 1.0)  # = exp(-hnu/kT) -- hkt is h/(kT) in 1/Hz units
    # Fortran: EHVKT(J) is exp(-h*nu/(kT)) already stored per layer per freq.
    # Here we receive hkt = h/(k*T), so ehvkt = exp(-freq*hkt).
    rj   = rfrwt * jnu[:n]
    rje  = rfrwt * ehvkt * (jnu[:n] + hvc)
    rjedt = rje * hkt[:n] * freq / temperature_k[:n]

    for i_level in range(1, 6):  # N=2..6, 0-based i_level=1..5
        acc.qradik[:n, i_level] += hcont[i_level] * rj
        acc.dqrad[:n, i_level]  += hcont[i_level] * rjedt
        acc.qradki[:n, i_level] += hcont[i_level] * rje

    acc.qrdhmk[:n] += hminbf * rj
    acc.dqrd[:n]   += hminbf * rjedt
    acc.qrdkhm[:n] += hminbf * rje


def stateq_solve(
    acc: StateqAccumulator,
    *,
    temperature_k: np.ndarray,
    xne: np.ndarray,
    tkev: np.ndarray,
    xnfp: np.ndarray,
    bhyd: np.ndarray,
    bmin: np.ndarray,
) -> None:
    """MODE=3: Solve H-minus and H-level departure coefficients (atlas12.for 1064-1150).

    Modifies ``bhyd`` (shape ``(nrhox, 6)``) and ``bmin`` (shape ``(nrhox,)``) in place.

    Parameters
    ----------
    temperature_k:  Temperature (K).
    xne:            Electron number density (cm^-3).
    tkev:           kT in eV, shape ``(nrhox,)``.
    xnfp:           Ionic number densities, shape ``(nrhox, mion)``.
                    xnfp[:, 0] = H I (neutral H), xnfp[:, 1] = H II (H+).
    bhyd:           On input: current departure coefficients (or ones for LTE start).
                    On output: updated NLTE departure coefficients.
    bmin:           On output: H-minus departure coefficient at each layer.
    """
    n = acc.nrhox
    for j in range(n):
        dt = temperature_k[j] - acc.told[j]
        theta = 5040.0 / temperature_k[j]

        # H-minus departure coefficient (Saha-like ionization equilibrium).
        qelect = 10.0 ** (-8.7) * theta**1.5 * xne[j]
        qassoc = 10.0 ** (-8.7) * 2.0 * bhyd[j, 0] * xnfp[j, 0]
        qcharg = 10.0 ** (-7.4) * theta**0.333333 * xnfp[j, 1]

        acc.qrdkhm[j] += acc.dqrd[j] * dt
        denom = acc.qrdhmk[j] + qelect + qassoc + qcharg
        numer = acc.qrdkhm[j] + qelect + qassoc + qcharg
        bmin[j] = numer / max(denom, 1e-300)

        # Statistical equilibrium for H levels 1..6.
        th = 13.595 / max(tkev[j], 1e-300)
        t_j = temperature_k[j]

        # Build 8x8 collision matrix QCOLL.
        qcoll = np.zeros((8, 8), dtype=np.float64)
        for i in range(8):
            y = float(i + 1)
            # Bound-free collisional rate.
            qcoll[i, i] = 2.2e-8 * y**3 / math.sqrt(th) * math.exp(-th / y**2) * xne[j]
            if i == 7:
                continue
            for k in range(i + 1, 8):
                z = float(k + 1)
                gik = 1.0 / y**2 - 1.0 / z**2
                x0 = th * gik
                f_ik = _F_OSC[i, k]
                if f_ik == 0.0 or gik <= 0.0:
                    continue
                q = (
                    2.186e-10
                    * f_ik
                    / gik**2
                    * x0
                    * math.sqrt(t_j)
                    * (expi(1, x0) + 0.148 * x0 * expi(5, x0))
                )
                qcoll[i, k] = q * xne[j]
                qcoll[k, i] = qcoll[i, k] * (y / z) ** 2 * math.exp(x0)

        # Build 6x6 statistical equilibrium matrix A and RHS b.
        a_mat = np.zeros((6, 6), dtype=np.float64)
        right = np.zeros(6, dtype=np.float64)

        for idx in range(6):
            acc.qradki[j, idx] += acc.dqrad[j, idx] * dt
            right[idx] = acc.qradki[j, idx] + qcoll[idx, idx] + qcoll[idx, 6] + qcoll[idx, 7]
            a_mat[idx, idx] = acc.qradik[j, idx]
            for k in range(8):
                a_mat[idx, idx] += qcoll[idx, k]
            for k_other in range(idx + 1, 6):
                a_mat[idx, k_other] = -qcoll[idx, k_other]
                a_mat[k_other, idx] = -qcoll[k_other, idx]

        # Solve A * x = right → departure coefficients for levels 1..6.
        solution = _solvit(a_mat, 6, right)
        bhyd[j, :] = solution
