"""Numerical helpers used by ATLAS12 JOSH solver.

Fortran references:
- `atlas12.for` `SUBROUTINE PARCOE`
- `atlas12.for` `SUBROUTINE INTEG`
- `atlas12.for` `SUBROUTINE DERIV`
- `atlas12.for` `FUNCTION MAP1`
"""

from __future__ import annotations

import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def _parcoe(f: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = f.size
    a = np.zeros(n, dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)
    c = np.zeros(n, dtype=np.float64)

    if n == 0:
        return a, b, c
    if n == 1:
        a[0] = f[0]
        return a, b, c

    c[0] = 0.0
    denom = x[1] - x[0]
    b[0] = (f[1] - f[0]) / denom if denom != 0.0 else 0.0
    a[0] = f[0] - x[0] * b[0]

    n1 = n - 1
    c[-1] = 0.0
    denom = x[-1] - x[n1 - 1]
    b[-1] = (f[-1] - f[n1 - 1]) / denom if denom != 0.0 else 0.0
    a[-1] = f[-1] - x[-1] * b[-1]

    if n == 2:
        return a, b, c

    for j in range(1, n1):
        j1 = j - 1
        denom = x[j] - x[j1]
        d = (f[j] - f[j1]) / denom if denom != 0.0 else 0.0
        denom1 = (x[j + 1] - x[j]) * (x[j + 1] - x[j1])
        term1 = f[j + 1] / denom1 if denom1 != 0.0 else 0.0
        denom2 = x[j + 1] - x[j1]
        denom3 = x[j + 1] - x[j]
        denom4 = x[j] - x[j1]
        part = 0.0
        if denom4 != 0.0:
            t1 = f[j1] / denom2 if denom2 != 0.0 else 0.0
            t2 = f[j] / denom3 if denom3 != 0.0 else 0.0
            part = (t1 - t2) / denom4
        c[j] = term1 + part
        b[j] = d - (x[j] + x[j1]) * c[j]
        a[j] = f[j1] - x[j1] * d + x[j] * x[j1] * c[j]

    c[1] = 0.0
    denom = x[2] - x[1]
    b[1] = (f[2] - f[1]) / denom if denom != 0.0 else 0.0
    a[1] = f[1] - x[1] * b[1]

    if n > 3:
        c[2] = 0.0
        denom = x[3] - x[2]
        b[2] = (f[3] - f[2]) / denom if denom != 0.0 else 0.0
        a[2] = f[2] - x[2] * b[2]

    for j in range(1, n1):
        if c[j] == 0.0:
            continue
        j1 = min(j + 1, n - 1)
        denom = abs(c[j1]) + abs(c[j])
        wt = abs(c[j1]) / denom if denom > 0.0 else 0.0
        a[j] = a[j1] + wt * (a[j] - a[j1])
        b[j] = b[j1] + wt * (b[j] - b[j1])
        c[j] = c[j1] + wt * (c[j] - c[j1])

    a[n1 - 1] = a[-1]
    b[n1 - 1] = b[-1]
    c[n1 - 1] = c[-1]
    return a, b, c


@jit(nopython=True, cache=True)
def _integ(x: np.ndarray, f: np.ndarray, start: float) -> np.ndarray:
    n = f.size
    fint = np.zeros(n, dtype=np.float64)
    if n == 0:
        return fint
    a, b, c = _parcoe(f, x)
    fint[0] = start
    if n == 1:
        return fint
    for i in range(n - 1):
        dx = x[i + 1] - x[i]
        term = a[i] + 0.5 * b[i] * (x[i + 1] + x[i])
        term += (c[i] / 3.0) * ((x[i + 1] + x[i]) * x[i + 1] + x[i] * x[i])
        fint[i + 1] = fint[i] + term * dx
    return fint


@jit(nopython=True, cache=True)
def _deriv(x: np.ndarray, f: np.ndarray) -> np.ndarray:
    n = f.size
    dfdx = np.zeros(n, dtype=np.float64)
    if n < 2:
        return dfdx
    dfdx[0] = (f[1] - f[0]) / (x[1] - x[0])
    dfdx[-1] = (f[-1] - f[-2]) / (x[-1] - x[-2])
    if n == 2:
        return dfdx

    s = abs(x[1] - x[0]) / (x[1] - x[0]) if x[1] != x[0] else 1.0
    for j in range(1, n - 1):
        scale = max(abs(f[j - 1]), abs(f[j]), abs(f[j + 1]))
        scale = scale / abs(x[j]) if x[j] != 0.0 else scale
        if scale == 0.0:
            scale = 1.0
        d1 = (f[j + 1] - f[j]) / (x[j + 1] - x[j]) / scale
        d0 = (f[j] - f[j - 1]) / (x[j] - x[j - 1]) / scale
        tan1 = d1 / (s * np.sqrt(1.0 + d1 * d1) + 1.0)
        tan0 = d0 / (s * np.sqrt(1.0 + d0 * d0) + 1.0)
        dfdx[j] = (tan1 + tan0) / (1.0 - tan1 * tan0) * scale
    return dfdx


@jit(nopython=True)
def _map1_kernel(
    xold: np.ndarray, fold: np.ndarray, xnew: np.ndarray
) -> tuple[np.ndarray, int]:
    """Piecewise-quadratic remapping matching Fortran MAP1 (atlas12.for 1299-1356).

    Fortran flow (1-based L, LL):
      label 10: advance L until XNEW(K) < XOLD(L)
      label 20: if L==LL skip; if L<=3 linear fallback (label 30)
      lines 1314-1315: if L>LL+1 or L==3 or L==4 → label 21 (fresh backward)
      else: save old forward as backward (lines 1316-1318)
      label 21: compute CBAC from L-2,L-1,L
      if L<NOLD → label 25; else label 22 (backward-only)
      label 25: compute CFOR from L-1,L,L+1; blend A,B,C
      label 30: linear fallback at boundaries
    """
    nold = xold.size
    nnew = xnew.size
    fnew = np.zeros(nnew, dtype=np.float64)
    if nold == 0 or nnew == 0:
        return fnew, 0

    xold1 = np.empty(nold + 1, dtype=np.float64)
    fold1 = np.empty(nold + 1, dtype=np.float64)
    xold1[1:] = xold
    fold1[1:] = fold

    l = 2
    ll = 0
    cfor = bfor = afor = 0.0
    cbac = bbac = abac = 0.0
    a = b = c = 0.0

    for k in range(1, nnew + 1):
        xk = xnew[k - 1]
        while True:
            if xk < xold1[l]:
                if l == ll:
                    break
                if l == 2 or l == 3:
                    l = min(nold, l)
                    c = 0.0
                    b = (fold1[l] - fold1[l - 1]) / (xold1[l] - xold1[l - 1])
                    a = fold1[l] - xold1[l] * b
                    ll = l
                    break
                l1 = l - 1
                if l > ll + 1 or l == 3 or l == 4:
                    # Fortran label 21: compute backward quadratic from L-2,L-1,L
                    d = (fold1[l1] - fold1[l - 2]) / (xold1[l1] - xold1[l - 2])
                    cbac = (
                        fold1[l]
                        / ((xold1[l] - xold1[l1]) * (xold1[l] - xold1[l - 2]))
                        + (fold1[l - 2] / (xold1[l] - xold1[l - 2]) - fold1[l1] / (xold1[l] - xold1[l1]))
                        / (xold1[l1] - xold1[l - 2])
                    )
                    bbac = d - (xold1[l1] + xold1[l - 2]) * cbac
                    abac = fold1[l - 2] - xold1[l - 2] * d + xold1[l1] * xold1[l - 2] * cbac
                else:
                    # Fortran lines 1316-1318: reuse previous forward as backward
                    cbac = cfor
                    bbac = bfor
                    abac = afor
                if l >= nold:
                    # Fortran label 22: backward-only (no forward stencil available)
                    c = cbac
                    b = bbac
                    a = abac
                    ll = l
                    break
                # Fortran label 25: compute forward quadratic from L-1,L,L+1
                d = (fold1[l] - fold1[l1]) / (xold1[l] - xold1[l1])
                cfor = (
                    fold1[l + 1] / ((xold1[l + 1] - xold1[l]) * (xold1[l + 1] - xold1[l1]))
                    + (fold1[l1] / (xold1[l + 1] - xold1[l1]) - fold1[l] / (xold1[l + 1] - xold1[l]))
                    / (xold1[l] - xold1[l1])
                )
                bfor = d - (xold1[l] + xold1[l1]) * cfor
                afor = fold1[l1] - xold1[l1] * d + xold1[l] * xold1[l1] * cfor
                wt = 0.0
                if abs(cfor) != 0.0:
                    wt = abs(cfor) / (abs(cfor) + abs(cbac))
                a = afor + wt * (abac - afor)
                b = bfor + wt * (bbac - bfor)
                c = cfor + wt * (cbac - cfor)
                ll = l
                break

            l += 1
            if l > nold:
                l = min(nold, l)
                c = 0.0
                b = (fold1[l] - fold1[l - 1]) / (xold1[l] - xold1[l - 1])
                a = fold1[l] - xold1[l] * b
                ll = l
                break

        fnew[k - 1] = a + (b + c * xk) * xk

    return fnew, ll


def _map1(
    xold: np.ndarray, fold: np.ndarray, xnew: np.ndarray
) -> tuple[np.ndarray, int]:
    fnew, maxj = _map1_kernel(xold, fold, xnew)
    # Fortran MAP1 returns LL-1 (atlas12.for line 1354).
    return fnew, max(maxj - 1, 0)
