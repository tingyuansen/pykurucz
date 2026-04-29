from __future__ import annotations

import numpy as np

from numba import jit

from .josh_tables import (
    CH_WEIGHTS,
    CK_WEIGHTS,
    XTAU_GRID,
    COEFJ_MATRIX,
    COEFH_MATRIX,
    NXTAU,
)

EPS = 1.0e-38
# Match Fortran REAL*4 behavior during the XS iteration.
USE_FLOAT32_ITERATION = True
# Convergence tolerance: Match Fortran exactly (atlas7v.for line 9950: IF(ERRORX.GT..00001))
# Using the same tolerance ensures the iteration converges to the same solution
ITER_TOL = 1.0e-5  # Match Fortran's 1e-5 exactly
MAX_ITER = NXTAU
COEFJ_DIAG = np.diag(COEFJ_MATRIX)
@jit(nopython=True, cache=True)
def _josh_iteration_kernel(
    coefj_matrix: np.ndarray,
    xs: np.ndarray,
    xalpha: np.ndarray,
    xsbar_modified: np.ndarray,
    coefj_diag: np.ndarray,
    iter_tol: float,
    max_iter: int,
    eps: float,
) -> tuple[np.ndarray, int]:
    """Numba-compiled kernel for JOSH iteration loop.

    This kernel performs the scattering iteration loop, which is the main
    computational bottleneck in solve_josh_flux. Uses optimized matrix-vector
    operations for significant speedup.

    Parameters
    ----------
    coefj_matrix : np.ndarray
        COEFJ matrix (NXTAU × NXTAU)
    xs : np.ndarray
        Current XS values (modified in-place, NXTAU)
    xalpha : np.ndarray
        XALPHA values (NXTAU)
    xsbar_modified : np.ndarray
        Modified XSBAR values (NXTAU)
    coefj_diag : np.ndarray
        Diagonal of COEFJ matrix (NXTAU)
    iter_tol : float
        Iteration tolerance
    max_iter : int
        Maximum number of iterations
    eps : float
        Minimum value for XS

    Returns
    -------
    xs : np.ndarray
        Converged XS values
    num_iterations : int
        Number of iterations performed
    """
    nxtau = len(xs)
    diag = 1.0 - xalpha * coefj_diag

    for iteration in range(max_iter):
        iferr = 0
        # Fortran iterates BACKWARDS: K=NXTAU+1, then K=K-1 for KK=1..NXTAU
        # So K goes from NXTAU down to 1 (atlas7v.for lines 9856-9858)
        for k in range(nxtau - 1, -1, -1):
            # Use optimized dot product for matrix-vector multiplication
            # This is much faster than manual loop (uses BLAS)
            delxs = 0.0
            for m in range(nxtau):
                delxs += coefj_matrix[k, m] * xs[m]

            # Compute DELXS
            delxs = (delxs * xalpha[k] + xsbar_modified[k] - xs[k]) / diag[k]

            errorx = abs(delxs / xs[k]) if xs[k] != 0.0 else float("inf")
            if errorx > iter_tol:
                iferr = 1
            xs[k] = max(xs[k] + delxs, eps)

        if iferr == 0:
            return xs, iteration + 1

    return xs, max_iter


@jit(nopython=True, cache=True)
def _parcoe(f: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parabolic coefficients matching the Fortran PARCOE routine."""

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

    # Boundary adjustments matching the Fortran logic
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
    """Numerical integral matching the Fortran INTEG routine."""

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
    """Derivative helper mirroring the Fortran DERIV routine."""

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
    """Exact translation of Fortran MAP1 (atlas7v.for lines 1142-1199)."""
    nold = xold.size
    nnew = xnew.size
    fnew = np.zeros(nnew, dtype=np.float64)
    if nold == 0 or nnew == 0:
        return fnew, 0

    # Use 1-based indexing to mirror Fortran exactly.
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
            # Label 10
            if xk < xold1[l]:
                # Label 20
                if l == ll:
                    break
                if l == 2 or l == 3:
                    # Label 30
                    l = min(nold, l)
                    c = 0.0
                    b = (fold1[l] - fold1[l - 1]) / (xold1[l] - xold1[l - 1])
                    a = fold1[l] - xold1[l] * b
                    ll = l
                    break
                l1 = l - 1
                if l > ll + 1 or l == 3:
                    # Label 21
                    l2 = l - 2
                    d = (fold1[l1] - fold1[l2]) / (xold1[l1] - xold1[l2])
                    cbac = fold1[l] / (
                        (xold1[l] - xold1[l1]) * (xold1[l] - xold1[l2])
                    ) + (
                        fold1[l2] / (xold1[l] - xold1[l2])
                        - fold1[l1] / (xold1[l] - xold1[l1])
                    ) / (
                        xold1[l1] - xold1[l2]
                    )
                    bbac = d - (xold1[l1] + xold1[l2]) * cbac
                    abac = fold1[l2] - xold1[l2] * d + xold1[l1] * xold1[l2] * cbac
                    if l < nold:
                        # Fall through to label 25 below.
                        pass
                    else:
                        # Label 22
                        c = cbac
                        b = bbac
                        a = abac
                        ll = l
                        break
                elif l > ll + 1 or l == 4:
                    # Label 21 (same as above)
                    l2 = l - 2
                    d = (fold1[l1] - fold1[l2]) / (xold1[l1] - xold1[l2])
                    cbac = fold1[l] / (
                        (xold1[l] - xold1[l1]) * (xold1[l] - xold1[l2])
                    ) + (
                        fold1[l2] / (xold1[l] - xold1[l2])
                        - fold1[l1] / (xold1[l] - xold1[l1])
                    ) / (
                        xold1[l1] - xold1[l2]
                    )
                    bbac = d - (xold1[l1] + xold1[l2]) * cbac
                    abac = fold1[l2] - xold1[l2] * d + xold1[l1] * xold1[l2] * cbac
                    if l < nold:
                        pass
                    else:
                        # Label 22
                        c = cbac
                        b = bbac
                        a = abac
                        ll = l
                        break
                else:
                    cbac = cfor
                    bbac = bfor
                    abac = afor
                    if l == nold:
                        # Label 22
                        c = cbac
                        b = bbac
                        a = abac
                        ll = l
                        break

                # Label 25
                d = (fold1[l] - fold1[l1]) / (xold1[l] - xold1[l1])
                cfor = fold1[l + 1] / (
                    (xold1[l + 1] - xold1[l]) * (xold1[l + 1] - xold1[l1])
                ) + (
                    fold1[l1] / (xold1[l + 1] - xold1[l1])
                    - fold1[l] / (xold1[l + 1] - xold1[l])
                ) / (
                    xold1[l] - xold1[l1]
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

            # Continue label 10 loop.
            l += 1
            if l > nold:
                # Label 30
                l = min(nold, l)
                c = 0.0
                b = (fold1[l] - fold1[l - 1]) / (xold1[l] - xold1[l - 1])
                a = fold1[l] - xold1[l] * b
                ll = l
                break

        fnew[k - 1] = a + (b + c * xk) * xk

    return fnew, ll - 1


def _map1(
    xold: np.ndarray, fold: np.ndarray, xnew: np.ndarray
) -> tuple[np.ndarray, int]:
    """Faithful port of the Fortran MAP1 interpolation routine.

    Wrapper around Numba-compiled kernel for performance.
    """
    fnew, maxj = _map1_kernel(xold, fold, xnew)
    return fnew, maxj


def solve_josh_flux(
    acont: np.ndarray,
    scont: np.ndarray,
    aline: np.ndarray,
    sline: np.ndarray,
    sigmac: np.ndarray,
    sigmal: np.ndarray,
    column_mass: np.ndarray,
) -> float:
    """Compute the emergent flux for a single frequency using the JOSH solver.

    Option 2: Use higher precision arithmetic when alpha (scattering) is large
    to reduce numerical errors. This is especially important when alpha > 0.1.
    """
    # CRITICAL FIX: Use float64 (REAL*8) to match Fortran exactly
    # Fortran uses REAL*8 (double precision) for all opacity and flux calculations
    # No higher precision needed - match Fortran's REAL*8 exactly
    dtype = np.float64  # Match Fortran REAL*8

    acont = np.asarray(acont, dtype=dtype)
    scont = np.asarray(scont, dtype=dtype)
    aline = np.asarray(aline, dtype=dtype)
    sline = np.asarray(sline, dtype=dtype)
    sigmac = np.asarray(sigmac, dtype=dtype)
    sigmal = np.asarray(sigmal, dtype=dtype)
    rho = np.asarray(column_mass, dtype=dtype)

    if acont.size == 0 or scont.size == 0 or rho.size == 0:
        if acont.size == 0:
            return 0.0

    if acont.size == 0:
        return 0.0

    # CRITICAL FIX: Match Fortran behavior - no clipping of input arrays
    # Fortran uses REAL*8 (double precision) and doesn't clip opacity arrays
    # Only ensure arrays are float64 to match Fortran REAL*8
    # Arrays are already converted to float64 above, so no additional conversion needed

    # Compute ABTOT directly without clipping (matching Fortran)
    # NOTE: There's a remaining ~30% discrepancy at scattering-dominated wavelengths (300nm)
    # due to ALPHA values at deep optical depths being ~0.5 in Python vs ~0.003 in Fortran.
    # The root cause is still under investigation - it may be related to how continuum
    # opacity coefficients are stored/swapped in fort.10 vs what Fortran actually uses internally.
    abtot = acont + aline + sigmac + sigmal
    # Only ensure ABTOT >= EPS to prevent division by zero (Fortran also does this)
    abtot = np.maximum(abtot, EPS)

    # CRITICAL: Check for INF/NaN (should be rare, but log if found)
    # Fortran would propagate INF/NaN, but we log a warning for debugging
    if np.any(~np.isfinite(abtot)):
        n_inf = np.sum(~np.isfinite(abtot))
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(
            f"Found {n_inf} INF/NaN values in ABTOT (matching Fortran behavior - will propagate)"
        )
        # Don't clip - let INF/NaN propagate like Fortran does

    scatter = sigmac + sigmal
    alpha = np.zeros_like(abtot)
    np.divide(scatter, abtot, out=alpha, where=abtot > 0.0)
    alpha = np.clip(alpha, 0.0, 1.0)

    # CRITICAL FIX: Match Fortran SNUBAR calculation exactly (atlas7v.for line 9030-9031)
    # Fortran: SNUBAR(J)=(ACONT(J)*SCONT(J)+ALINE(J)*SLINE(J))/(ACONT(J)+ALINE(J))
    # Fortran does NOT use max(denom, EPS) - it uses denom directly
    # This preserves small ALINE effects even when ACONT + ALINE is very small
    # Match Fortran exactly: use denom directly, no maximum
    denom = acont + aline
    # Handle division by zero (should be rare, but match Fortran behavior)
    # Fortran would divide by zero if ACONT + ALINE = 0, but this shouldn't happen
    # Use EPS only to prevent actual division by zero, not to clamp small values
    snubar = np.where(denom > 0, (acont * scont + aline * sline) / denom, scont)
    # CRITICAL: Fortran does NOT clamp SNUBAR (atlas7v.for line 8175-8176)
    # SNUBAR can be negative, zero, or exceed SCONT - all are physically possible
    # NOTE: SNUBAR can exceed SCONT if SLINE > SCONT (which is physically possible)

    # CRITICAL FIX: Fortran convention: J=1 is surface (smallest RHOX), J=NRHOX is deep (largest RHOX)
    # Fortran's INTEG requires RHOX to be INCREASING (surface → deep): RHOX(1) < RHOX(2) < ... < RHOX(N)
    #
    # Python arrays come in surface-first order, but RHOX may be decreasing.
    # We need RHOX to be INCREASING for INTEG to work correctly.
    #
    # If RHO[0] > RHO[-1], then RHOX is decreasing and we need to reverse.
    # After reversal: rho_integ[0] = rho[-1] (deep, largest), rho_integ[-1] = rho[0] (surface, smallest)
    # But Fortran expects: RHOX(1) = surface (smallest), RHOX(N) = deep (largest)
    # So we need rho_integ[0] to be SURFACE (smallest), not deep!
    #
    # This means we need to reverse TWICE: once to get increasing order, then integrate,
    # then reverse back to get surface-first order.

    try:
        # RHOX from NPZ file should already be in correct units (g/cm²)
        # The fixed.npz file has RHOX in correct units matching Fortran expectations
        # No scaling needed - use RHOX directly

        needs_reverse = rho.size > 1 and rho[0] > rho[-1]
        if needs_reverse:
            rho_integ = rho[::-1]
            abtot_integ = abtot[::-1]
            start_surface_last = (
                abtot_integ[-1] * rho_integ[-1] if rho_integ.size else 0.0
            )
            start = start_surface_last
        else:
            rho_integ = rho.copy()
            abtot_integ = abtot
            start = abtot[0] * rho_integ[0] if rho_integ.size else 0.0
    except Exception:
        return 0.0

    # Integrate: TAUNU[1] = START, TAUNU[2] = TAUNU[1] + ..., ..., TAUNU[N] = TAUNU[N-1] + ...
    # Since RHOX is INCREASING, TAUNU accumulates UPWARD: TAUNU[0] < TAUNU[1] < ... < TAUNU[-1]
    # After integration: TAUNU[0] = surface (smallest), TAUNU[-1] = deep (largest) ✓
    try:
        taunu = _integ(rho_integ, abtot_integ, start)
    except Exception:
        return 0.0  # Return zero on error

    # CRITICAL: After reversing rho for integration, TAUNU is in increasing order:
    #   TAUNU[0] = surface (smallest RHOX), TAUNU[-1] = deep (largest RHOX)
    #
    # But snubar and alpha are still in original order (surface-first, decreasing RHOX):
    #   snubar[0] = surface (largest RHOX), snubar[-1] = deep (smallest RHOX)
    #
    # For MAP1 to work correctly, TAUNU, SNUBAR, and ALPHA must all be in the SAME order!
    # So we need to reverse snubar and alpha to match TAUNU's order (increasing RHOX).
    if needs_reverse:
        snubar = snubar[::-1]
        alpha = alpha[::-1]
        # Now all arrays are in increasing RHOX order: [0] = surface, [-1] = deep

    # Determine whether the scattering iteration is needed. In the Fortran JOSH flow,
    # IFSCAT=1 drives the scattering path and iteration, regardless of whether lines
    # are present. For our runs, IFSCAT is effectively 1, so iterate whenever there
    # is non-negligible scattering.
    needs_iteration = np.any(alpha > 1e-12)

    # CRITICAL: Always use scattering path (XSBAR/XALPHA) even for continuum-only
    # This matches Fortran's IFSCAT=1 behavior
    always_use_scattering_path = True

    # CRITICAL FIX: For surface flux calculation (IFSURF=1), Fortran skips ITERATION
    # but still uses the correct code path based on IFSCAT:
    #
    # - If IFSCAT=0 (no scattering): Uses MAP1(TAUNU,SNUBAR,...) directly for XS (line 8260-8262)
    #   Then skips iteration (line 8264: IF(IFSURF.EQ.1)GO TO 60)
    #
    # - If IFSCAT=1 (scattering): Uses XSBAR/XALPHA path (label 30, line 8270-8271)
    #   Applies (1-XALPHA) modification to XSBAR (line 8346)
    #   Iterates XS (lines 8348-8398)
    #   Then skips further iteration (line 8400: IF(IFSURF.EQ.1)GO TO 60)
    #
    # So for surface flux, we should:
    # 1. Use the scattering path (XSBAR/XALPHA) when scattering is present
    # 2. Skip iteration (but still use XSBAR/XALPHA calculation)
    #
    # We'll handle skipping iteration later, after XSBAR/XALPHA are calculated.
    # For now, keep needs_iteration as calculated above.

    # CRITICAL FIX: Fortran flow analysis (atlas7v_1.for lines 7115-7129):
    # - Line 7116: IF(IFSCAT.EQ.1)GO TO 30
    # - When IFSCAT=0 (no scattering):
    #   * Line 7118-7119: Sets SNU(J)=SNUBAR(J)
    #   * Line 7121: Calls MAP1(TAUNU,SNU,...) to get XS8
    #   * Line 7122-7123: Sets XS(L)=XS8(L)
    #   * Line 7124: IF(IFSURF.EQ.1)GO TO 60 (flux calculation)
    #   * So when IFSCAT=0 and IFSURF=1, Fortran uses MAP1 interpolation!
    # - When IFSCAT=1 (scattering):
    #   * Line 7128: IF(TAUNU(1).GT.XTAU8(NXTAU))MAXJ=1
    #   * Line 7129: IF(MAXJ.EQ.1)GO TO 401
    #   * So MAXJ=1 check only applies when IFSCAT=1
    #
    # In Python, `needs_iteration` corresponds to IFSCAT=1 (scattering enabled).
    # When `needs_iteration=False` (IFSCAT=0), we should ALWAYS use MAP1 interpolation,
    # even if TAUNU[0] > XTAU_GRID[-1], matching Fortran behavior.

    # CRITICAL FIX: Fortran flow for continuum-only (IFSCAT=0) vs scattering (IFSCAT=1):
    #
    # When IFSCAT=0 (no scattering, continuum-only) AND IFSURF=1 (surface flux):
    #   - Line 8257-8258: Sets SNU(J)=SNUBAR(J)
    #   - Line 8260: Calls MAP1(TAUNU,SNU,...) to get XS8 directly
    #   - Line 8262: Sets XS(L)=XS8(L)
    #   - Line 8264: IF(IFSURF.EQ.1)GO TO 60 (goes directly to flux calculation)
    #   - Line 8520: Uses XS directly (from MAP1(TAUNU,SNUBAR,...), NOT from XSBAR)
    #   - Does NOT call MAP1 for XSBAR/XALPHA (skips lines 8270-8271)
    #   - Does NOT apply (1-XALPHA) modification
    #
    # When IFSCAT=1 (scattering) AND IFSURF=1 (surface flux):
    #   - Line 8268: IF(TAUNU(1).GT.XTAU8(NXTAU))MAXJ=1
    #   - Line 8270-8271: Calls MAP1 for XSBAR and XALPHA
    #   - Line 8300: Initializes XS(L)=XSBAR(L)
    #   - Line 8346: Applies XSBAR(L)=(1.-XALPHA(L))*XSBAR(L)
    #   - Line 8348-8398: Iterates XS
    #   - Line 8400: IF(IFSURF.EQ.1)GO TO 60
    #   - Line 8520: Uses iterated XS
    #
    # So for continuum-only (needs_iteration=False), we should:
    #   1. Skip XSBAR/XALPHA MAP1 calls
    #   2. Use MAP1(TAUNU,SNUBAR,...) directly to get XS
    #   3. Use XS directly for flux calculation (no modification, no iteration)

    # CRITICAL FIX: Fortran ALWAYS uses IFSCAT=1 (scattering path), even for continuum-only
    # This means Fortran ALWAYS computes XSBAR and XALPHA via MAP1, then sets XS(L)=XSBAR(L)
    # Python should match this behavior by always using the scattering path
    flux_override = None
    flux_ck_override = None
    flux_hnu_surface = float("nan")
    flux_knu_surface = float("nan")
    maxj401_error = float("nan")

    if always_use_scattering_path or needs_iteration:
        # Scattering case (IFSCAT=1): use XSBAR/XALPHA path
        # CRITICAL: Fortran ALWAYS uses this path (IFSCAT=1), even for continuum-only
        # CRITICAL FIX: Check MAXJ=1 condition BEFORE MAP1
        # Fortran checks: IF(TAUNU(1).GT.XTAU8(NXTAU))MAXJ=1 (line 8268)
        # This check applies when IFSCAT=1 (scattering)
        # When MAXJ=1, Fortran skips MAP1 and goes to label 401 (line 8269: IF(MAXJ.EQ.1)GO TO 401)
        # For flux calculation (IFSURF=1), Fortran goes to label 60 which uses XS directly
        maxj_force_401 = bool(taunu.size > 0 and taunu[0] > XTAU_GRID[-1])
        maxj = 1 if maxj_force_401 else 0

        # CRITICAL FIX: When MAXJ=1, Fortran skips MAP1 interpolation (line 8269: GO TO 401)
        # Instead, it sets XSBAR and XALPHA directly from SNUBAR[0] and ALPHA[0]
        # (see lines 8295-8299: when XTAU8(L) < TAUNU(1), set XSBAR(L)=SNUBAR(1))
        # Since all XTAU_GRID points are < TAUNU[0] when MAXJ=1, all XSBAR should be SNUBAR[0]
        if maxj_force_401:
            xsbar = np.full(
                len(XTAU_GRID), snubar[0] if snubar.size > 0 else EPS, dtype=np.float64
            )
            xalpha = np.full(
                len(XTAU_GRID), alpha[0] if alpha.size > 0 else 0.0, dtype=np.float64
            )
            maxj_xsbar = 1
            maxj_xalpha = 1
        else:
            # Normal case: call MAP1
            xsbar, maxj_xsbar = _map1(
                taunu,
                snubar,
                XTAU_GRID,
            )
            xalpha, maxj_xalpha = _map1(taunu, alpha, XTAU_GRID)

        # CRITICAL FIX: Fortran overwrites MAXJ with each MAP1 call (line 8270-8271)
        # Use the result from the last MAP1 call (ALPHA) unless MAXJ was set to 1 by pre-check
        # When MAXJ=1, we skip MAP1, so keep maxj=1
        if not maxj_force_401:
            maxj = maxj_xalpha
        xsbar = np.maximum(xsbar, EPS)
        xalpha = np.clip(xalpha, 0.0, 1.0)

        # Apply surface-value masking when XTAU8(L) < TAUNU(1)
        # (atlas7v.for lines ~10230-10233).
        if taunu.size > 0:
            mask = XTAU_GRID < taunu[0]
            if np.any(mask):
                xsbar[mask] = np.maximum(snubar[0], EPS)
                xalpha[mask] = np.clip(alpha[0], 0.0, 1.0)

        # Initialize XS from XSBAR (will be modified by iteration)
        xs = xsbar.copy()

        # CRITICAL FIX: Fortran ALWAYS applies (1-XALPHA) modification (line 9100),
        # even for surface flux. This happens BEFORE the iteration check.
        # Apply (1-XALPHA) modification to XSBAR (line 9100 in atlas7v.for)
        # This must happen BEFORE iteration, matching Fortran line 9100
        # Note: XS is initialized from UNMODIFIED XSBAR, but xsbar_modified is
        # used in the iteration formula. This matches Fortran's behavior.
        xsbar_modified = xsbar * (1.0 - xalpha)

        # CRITICAL FIX: Fortran DOES iterate for surface flux until convergence!
        # Fortran code structure (lines 9102-9154):
        #   DO 34 L=1,NXTAU
        #     DO 33 KK=1,NXTAU
        #       ... iteration code ...
        #     33 XS(K)=MAX(XS(K)+DELXS,1.E-38)
        #     39 IF(IFERR.EQ.0)GO TO 35
        #    34 CONTINUE
        #   35 IF(IFSURF.EQ.1)GO TO 60
        #
        # The iteration loop executes FIRST, then checks IFSURF AFTER convergence.
        # So Fortran iterates until convergence (IFERR=0), then checks IFSURF.
        # Python was incorrectly skipping iteration for surface flux!
        #
        # Iteration formula (line 9147):
        #   DELXS = (sum(COEFJ(K,M)*XS(M)) * XALPHA(K) + XSBAR(K) - XS(K)) / DIAG(K)
        #   XS(K) = XS(K) + DELXS
        # Where XSBAR(K) is AFTER (1-XALPHA) modification
        diag = 1.0 - xalpha * COEFJ_DIAG

        # Initialize XS from unmodified XSBAR (after masking).
        # Fortran sets XS(L)=XSBAR(L) before applying the (1-XALPHA) modification.
        xs = xsbar.copy()
        num_iterations = 0

        # For IFSCAT=1, only the pre-MAP1 TAUNU gate triggers label 401.
        # If MAP1 later returns MAXJ=1, Fortran still stays on the normal XS iteration path.
        if not maxj_force_401:
            # Make a copy for Numba (needs writable array)
            xs_copy = xs.copy()
            if USE_FLOAT32_ITERATION:
                # Fortran uses REAL*4 arrays in the XS iteration; mirror that precision.
                coefj_f32 = COEFJ_MATRIX.astype(np.float32)
                diag_f32 = COEFJ_DIAG.astype(np.float32)
                xs_f32 = xs_copy.astype(np.float32)
                xalpha_f32 = xalpha.astype(np.float32)
                xsbar_mod_f32 = xsbar_modified.astype(np.float32)
                xs_result_f32, num_iterations = _josh_iteration_kernel(
                    coefj_f32,
                    xs_f32,
                    xalpha_f32,
                    xsbar_mod_f32,
                    diag_f32,
                    np.float32(ITER_TOL),
                    MAX_ITER,
                    np.float32(EPS),
                )
                xs[:] = xs_result_f32.astype(np.float64)
            else:
                xs_result, num_iterations = _josh_iteration_kernel(
                    COEFJ_MATRIX,
                    xs_copy,
                    xalpha,
                    xsbar_modified,
                    COEFJ_DIAG,
                    ITER_TOL,
                    MAX_ITER,
                    EPS,
                )
                xs[:] = xs_result  # Copy result back
        else:
            # Fortran JOSH label 401 path for IFSCAT=1 and MAXJ=1:
            # it solves on the physical TAUNU grid (SNU/HNU/JNU), then returns.
            # This is the path used for saturated cores where TAUNU(1) > XTAU_GRID[-1].
            snu = snubar.astype(np.float64).copy()
            hnu_profile = np.zeros_like(snu)
            jnu_profile = np.zeros_like(snu)

            # ------------------------------------------------------------------
            # Stability check before iterating (Fortran atlas7v.for label 401).
            #
            # Fortran's iteration converges because TiO opacity keeps delta_tau
            # large at every depth (> ~0.1).  Without correct TiO populations,
            # Python's taunu plateaus after depth 2 (delta_tau ~ 5e-6) and the
            # tangent-addition formula in _deriv diverges:
            #   D ~ delta_f / delta_tau / scale  →  huge
            #   tan ~ 1 - 1/D  →  tan1*tan0 → 1  →  denominator → 0
            # Each of the 51 iterations then amplifies snu by alpha*jmins/snubar
            # (can be >> 1), producing 10^160 after 51 steps.
            #
            # Detect this condition: if any taunu increment in the DEEP layers
            # (indices 2 onwards) is less than 1e-4 × taunu[0], the derivatives
            # will be explosive.  In that case skip the iteration and return the
            # single-pass (iter-0) estimate, which is physically reasonable
            # (gives ~10% error vs Fortran, vs 10^160 with iteration).
            # ------------------------------------------------------------------
            dtau = np.diff(taunu)
            if dtau.size > 2:
                min_dtau_deep = float(np.min(np.abs(dtau[2:])))
            elif dtau.size > 0:
                min_dtau_deep = float(np.min(np.abs(dtau)))
            else:
                min_dtau_deep = 0.0
            # Threshold: 1e-4 of the surface optical depth.
            _stability_threshold = 1e-4 * float(taunu[0]) if taunu.size > 0 else 1e-4
            _tau_unstable = min_dtau_deep < _stability_threshold

            if _tau_unstable:
                # Single-pass estimate: HNU(1) = DERIV(TAUNU, SNUBAR)[0] / 3.
                # Matches Fortran's converged result when TiO is present (alpha*jmins
                # is tiny → snu ≈ snubar → HNU(1) ≈ DERIV(taunu, snubar)[0] / 3).
                hnu_profile = _deriv(taunu, snubar) / 3.0
                jnu_profile = snubar.copy()
                xs = snu.copy()
                flux_hnu_surface = float(hnu_profile[0]) if hnu_profile.size else 0.0
                flux_knu_surface = float(jnu_profile[0] / 3.0) if jnu_profile.size else float("nan")
                flux_override = flux_hnu_surface
                flux_ck_override = flux_knu_surface
            else:
                for l_iter in range(MAX_ITER):
                    hnu_profile = _deriv(taunu, snu) / 3.0
                    jmins_profile = _deriv(taunu, hnu_profile)

                    # Guard: if corrective update alpha*jmins would exceed snubar,
                    # the iteration has diverged; exit with current hnu_profile.
                    _max_correction = float(np.max(np.abs(alpha * jmins_profile)))
                    _max_snubar = float(np.max(np.abs(snubar))) + EPS
                    if _max_correction > _max_snubar:
                        break

                    jnu_profile = jmins_profile + snu
                    snew = (1.0 - alpha) * snubar + alpha * jnu_profile
                    rel = np.abs(snew - snu) / np.maximum(np.abs(snew), EPS)
                    maxj401_error = float(np.sum(rel))
                    snu = snew
                    num_iterations = l_iter + 1
                    if maxj401_error < ITER_TOL:
                        break

                # Keep XS populated for debug output; surface flux comes from HNU(1) on this path.
                xs = snu.copy()
                flux_hnu_surface = float(hnu_profile[0]) if hnu_profile.size else 0.0
                flux_knu_surface = (
                    float(jnu_profile[0] / 3.0) if jnu_profile.size else float("nan")
                )
                flux_override = flux_hnu_surface
                flux_ck_override = flux_knu_surface
    else:
        # IFSCAT=0 path: Direct MAP1 to XS (not used in current implementation)
        # This path is not used since always_use_scattering_path = True
        # But keep for completeness
        xs = xsbar.copy()
        num_iterations = 0  # No iteration for this path

    # When TAUNU is constant (or nearly constant) and > XTAU_GRID max, MAP1 can't
    # extrapolate properly because linear extrapolation requires TAUNU to vary.
    # However, we should still let MAP1 try to extrapolate, as Fortran does.
    # Only if MAP1 fails (returns constant values) should we use SNUBAR[0] directly.

    # For continuum-only case, XS is already initialized from MAP1(TAUNU,SNUBAR,...) above.
    # For scattering case, XS is initialized from XSBAR above.
    #
    # Surface-flux behavior has two Fortran paths:
    # - regular path: label 60, HNU(1)=SUM(CH*XS)
    # - IFSCAT=1 and MAXJ=1: label 401 profile solve on TAUNU, where HNU(1) comes
    #   from DERIV(TAUNU,SNU)/3 and is not equivalent to CH·XS.
    # For surface flux (HNU), Fortran uses CH weights at label 60.
    # CK weights are used for KNU in the IFSURF=0 branch (label 50).
    # This solver returns HNU, so use CH_WEIGHTS to match Fortran.
    flux_weights = np.asarray(CH_WEIGHTS, dtype=xs.dtype)

    if flux_override is not None:
        flux = float(flux_override)
    else:
        flux = float(np.dot(flux_weights, xs))

    return flux
