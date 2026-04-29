#!/usr/bin/env python3
"""Exact implementation of NMOLEC subroutine for molecular equilibrium.

From atlas7v.for lines 4308-4641 (NMOLEC subroutine).
This computes molecular XNATOM including molecular contributions.
"""

from __future__ import annotations

import math
import os
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Decimal precision for extended range handling
# We use 50 digits for range (avoids float64 overflow at ~1e308)
# but compute element equations with float64 to match Fortran's binary rounding
getcontext().prec = 50

from numba import jit, prange

# Constants
MAXMOL = 200
MAXEQ = 30
MAXLOC = 3 * MAXMOL

# CRITICAL: Use np.float64 to EXACTLY match Fortran's REAL*8 precision
# Fortran uses REAL*8 (64-bit double precision)
# Using extended precision (np.longdouble) causes different rounding → different pivot selection → instability
# Always use float64 to match Fortran exactly
EXTENDED_DTYPE = np.float64  # Keep name for compatibility, but always use float64

# Standard format for float64 logging (17 digits after decimal = ~15-17 significant digits)
# This ensures bit-by-bit comparison accuracy between Python and Fortran logs
FLOAT64_FMT = ".17E"  # Scientific notation with 17 digits after decimal

ROW_NORM_MAX_EXP = 512  # keep DEQ/EQ rows within ~2**512 magnitude
ROW_NORM_MIN_EXP = -512
ROW_NORM_TARGET_EXP = 0

HAS_FMA = hasattr(math, "fma")

_FLOAT64_TINY = np.finfo(np.float64).tiny

# Minimum acceptable seed value when copying XN across layers.
# Defaults to float64 tiny (≈2.22e-308) but can be raised via NM_SEED_MIN_VALUE.
_SEED_MIN_VALUE = _FLOAT64_TINY
_seed_min_env = os.environ.get("NM_SEED_MIN_VALUE")
if _seed_min_env:
    try:
        _seed_min_candidate = float(_seed_min_env)
        if _seed_min_candidate > _SEED_MIN_VALUE:
            _SEED_MIN_VALUE = _seed_min_candidate
    except ValueError:
        print(
            f"WARNING: Ignoring invalid NM_SEED_MIN_VALUE='{_seed_min_env}'. "
            "Using float64 tiny."
        )


def _mul_preserving_precision(a: float, b: float) -> np.float64:
    """Multiply two float64 values using power-of-two scaling to reduce overflow/underflow."""
    a_val = np.float64(a)
    b_val = np.float64(b)
    if a_val == 0.0 or b_val == 0.0:
        return np.float64(0.0)
    mant_a, exp_a = math.frexp(a_val)
    mant_b, exp_b = math.frexp(b_val)
    mant_prod = mant_a * mant_b
    exp_prod = exp_a + exp_b
    mant_prod, adjust = math.frexp(mant_prod)
    exp_prod += adjust
    try:
        return np.float64(math.ldexp(mant_prod, exp_prod))
    except OverflowError:
        return np.float64(np.copysign(np.inf, mant_prod))


def _div_preserving_precision(numerator: float, denominator: float) -> np.float64:
    """Divide two float64 values using power-of-two scaling to reduce overflow/underflow."""
    num = np.float64(numerator)
    den = np.float64(denominator)
    if den == 0.0:
        if num == 0.0:
            return np.float64(np.nan)
        return np.float64(np.copysign(np.inf, num))
    if num == 0.0:
        return np.float64(0.0)
    mant_num, exp_num = math.frexp(num)
    mant_den, exp_den = math.frexp(den)
    mant_ratio = mant_num / mant_den
    exp_ratio = exp_num - exp_den
    mant_ratio, adjust = math.frexp(mant_ratio)
    exp_ratio += adjust
    try:
        return np.float64(math.ldexp(mant_ratio, exp_ratio))
    except OverflowError:
        return np.float64(np.copysign(np.inf, mant_ratio))


def _stable_subtract(minuend: float, subtrahend: float) -> float:
    """Return minuend - subtrahend using ratio form when safe (float64 only)."""
    a = np.float64(minuend)
    b = np.float64(subtrahend)
    if not (np.isfinite(a) and np.isfinite(b)):
        return a - b
    if abs(a) > _FLOAT64_TINY:
        return np.float64(a * (1.0 - b / a))
    return np.float64(a - b)


def _ratio_preserving_precision(numerator: float, denominator: float) -> float:
    """Compute abs(numerator / denominator) with power-of-two scaling to avoid overflow."""
    num = np.float64(numerator)
    den = np.float64(denominator)
    if den == 0.0:
        if num == 0.0:
            return 0.0
        return np.inf
    if num == 0.0:
        return 0.0
    ratio = _div_preserving_precision(num, den)
    return float(abs(ratio))


def _safe_add(a: float, b: float) -> float:
    """Accumulate two floats while avoiding overflow."""
    a_val = np.float64(a)
    b_val = np.float64(b)
    if not np.isfinite(a_val) or not np.isfinite(b_val):
        return a_val + b_val
    abs_max = max(abs(a_val), abs(b_val))
    if abs_max == 0.0 or abs_max < 1e300:
        return np.float64(a_val + b_val)
    mant, exp = math.frexp(abs_max)
    shift = max(exp - 900, 0)
    if shift == 0:
        return np.float64(a_val + b_val)
    scale = math.ldexp(1.0, -shift)
    scaled_sum = np.float64(a_val * scale + b_val * scale)
    try:
        return np.float64(math.ldexp(scaled_sum, shift))
    except OverflowError:
        return np.copysign(np.inf, scaled_sum)


# =============================================================================
# LOG-SPACE ARITHMETIC FOR NEWTON ITERATION
# =============================================================================
# The Newton iteration solves F(XN) = 0 where XN are number densities.
# In log-space, we work with Y = log(XN), so XN = exp(Y).
# This allows us to represent values from ~1e-4000 to ~1e+4000 (as Y from -9210 to +9210)
# which far exceeds float64's range of ~1e-308 to ~1e+308.
#
# The Jacobian transforms as: DEQ_log[i,j] = DEQ[i,j] * XN[j]
# (chain rule: ∂F/∂Y = ∂F/∂X * ∂X/∂Y = ∂F/∂X * exp(Y) = DEQ * XN)
#
# Newton step: dY = solve(DEQ_log, -EQ)
# Update: Y_new = Y + dY, XN_new = exp(Y_new)

# Threshold for switching to log-space (when |log(XN)| exceeds this)
LOG_SPACE_THRESHOLD = 650.0  # Well below ln(float64_max) ≈ 709.78

# Minimum value for XN to avoid log(0)
XN_MIN_VALUE = 1e-300

# Maximum |log(XN)| value (beyond this, clamp)
LOG_XN_MAX = 700.0


def _to_log_space(xn: np.ndarray) -> np.ndarray:
    """Convert XN array to log-space: log_xn = log(XN).

    Handles zeros and negative values by clamping to XN_MIN_VALUE.
    Returns log(XN) which can represent values from ~1e-4000 to ~1e+4000.
    """
    xn_safe = np.maximum(np.abs(xn), XN_MIN_VALUE)
    return np.log(xn_safe)


def _from_log_space(log_xn: np.ndarray) -> np.ndarray:
    """Convert log-space back to linear: XN = exp(log_xn).

    Handles extreme values by clamping log_xn to [-LOG_XN_MAX, LOG_XN_MAX]
    to prevent overflow/underflow in exp().
    """
    log_xn_clamped = np.clip(log_xn, -LOG_XN_MAX, LOG_XN_MAX)
    return np.exp(log_xn_clamped)


def _to_log_space_scalar(xn: float) -> float:
    """Convert single XN value to log-space."""
    xn_safe = max(abs(xn), XN_MIN_VALUE)
    return np.log(xn_safe)


def _from_log_space_scalar(log_xn: float) -> float:
    """Convert single log-space value back to linear."""
    log_xn_clamped = max(-LOG_XN_MAX, min(LOG_XN_MAX, log_xn))
    return np.exp(log_xn_clamped)


def _scale_jacobian_for_log_space(
    deq: np.ndarray, xn: np.ndarray, nequa: int
) -> np.ndarray:
    """Scale Jacobian for log-space Newton: DEQ_log[i,j] = DEQ[i,j] * XN[j].

    This accounts for the chain rule when working with Y = log(XN):
    ∂F/∂Y[j] = ∂F/∂X[j] * ∂X/∂Y[j] = DEQ[i,j] * XN[j]

    Args:
        deq: Original Jacobian in flat column-major format (nequa*nequa)
        xn: Current XN values (nequa)
        nequa: Number of equations

    Returns:
        Scaled Jacobian for log-space Newton iteration
    """
    deq_log = deq.copy()
    for j in range(nequa):
        # Scale column j by XN[j]
        col_start = j * nequa
        for i in range(nequa):
            deq_log[col_start + i] *= xn[j]
    return deq_log


def _log_space_newton_update(
    log_xn: np.ndarray, delta_log: np.ndarray, damping: float = 1.0
) -> np.ndarray:
    """Update log-space XN values: log_xn_new = log_xn + damping * delta_log.

    This preserves positivity of XN (exp of anything is positive).

    Args:
        log_xn: Current log(XN) values
        delta_log: Newton step in log-space (from SOLVIT)
        damping: Damping factor (0 < damping <= 1)

    Returns:
        Updated log(XN) values, clamped to valid range
    """
    log_xn_new = log_xn + damping * delta_log
    # Clamp to prevent overflow/underflow when converting back
    return np.clip(log_xn_new, -LOG_XN_MAX, LOG_XN_MAX)


# =============================================================================
# SIGNED LOG-SPACE ACCUMULATION
# =============================================================================
# For accumulating TERM values into EQ without overflow.
# TERM values can be ~10^400+ which overflows float64, but in log-space
# we just store 400*ln(10) ≈ 921 which is easily representable.
#
# Since EQ values can be positive or negative, we track (sign, log_abs) pairs.


def _log_add_exp(log_a: float, log_b: float) -> float:
    """Compute log(exp(log_a) + exp(log_b)) stably.

    Uses the log-sum-exp trick: log(a + b) = log(a) + log(1 + b/a)
                                           = log_a + log1p(exp(log_b - log_a))
    """
    if not np.isfinite(log_a) and log_a < 0:
        return log_b
    if not np.isfinite(log_b) and log_b < 0:
        return log_a
    if log_a > log_b:
        return log_a + math.log1p(math.exp(log_b - log_a))
    else:
        return log_b + math.log1p(math.exp(log_a - log_b))


def _log_sub_exp(log_a: float, log_b: float) -> tuple[int, float]:
    """Compute log(|exp(log_a) - exp(log_b)|) stably.

    Returns (sign, log_abs) where sign is +1 if a > b, -1 if b > a.
    For a == b, returns (+1, -inf) representing 0.
    """
    if log_a > log_b:
        diff = log_b - log_a
        if diff < -40:  # exp(diff) ≈ 0
            return (+1, log_a)
        return (+1, log_a + math.log1p(-math.exp(diff)))
    elif log_b > log_a:
        diff = log_a - log_b
        if diff < -40:
            return (-1, log_b)
        return (-1, log_b + math.log1p(-math.exp(diff)))
    else:
        # Equal: result is 0
        return (+1, float("-inf"))


def _add_signed_log(
    sign_a: int, log_a: float, sign_b: int, log_b: float
) -> tuple[int, float]:
    """Add two signed log-space values: (sign_a * exp(log_a)) + (sign_b * exp(log_b)).

    Returns (result_sign, result_log_abs).

    This allows accumulation of values that would overflow float64:
    - Values up to ~10^4000 can be represented (log_abs up to ~9200)
    - Handles mixed signs correctly
    - No overflow during intermediate calculations
    """
    # Handle zeros (log = -inf means value is 0)
    if not np.isfinite(log_a) and log_a < 0:
        return (sign_b, log_b)
    if not np.isfinite(log_b) and log_b < 0:
        return (sign_a, log_a)

    if sign_a == sign_b:
        # Same sign: |a + b| = |a| + |b|
        return (sign_a, _log_add_exp(log_a, log_b))
    else:
        # Different signs: compute |a| - |b| or |b| - |a|
        result_sign, result_log = _log_sub_exp(log_a, log_b)
        # Adjust sign based on which term was larger
        if sign_a == +1:
            # Computing (+a) + (-b) = a - b
            return (result_sign, result_log)
        else:
            # Computing (-a) + (+b) = b - a = -(a - b)
            return (-result_sign, result_log)


def _signed_log_to_linear(sign: int, log_abs: float, clamp_max: float = 1e307) -> float:
    """Convert signed log-space value back to linear, with clamping.

    Args:
        sign: +1 or -1
        log_abs: log(|value|)
        clamp_max: Maximum absolute value to return (prevents overflow)

    Returns:
        sign * exp(log_abs), clamped to [-clamp_max, clamp_max]
    """
    if not np.isfinite(log_abs) and log_abs < 0:
        return 0.0
    if log_abs > 708:  # Would overflow exp()
        return sign * clamp_max
    if log_abs < -745:  # Would underflow to 0
        return 0.0
    return sign * math.exp(log_abs)


def _linear_to_signed_log(value: float) -> tuple[int, float]:
    """Convert linear value to signed log-space representation.

    Args:
        value: Any finite float64 value

    Returns:
        (sign, log_abs) where sign is +1 or -1, log_abs is log(|value|)
        For value == 0, returns (+1, -inf)
    """
    if value == 0.0:
        return (+1, float("-inf"))
    if not np.isfinite(value):
        if np.isnan(value):
            return (+1, float("nan"))
        return (+1 if value > 0 else -1, float("inf"))
    sign = +1 if value > 0 else -1
    log_abs = math.log(abs(value))
    return (sign, log_abs)


def _sl_multiply(
    sign_a: int, log_a: float, sign_b: int, log_b: float
) -> tuple[int, float]:
    """Multiply two signed log values: (sign_a * exp(log_a)) * (sign_b * exp(log_b)).

    Returns (result_sign, result_log_abs).
    In log-space: log(|a*b|) = log|a| + log|b|, sign = sign_a * sign_b
    """
    return (sign_a * sign_b, log_a + log_b)


def _sl_divide(
    sign_a: int, log_a: float, sign_b: int, log_b: float
) -> tuple[int, float]:
    """Divide two signed log values: (sign_a * exp(log_a)) / (sign_b * exp(log_b)).

    Returns (result_sign, result_log_abs).
    In log-space: log(|a/b|) = log|a| - log|b|, sign = sign_a * sign_b
    """
    if log_b == float("-inf"):  # Division by zero
        return (sign_a * sign_b, float("inf"))
    return (sign_a * sign_b, log_a - log_b)


def _two_sum(a: float, b: float) -> tuple:
    """Compute s = a + b and error e such that a + b = s + e exactly (Knuth's TwoSum).

    This is an error-free transformation that computes the exact rounding error
    in a floating-point addition, allowing us to track and compensate for it.
    """
    s = a + b
    a_prime = s - b
    b_prime = s - a_prime
    delta_a = a - a_prime
    delta_b = b - b_prime
    e = delta_a + delta_b
    return s, e


def _two_product_fma(a: float, b: float) -> tuple:
    """Compute p = a * b and error e such that a * b = p + e exactly (using FMA).

    Requires FMA (fused multiply-add) instruction for exact error computation.
    """
    p = a * b
    if HAS_FMA:
        try:
            e = math.fma(a, b, -p)  # Compute a*b - p exactly
        except OverflowError:
            # FMA overflows - return 0 error (p is already inf or very large)
            e = 0.0
    else:
        # Fallback: split method (less accurate but still better than nothing)
        # This is Dekker's algorithm
        factor = 134217729.0  # 2^27 + 1
        ah = a * factor
        ah = ah - (ah - a)
        al = a - ah
        bh = b * factor
        bh = bh - (bh - b)
        bl = b - bh
        e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, e


def _accurate_diff(a: float, b: float) -> float:
    """Compute a - b accurately even when a ≈ b using compensated arithmetic.

    Uses TwoSum to capture the rounding error and add it back.
    """
    # Compute s = a - b and error e
    s, e = _two_sum(a, -b)
    # The exact result is s + e, but s + e rounds to s in float64
    # However, if we're subtracting nearly equal numbers, s might be inaccurate
    # The key insight: if |s| < |e|, then s is dominated by rounding error
    # In that case, return e (which captures the true small difference)
    if abs(s) < abs(e) * 1e10:
        # s is unreliable, use e as the correction
        return s + e
    return s


def _accurate_element_residual(xn_k: float, xab_k: float, xn0: float) -> float:
    """Compute XN(K) - XAB(K)*XN(1) accurately using double-double arithmetic.

    When XN(K) ≈ XAB(K)*XN(1), direct subtraction loses all precision because
    we're subtracting two ~1E+25 numbers to get a result that should be ~0.
    Fortran uses 80-bit extended precision which has 3-4 extra digits, but
    Python's float64 returns noise.

    Solution: Use double-double arithmetic (error-free transformations) to
    compute the product and subtraction with ~30 digits of precision.
    This uses only standard float64 operations - no external libraries.

    The algorithm:
    1. Compute xab_k * xn0 exactly as (prod_hi, prod_lo) using TwoProduct
    2. Compute xn_k - prod_hi exactly as (diff_hi, diff_lo) using TwoSum
    3. Combine: result = (diff_hi + diff_lo) - prod_lo

    When at the float64 precision floor, return a tiny sign-preserving value
    to ensure correct Newton damping behavior.
    """
    if xn0 == 0.0 or not np.isfinite(xn0):
        return xn_k - xab_k * xn0

    if not np.isfinite(xn_k) or not np.isfinite(xab_k):
        return xn_k - xab_k * xn0

    # Step 1: Compute xab_k * xn0 exactly as double-double (prod_hi, prod_lo)
    # True product = prod_hi + prod_lo (exact to ~30 digits)
    prod_hi, prod_lo = _two_product_fma(xab_k, xn0)

    # Handle overflow in product
    if not np.isfinite(prod_hi):
        return xn_k - xab_k * xn0

    # Step 2: Compute xn_k - prod_hi exactly as double-double (diff_hi, diff_lo)
    # xn_k - prod_hi = diff_hi + diff_lo (exact)
    diff_hi, diff_lo = _two_sum(xn_k, -prod_hi)

    # Key insight: diff_hi is the primary difference in float64.
    # If diff_hi == 0, then xn_k and prod_hi are bit-identical, meaning
    # xn_k was computed as xab_k * xn0. In this case, the "true" residual
    # for Newton iteration purposes is 0, not the product's rounding error.
    #
    # However, if diff_hi != 0, we have a real difference to compute accurately.

    if diff_hi == 0.0 and diff_lo == 0.0:
        # xn_k == prod_hi exactly (bit-identical)
        # For Newton iteration: residual = 0 (at equilibrium)
        # Return tiny sign-preserving value (sign arbitrary, magnitude negligible)
        return 0.0

    # Step 3: Combine the double-double result
    # Full difference = diff_hi + diff_lo - prod_lo
    # But prod_lo is the error in xab_k*xn0 computation, not relevant to
    # comparing xn_k (which is a stored value) to prod_hi (the same computation).
    #
    # For Newton iteration, we want: xn_k - (xab_k * xn0 as computed in float64)
    # This is just: diff_hi + diff_lo (ignoring prod_lo)

    result = diff_hi + diff_lo

    # Check if we're at the precision floor
    if prod_hi != 0.0:
        relative_residual = abs(result) / abs(prod_hi)
        if relative_residual < 1e-14:
            # At precision floor - the result is meaningful but tiny
            # Preserve sign with controlled magnitude for Newton damping
            if result == 0.0:
                # Exactly zero - use ratio to determine sign
                ratio = xn_k / xn0
                sign = 1.0 if ratio >= xab_k else -1.0
            else:
                sign = 1.0 if result > 0 else -1.0
            # Return a tiny value: sign * (1e-15 relative to product magnitude)
            tiny_residual = sign * abs(prod_hi) * 1e-15
            return tiny_residual

    return result


@jit(nopython=True, cache=True)
def _two_sum_numba(a: float, b: float) -> tuple:
    """Numba-compatible TwoSum for exact error computation."""
    s = a + b
    a_prime = s - b
    b_prime = s - a_prime
    delta_a = a - a_prime
    delta_b = b - b_prime
    e = delta_a + delta_b
    return s, e


def _kahan_add(sum_val: float, compensation: float, addend: float) -> tuple:
    """Kahan summation step: adds 'addend' to 'sum_val' with error compensation.

    Returns (new_sum, new_compensation).
    This recovers precision lost in floating-point addition by tracking
    the small error term and incorporating it into the next addition.
    """
    y = addend - compensation  # Compensate for previous error
    t = sum_val + y  # New sum
    compensation = (t - sum_val) - y  # Compute new error (what was lost)
    return t, compensation


@jit(nopython=True, cache=True)
def _kahan_add_numba(sum_val: float, compensation: float, addend: float) -> tuple:
    """Numba-compatible Kahan summation step."""
    y = addend - compensation
    t = sum_val + y
    new_compensation = (t - sum_val) - y
    return t, new_compensation


# ---- Internal state used by _solvit tracing (kept for global state tracking) ----
_current_solvit_layer: int = -1
_current_solvit_iter: int = -1
_current_solvit_call: int = -1

# ---- All trace flags disabled ----
MAX_SOLVIT_LOG_ITER: int = 0
TRACE_A9_ENABLED: bool = False
TRACE_A9_LAYER: int = 0
TRACE_A9_CALL: int = 0
TRACE_PIVOT_SEARCH: bool = False
TRACE_MOLECULES_ZERO: set = set()
TRACE_MOLECULES_FORCE: set = set()
_DEQ_TRACE_ROWS: tuple = ()
_DEQ_TRACE_COLS: tuple = ()
_PFSAHA_TRACKED_JMOLS: set = set()
_PFSAHA_TRACE_JMOLS: set = set()
_TRACE_MOLECULE_CODES: set = set()
_TRACE_XN_SEED_LAYERS: set = set()
_TRACE_DEQ_COLS: set = set()
_DEQ_TRACE_THRESHOLD: Optional[float] = None
_TERM_TRACE_THRESHOLD: Optional[float] = None
_TRACE_SOLVIT_LAYERS: set = set()
_TRACE_SOLVIT_DETAILED: bool = False
_TRACE_XN_INDICES: set = set()
_TRACE_XN_ALL_LAYERS_FLAG: bool = False
_TRACE_XN_LAYERS: set = set()
_TRACE_EQ_TARGETS: set = set()
_TRACE_EQ_COMPONENTS: set = set()
_TRACE_EQ_COMPONENTS_ALL_LAYERS: bool = False
_TRACE_EQ_COMPONENT_LAYERS: set = set()
_TRACE_NEWTON_UPDATES: bool = False
_TRACE_NEWTON_ALL_LAYERS: bool = False
_TRACE_NEWTON_LAYERS: set = set()
_TRACE_ELECTRON_TERMS: bool = False
_TRACE_ELECTRON_LAYERS: set = set()
_TRACE_ELECTRON_CONTRIBS: bool = False
_TRACE_ELECTRON_CONTRIB_LAYERS: set = set()
_TRACE_EQ_STAGE: bool = False
_TRACE_EQ_STAGE_ALL_LAYERS: bool = False
_TRACE_EQ_STAGE_LAYERS: set = set()
_TRACE_EQ_STAGE_ITERS: set = set()
_LOG_ELECTRON_MOL: bool = False
_LOG_ELECTRON_CODES: set = set()
_TRACE_EQUILJ: bool = False
_TRACE_TERM: bool = False
_TRACE_DEQ_FULL: bool = False
_TRACE_EQ_FULL: bool = False
_TRACE_XN_FULL: bool = False
_TRACE_RATIO: bool = False
_TRACE_SOLVIT_MATRIX: bool = False
_TRACE_ITERATIONS: set = set()
_TRACKED_DEQ_KS: set = set()
_TRACKED_DEQ_CROSS: set = set()
TRACE_MOLECULE_IDS: set = set()
_TRACE_XN_ITERATIONS: None = None
_TRACE_ITERATIONS_ENV_SET: bool = False
MIN_NEWTON_ITER: int = 0


def _should_trace_molecule(jmol: int, code: float) -> bool:
    return False


def _should_trace_eq_target(k_idx: int) -> bool:
    return False


def _should_trace_electron_layer(layer_idx: int) -> bool:
    return False


def _should_trace_electron_contrib(layer_idx: int) -> bool:
    return False


def _should_trace_newton_layer(layer_idx: int) -> bool:
    return False


def _should_trace_eq_stage(layer_idx: int, iteration: int) -> bool:
    return False


def _should_trace_deq_full(layer_idx: int, iteration: int) -> bool:
    return False


def _should_trace_eq_full(layer_idx: int, iteration: int) -> bool:
    return False


def _should_dump_solvit_state(layer_idx: int, iteration: int) -> bool:
    return False


def _should_dump_premol_state(layer_idx: int, iteration: int) -> bool:
    return False


def _should_trace_pfsa(j_layer: int, jmol_index: int) -> bool:
    return False


def _should_trace_eq_accum(layer_index: int, iteration: int) -> bool:
    return False


def _should_log_electron_molecule(molecule_code: float) -> bool:
    return False


def _noop_write(filename: str, line: str) -> None:
    pass


def _append_nmolec_log(line: str) -> None:
    pass


def _log_equilj_event(*args: Any, **kwargs: Any) -> None:
    pass


def _log_electron_event(*args: Any, **kwargs: Any) -> None:
    pass


def _log_electron_term_stage(*args: Any, **kwargs: Any) -> None:
    pass


def _log_electron_equilj(*args: Any, **kwargs: Any) -> None:
    pass


def _log_electron_state_snapshot(*args: Any, **kwargs: Any) -> None:
    pass


def _log_newton_update(*args: Any, **kwargs: Any) -> None:
    pass


def _log_electron_term_step(*args: Any, **kwargs: Any) -> None:
    pass


def _log_electron_eq_update(*args: Any, **kwargs: Any) -> None:
    pass


def _log_electron_deq_update(*args: Any, **kwargs: Any) -> None:
    pass


def _log_eq_stage(*args: Any, **kwargs: Any) -> None:
    pass


def _dump_solvit_state(*args: Any, **kwargs: Any) -> None:
    pass


def _dump_premol_state(*args: Any, **kwargs: Any) -> None:
    pass


def _log_eq_accum(*args: Any, **kwargs: Any) -> None:
    pass


def _log_eq_accum_ext(*args: Any, **kwargs: Any) -> None:
    pass


def _log_molecule_metadata(*args: Any, **kwargs: Any) -> None:
    pass


def _log_molecule_term(*args: Any, **kwargs: Any) -> None:
    pass


def _log_deq_snapshot(*args: Any, **kwargs: Any) -> None:
    pass


def _log_eq_components(*args: Any, **kwargs: Any) -> None:
    pass


def _setup_element_equations_kernel(
    eq: np.ndarray,
    deq: np.ndarray,
    xn: np.ndarray,
    xab: np.ndarray,
    nequa: int,
    nequa1: int,
    xntot: float,
    idequa: np.ndarray,
) -> None:
    """
    Numba-compiled kernel for setting up element equations EQ and DEQ.
    This is the hot loop that initializes equations before molecular terms.

    Matches Fortran logic from atlas7v.for lines 5205-5221.
    """
    eq[0] = -xntot
    kk = 0
    xn0 = xn[0]

    for k in range(1, nequa):  # k=2..NEQUA (1-based), k=1..nequa-1 (0-based)
        eq[0] = eq[0] + xn[k]
        k1 = k * nequa  # 0-based index for DEQ(1, k+1)
        deq[k1] = 1.0  # DEQ(1, k) = 1

        xn_k = xn[k]
        xab_k = xab[k]

        # Compute element residual: EQ(K) = XN(K) - XAB(K)*XN(1)
        # Direct computation (Fortran-compatible)
        eq[k] = xn_k - xab_k * xn0

        kk = kk + nequa1  # kk = k * nequa1 (0-based: DEQ(k, k) in column-major)
        deq[kk] = 1.0  # DEQ(k, k) = 1
        deq[k] = -xab_k  # DEQ(k+1, 1) = -XAB(K)

    # CRITICAL: Electron equation initialization (Fortran lines 5219-5221)
    # IF(IDEQUA(NEQUA).LT.100)GO TO 62
    # EQ(NEQUA)=-XN(NEQUA)
    # DEQ(NEQNEQ)=-1.
    electron_idx = nequa - 1  # 0-based index for NEQUA
    if idequa[electron_idx] >= 100:  # Electron equation (ID=100)
        eq[electron_idx] = -xn[electron_idx]
        neqneq_idx = nequa * nequa - 1  # 0-based index for DEQ(NEQUA, NEQUA)
        deq[neqneq_idx] = -1.0


@jit(nopython=True, cache=True)
def _accumulate_molecules_kernel(
    eq: np.ndarray,
    deq: np.ndarray,
    xn: np.ndarray,
    equilj: np.ndarray,
    locj: np.ndarray,
    kcomps: np.ndarray,
    idequa: np.ndarray,
    nequa: int,
    nummol: int,
    eq_comp: np.ndarray,  # Kahan compensation array for EQ
) -> None:
    """
    Numba-compiled kernel for accumulating molecular terms into EQ/DEQ.
    This is the hot loop that processes all molecules without tracing overhead.

    Uses Kahan summation to reduce floating-point accumulation errors.
    """
    for jmol in range(nummol):
        ncomp = int(locj[jmol + 1] - locj[jmol])
        locj1 = int(locj[jmol])
        locj2 = int(locj[jmol + 1] - 1)

        if ncomp <= 1:
            continue

        equilj_val = equilj[jmol]
        if not np.isfinite(equilj_val):
            continue

        # Fortran multiplies TERM in linear space and allows inf/nan to propagate.
        term = equilj_val
        for lock in range(locj1, locj2 + 1):
            k_raw = int(kcomps[lock])
            if k_raw >= nequa:
                k_idx = nequa - 1
                term = term / xn[k_idx]
            else:
                k_idx = k_raw
                term = term * xn[k_idx]

        # Accumulate into EQ[0] using Kahan summation
        y = term - eq_comp[0]
        t = eq[0] + y
        eq_comp[0] = (t - eq[0]) - y
        eq[0] = t

        # Accumulate into EQ and DEQ for each component
        for lock in range(locj1, locj2 + 1):
            k_raw = int(kcomps[lock])
            if k_raw == nequa:
                k_idx = nequa - 1
            else:
                k_idx = k_raw

            xn_val = xn[k_idx]
            if not np.isfinite(xn_val) or xn_val == 0.0:
                continue

            if k_raw == nequa:
                d = -term / xn_val
            else:
                d = term / xn_val

            if not np.isfinite(d):
                continue

            # Accumulate into EQ[k_idx] using Kahan summation
            y_k = term - eq_comp[k_idx]
            t_k = eq[k_idx] + y_k
            eq_comp[k_idx] = (t_k - eq[k_idx]) - y_k
            eq[k_idx] = t_k

            # Accumulate into DEQ column k_idx
            nequak = nequa * k_idx
            deq[nequak] = deq[nequak] + d

            # Accumulate into DEQ entries for all components
            for locm in range(locj1, locj2 + 1):
                m_raw = int(kcomps[locm])
                m_idx = nequa - 1 if m_raw == nequa else m_raw
                mk = m_idx + nequak
                deq[mk] = deq[mk] + d

        # Correction to charge equation for negative ions
        # FIX: Only apply if last component is REGULAR electron (kcomps = nequa-1),
        # NOT inverse electron sentinel (kcomps = nequa).
        last_comp_raw = int(kcomps[locj2])
        if (
            last_comp_raw == nequa - 1  # Regular electron, not inverse electron
            and idequa[nequa - 1] == 100  # Confirm it's the electron equation
        ):
            for lock in range(locj1, locj2 + 1):
                k_corr_raw = int(kcomps[lock])
                k_corr_idx = nequa - 1 if k_corr_raw >= nequa else k_corr_raw
                xn_val = xn[k_corr_idx]
                if not np.isfinite(xn_val) or xn_val == 0.0:
                    continue
                term_corr = term
                if not np.isfinite(term_corr):
                    continue
                d_corr = term_corr / xn_val
                if not np.isfinite(d_corr):
                    continue
                if k_corr_idx == nequa - 1:
                    eq[k_corr_idx] = eq[k_corr_idx] - term_corr - term_corr
                delta = -d_corr - d_corr
                for locm in range(locj1, locj2 + 1):
                    m_corr_raw = int(kcomps[locm])
                    # Map raw value to 0-based index (Fortran: IF(M.GE.NEQUA1)M=NEQUA)
                    m_corr_idx = nequa - 1 if m_corr_raw >= nequa else m_corr_raw
                    # Only update DEQ when M is the electron equation (Fortran: IF(M.NE.NEQUA)GO TO 93)
                    if m_corr_idx != nequa - 1:
                        continue
                    mk = m_corr_idx + nequa * k_corr_idx
                    deq[mk] = deq[mk] + delta


def _accumulate_molecules_atlas7(
    *,
    eq: np.ndarray,
    deq: np.ndarray,
    xn: np.ndarray,
    equilj: np.ndarray,
    locj: np.ndarray,
    kcomps: np.ndarray,
    idequa: np.ndarray,
    nequa: int,
    nummol: int,
    code_mol: np.ndarray,
    pending_solvit_call: int,
    layer_index: int,
    iteration: int,
    trace_callback: Optional[Callable[..., None]] = None,
    nonfinite_callback: Optional[Callable[..., None]] = None,
    log_xn: Optional[np.ndarray] = None,  # Full log-space: log(XN) values
) -> Optional[list[tuple[int, float, float, float, int, int]]]:
    """
    Port of atlas7v.for SUBROUTINE NMOLEC (lines 3772–3835) for EQ/DEQ assembly.
    """
    term_trace: Optional[list[tuple[int, float, float, float, int, int]]] = (
        [] if (layer_index == 0 and iteration == 0) else None
    )

    trace_eq_accum = _should_trace_eq_accum(layer_index, iteration)
    trace_electron_layer = _should_trace_electron_layer(layer_index)
    trace_electron_contrib = _should_trace_electron_contrib(layer_index)
    trace_metadata = trace_eq_accum

    # Fast path: Use Numba kernel when tracing is disabled
    use_numba_kernel = (
        not trace_eq_accum
        and not trace_electron_layer
        and trace_callback is None
        and nonfinite_callback is None
        and _TERM_TRACE_THRESHOLD is None
        and _DEQ_TRACE_THRESHOLD is None
        and len(_TRACE_DEQ_COLS) == 0
    )

    if use_numba_kernel:
        # Make copies for Numba (needs writable arrays)
        eq_copy = eq.copy()
        deq_copy = deq.copy()
        # Kahan compensation array for EQ accumulation
        eq_comp = np.zeros(nequa, dtype=np.float64)

        # Call Numba kernel
        _accumulate_molecules_kernel(
            eq_copy, deq_copy, xn, equilj, locj, kcomps, idequa, nequa, nummol, eq_comp
        )

        # Copy results back
        eq[:] = eq_copy
        deq[:] = deq_copy

        return term_trace

    # Kahan compensation for Python path EQ accumulation
    eq_comp_py = np.zeros(nequa, dtype=np.float64)

    for jmol in range(nummol):
        ncomp = int(locj[jmol + 1] - locj[jmol])
        locj1 = int(locj[jmol])
        locj2 = int(locj[jmol + 1] - 1)
        molecule_code = float(code_mol[jmol])
        if trace_metadata:
            comps = [int(kcomps[idx]) for idx in range(locj1, locj2 + 1)]
            _log_molecule_metadata(
                layer_index=layer_index,
                iteration=iteration,
                molecule_index=jmol + 1,
                ncomp=ncomp,
                locj1=locj1,
                locj2=locj2,
                components=comps,
            )
        if ncomp <= 1:
            continue

        default_trace = jmol in TRACE_MOLECULES_ZERO
        env_trace = False
        if _TRACE_MOLECULE_CODES:
            for target in _TRACE_MOLECULE_CODES:
                if abs(code_mol[jmol] - target) < 0.5:
                    env_trace = True
                    break
        iteration_traced = (iteration + 1) in _TRACE_ITERATIONS
        force_trace = jmol in TRACE_MOLECULES_FORCE
        has_target_component = bool(_TRACE_EQ_TARGETS) and any(
            int(kcomps[idx]) in _TRACE_EQ_TARGETS for idx in range(locj1, locj2 + 1)
        )
        trace_scope_ok = layer_index == 0 or force_trace or env_trace
        trace_this_molecule = trace_scope_ok and (
            env_trace
            or force_trace
            or (
                layer_index == 0
                and (
                    (iteration <= 5 and default_trace)
                    or (iteration_traced and has_target_component)
                )
            )
        )
        component_logs: list[dict[str, float]] = []
        electron_logs: list[dict[str, float]] = []
        eq0_before_trace = float(eq[0]) if trace_this_molecule else 0.0

        # CRITICAL: Guard against NaN/Inf equilibrium values
        # Fortran keeps these in extended precision, but once they become NaN/Inf in
        # Python we skip the molecule entirely (matching the Fortran "TERM=0" effect
        # when a component underflows).
        equilj_val = np.float64(equilj[jmol])
        if not np.isfinite(equilj_val) or equilj_val <= 0.0:
            if nonfinite_callback is not None:
                nonfinite_callback(
                    stage="equilj",
                    row=None,
                    col=None,
                    value=float(equilj_val),
                    delta=float("nan"),
                    previous=float("nan"),
                    molecule_index=jmol + 1,
                    molecule_code=float(code_mol[jmol]),
                )
            continue
        # Use log-space for TERM calculation to preserve precision
        log_term = np.log(equilj_val)
        term = equilj_val  # Keep for logging compatibility
        if trace_this_molecule and (iteration + 1) in _TRACE_ITERATIONS:
            _log_equilj_event(
                layer_idx=layer_index,
                iteration=iteration,
                molecule_index=jmol,
                molecule_code=float(code_mol[jmol]),
                equilj_value=float(equilj_val),
            )
        log_term_steps = (
            [] if (trace_this_molecule or (jmol + 1) in TRACE_MOLECULE_IDS) else None
        )
        term_step_logs: list[dict[str, Any]] | None = log_term_steps
        if term_step_logs is not None:
            term_step_logs.append(
                {
                    "operation": "seed",
                    "component_idx": 0,
                    "k_raw": None,
                    "k_idx": None,
                    "xn": float("nan"),
                    "term_before": float("nan"),
                    "term_after": float(term),
                }
            )
        log_electron_molecule = _should_log_electron_molecule(molecule_code)
        if log_electron_molecule:
            _log_electron_term_stage(
                stage="start",
                layer_idx=layer_index,
                iteration=iteration,
                molecule_index=jmol,
                molecule_code=molecule_code,
                term_after=float(term),
            )
        term_invalid = False
        for lock in range(locj1, locj2 + 1):
            k_raw = int(kcomps[lock])
            # Determine xn index before modifying term
            if k_raw >= nequa:
                k_idx = nequa - 1
            else:
                k_idx = k_raw

            # FULL LOG-SPACE: Use log_xn directly to avoid any conversion
            if log_xn is not None:
                # Get log(XN[k]) directly - no conversion, no overflow possible
                log_xn_k = log_xn[k_idx]
                if not np.isfinite(log_xn_k):
                    term_invalid = True
                    break

                term_before_float = (
                    _signed_log_to_linear(1, log_term) if log_term < 700 else 1e307
                )

                # Update log_term purely in log-space
                if k_raw >= nequa:
                    log_term = (
                        log_term - log_xn_k
                    )  # Division: log(a/b) = log(a) - log(b)
                else:
                    log_term = (
                        log_term + log_xn_k
                    )  # Multiplication: log(a*b) = log(a) + log(b)

                # For logging: compute clamped linear term
                if log_term > 709.0:
                    term = np.finfo(np.float64).max
                elif log_term < -745.0:
                    term = 0.0
                else:
                    term = np.exp(log_term)

                # For xn_val used in DEQ: get from linear xn (already clamped on storage)
                xn_val = np.float64(xn[k_idx])
            else:
                # LEGACY PATH: Convert xn to log (may overflow if xn is huge)
                xn_val = np.float64(xn[k_idx])
                if not np.isfinite(xn_val) or xn_val <= 0.0:
                    term_invalid = True
                    break

                term_before_float = float(term)
                # Update log_term in log-space for precision
                if k_raw >= nequa:
                    log_term = log_term - np.log(xn_val)  # Division
                else:
                    log_term = log_term + np.log(xn_val)  # Multiplication

                # Convert to linear space for logging (clamped)
                if log_term > 709.0:
                    term = np.finfo(np.float64).max
                elif log_term < -745.0:
                    term = 0.0
                else:
                    term = np.exp(log_term)

            if term_step_logs is not None:
                step_log = {
                    "operation": "div" if k_raw >= nequa else "mul",
                    "component_idx": lock - locj1 + 1,
                    "k_raw": k_raw,
                    "k_idx": k_idx,
                    "xn": float(xn[k_idx]),
                    "xn_safe": float(xn_val),
                    "term_before": term_before_float,
                    "term_after": float(term),
                }
                term_step_logs.append(step_log)

            if trace_electron_layer and k_idx == nequa - 1:
                _log_electron_term_step(
                    layer_idx=layer_index,
                    iteration=iteration,
                    molecule_index=jmol,
                    molecule_code=float(code_mol[jmol]),
                    component_idx=lock - locj1 + 1,
                    operation="div" if k_raw >= nequa else "mul",
                    term_before=term_before_float,
                    term_after=float(term),
                    xn_raw=float(xn[k_idx]),
                    xn_safe=float(xn_val),
                )

            if log_electron_molecule:
                if k_raw < nequa:
                    _log_electron_term_stage(
                        stage="multiply",
                        layer_idx=layer_index,
                        iteration=iteration,
                        molecule_index=jmol,
                        molecule_code=molecule_code,
                        lock_idx=lock - locj1 + 1,
                        k_raw=k_raw + 1,
                        xn_value=float(xn_val),
                        term_before=term_before_float,
                        term_after=float(term),
                    )
                else:
                    _log_electron_term_stage(
                        stage="div_pre",
                        layer_idx=layer_index,
                        iteration=iteration,
                        molecule_index=jmol,
                        molecule_code=molecule_code,
                        xn_value=float(xn_val),
                        term_before=term_before_float,
                    )
                    _log_electron_term_stage(
                        stage="div_post",
                        layer_idx=layer_index,
                        iteration=iteration,
                        molecule_index=jmol,
                        molecule_code=molecule_code,
                        term_after=float(term),
                    )

        if term_invalid or not np.isfinite(log_term):
            if nonfinite_callback is not None:
                nonfinite_callback(
                    stage="term",
                    row=None,
                    col=None,
                    value=float("nan"),
                    delta=float("nan"),
                    previous=float(equilj_val),
                    molecule_index=jmol + 1,
                    molecule_code=float(code_mol[jmol]),
                )
            continue

        # Final conversion from log-space to linear space
        if log_term > 709.0:
            term = np.finfo(np.float64).max
        elif log_term < -745.0:
            term = 0.0
        else:
            term = np.exp(log_term)

        if not np.isfinite(term):
            if nonfinite_callback is not None:
                nonfinite_callback(
                    stage="term",
                    row=None,
                    col=None,
                    value=float(term),
                    delta=float("nan"),
                    previous=float(equilj[jmol]),
                    molecule_index=jmol + 1,
                    molecule_code=float(code_mol[jmol]),
                )
            continue

        log_eq0 = trace_eq_accum or trace_this_molecule
        if log_eq0:
            eq0_before = float(eq[0])
        # Use signed log-space accumulation to handle extreme TERM values
        # This prevents overflow when TERM is ~10^400+ (log_term > 709)
        eq0_sign, log_eq0_abs = _linear_to_signed_log(float(eq[0]))
        eq0_sign, log_eq0_abs = _add_signed_log(eq0_sign, log_eq0_abs, +1, log_term)
        eq[0] = np.float64(_signed_log_to_linear(eq0_sign, log_eq0_abs))
        if trace_eq_accum:
            _log_eq_accum(
                layer_index=layer_index,
                iteration=iteration,
                molecule_index=jmol + 1,
                k_index=0,
                term_value=float(term),
                eq_before=eq0_before,
                eq_after=float(eq[0]),
            )
        if trace_this_molecule:
            eq0_before_trace = eq0_before
        if term_trace is not None and jmol < 20:
            term_trace.append(
                (
                    jmol,
                    float(code_mol[jmol]),
                    float(equilj[jmol]),
                    float(term),
                    locj1,
                    locj2,
                )
            )

        def _enforce_electron_coupling_local() -> None:
            if idequa[nequa - 1] != 100:
                return
            # Fortran does not inject any extra coupling here; electron
            # contributions arise solely from molecule terms.
            return

        for lock in range(locj1, locj2 + 1):
            k_raw = int(kcomps[lock])
            if k_raw == nequa:
                k_idx = nequa - 1
            else:
                k_idx = k_raw

            # CRITICAL: Guard against zero/non-finite xn values to prevent NaN
            xn_val = np.float64(xn[k_idx])
            if not np.isfinite(xn_val):
                continue
            if xn_val == 0.0:
                continue

            if k_raw == nequa:
                d = -_div_preserving_precision(term, xn_val)
            else:
                d = _div_preserving_precision(term, xn_val)

            if not np.isfinite(d):
                continue

            if nonfinite_callback is not None and not np.isfinite(d):
                nonfinite_callback(
                    stage="delta",
                    row=None,
                    col=k_idx,
                    value=float(d),
                    delta=float(d),
                    previous=float("nan"),
                    molecule_index=jmol + 1,
                    molecule_code=float(code_mol[jmol]),
                )

            eq_before = float(eq[k_idx])
            log_eq_component = trace_eq_accum or trace_this_molecule
            # Use signed log-space accumulation to handle extreme TERM values
            eq_k_val = np.float64(eq[k_idx])  # Save for tracing
            eq_k_sign, log_eq_k_abs = _linear_to_signed_log(float(eq[k_idx]))
            eq_k_sign, log_eq_k_abs = _add_signed_log(
                eq_k_sign, log_eq_k_abs, +1, log_term
            )
            eq[k_idx] = np.float64(_signed_log_to_linear(eq_k_sign, log_eq_k_abs))
            electron_ratio_val: float | None = None
            if trace_electron_layer and k_idx == nequa - 1 and eq_k_val != 0.0:
                electron_ratio_val = float(term / eq_k_val)
            if trace_eq_accum:
                _log_eq_accum(
                    layer_index=layer_index,
                    iteration=iteration,
                    molecule_index=jmol + 1,
                    k_index=k_idx,
                    term_value=float(term),
                    eq_before=eq_before,
                    eq_after=float(eq[k_idx]),
                )
                _log_eq_accum_ext(
                    layer_index=layer_index,
                    iteration=iteration,
                    molecule_index=jmol + 1,
                    k_index=k_idx,
                    term_value=float(term),
                    d_value=float(d),
                    eq_before=eq_before,
                    eq_after=float(eq[k_idx]),
                )
            if trace_this_molecule:
                component_logs.append(
                    {
                        "k_idx": k_idx,
                        "xn": float(xn[k_idx]),
                        "eq_before": eq_before,
                        "eq_after": float(eq[k_idx]),
                        "delta": float(d),
                    }
                )
            if trace_this_molecule and k_idx == nequa - 1:
                electron_logs.append(
                    {
                        "k_idx": k_idx,
                        "eq_before": eq_before,
                        "eq_after": float(eq[k_idx]),
                        "adjustment": float(term),
                    }
                )
            if trace_electron_layer and k_idx == nequa - 1:
                _log_electron_eq_update(
                    layer_idx=layer_index,
                    iteration=iteration,
                    molecule_index=jmol,
                    molecule_code=float(code_mol[jmol]),
                    term_value=float(term),
                    eq_before=eq_before,
                    eq_after=float(eq[k_idx]),
                    denom=float(eq_k_val),
                    ratio=electron_ratio_val,
                )
            nequak = nequa * k_idx
            prev_col_val = deq[nequak]
            new_col_val = np.float64(prev_col_val + d)
            if trace_electron_contrib and k_idx == nequa - 1:
                _log_electron_event(
                    "PY_EQ_CORR layer={layer:3d} iter={iter:3d} code={code:8.3f} "
                    "stage=deq_col row={row:3d} col={col:3d} prev={prev:.17E} "
                    "delta={delta:.17E} new={new:.17E}".format(
                        layer=layer_index + 1,
                        iter=iteration + 1,
                        code=float(code_mol[jmol]),
                        row=1,
                        col=k_idx + 1,
                        prev=float(prev_col_val),
                        delta=float(d),
                        new=float(new_col_val),
                    )
                )
            deq[nequak] = new_col_val
            if nonfinite_callback is not None and not np.isfinite(deq[nequak]):
                nonfinite_callback(
                    stage="deq_col",
                    row=0,
                    col=k_idx,
                    value=float(deq[nequak]),
                    delta=float(d),
                    previous=float(prev_col_val),
                    molecule_index=jmol + 1,
                    molecule_code=float(code_mol[jmol]),
                )

            for locm in range(locj1, locj2 + 1):
                m_raw = int(kcomps[locm])
                m_idx = nequa - 1 if m_raw == nequa else m_raw
                mk = m_idx + nequak
                prev_val = deq[mk]
                new_val = np.float64(prev_val + d)
                if trace_electron_contrib and (
                    m_idx == nequa - 1 or k_idx == nequa - 1
                ):
                    _log_electron_event(
                        "PY_EQ_CORR layer={layer:3d} iter={iter:3d} code={code:8.3f} "
                        "stage=deq_entry row={row:3d} col={col:3d} prev={prev:.17E} "
                        "delta={delta:.17E} new={new:.17E}".format(
                            layer=layer_index + 1,
                            iter=iteration + 1,
                            code=float(code_mol[jmol]),
                            row=m_idx + 1,
                            col=k_idx + 1,
                            prev=float(prev_val),
                            delta=float(d),
                            new=float(new_val),
                        )
                    )
                deq[mk] = new_val
                if trace_electron_layer and m_idx == nequa - 1:
                    _log_electron_deq_update(
                        layer_idx=layer_index,
                        iteration=iteration,
                        row_idx=m_idx,
                        col_idx=k_idx,
                        prev_val=float(prev_val),
                        delta=float(d),
                        new_val=float(deq[mk]),
                        stage="general",
                        molecule_index=jmol,
                        molecule_code=float(code_mol[jmol]),
                    )
                if nonfinite_callback is not None and not np.isfinite(deq[mk]):
                    nonfinite_callback(
                        stage="deq_entry",
                        row=m_idx,
                        col=k_idx,
                        value=float(deq[mk]),
                        delta=float(d),
                        previous=float(prev_val),
                        molecule_index=jmol + 1,
                        molecule_code=float(code_mol[jmol]),
                    )

            if trace_callback is not None:
                trace_callback(
                    layer=layer_index + 1,
                    iteration=iteration,
                    call_idx=pending_solvit_call,
                    molecule_index=jmol + 1,
                    molecule_code=code_mol[jmol],
                    m=k_idx,
                    d=d,
                    deq=deq,
                    nequa=nequa,
                )

        if term_step_logs is not None:
            _log_molecule_term(
                layer_index=layer_index,
                iteration=iteration + 1,
                molecule_index=jmol + 1,
                molecule_code=float(code_mol[jmol]),
                term_value=float(term),
                eq0_before=eq0_before_trace,
                eq0_after=float(eq[0]),
                component_logs=component_logs,
                electron_logs=electron_logs,
                term_steps=term_step_logs,
            )

        if log_electron_molecule:
            _log_electron_term_stage(
                stage="term_final",
                layer_idx=layer_index,
                iteration=iteration,
                molecule_index=jmol,
                molecule_code=molecule_code,
                term_after=float(term),
            )

        # Fortran lines 5378-5390: correction to charge equation for negative ions.
        # In Fortran: K=KCOMPS(LOCJ2), IF(IDEQUA(K).NE.100) GO TO 99
        # This checks if the LAST component is a REGULAR electron (ID=100), not inverse electron (ID=101).
        # In Python's kcomps:
        #   - Regular electron (ID 100) → kcomps = nequa - 1 (e.g., 22)
        #   - Inverse electron (ID 101) → kcomps = nequa (e.g., 23, the sentinel value)
        # The neg_ion correction applies ONLY to molecules ending with a regular electron,
        # NOT to molecules ending with an inverse electron (like H+ = CODE 1.01).
        last_comp_raw = int(kcomps[locj2])
        # FIX: Only apply if last component is the REGULAR electron equation index (nequa-1),
        # NOT the inverse electron sentinel (nequa or higher).
        if (
            last_comp_raw
            == nequa - 1  # Regular electron, not inverse electron sentinel
            and idequa[nequa - 1] == 100  # Confirm it's the electron equation
        ):
            for lock in range(locj1, locj2 + 1):
                k_corr_raw = int(kcomps[lock])
                # Map raw value to 0-based index (same as Fortran's K mapping)
                # Fortran: IF(K.GE.NEQUA1)K=NEQUA means K >= 24 maps to K=23
                # Python: raw >= nequa (23) maps to index 22
                k_corr_idx = nequa - 1 if k_corr_raw >= nequa else k_corr_raw
                xn_val = np.float64(xn[k_corr_idx])
                if not np.isfinite(xn_val) or xn_val == 0.0:
                    continue
                term_corr = np.float64(term)
                if not np.isfinite(term_corr):
                    continue
                d_corr = _div_preserving_precision(term_corr, xn_val)
                if not np.isfinite(d_corr):
                    continue
                if k_corr_idx == nequa - 1:
                    prev_eq = eq[k_corr_idx]
                    # Use signed log-space to prevent overflow
                    eq_sign, log_eq_abs = _linear_to_signed_log(float(eq[k_corr_idx]))
                    # BUG FIX: Use log(term) instead of log_term to ensure sync
                    # log_2_term = log_term + np.log(2.0)  # OLD: uses potentially stale log_term
                    log_2_term = (
                        np.log(term_corr) + np.log(2.0)
                        if term_corr > 0
                        else float("-inf")
                    )  # NEW: compute from term_corr
                    eq_sign, log_eq_abs = _add_signed_log(
                        eq_sign, log_eq_abs, -1, log_2_term
                    )
                    new_eq_val = np.float64(_signed_log_to_linear(eq_sign, log_eq_abs))
                    if trace_electron_contrib:
                        _log_electron_event(
                            "PY_EQ_CORR layer={layer:3d} iter={iter:3d} code={code:8.3f} "
                            "stage=neg_eq row={row:3d} col={col:3d} prev={prev:.17E} "
                            "delta={delta:.17E} new={new:.17E}".format(
                                layer=layer_index + 1,
                                iter=iteration + 1,
                                code=float(code_mol[jmol]),
                                row=k_corr_idx + 1,
                                col=k_corr_idx + 1,
                                prev=float(prev_eq),
                                delta=float(-2.0 * term_corr),
                                new=float(new_eq_val),
                            )
                        )
                    eq[k_corr_idx] = new_eq_val
                base_idx = k_corr_idx * nequa
                for locm in range(locj1, locj2 + 1):
                    m_raw = int(kcomps[locm])
                    # Map raw value to 0-based index (same as Fortran's M mapping)
                    m_idx = nequa - 1 if m_raw >= nequa else m_raw
                    # Only update DEQ when M is the electron equation (Fortran: IF(M.NE.NEQUA)GO TO 93)
                    if m_idx != nequa - 1:
                        continue
                    mk = m_idx + base_idx
                    prev_val = deq[mk]
                    delta = np.float64(-2.0 * d_corr)
                    if not np.isfinite(delta):
                        continue
                    if not np.isfinite(prev_val):
                        deq[mk] = prev_val
                    else:
                        new_val = np.float64(prev_val + delta)
                        if trace_electron_contrib:
                            _log_electron_event(
                                "PY_EQ_CORR layer={layer:3d} iter={iter:3d} code={code:8.3f} "
                                "stage=neg_deq row={row:3d} col={col:3d} prev={prev:.17E} "
                                "delta={delta:.17E} new={new:.17E}".format(
                                    layer=layer_index + 1,
                                    iter=iteration + 1,
                                    code=float(code_mol[jmol]),
                                    row=m_idx + 1,
                                    col=k_corr_idx + 1,
                                    prev=float(prev_val),
                                    delta=float(delta),
                                    new=float(new_val),
                                )
                            )
                        deq[mk] = new_val
                    if trace_electron_layer:
                        _log_electron_deq_update(
                            layer_idx=layer_index,
                            iteration=iteration,
                            row_idx=m_idx,
                            col_idx=k_corr_idx,
                            prev_val=float(prev_val),
                            delta=float(delta),
                            new_val=float(deq[mk]),
                            stage="neg_ion",
                            molecule_index=jmol,
                            molecule_code=float(code_mol[jmol]),
                        )
    return term_trace


def _log_xn_update(
    *,
    layer_idx: int,
    iteration: int,
    k_idx: int,
    xn_before: float,
    xn_after: float,
    eq_value: float,
    xneq: float,
    xn100: float,
    ratio: float,
    branch: str,
    scale_value: float,
) -> None:
    pass


def _log_eq_components(
    *,
    layer_idx: int,
    iteration: int,
    eq_vec: np.ndarray,
    xn_vec: np.ndarray,
) -> None:
    if not _TRACE_EQ_COMPONENTS:
        return
    log_path = os.path.join(os.getcwd(), "logs/eq_component_trace.log")
    with open(log_path, "a") as f:
        iter_one_based = iteration + 1
        for idx in sorted(_TRACE_EQ_COMPONENTS):
            eq_val = eq_vec[idx] if idx < eq_vec.size else float("nan")
            xn_val = xn_vec[idx] if idx < xn_vec.size else float("nan")
            f.write(
                "PY_EQ_COMP layer={layer:3d} iter={iter:3d} k={k:2d} "
                "EQ={eq: .17E} XN={xn: .17E}\n".format(
                    layer=layer_idx + 1,
                    iter=iter_one_based,
                    k=idx + 1,
                    eq=eq_val,
                    xn=xn_val,
                )
            )


def _should_trace_xn(layer_idx: int, iteration: int, k_idx: int) -> bool:
    if not _TRACE_XN_INDICES:
        return False
    allow_layer = False
    if _TRACE_XN_ALL_LAYERS_FLAG:
        allow_layer = True
    elif layer_idx == 0:
        allow_layer = True
    elif layer_idx in _TRACE_XN_LAYERS:
        allow_layer = True
    if not allow_layer:
        return False
    if _TRACE_XN_ITERATIONS is not None:
        iter_one_based = iteration + 1
        if iter_one_based not in _TRACE_XN_ITERATIONS:
            return False
    return k_idx in _TRACE_XN_INDICES


def _trace_deq_update(
    *,
    layer: int,
    iteration: int,
    call_idx: int | None,
    molecule_index: int,
    molecule_code: float,
    m: int,
    d: float,
    deq: np.ndarray,
    nequa: int,
) -> None:
    """Detailed logging for targeted DEQ columns/rows during accumulation."""
    if not _TRACKED_DEQ_KS and not _TRACKED_DEQ_CROSS:
        return

    should_log_column = m in _TRACKED_DEQ_KS
    should_log_cross = bool(_TRACKED_DEQ_CROSS)
    if not should_log_column and not should_log_cross:
        return

    col_offset = m * nequa
    entries = []
    if should_log_column:
        deq_1k = deq[col_offset] if col_offset < len(deq) else float("nan")
        diag_idx = col_offset + m
        diag_val = deq[diag_idx] if diag_idx < len(deq) else float("nan")
        entries.append(f"DEQ(1,{m+1})={deq_1k: .17E}")
        entries.append(f"DEQ({m+1},{m+1})={diag_val: .17E}")

    if should_log_cross:
        for row_idx in sorted(_TRACKED_DEQ_CROSS):
            if row_idx < 0 or row_idx >= nequa:
                continue
            idx = row_idx + col_offset
            if idx >= len(deq):
                continue
            entries.append(f"DEQ({row_idx+1},{m+1})={deq[idx]: .17E}")


def _set_solvit_context(layer: int, iteration: int, call_idx: int | None) -> None:
    global _current_solvit_layer, _current_solvit_iter, _current_solvit_call
    _current_solvit_layer = layer
    _current_solvit_iter = iteration
    _current_solvit_call = call_idx


def nmolec_exact(
    n_layers: int,
    temperature: np.ndarray,
    tkev: np.ndarray,
    tk: np.ndarray,
    tlog: np.ndarray,
    gas_pressure: np.ndarray,
    electron_density: np.ndarray,
    xabund: np.ndarray,  # Element abundances (99,)
    xnatom_atomic: np.ndarray,  # Atomic-only XNATOM = P/TK - XNE
    # Molecular data (from READMOL)
    nummol: int,
    code_mol: np.ndarray,  # (MAXMOL,) molecular codes
    equil: np.ndarray,  # (7, MAXMOL) equilibrium constants
    locj: np.ndarray,  # (MAXMOL+1,) component locations
    kcomps: np.ndarray,  # (MAXLOC,) component indices (0-based equation numbers)
    idequa: np.ndarray,  # (MAXEQ,) equation element IDs
    nequa: int,  # Number of equations
    # Partition function data (required, initialized to 1.0 for LTE, matching Fortran DATA statements)
    bhyd: np.ndarray,  # (n_layers, 8) H partition functions
    bc1: np.ndarray,  # (n_layers, 14) C partition functions
    bo1: np.ndarray,  # (n_layers, 13) O partition functions
    bmg1: np.ndarray,  # (n_layers, 11) Mg partition functions
    bal1: np.ndarray,  # (n_layers, 9) Al partition functions
    bsi1: np.ndarray,  # (n_layers, 11) Si partition functions
    bca1: np.ndarray,  # (n_layers, 8) Ca partition functions
    # PFSAHA function: (j, iz, nion, mode, frac, nlte_on) -> None
    # frac is (n_layers, 31) array, modified in-place
    pfsaha_func: Optional[PFSAHAFunc] = None,
    # Output
    xnatom_molecular: Optional[np.ndarray] = None,  # Output: molecular XNATOM
    xnmol: Optional[
        np.ndarray
    ] = None,  # Output: molecular number densities (n_layers, MAXMOL)
    # Zero pivot fix options
    zero_pivot_fix: str = "none",  # "none", "pivot_early", "perturbation"
    # Optional: xnatom array to update in-place during iterations (for PFSAHA access)
    xnatom_inout: Optional[
        np.ndarray
    ] = None,  # Modified in-place: xnatom_inout[j] = XN[0] during iterations
    # Log-space Newton iteration (experimental)
    use_log_space: bool = False,  # Enable log-space Jacobian scaling and XN updates
    # Full Decimal precision Newton (for extreme values)
    use_decimal_newton: bool = False,  # Use 50-digit Decimal for entire Newton iteration
    # Bounded Newton with trust region (for cool atmospheres)
    use_bounded_newton: bool = False,  # Enable trust-region bounded Newton for robust convergence
    # Gibbs minimization (DEFAULT: True - avoids Newton basin-of-attraction issues)
    use_gibbs: bool = True,  # Use Gibbs free energy minimization (recommended)
    gibbs_temperature_threshold: float = 5000.0,  # Auto-enable Gibbs below this T (if auto_gibbs)
    auto_gibbs: bool = False,  # Only matters if use_gibbs=False
    # Continuation method: process layers hot-to-cool to avoid bifurcation
    # CRITICAL FIX: Default to False to match Fortran's layer order (surface first)
    # Fortran atlas7v.for line 4973: DO 110 J=JSTART,NRHOX processes layers 1→80
    # Continuation mode caused Layer 0 to inherit wrong XN values from hot layers
    use_continuation: bool = False,  # Disabled to match Fortran layer order
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute molecular XNATOM using NMOLEC algorithm.

    Returns:
        (xnatom_molecular, xnmol) where:
        - xnatom_molecular: (n_layers,) molecular XNATOM = XN(1)
        - xnmol: (n_layers, MAXMOL) molecular number densities
    """
    trace_xne_layer_env = os.environ.get("NM_TRACE_XNE_LAYER", "").strip()
    trace_xne_layer = None
    if trace_xne_layer_env.lstrip("+-").isdigit():
        trace_xne_layer = int(trace_xne_layer_env)
    # DISABLED: Experimental features that cause divergence
    # These are kept for reference but not triggered by environment variables
    # The LOG-SPACE and DECIMAL Newton modes require more work to be stable
    use_log_space = False  # Force disabled - causes overflow/underflow
    use_decimal_newton = (
        False  # Force disabled - produces different trajectory than Fortran
    )

    if xnatom_molecular is None:
        xnatom_molecular = np.zeros(n_layers, dtype=np.float64)
    if xnmol is None:
        xnmol = np.zeros((n_layers, MAXMOL), dtype=np.float64)

    # Check if we should use Gibbs solver
    # Auto-enable if any layer temperature is below threshold
    if auto_gibbs:
        min_temp = np.min(temperature)
        if min_temp < gibbs_temperature_threshold:
            use_gibbs = True

    # Use Gibbs minimization if requested (recommended for cool atmospheres)
    if use_gibbs:
        from synthe_py.tools.nmolec_gibbs import nmolec_gibbs

        xnatom_result, xne_result, xnz_result = nmolec_gibbs(
            n_layers=n_layers,
            temperature=temperature,
            tkev=tkev,
            tk=tk,
            tlog=tlog,
            gas_pressure=gas_pressure,
            electron_density=electron_density,
            xabund=xabund,
            xnatom_atomic=xnatom_atomic,
            nummol=nummol,
            code_mol=code_mol,
            equil=equil,
            locj=locj,
            kcomps=kcomps,
            idequa=idequa,
            nequa=nequa,
            # Partition function data for CPF corrections (required by Gibbs)
            bhyd=bhyd,
            bc1=bc1,
            bo1=bo1,
            bmg1=bmg1,
            bal1=bal1,
            bsi1=bsi1,
            bca1=bca1,
            pfsaha_func=pfsaha_func,
            xnatom_molecular=xnatom_molecular,
            verbose=False,
        )

        # Update outputs
        xnatom_molecular[:] = xnatom_result
        electron_density[:] = xne_result

        # Return early - skip Newton iteration
        return xnatom_molecular, electron_density, xnz_result

    # Initialize XNZ array to store XN values after each layer (Fortran line 5049-5050)
    # This matches Fortran's XNZ(J,K) array used to persist XN across layers
    xnz_molecular = np.zeros((n_layers, MAXEQ), dtype=np.float64)

    # CONTINUATION METHOD: Process layers from hot to cool
    # This avoids the bifurcation problem at cool temperatures by starting
    # from a high-T solution where only one basin exists, then tracking it
    # down to lower temperatures.
    if use_continuation:
        # Sort layers by temperature (descending: hot first)
        layer_order = np.argsort(-temperature)  # Negative for descending
    else:
        # Original order (as in .atm file)
        layer_order = np.arange(n_layers)

    # XNE ITERATION: Compute self-consistent electron density before Newton iteration
    # This matches Fortran's POPS XNE iteration loop (atlas7v.for lines 2956-2980)
    # Key elements that contribute electrons: H, He, C, Na, Mg, Al, Si, K, Ca, Fe
    xne_electron_donors = [1, 2, 6, 11, 12, 13, 14, 19, 20, 26]  # Element Z numbers
    xne_nions = [1, 2, 2, 2, 2, 2, 2, 2, 2, 2]  # Max ionization stages to consider

    trace_xne_layer_env = os.environ.get("NM_TRACE_XNE_LAYER", "").strip()
    trace_xne_layer = None
    if trace_xne_layer_env.lstrip("+-").isdigit():
        trace_xne_layer = int(trace_xne_layer_env)

    def _iterate_xne_for_layer(j_layer: int, pfsaha_fn, max_iter: int = 200) -> float:
        """Iterate XNE to self-consistency using PFSAHA mode 4 electron calculation.

        CRITICAL: Fortran initializes XNE = XNTOT/2 (atlas7v.for line 2956),
        NOT the value from .atm file. This makes a huge difference because
        XNTOT/2 ≈ 6.9e12 while .atm XNE ≈ 2e7.
        """
        xntot_j = gas_pressure[j_layer] / tk[j_layer]
        # CRITICAL: Initialize XNE to XNTOT/2, matching Fortran line 2956
        xne_j = xntot_j / 2.0
        xnatom_j = xntot_j - xne_j

        if pfsaha_fn is None:
            return electron_density[j_layer]  # Can't iterate without PFSAHA

        elec_arr = np.zeros((n_layers, 31), dtype=np.float64)
        mask = [1] * len(
            xne_electron_donors
        )  # Track which elements still contribute significantly

        for iteration in range(max_iter):
            xne_new = 0.0

            # CRITICAL: Update electron_density[j] so that pfsaha_wrapper closure
            # uses the current XNE value when computing ionization fractions.
            # Without this, PFSAHA uses the .atm value and gives wrong results.
            electron_density[j_layer] = xne_j

            for i, (iz, nion) in enumerate(zip(xne_electron_donors, xne_nions)):
                if mask[i] == 0:
                    continue
                # Call PFSAHA mode 4 to get electron contribution per atom
                elec_contribution = 0.0
                try:
                    pfsaha_fn(j_layer, iz, nion, 4, elec_arr, 0)
                    elec_contribution = elec_arr[j_layer, 0]
                except Exception:
                    continue

                # Electron contribution = elec_per_atom * xnatom * abundance
                if iz - 1 < len(xabund) and xabund[iz - 1] > 0:
                    x_contrib = elec_contribution * xnatom_j * xabund[iz - 1]
                    xne_new += x_contrib

                    # Mask out negligible contributors
                    if iteration > 0 and x_contrib < 1e-5 * xne_j:
                        mask[i] = 0

            # Damped update: XNE = (XNENEW + XNE) / 2
            xne_new = (xne_new + xne_j) / 2.0

            # Check convergence
            if xne_j > 0:
                error = (
                    abs((xne_j - xne_new) / xne_new)
                    if xne_new > 0
                    else abs(xne_j - xne_new)
                )
            else:
                error = abs(xne_new)

            xne_j = xne_new
            xnatom_j = xntot_j - xne_j

            if error < 0.0005:  # Fortran uses 0.0005 tolerance
                break

        return xne_j

    # Initialize arrays
    nequa1 = nequa + 1
    neqneq = nequa * nequa

    nonfinite_term_hits = 0
    nonfinite_d_hits = 0
    nonfinite_xn_hits = 0
    NONFINITE_LOG_LIMIT = 10

    def _log_nonfinite_event(tag: str, message: str) -> None:
        pass

    def _log_eq_vector(
        stage_header: str, entry_label: str, iteration_idx: int, vec: np.ndarray
    ) -> None:
        """Log the full EQ vector for early iterations (mirrors Fortran behavior)."""
        if vec is None:
            return

    def _log_tracked_deq_columns(
        layer_idx: int,
        iteration_idx: int,
        deq_matrix: np.ndarray,
        eq_vec: np.ndarray,
        call_idx: Optional[int] = None,
    ) -> None:
        """Log selected DEQ columns (1,8,9,10,16) before SOLVIT for comparison with Fortran."""
        if deq_matrix is None or eq_vec is None:
            return

    def _log_xn_seed(
        layer_idx: int,
        xn_seed: np.ndarray,
        nequa_local: int,
        ratio: float | None,
        xntot_val: float,
        electron_val: float,
    ) -> None:
        """Log the XN seed vector for selected layers (entire active system)."""
        if layer_idx not in _TRACE_XN_SEED_LAYERS:
            return
        seed_path = os.path.join(os.getcwd(), "xn_seed_trace.log")
        with open(seed_path, "a") as seed_log:
            seed_log.write(
                f"PY_XN_SEED layer={layer_idx+1:3d} xntot={xntot_val: .17E} "
                f"ratio={ratio if ratio is not None else 1.0: .17E} "
                f"electron_seed={electron_val: .17E}\n"
            )
            for idx in range(nequa_local):
                seed_log.write(f"  XN[{idx+1:2d}]={xn_seed[idx]: .17E}\n")
            finite_seed = xn_seed[:nequa_local]
            seed_log.write(
                f"  XN stats: min={np.min(finite_seed): .17E} max={np.max(finite_seed): .17E}\n"
            )

    def _log_seed_reset(
        *,
        layer_idx: int,
        reason: str,
        ratio: float | None,
        component_index: Optional[int],
        offending_value: Optional[float],
        electron_seed_value: Optional[float],
    ) -> None:
        pass

    def _seed_min_threshold(
        component_index: int, electron_equation_index: Optional[int]
    ) -> float:
        if component_index == 0:
            return _SEED_MIN_VALUE
        if (
            electron_equation_index is not None
            and component_index == electron_equation_index
        ):
            return _SEED_MIN_VALUE
        return 0.0

    def _seed_value_valid(value: float, min_allowed: float) -> bool:
        return np.isfinite(value) and value >= min_allowed

    # XAB: abundances for each equation variable
    xab = np.zeros(MAXEQ, dtype=np.float64)
    for k in range(1, nequa):  # k=2..NEQUA (1-based), k=1..nequa-1 (0-based)
        id_elem = idequa[k]  # 0-based: idequa[k] = element ID
        if id_elem < 100:
            xab[k] = max(xabund[id_elem - 1], 1e-20)  # 1-based to 0-based

    # Check if last equation is for electrons
    if idequa[nequa - 1] == 100:  # 0-based
        xab[nequa - 1] = 0.0

    # XN working array (Fortran XN)
    xn = np.zeros(MAXEQ, dtype=np.float64)

    # Process each layer
    # CRITICAL: Match Fortran's exact sequence
    solvit_call_counter = 0

    # Persistent state matching Fortran's XNZ array, initialized on first layer
    xnz_prev = np.zeros(MAXEQ, dtype=np.float64)
    electron_density_atm = electron_density.copy()
    x_prev_seeded = False

    # Track computed XNE for layer-to-layer scaling (Fortran uses XNE array, not .atm values)
    xne_computed = np.zeros(n_layers, dtype=np.float64)
    # Track seed XNE used before Newton iteration (Fortran uses this in PFSAHA mode=12)
    xne_seed = np.zeros(n_layers, dtype=np.float64)

    # Track previous layer index for continuation seeding
    prev_layer_idx = -1

    for iter_idx, j in enumerate(layer_order):
        xntot = gas_pressure[j] / tk[j]
        electron_idx = nequa - 1 if idequa[nequa - 1] == 100 else None

        # For continuation: first iteration (iter_idx=0) is the hottest layer
        # For original order: first iteration is layer 0
        is_first_iteration = iter_idx == 0

        if is_first_iteration:
            # Fortran layer-1 initialization (atlas7v.for NMOLEC, lines 4106-4112):
            #   XNTOT = P(JSTART)/TK(JSTART)
            #   XN(1) = XNTOT/2.          ← always XNTOT/2, no cool-star override
            #   X = XN(1)/10.
            #   XN(K) = X*XAB(K)  for K=2..NEQUA
            #   IF(ID.EQ.100) XN(NEQUA) = X
            #   XNE(1) = X
            # NOTE: atlas12.for has IF(T(1).LT.4000.)XN(1)=XNTOT but atlas7v.for
            # (used by xnfpelsyn) does NOT have this override.
            seed_ratio = None
            xn[0] = xntot / 2.0
            base_x = xn[0] / 10.0  # Fortran: X = XN(1)/10
            for k in range(1, nequa):
                xn[k] = base_x * xab[k]  # Fortran: XN(K) = X*XAB(K)
            # Fortran: IF(ID.EQ.100) XN(NEQUA) = X  and  XNE(1) = X
            if electron_idx is not None:
                xn[electron_idx] = base_x
            xne_computed[j] = base_x  # Fortran: XNE(1) = X
            electron_density[j] = base_x
            xne_seed[j] = base_x
            x_prev_seeded = True
        else:
            # Subsequent layers: use previous layer's solution as initial guess
            # For continuation: prev_layer_idx is the previous iteration's layer (hotter)
            # For original order: prev_layer_idx is j-1 (adjacent layer)
            if use_continuation:
                # Use the XN solution from the previous iteration (stored in xnz_prev)
                # Scale by pressure ratio between current and previous processed layer
                prev_pressure = gas_pressure[prev_layer_idx]
                seed_ratio = (
                    gas_pressure[j] / prev_pressure
                    if prev_pressure not in (0.0, np.inf)
                    else 1.0
                )
            else:
                # Fortran subsequent layers (atlas7v.for lines 5001-5005):
                #   RATIO = P(J)/P(J-1)
                prev_pressure = gas_pressure[j - 1]
                seed_ratio = (
                    gas_pressure[j] / prev_pressure
                    if prev_pressure not in (0.0, np.inf)
                    else 1.0
                )

            nonfinite_seed = False
            invalid_component_idx: Optional[int] = None
            invalid_value: Optional[float] = None
            invalid_reason: Optional[str] = None

            # Scale all XN components by ratio (Fortran: DO 33 K=1,NEQUA; XN(K)=XN(K)*RATIO)
            for k in range(nequa):
                xn_val = xnz_prev[k] * seed_ratio
                if not np.isfinite(xn_val):
                    nonfinite_seed = True
                    invalid_component_idx = k
                    invalid_value = xn_val
                    invalid_reason = "nonfinite_component"
                    break
                xn[k] = xn_val

            # Scale XNE by ratio (but we'll use base_x_layer for electron_density)
            if use_continuation:
                xne_scaled = xne_computed[prev_layer_idx] * seed_ratio
            else:
                xne_scaled = xne_computed[j - 1] * seed_ratio

            if not np.isfinite(xne_scaled):
                nonfinite_seed = True
                invalid_reason = "nonfinite_xne"

            if nonfinite_seed:
                # Fallback: reinitialize like layer 1 (atlas12.for lines 4185-4191)
                failed_ratio = seed_ratio
                seed_ratio = None
                is_cool_outer_layer = temperature[0] < 4000.0  # atlas12.for line 4187
                xn[0] = xntot if is_cool_outer_layer else xntot / 2.0
                base_x = xn[0] / 10.0
                for k in range(1, nequa):
                    xn[k] = base_x * xab[k]
                if electron_idx is not None:
                    xn[electron_idx] = base_x
                xne_computed[j] = base_x
                electron_density[j] = base_x
                xne_seed[j] = base_x
                _log_seed_reset(
                    layer_idx=j,
                    reason=invalid_reason or "invalid_seed",
                    ratio=failed_ratio,
                    component_index=invalid_component_idx,
                    offending_value=invalid_value,
                    electron_seed_value=base_x,
                )
            else:
                xne_computed[j] = xne_scaled
                electron_density[j] = xne_scaled
                xne_seed[j] = xne_scaled

        xnz_prev[:nequa] = xn[:nequa]
        _log_xn_seed(
            layer_idx=j,
            xn_seed=xn[:nequa],
            nequa_local=nequa,
            ratio=seed_ratio,
            xntot_val=xntot,
            electron_val=electron_density[j],
        )

        # Fortran NMOLEC does not iterate XNE via PFSAHA; keep off by default.
        # Enable explicitly with NM_ENABLE_XNE_ITER=1 if needed for experiments.
        if pfsaha_func is not None and os.environ.get("NM_ENABLE_XNE_ITER", "0") != "0":
            electron_density[j] = _iterate_xne_for_layer(j, pfsaha_func)
            xne_computed[j] = electron_density[j]
            if electron_idx is not None:
                xn[electron_idx] = electron_density[j]

        # NOTE: XNE iteration is enabled by default for parity with xnfpelsyn.

        # Compute partition function corrections for NLTE
        # From atlas7v.for lines 4556-4592
        # B arrays are always required (initialized to 1.0 for LTE, matching Fortran DATA statements)
        # CPF corrections are computed if PFSAHA is available, otherwise default to 1.0
        cpfh = 1.0
        cpfc = 1.0
        cpfo = 1.0
        cpfmg = 1.0
        cpfal = 1.0
        cpfsi = 1.0
        cpfca = 1.0

        if pfsaha_func is not None:
            # NLTE partition functions (NLTEON = -1)
            pf = np.zeros((n_layers, 31), dtype=np.float64)
            pfsaha_func(j, 1, 1, 3, pf, -1)
            pfh = pf[j, 0]
            pfsaha_func(j, 6, 1, 3, pf, -1)
            pfc = pf[j, 0]
            pfsaha_func(j, 8, 1, 3, pf, -1)
            pfo = pf[j, 0]
            pfsaha_func(j, 12, 1, 3, pf, -1)
            pfmg = pf[j, 0]
            pfsaha_func(j, 13, 1, 3, pf, -1)
            pfal = pf[j, 0]
            pfsaha_func(j, 14, 1, 3, pf, -1)
            pfsi = pf[j, 0]
            pfsaha_func(j, 20, 1, 3, pf, -1)
            pfca = pf[j, 0]

            # LTE partition functions (NLTEON = 0)
            bpf = np.zeros((n_layers, 31), dtype=np.float64)
            pfsaha_func(j, 1, 1, 3, bpf, 0)
            bpfh = bpf[j, 0]
            pfsaha_func(j, 6, 1, 3, bpf, 0)
            bpfc = bpf[j, 0]
            pfsaha_func(j, 8, 1, 3, bpf, 0)
            bpfo = bpf[j, 0]
            pfsaha_func(j, 12, 1, 3, bpf, 0)
            bpfmg = bpf[j, 0]
            pfsaha_func(j, 13, 1, 3, bpf, 0)
            bpfal = bpf[j, 0]
            pfsaha_func(j, 14, 1, 3, bpf, 0)
            bpfsi = bpf[j, 0]
            pfsaha_func(j, 20, 1, 3, bpf, 0)
            bpfca = bpf[j, 0]

            # Compute corrections
            # CRITICAL: Match Fortran exactly (lines 4586-4592): CPF = PF/BPF * B
            # Fortran always has B arrays in COMMON blocks (initialized to 1.0 by default)
            # No fallback logic - arrays must always be provided
            if bpfh != 0:
                cpfh = pfh / bpfh * bhyd[j, 0]
            if bpfc != 0:
                cpfc = pfc / bpfc * bc1[j, 0]
            if bpfo != 0:
                cpfo = pfo / bpfo * bo1[j, 0]
            if bpfmg != 0:
                cpfmg = pfmg / bpfmg * bmg1[j, 0]
            if bpfal != 0:
                cpfal = pfal / bpfal * bal1[j, 0]
            if bpfsi != 0:
                cpfsi = pfsi / bpfsi * bsi1[j, 0]
            if bpfca != 0:
                cpfca = pfca / bpfca * bca1[j, 0]

        # Compute equilibrium constants EQUILJ for each molecule
        equilj = np.zeros(MAXMOL, dtype=np.float64)

        for jmol in range(nummol):
            ncomp = locj[jmol + 1] - locj[jmol]

            if equil[0, jmol] == 0.0:
                # Use PFSAHA-based equilibrium
                if ncomp > 1:
                    id_elem = int(code_mol[jmol])
                    ion = ncomp - 1
                    # Call PFSAHA in mode 12 (ionization fractions)
                    if pfsaha_func is not None:
                        frac = np.zeros((n_layers, 31), dtype=np.float64)

                        # Match Fortran NMOLEC/PFSAHA: use the live XNE(J) state directly.
                        # Do not substitute a separate seed value before PFSAHA calls.
                        pfsaha_func(j, id_elem, ncomp, 12, frac, 0)

                        frac0 = np.float64(frac[j, 0])
                        fracn = np.float64(frac[j, ncomp - 1])
                        if (
                            frac0 == 0.0
                            or abs(frac0) < 1e-300
                            or not np.isfinite(frac0)
                        ):
                            equilj_before = 0.0
                            equilj_str = "0.00000000000000000E+00"
                        else:
                            # Use log-space to preserve precision: log(fracn) - log(frac0) + ion*log(xne)
                            if fracn > 0.0:
                                log_frac_ratio = np.log(fracn) - np.log(frac0)
                                log_term = log_frac_ratio + ion * np.log(
                                    electron_density[j]
                                )
                                equilj_before = np.exp(log_term)
                            else:
                                equilj_before = (
                                    fracn / frac0 * (electron_density[j] ** ion)
                                )
                            if np.isnan(equilj_before):
                                equilj_str = "NaN"
                            elif np.isinf(equilj_before):
                                equilj_str = "Inf"
                            else:
                                equilj_str = f"{equilj_before:.17E}"

                        # CRITICAL: Fortran (line 4649) does NOT check for zero denominators:
                        #   EQUILJ(JMOL)=FRAC(J,NCOMP)/FRAC(J,1)*XNE(J)**ION
                        # If FRAC(J,1)=0, Fortran produces Inf/NaN but continues execution.
                        # Match Fortran behavior exactly: do NOT add protection checks!
                        # Fortran allows INF/NaN EQUILJ to propagate through TERM calculation
                        # Use np.errstate to match Fortran's silent division by zero behavior
                        with np.errstate(divide="ignore", invalid="ignore"):
                            if frac0 > 0.0 and fracn > 0.0:
                                log_frac_ratio = np.log(fracn) - np.log(frac0)
                                log_term = log_frac_ratio + ion * np.log(
                                    electron_density[j]
                                )
                                equilj[jmol] = np.exp(log_term)
                            else:
                                equilj[jmol] = (
                                    fracn / frac0 * (electron_density[j] ** ion)
                                )
                        # CRITICAL: Do NOT check for NaN/Inf - Fortran doesn't check either!
                        # Fortran allows INF/NaN EQUILJ to propagate through TERM calculation

                        # CRITICAL FIX: Do NOT apply CPFC corrections in PFSAHA path!
                        # Fortran does NOT apply CPFC corrections to PFSAHA molecules (lines 4544-4554
                        # are BEFORE label 35, so they're only in polynomial path)
                    else:
                        equilj[jmol] = 1.0
                else:
                    equilj[jmol] = 1.0
            else:
                # Use EQUIL polynomial
                # CRITICAL: Match Fortran's ION calculation exactly (line 4510):
                #   ION=(CODE(JMOL)-DBLE( INT(CODE(JMOL))))*100.+.5
                # Fortran uses DBLE() to ensure double precision, then adds 0.5 and truncates
                # Python: Use np.float64 to ensure double precision, then add 0.5 and convert to int
                code_int = int(code_mol[jmol])
                code_frac = np.float64(code_mol[jmol]) - np.float64(code_int)
                ion = int(code_frac * 100.0 + 0.5)
                equilj[jmol] = np.float64(0.0)

                if temperature[j] > 10000.0:
                    continue

                is_h_minus = abs(code_mol[jmol] - 101.0) < 1e-9
                if is_h_minus:
                    # Special case for H- (HMINUS)
                    exp_arg = (
                        4.478 / tkev[j]
                        - 46.4584
                        + (
                            1.63660e-3
                            + (
                                -4.93992e-7
                                + (
                                    1.11822e-10
                                    + (
                                        -1.49567e-14
                                        + (1.06206e-18 - 3.08720e-23 * temperature[j])
                                        * temperature[j]
                                    )
                                    * temperature[j]
                                )
                                * temperature[j]
                            )
                            * temperature[j]
                        )
                        * temperature[j]
                        - 1.5 * tlog[j]
                    )
                    # Fortran does NOT clamp exp() arguments - it allows inf/nan
                    # Match Fortran behavior exactly: use exp() directly
                    equilj_val = np.exp(exp_arg)
                    # Apply CPFH exactly once (Fortran multiplies inside the component loop;
                    # we apply it here and skip reapplying for the H component later)
                    equilj_val *= cpfh
                    equilj[jmol] = equilj_val
                else:
                    # General polynomial equilibrium constant
                    # Calculate polynomial step-by-step
                    # CRITICAL: Match Fortran's polynomial calculation exactly (lines 4552-4555):
                    #   EQUIL(1,JMOL)/TKEV(J)-EQUIL(2,JMOL)+
                    #   (EQUIL(3,JMOL)+(-EQUIL(4,JMOL)+(EQUIL(5,JMOL)+(-EQUIL(6,JMOL)+
                    #   +EQUIL(7,JMOL)*T(J))*T(J))*T(J))*T(J))*T(J)
                    # Ensure all operations use np.float64 for double precision
                    poly_term = (
                        np.float64(equil[0, jmol]) / np.float64(tkev[j])
                        - np.float64(equil[1, jmol])
                        + (
                            np.float64(equil[2, jmol])
                            + (
                                -np.float64(equil[3, jmol])
                                + (
                                    np.float64(equil[4, jmol])
                                    + (
                                        -np.float64(equil[5, jmol])
                                        + np.float64(equil[6, jmol])
                                        * np.float64(temperature[j])
                                    )
                                    * np.float64(temperature[j])
                                )
                                * np.float64(temperature[j])
                            )
                            * np.float64(temperature[j])
                        )
                        * np.float64(temperature[j])
                    )
                    # CRITICAL: Match Fortran's TLOG_TERM calculation exactly (line 4555):
                    #   -1.5*(DBLE(NCOMP-ION-ION-1))*TLOG(J)
                    # Fortran uses DBLE() to ensure double precision
                    # Python: Use np.float64 to ensure double precision
                    tlog_term = (
                        -1.5 * np.float64(ncomp - ion - ion - 1) * np.float64(tlog[j])
                    )
                    exp_arg_poly = np.float64(poly_term) + tlog_term

                    # Fortran does NOT clamp exp() arguments - it allows inf/nan
                    # Match Fortran behavior exactly: use exp() directly
                    # CRITICAL: Ensure exp_arg_poly is np.float64 for double precision
                    equilj[jmol] = np.exp(np.float64(exp_arg_poly))

                # Apply partition function corrections
                # CRITICAL: These corrections are ONLY applied in polynomial path!
                # Fortran applies them BEFORE label 35 (PFSAHA path), so they're only in polynomial path
                locj1 = locj[jmol]
                locj2 = locj[jmol + 1] - 1
                for lock in range(locj1, locj2 + 1):
                    k = kcomps[lock]  # 0-based equation number
                    k_raw = k
                    id_elem = idequa[k] if k < nequa else 100

                    # For H- we already applied CPFH above; skip the hydrogen component here
                    if is_h_minus and id_elem == 1:
                        continue

                    if id_elem == 1:
                        equilj[jmol] = equilj[jmol] * cpfh
                    elif id_elem == 6:
                        equilj[jmol] = equilj[jmol] * cpfc
                    elif id_elem == 8:
                        equilj[jmol] = equilj[jmol] * cpfo
                    elif id_elem == 12:
                        equilj[jmol] = equilj[jmol] * cpfmg
                    elif id_elem == 13:
                        equilj[jmol] = equilj[jmol] * cpfal
                    elif id_elem == 14:
                        equilj[jmol] = equilj[jmol] * cpfsi
                    elif id_elem == 20:
                        equilj[jmol] = equilj[jmol] * cpfca

                # CRITICAL: Fortran does NOT clamp EQUILJ - it allows inf/nan/negative values!
                # Fortran (line 4649) does: EQUILJ(JMOL)=FRAC(J,NCOMP)/FRAC(J,1)*XNE(J)**ION
                # No checks, no clamping - Fortran allows INF/NaN/NEGATIVE EQUILJ to propagate
                # Match Fortran behavior exactly: do NOT modify EQUILJ after calculation!
        # Newton-Raphson iteration
        eqold = np.zeros(nequa, dtype=np.float64)
        max_iter = 200
        converged = False


        # BOUNDED NEWTON: Add step limiting to prevent chaotic divergence
        # This modifies the existing Newton loop rather than replacing it
        # The bounds are enforced in the XN update section below
        bounded_newton_active = use_bounded_newton

        for iteration in range(max_iter):
            pending_solvit_call = solvit_call_counter + 1

            # Set up equations EQ and Jacobian DEQ
            # DEQ is stored column-major (1D array): DEQ(K1) = DEQ(1, K) where K1 = NEQUA*K - NEQUA + 1
            deq = np.zeros(neqneq, dtype=np.float64)
            eq = np.zeros(nequa, dtype=np.float64)

            xntot = gas_pressure[j] / tk[j]

            use_numba_element_setup = not (j == 0 and iteration < 5)

            if use_numba_element_setup:
                # Use Numba kernel for element equation setup
                _setup_element_equations_kernel(
                    eq, deq, xn, xab, nequa, nequa1, xntot, idequa
                )
            else:
                # Python path
                eq[0] = -xntot
                kk = 0  # 0-based: DEQ(k, k) = DEQ[kk] where kk = k * nequa1

                xn0 = xn[0]
                for k in range(
                    1, nequa
                ):  # k=2..NEQUA (1-based), k=1..nequa-1 (0-based)
                    eq[0] = eq[0] + xn[k]
                    k1 = k * nequa  # 0-based index for DEQ(1, k+1)
                    deq[k1] = 1.0  # DEQ(1, k) = 1 (0-based: row 0, col k)
                    xn_k = xn[k]
                    xab_k = xab[k]
                    if np.isfinite(xn0) and np.isfinite(xn_k) and np.isfinite(xab_k):
                        # Use compensated arithmetic to avoid catastrophic cancellation
                        # when XN(K) ≈ XAB(K)*XN(1)
                        element_residual = _accurate_element_residual(xn_k, xab_k, xn0)
                    else:
                        element_residual = xn_k - xab_k * xn0
                    eq[k] = element_residual

                    kk = (
                        kk + nequa1
                    )  # kk = k * nequa1 (0-based: DEQ(k, k) in column-major)
                    deq[kk] = 1.0  # DEQ(k, k) = 1 (0-based: row k, col k)
                    deq[k] = -xab[k]  # DEQ(k+1, 1) = -XAB(K) (0-based: row k, col 0)
                    # CRITICAL: DEQ(1,1) stays ZERO in Fortran - do NOT accumulate it!

                # CRITICAL: Electron equation initialization (Fortran lines 5219-5221)
                # IF(IDEQUA(NEQUA).LT.100)GO TO 62
                # EQ(NEQUA)=-XN(NEQUA)
                # DEQ(NEQNEQ)=-1.
                if electron_idx is not None and idequa[electron_idx] >= 100:
                    eq[electron_idx] = -xn[electron_idx]
                    neqneq_idx = (
                        nequa * nequa - 1
                    )  # 0-based index for DEQ(NEQUA, NEQUA)
                    deq[neqneq_idx] = -1.0

            _log_eq_stage(
                "post_elements",
                layer_idx=j,
                iteration=iteration,
                eq_vec=eq,
                xn_vec=xn,
                nequa=nequa,
                electron_idx=electron_idx,
            )

            if j == 0 and iteration < 5:
                _log_deq_snapshot(
                    label="post_elements",
                    layer_idx=j,
                    iteration=iteration,
                    deq=deq,
                    eq=eq,
                    nequa=nequa,
                )

            # Charge equation (if electrons are included)
            if idequa[nequa - 1] == 100:  # 0-based
                eq[nequa - 1] = -xn[nequa - 1]
                deq[neqneq - 1] = -1.0

            # CRITICAL: DEQ(1,1) is only set through molecular terms!
            # If no molecules contain XN(1), DEQ(1,1) = 0, causing zero pivot

            def _enforce_electron_coupling() -> None:
                """Match Fortran relation DEQ(1,NEQUA) = -DEQ(NEQUA,NEQUA)."""
                if idequa[nequa - 1] != 100:
                    return
                # Original Fortran relies solely on molecule/negative-ion
                # updates; there is no extra coupling step.
                return

            def _record_nonfinite_deq(**info: Any) -> None:
                nonlocal nonfinite_term_hits, nonfinite_d_hits
                stage = str(info.get("stage", "unknown"))
                row = info.get("row")
                col = info.get("col")
                value = info.get("value", float("nan"))
                delta = info.get("delta", float("nan"))
                previous = info.get("previous", float("nan"))
                molecule_index = info.get("molecule_index", -1)
                molecule_code = info.get("molecule_code", float("nan"))

                row_str = str(row + 1) if isinstance(row, int) else "N/A"
                col_str = str(col + 1) if isinstance(col, int) else "N/A"
                log_message = (
                    f"layer={j+1} iter={iteration} call={pending_solvit_call} stage={stage} "
                    f"row={row_str} col={col_str} value={value:.6e} delta={delta:.6e} "
                    f"prev={previous:.6e} mol={molecule_index} code={molecule_code:.2f}"
                )

                if stage == "term":
                    if nonfinite_term_hits < NONFINITE_LOG_LIMIT:
                        _log_nonfinite_event("term", log_message)
                    nonfinite_term_hits += 1
                else:
                    if nonfinite_d_hits < NONFINITE_LOG_LIMIT:
                        _log_nonfinite_event("deq", log_message)
                    nonfinite_d_hits += 1

            trace_callback = (
                _trace_deq_update if (_TRACKED_DEQ_KS or _TRACKED_DEQ_CROSS) else None
            )

            if _should_dump_premol_state(j, iteration):
                premol_matrix = np.array(deq, copy=True).reshape(
                    (nequa, nequa), order="F"
                )
                _dump_premol_state(
                    layer_idx=j,
                    iteration=iteration,
                    matrix=premol_matrix,
                    rhs=np.array(eq[:nequa], copy=True),
                    xn_vec=np.array(xn[:nequa], copy=True),
                )

            if _LOG_ELECTRON_MOL:
                electron_density_val = float(
                    electron_density[j] if j < len(electron_density) else float("nan")
                )
                _log_electron_state_snapshot(
                    stage="pre_terms",
                    layer_idx=j,
                    iteration=iteration,
                    xn=xn,
                    electron_idx=electron_idx,
                    electron_density_val=electron_density_val,
                    locj=locj,
                    kcomps=kcomps,
                    code_mol=code_mol,
                    nequa=nequa,
                )
                for jmol in range(nummol):
                    molecule_code = float(code_mol[jmol])
                    if not _should_log_electron_molecule(molecule_code):
                        continue
                    _log_electron_equilj(
                        layer_idx=j,
                        iteration=iteration,
                        molecule_index=jmol,
                        molecule_code=molecule_code,
                        equilj_value=float(equilj[jmol]),
                        xn_total=float(xn[0]),
                        electron_density_val=electron_density_val,
                    )

            _ = _accumulate_molecules_atlas7(
                eq=eq,
                deq=deq,
                xn=xn,
                equilj=equilj,
                locj=locj,
                kcomps=kcomps,
                idequa=idequa,
                nequa=nequa,
                nummol=nummol,
                code_mol=code_mol,
                pending_solvit_call=pending_solvit_call,
                layer_index=j,
                iteration=iteration,
                trace_callback=trace_callback,
                nonfinite_callback=_record_nonfinite_deq,
                log_xn=log_xn if use_log_space else None,  # Full log-space mode
            )

            _log_eq_stage(
                "post_terms",
                layer_idx=j,
                iteration=iteration,
                eq_vec=eq,
                xn_vec=xn,
                nequa=nequa,
                electron_idx=electron_idx,
            )

            # Solve linear system: DEQ * delta_XN = EQ
            # Use SOLVIT algorithm (Gaussian elimination with complete pivoting)
            # From atlas7v_1.for lines 1200-1262
            # SOLVIT modifies DEQ and EQ in-place

            # CRITICAL: DEQ is stored column-major (Fortran order)
            # numpy reshape defaults to C-order (row-major), so we need order='F'
            deq_2d = deq[:neqneq].reshape(nequa, nequa, order="F").copy()

            if not np.isfinite(deq_2d).all():
                bad_indices = np.argwhere(~np.isfinite(deq_2d))
                first_bad_row, first_bad_col = bad_indices[0]
                bad_value = deq_2d[first_bad_row, first_bad_col]
                if nonfinite_d_hits < NONFINITE_LOG_LIMIT:
                    _log_nonfinite_event(
                        "deq-pre-solvit",
                        f"layer={j+1} iter={iteration} call={pending_solvit_call} "
                        f"row={first_bad_row+1} col={first_bad_col+1} value={bad_value}",
                    )
                nonfinite_d_hits += 1

            eq_copy = eq.copy()

            # Option 2: Add small perturbation to DEQ[0,0] to prevent exact zero
            # CRITICAL: DEQ[0,0] starts at 0.0 and never accumulates molecular contributions
            # Adding perturbation makes it pivotable early, preventing zero pivot at iteration 20
            if zero_pivot_fix == "perturbation":
                # Use scale-relative perturbation: eps * matrix_size * max_entry
                # This is principled regularization, not ad-hoc
                finite_mask = np.isfinite(deq_2d)
                deq_max = (
                    np.max(np.abs(deq_2d[finite_mask])) if np.any(finite_mask) else 1.0
                )
                eps = np.finfo(np.float64).eps
                # Scale-relative perturbation: machine epsilon * matrix size * max entry
                perturbation = max(1e-12, eps * nequa * deq_max)
                deq_2d[0, 0] += perturbation

            # CRITICAL: Align with Fortran behavior
            # Fortran does NOT check for NaN/Inf before SOLVIT
            # Fortran allows Inf/NaN and continues execution
            # Python should match this behavior - allow solver to proceed even with Inf/NaN
            # The solver will handle Inf/NaN operations (may produce Inf/NaN results, but continues)
            # Only skip if we can't even call the solver (but Fortran doesn't skip)
            # Remove NaN check to match Fortran's behavior

            _enforce_electron_coupling()

            if j == 0 and iteration < 5:
                _log_deq_snapshot(
                    label="post_enforce",
                    layer_idx=j,
                    iteration=iteration,
                    deq=deq,
                    eq=eq,
                    nequa=nequa,
                )

            solvit_call_counter += 1
            current_call_idx = solvit_call_counter

            _log_eq_stage(
                "pre_solvit",
                layer_idx=j,
                iteration=iteration,
                eq_vec=eq,
                xn_vec=xn,
                nequa=nequa,
                electron_idx=electron_idx,
            )

            if j == 0 and iteration <= MAX_SOLVIT_LOG_ITER:
                _log_tracked_deq_columns(
                    j, iteration, deq_2d, eq_copy[:nequa], current_call_idx
                )
                _log_eq_vector(
                    "Before SOLVIT, EQ vector",
                    "EQ_before",
                    iteration,
                    eq_copy[:nequa],
                )

            # CRITICAL: Create a deep copy right before the call to ensure it's not modified
            eq_copy_final = eq_copy.copy()
            if _should_dump_solvit_state(j, iteration):
                _dump_solvit_state(
                    layer_idx=j,
                    iteration=iteration,
                    call_idx=current_call_idx,
                    matrix=deq_2d,
                    rhs=eq_copy_final[:nequa],
                )
            # Solve using a working copy so elimination does not corrupt the matrix
            # that subsequent Newton iterations will rebuild.
            matrix_for_solvit = np.array(deq_2d, copy=True)

            # LOG-SPACE: Scale Jacobian for log-space Newton
            # DEQ_log[i,j] = DEQ[i,j] * XN[j]
            # This accounts for the chain rule: ∂F/∂(log X) = ∂F/∂X * X
            if use_log_space:
                for col_j in range(nequa):
                    xn_j = xn[col_j]
                    # Use log-space scaling to prevent overflow when XN is huge
                    log_xn_j = (
                        log_xn[col_j]
                        if log_xn is not None
                        else np.log(max(abs(xn_j), 1e-300))
                    )
                    for row_i in range(nequa):
                        deq_val = matrix_for_solvit[row_i, col_j]
                        if deq_val == 0.0 or not np.isfinite(deq_val):
                            continue
                        # Compute log(|DEQ * XN|) = log(|DEQ|) + log(XN)
                        sign_deq = 1 if deq_val >= 0 else -1
                        log_deq = np.log(abs(deq_val))
                        log_scaled = log_deq + log_xn_j
                        # Convert back with clamping
                        if log_scaled > 700:
                            matrix_for_solvit[row_i, col_j] = sign_deq * 1e307
                        elif log_scaled < -700:
                            matrix_for_solvit[row_i, col_j] = 0.0
                        else:
                            matrix_for_solvit[row_i, col_j] = sign_deq * np.exp(
                                log_scaled
                            )

            # CRITICAL: Pass eq_copy_final directly - don't modify it
            _set_solvit_context(j + 1, iteration, current_call_idx)

            # Check if we should use Decimal precision SOLVIT
            # DISABLED: The _solvit_decimal function can't handle NaN/inf values properly,
            # and automatic fallback causes InvalidOperation errors. Keep using float64 SOLVIT.
            use_decimal_solvit = False

            if use_decimal_solvit:
                # Use Decimal-precision SOLVIT for extreme values
                # Convert 2D matrix back to flat column-major for _solvit_decimal
                matrix_flat = matrix_for_solvit.T.flatten()  # Column-major
                delta_xn = _solvit_decimal(
                    matrix_flat,
                    nequa,
                    eq_copy_final,
                )
            else:
                delta_xn = _solvit(
                    matrix_for_solvit,
                    nequa,
                    eq_copy_final,
                    zero_pivot_fix=zero_pivot_fix,
                )

            # Note: SOLVIT modifies DEQ_2d in-place, so we can print it after
            if j == 0 and iteration <= MAX_SOLVIT_LOG_ITER and delta_xn is not None:
                _log_eq_vector(
                    "After SOLVIT, EQ vector",
                    "EQ_after",
                    iteration,
                    delta_xn[:nequa],
                )
            # CRITICAL: SOLVIT should never return None now - it continues even with zero pivot
            # But keep this check for safety
            if delta_xn is None:
                # SOLVIT failed unexpectedly - exit loop and use last XN values
                break

            # CRITICAL: Fortran does NOT check for Inf/NaN before using EQ(K)!
            # Fortran code (atlas7v.for lines 5039-5054) uses EQ(K) directly:
            #   RATIO=ABS(EQ(K)/XN(K))  - no check for Inf/NaN
            #   XNEQ=XN(K)-EQ(K)        - no check for Inf/NaN
            #   XN(K)=XNEQ              - sets XN to Inf if XNEQ=Inf
            # Fortran continues iteration even with Inf/NaN in XN
            # Python must match this behavior exactly!

            # After SOLVIT, eq_copy (now delta_xn) contains the solution
            # Fortran modifies EQ in-place, so we update eq to match while
            # keeping XN as the pre-solve values until the damping loop updates them.
            eq[:] = delta_xn

            log_eq_components_now = (
                _TRACE_EQ_COMPONENTS_ALL_LAYERS
                or j == 0
                or j in _TRACE_EQ_COMPONENT_LAYERS
            )
            if _TRACE_EQ_COMPONENTS and log_eq_components_now:
                _log_eq_components(
                    layer_idx=j,
                    iteration=iteration,
                    eq_vec=eq,
                    xn_vec=xn,
                )

            # Update XN and check convergence
            # From atlas7v_1.for lines 3806-3824
            iferr = 0
            scale = 100.0

            for k in range(nequa):
                xn_before = xn[k]
                scale_before_k = scale  # Capture SCALE value before processing this K
                eqold_before_k = eqold[k]

                # After SOLVIT, EQ(K) contains the solution (delta XN)
                # Fortran: RATIO=ABS(EQ(K)/XN(K)) - NO CHECK FOR ZERO/INF!
                # Python must match: calculate ratio even if xn[k]=0 or eq[k]=Inf
                # Use try/except to handle division by zero like Fortran (produces Inf)
                ratio = _ratio_preserving_precision(eq[k], xn[k])
                if ratio > 0.001:
                    iferr = 1

                # Damping
                eq_before_damping = eq[k]
                damping_applied = False
                # Use sign comparison instead of multiplication to avoid overflow
                sign_change = (eqold[k] > 0 and eq[k] < 0) or (
                    eqold[k] < 0 and eq[k] > 0
                )
                if sign_change:
                    eq_k = np.float64(eq[k])
                    if np.isfinite(eq_k):
                        eq[k] = eq_k * 0.69
                    else:
                        eq[k] = eq_k
                    damping_applied = True

                # Fortran: XNEQ = XN(K) - EQ(K) where EQ(K) is the solution from SOLVIT
                xn_val = np.float64(xn[k])
                eq_val = np.float64(eq[k])
                xneq = _stable_subtract(xn_val, eq_val)
                xn100 = xn[k] / 100.0

                # Note: xnatom_inout will be updated after xn[k] is assigned (see below)

                scale_used = 1.0
                branch = "direct"
                scale_modified = False

                # LOG-SPACE UPDATE: Use additive update in log-space
                # This prevents overflow because we add to log(XN) instead of subtracting from XN
                if use_log_space:
                    # In log-space, eq[k] is the change in log(XN[k])
                    # (because we scaled the Jacobian by XN[j])
                    delta_log_k = eq[k]

                    # Apply damping in log-space
                    if damping_applied:  # Sign flip detected above
                        delta_log_k = delta_log_k * 0.69

                    # Clamp delta to prevent extreme jumps
                    max_delta_log = 10.0  # Max ~22000x change per iteration
                    delta_log_k = max(-max_delta_log, min(max_delta_log, delta_log_k))

                    # Update log-space
                    log_xn[k] = log_xn[k] + delta_log_k

                    # Clamp log_xn to valid range
                    log_xn[k] = max(-LOG_XN_MAX, min(LOG_XN_MAX, log_xn[k]))

                    # Convert back to linear space
                    xn[k] = _from_log_space_scalar(log_xn[k])
                    branch = "log_space"

                elif xneq < xn100:
                    branch = "scale"
                    scale_used = scale
                    xn[k] = xn[k] / scale_used
                    # Use sign comparison instead of multiplication to avoid overflow
                    sign_change_scale = (eqold[k] > 0 and eq[k] < 0) or (
                        eqold[k] < 0 and eq[k] > 0
                    )
                    if sign_change_scale:
                        scale_old = scale
                        scale = np.sqrt(scale)
                        scale_modified = True
                    else:
                        scale_modified = False
                else:
                    xn[k] = xneq

                # BOUNDED NEWTON: Enforce physical bounds to prevent divergence
                if bounded_newton_active:
                    xn_min = 1e-100  # Minimum physical value
                    xn_max = 10.0 * xntot  # Maximum reasonable value
                    if xn[k] < xn_min:
                        xn[k] = xn_min
                    elif xn[k] > xn_max:
                        xn[k] = xn_max

                # BUG FIX: Do NOT update xnatom_inout with XN[0]!
                # XN[0] represents NUCLEI (atoms + extra atoms in molecules), but
                # xnatom should be PARTICLES (P/TK - XNE) for:
                #   - RHO = XNATOM * WTMOLE * 1.66e-24 (mass density)
                #   - Element populations = XNATOM * XABUND
                # The original code updated xnatom_inout with XN[0], causing a 1.85x error
                # in cool atmospheres where H2 fraction is high.
                #
                # if xnatom_inout is not None and k == 0:  # DISABLED
                #     xnatom_inout[j] = xn[k]  # REMOVED - XN[0] is NUCLEI, not PARTICLES!

                if _should_trace_xn(j, iteration, k):
                    _log_xn_update(
                        layer_idx=j,
                        iteration=iteration,
                        k_idx=k,
                        xn_before=xn_before,
                        xn_after=xn[k],
                        eq_value=eq[k],
                        xneq=xneq,
                        xn100=xn100,
                        ratio=ratio,
                        branch=branch,
                        scale_value=scale_used,
                    )

                if not np.isfinite(xn[k]):
                    if nonfinite_xn_hits < NONFINITE_LOG_LIMIT:
                        _log_nonfinite_event(
                            "xn",
                            f"layer={j} iter={iteration} k={k} eq={eq[k]:.6e} "
                            f"xneq={xneq:.6e} xn100={xn100:.6e} scale={scale:.6e} "
                            f"branch={'scale' if xneq < xn100 else 'direct'}",
                        )
                    nonfinite_xn_hits += 1

                # Fortran does NOT check for NaN/inf - it just assigns XN(K)=XNEQ
                # Match Fortran exactly: no checks, no clamping, allow inf/nan to propagate

                eqold[k] = eq[k]

                if _TRACE_NEWTON_UPDATES:
                    _log_newton_update(
                        layer_idx=j,
                        iteration=iteration,
                        k_idx=k,
                        xn_before=xn_before,
                        eq_before_damping=eq_before_damping,
                        eq_after_damping=eq[k],
                        eqold_before=eqold_before_k,
                        eqold_after=eqold[k],
                        xneq=xneq,
                        xn100=xn100,
                        ratio=ratio,
                        branch=branch,
                        scale_before=scale_before_k,
                        scale_used=scale_used,
                        scale_after=scale,
                        damping_applied=damping_applied,
                        scale_modified=scale_modified,
                    )

            iteration_one_based = iteration + 1
            if iferr == 0:
                if MIN_NEWTON_ITER > 0 and iteration_one_based < MIN_NEWTON_ITER:
                    continue
                converged = True
                break

        # CRITICAL: Fortran ALWAYS uses XN(1) as XNATOM(J) after the iteration loop
        # completes, regardless of whether convergence succeeded or failed
        # (atlas7v.for around line 5243). The IFERR flag only controls whether the
        # Newton loop keeps iterating; it does not change how XNATOM is stored.
        #
        # To match Fortran exactly, always store XN(1) as the molecular XNATOM for
        # this layer, without any additional thresholds or fallbacks.
        xnatom_molecular[j] = xn[0]

        # Fortran line 5049-5050: Store XN to XNZ after iteration
        # DO 107 K=1,NEQUA
        # XNZ(J,K)=XN(K)
        # 107 CONTINUE
        for k in range(nequa):
            xnz_molecular[j, k] = xn[k]
        xnz_prev[:nequa] = xn[:nequa]
        prev_layer_idx = j  # Track for continuation

        if idequa[nequa - 1] == 100:
            # Fortran atlas7v.for line 5847: XNE(J)=XN(NEQUA)
            # Store the Newton-converged electron density, NOT the initial guess
            # The previous xntot/20 was WRONG and caused XNE to be 4500× too high!
            electron_density[j] = xn[nequa - 1]
            xne_computed[j] = electron_density[j]

        # Compute molecular number densities
        # From atlas7v_1.for lines 3831-3842
        for jmol in range(nummol):
            ncomp = locj[jmol + 1] - locj[jmol]
            xnmol[j, jmol] = equilj[jmol]
            locj1 = locj[jmol]
            locj2 = locj[jmol + 1] - 1
            for lock in range(locj1, locj2 + 1):
                k = kcomps[lock]  # 0-based equation number
                # CRITICAL: kcomps = nequa (23) is sentinel for inverse electrons
                if k == nequa:  # Sentinel value for inverse electrons
                    k = nequa - 1  # Map to actual electron equation index
                    xnmol[j, jmol] = xnmol[j, jmol] / xn[k]
                else:
                    xnmol[j, jmol] = xnmol[j, jmol] * xn[k]

    return xnatom_molecular, xnmol, xnz_molecular


def _solvit_kernel_python(
    a_work: np.ndarray,
    b_work: np.ndarray,
    ipivot: np.ndarray,
    n: int,
) -> None:
    """Python fallback for SOLVIT kernel (used when Numba not available)."""
    # Use FMA for multiply-subtract operations when available to keep
    # elimination steps numerically close to Fortran's x87 intermediate precision.
    has_fma = hasattr(math, "fma")

    def _stable_submul(lhs: float, rhs: float, factor: float) -> float:
        """Compute lhs - rhs*factor using FMA when available.

        Note: math.fma() can throw OverflowError even for finite inputs when
        the result would overflow. We catch this and fall back to regular
        arithmetic which returns inf instead of throwing.
        """
        if np.isfinite(lhs) and np.isfinite(rhs) and np.isfinite(factor):
            if has_fma:
                try:
                    return math.fma(-rhs, factor, lhs)
                except OverflowError:
                    # FMA overflows - fall back to regular arithmetic (returns inf)
                    pass
            return lhs - rhs * factor
        return lhs - rhs * factor

    for iter_idx in range(1, n + 1):
        amax = 0.0
        irow = 1
        icolum = 1

        # Pivot search
        for row in range(1, n + 1):
            if ipivot[row] == 1:
                continue
            jk = row - n
            for col in range(1, n + 1):
                jk = jk + n
                if ipivot[col] == 1:
                    continue
                aa = abs(a_work[jk])
                if aa > amax:
                    amax = aa
                    irow = row
                    icolum = col

        ipivot[icolum] += 1

        # Row/column swap if needed
        if irow != icolum:
            irl = irow - n
            icl = icolum - n
            for _ in range(1, n + 1):
                irl += n
                swap_val = a_work[irl]
                icl += n
                a_work[irl] = a_work[icl]
                a_work[icl] = swap_val
            b_work[irow], b_work[icolum] = b_work[icolum], b_work[irow]

        # Normalize pivot row
        pivot_idx = icolum * n + icolum - n
        pivot = a_work[pivot_idx]
        a_work[pivot_idx] = 1.0
        icl = icolum - n
        for _ in range(1, n + 1):
            icl += n
            a_work[icl] = a_work[icl] / pivot
        b_work[icolum] = b_work[icolum] / pivot

        # Elimination
        l1ic = icolum * n - n
        for l1 in range(1, n + 1):
            l1ic += 1
            if l1 == icolum:
                continue
            t = a_work[l1ic]
            a_work[l1ic] = 0.0
            if t == 0.0:
                continue
            l1l = l1 - n
            icl = icolum - n
            for _ in range(1, n + 1):
                l1l += n
                icl += n
                # Use FMA for numerical stability (matches original _solvit behavior)
                a_work[l1l] = _stable_submul(a_work[l1l], a_work[icl], t)
            # Use FMA for numerical stability (matches original _solvit behavior)
            b_work[l1] = _stable_submul(b_work[l1], b_work[icolum], t)


@jit(nopython=True, cache=True)
def _solvit_kernel(
    a_work: np.ndarray,
    b_work: np.ndarray,
    ipivot: np.ndarray,
    n: int,
) -> None:
    """Core SOLVIT algorithm without I/O or logging (Numba-optimized).

    This is the pure numerical computation extracted from _solvit.
    Modifies a_work and b_work in-place.

    Args:
        a_work: 1-based column-major matrix array (size n*n+1)
        b_work: 1-based RHS vector (size n+1)
        ipivot: 1-based pivot tracking array (size n+1)
        n: Matrix dimension
    """
    # Note: Numba doesn't support math.fma directly, but modern CPUs
    # will use FMA instructions automatically when optimizing (-rhs * factor + lhs)
    # The compiler will recognize this pattern and use FMA if available.
    # For numerical stability, we use the same pattern as the original code.

    for iter_idx in range(1, n + 1):
        amax = 0.0
        irow = 1
        icolum = 1

        # Pivot search
        for row in range(1, n + 1):
            if ipivot[row] == 1:
                continue
            jk = row - n
            for col in range(1, n + 1):
                jk = jk + n
                if ipivot[col] == 1:
                    continue
                aa = abs(a_work[jk])
                if aa > amax:
                    amax = aa
                    irow = row
                    icolum = col

        ipivot[icolum] += 1

        # Row/column swap if needed
        if irow != icolum:
            irl = irow - n
            icl = icolum - n
            for _ in range(1, n + 1):
                irl += n
                swap_val = a_work[irl]
                icl += n
                a_work[irl] = a_work[icl]
                a_work[icl] = swap_val
            b_work[irow], b_work[icolum] = b_work[icolum], b_work[irow]

        # Normalize pivot row
        pivot_idx = icolum * n + icolum - n
        pivot = a_work[pivot_idx]
        a_work[pivot_idx] = 1.0
        icl = icolum - n
        for _ in range(1, n + 1):
            icl += n
            a_work[icl] = a_work[icl] / pivot
        b_work[icolum] = b_work[icolum] / pivot

        # Elimination
        l1ic = icolum * n - n
        for l1 in range(1, n + 1):
            l1ic += 1
            if l1 == icolum:
                continue
            t = a_work[l1ic]
            a_work[l1ic] = 0.0
            if t == 0.0:
                continue
            l1l = l1 - n
            icl = icolum - n
            for _ in range(1, n + 1):
                l1l += n
                icl += n
                # Compute lhs - rhs*factor (Numba will optimize to FMA if available)
                a_work[l1l] = a_work[l1l] - a_work[icl] * t
            # Compute lhs - rhs*factor (Numba will optimize to FMA if available)
            b_work[l1] = b_work[l1] - b_work[icolum] * t


# =============================================================================
# DECIMAL-PRECISION SOLVIT
# =============================================================================
# Uses Python's built-in Decimal module for extended precision (50 digits).
# This matches or exceeds Fortran's 80-bit precision (~18-19 digits).


def _to_decimal(value: float) -> Decimal:
    """Convert float to Decimal, handling special values."""
    if not np.isfinite(value):
        if np.isnan(value):
            return Decimal("NaN")
        return Decimal("Infinity") if value > 0 else Decimal("-Infinity")
    return Decimal(str(value))


def _from_decimal(d: Decimal, clamp_log: float = 300.0) -> float:
    """Convert Decimal back to float with clamping for extreme values."""
    if d.is_nan():
        return float("nan")
    if d.is_infinite():
        return float("inf") if d > 0 else float("-inf")
    if d == 0:
        return 0.0

    # Check for overflow/underflow
    sign, digits, exp = d.as_tuple()
    if digits == (0,):
        return 0.0
    log10_approx = exp + len(digits) - 1

    if log10_approx > clamp_log:
        return float("inf") if sign == 0 else float("-inf")
    if log10_approx < -clamp_log:
        return 0.0

    return float(d)


def _nmolec_newton_bounded(
    xn_init: np.ndarray,
    xab: np.ndarray,
    equilj: np.ndarray,
    locj: np.ndarray,
    kcomps: np.ndarray,
    idequa: np.ndarray,
    nequa: int,
    nummol: int,
    xntot: float,
    max_iter: int = 200,
    layer_idx: int = 0,
    temperature: float = 5000.0,
    tol: float = 1e-3,
) -> tuple:
    """
    Bounded Newton iteration for NMOLEC with trust-region step limiting.

    This prevents the chaotic divergence that causes the solver to land in the
    wrong basin of attraction (atomic solution instead of molecular solution)
    for cool atmospheres.

    Key features:
    1. Works in log-space to handle large dynamic range
    2. Trust region limits step sizes to prevent wild divergence
    3. Line search ensures each step improves the residual
    4. Bounds enforcement keeps solutions physical

    Returns:
        (xn_solution, converged): Solution array and convergence flag
    """
    import os

    # Initialize in log-space
    xn = np.maximum(xn_init.copy(), 1e-100)
    log_xn = np.log(xn)

    # Bounds for log(XN)
    log_xn_min = -230  # ~1e-100
    log_xn_max = np.log(10 * xntot)

    # Trust radius (in log-space, so exp(5) ~ 150x change max)
    trust_radius = 5.0

    def compute_eq_residual(log_xn_vec):
        """Compute equilibrium residuals matching NMOLEC structure."""
        xn_local = np.exp(log_xn_vec)
        eq = np.zeros(nequa)

        # First equation: mass balance
        # EQ(1) = -XNTOT + XN(2) + XN(3) + ... + XN(NEQUA)
        # Note: XN(1) is NOT included in the sum!
        eq[0] = -xntot
        for k in range(1, nequa):  # k=1..nequa-1 (0-based), NOT including k=0
            eq[0] += xn_local[k]

        # Element equations: EQ(K) = XN(K) - XAB(K)*XN(1)
        for k in range(1, nequa):
            eq[k] = xn_local[k] - xab[k] * xn_local[0]

        # Electron equation override (if electrons included)
        electron_idx = nequa - 1
        if idequa[electron_idx] >= 100:  # Electron equation
            eq[electron_idx] = -xn_local[electron_idx]

        # Add molecular contributions to equations
        for jmol in range(nummol):
            if equilj[jmol] == 0.0:
                continue

            ncomp = locj[jmol + 1] - locj[jmol]
            if ncomp <= 1:
                continue

            # Compute molecular term = EQUILJ * product(XN[k] for k in components)
            term = equilj[jmol]
            for iloc in range(locj[jmol], locj[jmol + 1]):
                k = (
                    kcomps[iloc] - 1
                )  # Convert to 0-based (kcomps is 1-based equation numbers)
                if 0 <= k < nequa:
                    term *= xn_local[k]

            # Add to relevant equations (each component contributes)
            for iloc in range(locj[jmol], locj[jmol + 1]):
                k = kcomps[iloc] - 1
                if 0 <= k < nequa:
                    eq[k] += term

        return eq

    def compute_jacobian(log_xn_vec):
        """Numerical Jacobian."""
        eps = 1e-8
        jac = np.zeros((nequa, nequa))
        f0 = compute_eq_residual(log_xn_vec)
        for i in range(nequa):
            log_xn_vec[i] += eps
            jac[:, i] = (compute_eq_residual(log_xn_vec) - f0) / eps
            log_xn_vec[i] -= eps
        return jac

    converged = False
    best_res_norm = float("inf")
    best_xn = xn.copy()

    for iteration in range(max_iter):
        # Compute residual
        eq = compute_eq_residual(log_xn)
        xn_current = np.exp(log_xn)

        # Scale residuals by XN for relative error
        scaled_eq = eq / np.maximum(xn_current, 1e-100)
        res_norm = np.sqrt(np.sum(scaled_eq**2))

        # Track best solution
        if res_norm < best_res_norm:
            best_res_norm = res_norm
            best_xn = xn_current.copy()

        # Check convergence (relative error < tol for all equations)
        max_ratio = np.max(np.abs(eq) / np.maximum(xn_current, 1e-100))
        if max_ratio < tol:
            converged = True
            break

        # Compute Jacobian
        jac = compute_jacobian(log_xn.copy())

        # Solve for Newton step
        try:
            # Add regularization for stability
            jac_reg = jac + 1e-10 * np.eye(nequa)
            delta = np.linalg.solve(jac_reg, -eq)
        except np.linalg.LinAlgError:
            # Singular Jacobian - use gradient descent step
            delta = -scaled_eq * 0.1

        # Limit step size (trust region)
        step_norm = np.sqrt(np.sum(delta**2))
        if step_norm > trust_radius:
            delta = delta * trust_radius / step_norm

        # Line search
        alpha = 1.0
        improved = False
        for _ in range(10):
            log_xn_new = log_xn + alpha * delta
            log_xn_new = np.clip(log_xn_new, log_xn_min, log_xn_max)

            eq_new = compute_eq_residual(log_xn_new)
            xn_new = np.exp(log_xn_new)
            scaled_eq_new = eq_new / np.maximum(xn_new, 1e-100)
            res_norm_new = np.sqrt(np.sum(scaled_eq_new**2))

            if res_norm_new < res_norm:
                log_xn = log_xn_new
                improved = True
                break

            alpha *= 0.5

        if not improved:
            # Take small step anyway
            log_xn = np.clip(log_xn + 0.1 * delta, log_xn_min, log_xn_max)

        # Adaptive trust radius
        if improved and res_norm_new < 0.5 * res_norm:
            trust_radius = min(trust_radius * 1.5, 20.0)
        elif not improved or res_norm_new > 0.9 * res_norm:
            trust_radius = max(trust_radius * 0.5, 0.5)

    # Use best solution found
    xn_solution = best_xn if not converged else np.exp(log_xn)

    return xn_solution, converged


def _solvit(
    a: np.ndarray,
    n: int,
    b: np.ndarray,
    use_extended_precision: bool = False,
    zero_pivot_fix: str = "none",
) -> Optional[np.ndarray]:
    """Port of ATLAS SOLVIT (atlas7v_1.for lines 1200-1295)."""

    def idx_cm(row: int, col: int) -> int:
        """Column-major offset mirroring JK = J + (K-1)*N."""
        return row + col * n

    def _log(message: str) -> None:
        if not log_file:
            return
        log_file.write(message + "\n")

    def _trace(message: str) -> None:
        if not trace_file:
            return
        trace_file.write(message + "\n")

    # Check if tracing is enabled
    # Only check flags - don't check file existence (too expensive for 16k calls)
    # If users want logging, they should set the environment variables
    tracing_enabled = (
        _TRACE_SOLVIT_MATRIX
        or TRACE_PIVOT_SEARCH
        or _current_solvit_layer in _TRACE_SOLVIT_LAYERS
    )

    log_path = os.path.join(os.getcwd(), "solvit_trace.log")
    trace_path = os.path.join(os.getcwd(), "solvit_trace.log")
    detail_path = os.path.join(os.getcwd(), "solvit_matrix_trace.log")
    pivot_trace_path = os.path.join(os.getcwd(), "solvit_pivot_trace.log")

    log_file = None
    trace_file = None
    detail_file = None
    pivot_trace_file = None

    # Only open log files if tracing is enabled
    if tracing_enabled:
        try:
            log_file = open(log_path, "a")
        except OSError:
            log_file = None
        trace_path = os.path.join(os.getcwd(), "solvit_trace.log")
        try:
            trace_file = open(trace_path, "a")
        except OSError:
            trace_file = None
        detail_path = os.path.join(os.getcwd(), "solvit_matrix_trace.log")
        detail_file = None
        if _TRACE_SOLVIT_MATRIX:
            try:
                detail_file = open(detail_path, "a")
            except OSError:
                detail_file = None
        if TRACE_PIVOT_SEARCH:
            pivot_trace_path = os.path.join(os.getcwd(), "solvit_pivot_trace.log")
            try:
                pivot_trace_file = open(pivot_trace_path, "a")
            except OSError:
                pivot_trace_file = None

    ctx_layer = _current_solvit_layer
    ctx_iter = _current_solvit_iter
    ctx_call = _current_solvit_call
    ctx_suffix = (
        f"(layer={ctx_layer:3d} iter={ctx_iter:3d} call={ctx_call:5d})"
        if ctx_layer >= 0 and ctx_iter >= 0 and ctx_call is not None
        else ""
    )

    a_fortran = np.array(a, dtype=np.float64, order="F", copy=True)
    a_vec = np.reshape(a_fortran, a_fortran.size, order="F")
    size = n * n
    # Use 1-based working arrays to mirror Fortran indexing exactly.
    a_work = np.zeros(size + 1, dtype=np.float64)
    a_work[1:] = a_vec
    b_work = np.zeros(n + 1, dtype=np.float64)
    b_work[1:] = np.array(b, dtype=np.float64, copy=True)
    ipivot = np.zeros(n + 1, dtype=np.int32)

    def idx1(row_1b: int, col_1b: int) -> int:
        """1-based column-major index helper."""
        return row_1b + (col_1b - 1) * n

    # Use FMA for multiply-subtract operations when available to keep
    # elimination steps numerically close to Fortran's x87 intermediate precision.
    has_fma = hasattr(math, "fma")

    def _stable_submul(lhs: float, rhs: float, factor: float) -> float:
        """Compute lhs - rhs*factor using FMA when available.

        Note: math.fma() can throw OverflowError even for finite inputs when
        the result would overflow. We catch this and fall back to regular
        arithmetic which returns inf instead of throwing.
        """
        if np.isfinite(lhs) and np.isfinite(rhs) and np.isfinite(factor):
            if has_fma:
                try:
                    return math.fma(-rhs, factor, lhs)
                except OverflowError:
                    # FMA overflows - fall back to regular arithmetic (returns inf)
                    pass
            return lhs - rhs * factor
        return lhs - rhs * factor

    # Fast path: Use kernel when tracing is disabled
    if not tracing_enabled:
        _solvit_kernel(a_work, b_work, ipivot, n)
        return b_work[1:]

    # Slow path: Original code with logging

    def _log_matrix_state(stage: str) -> None:
        if not detail_file:
            return
        detail_file.write(
            f"SOLVIT_MATRIX {stage} layer={ctx_layer} iter={ctx_iter} call={ctx_call}\n"
        )
        detail_file.write("  B:")
        for idx in range(1, n + 1):
            detail_file.write(f" {b_work[idx]: .17E}")
        detail_file.write("\n")
        for row in range(1, n + 1):
            row_vals = [a_work[idx1(row, col)] for col in range(1, n + 1)]
            joined = " ".join(f"{val: .17E}" for val in row_vals)
            detail_file.write(f"  ROW {row:2d}: {joined}\n")

    # Track A(9,2) (1-based) before eliminations for comparison with Fortran logs.
    if n >= 9 and log_file:
        a9_2 = a_work[idx1(9, 2)]
        _log(f"PY_MATRIX iter  0: A9_2 BEFORE eliminations = {a9_2: .17E} {ctx_suffix}")
    _log_matrix_state("pre_iteration")

    def _log_pivot_candidate(row: int, col: int, value: float) -> None:
        if not pivot_trace_file:
            return
        pivot_trace_file.write(
            "PY_PIVOT_CAND layer={layer} iter={iter} call={call} "
            "solvit_iter={solvit_iter} row={row} col={col} value={value:.17E} "
            "ipiv_row={ipiv_row} ipiv_col={ipiv_col}\n".format(
                layer=ctx_layer,
                iter=ctx_iter,
                call=ctx_call,
                solvit_iter=iter_idx,
                row=row,
                col=col,
                value=value,
                ipiv_row=int(ipivot[row]),
                ipiv_col=int(ipivot[col]),
            )
        )

    for iter_idx in range(1, n + 1):
        amax = 0.0
        irow = 1
        icolum = 1

        for row in range(1, n + 1):
            if ipivot[row] == 1:
                continue
            jk = row - n
            for col in range(1, n + 1):
                jk = jk + n
                if ipivot[col] == 1:
                    continue
                aa = abs(a_work[jk])
                if aa > amax:
                    amax = aa
                    irow = row
                    icolum = col
                    _log_pivot_candidate(row, col, aa)

        ipivot[icolum] += 1

        if irow != icolum:
            irl = irow - n
            icl = icolum - n
            for _ in range(1, n + 1):
                irl += n
                swap_val = a_work[irl]
                icl += n
                a_work[irl] = a_work[icl]
                a_work[icl] = swap_val
            b_work[irow], b_work[icolum] = b_work[icolum], b_work[irow]
        _log_matrix_state(f"post_swap_iter{iter_idx}")

        pivot_idx = icolum * n + icolum - n
        pivot = a_work[pivot_idx]

        if log_file:
            _log(
                "PY_SOLVIT iter{iter:3d}: pivot row={row:3d} col={col:3d} "
                "amax={amax: .17E} pivot_val={pivot: .17E} {suffix}".format(
                    iter=iter_idx + 1,
                    row=irow,
                    col=icolum,
                    amax=amax,
                    pivot=pivot,
                    suffix=ctx_suffix,
                )
            )
        _trace(
            f"TRACE_SOLVIT iter={iter_idx} {ctx_suffix} "
            f"pivot_row={irow} pivot_col={icolum} amax={amax:.17E} pivot={pivot:.17E} "
            f"b_col_before={b_work[icolum]:.17E}"
        )

        if log_file and n > 8:
            _log(
                "PY_SOLVIT iter{iter:3d}: b[9] BEFORE normalization = {val: .17E}".format(
                    iter=iter_idx,
                    val=b_work[9] if n >= 9 else float("nan"),
                )
            )

        a_work[pivot_idx] = 1.0
        icl = icolum - n
        for _ in range(1, n + 1):
            icl += n
            a_work[icl] = a_work[icl] / pivot
        b_work[icolum] = b_work[icolum] / pivot
        _log_matrix_state(f"post_normalize_iter{iter_idx}")
        _trace(
            f"TRACE_SOLVIT iter={iter_idx} {ctx_suffix} "
            f"b_col_after={b_work[icolum]:.17E}"
        )

        if log_file and n > 8:
            _log(
                "PY_SOLVIT iter{iter:3d}: b[9] AFTER normalization = {val: .17E}".format(
                    iter=iter_idx,
                    val=b_work[9] if n >= 9 else float("nan"),
                )
            )

        l1ic = icolum * n - n
        for l1 in range(1, n + 1):
            l1ic += 1
            if l1 == icolum:
                continue
            t = a_work[l1ic]
            a_work[l1ic] = 0.0
            if t == 0.0:
                continue
            l1l = l1 - n
            icl = icolum - n
            for _ in range(1, n + 1):
                l1l += n
                icl += n
                a_work[l1l] = _stable_submul(a_work[l1l], a_work[icl], t)
            b_work[l1] = _stable_submul(b_work[l1], b_work[icolum], t)
        _log_matrix_state(f"post_eliminate_iter{iter_idx}")

        row0_sum = 0.0
        row0_max = 0.0
        for col in range(1, n + 1):
            val = abs(a_work[idx1(1, col)])
            row0_sum += val
            row0_max = max(row0_max, val)
        if log_file:
            _log(
                "PY_SOLVIT iter{iter:3d}: row0 sum_abs={sumv: .17E} "
                "max={maxv: .17E} b[1]={b0: .17E} {suffix}".format(
                    iter=iter_idx,
                    sumv=row0_sum,
                    maxv=row0_max,
                    b0=b_work[1],
                    suffix=ctx_suffix,
                )
            )
            _log(
                "PY_SOLVIT iter{iter:3d} AFTER: row0 sum_abs={sumv: .17E} "
                "max={maxv: .17E} b[1]={b0: .17E} {suffix}".format(
                    iter=iter_idx,
                    sumv=row0_sum,
                    maxv=row0_max,
                    b0=b_work[1],
                    suffix=ctx_suffix,
                )
            )
            _log(
                "PY_SOLVIT iter{iter:3d} All b values after elimination "
                "(layer={layer:3d} iter={iter_ctx:3d} call={call:5d})".format(
                    iter=iter_idx,
                    layer=ctx_layer if ctx_layer >= 0 else -1,
                    iter_ctx=ctx_iter if ctx_iter >= 0 else -1,
                    call=ctx_call if ctx_call is not None else -1,
                )
            )
            for row in range(1, n + 1):
                _log(f"  b[{row:2d}] = {b_work[row]: .17E}")
        _trace(
            f"TRACE_SOLVIT iter={iter_idx} {ctx_suffix} "
            f"row0_sum={row0_sum:.17E} row0_max={row0_max:.17E} b1={b_work[1]:.17E}"
        )

    if log_file:
        log_file.close()
    if trace_file:
        trace_file.close()
    if detail_file:
        detail_file.close()
    if pivot_trace_file:
        pivot_trace_file.close()

    return b_work[1:]
