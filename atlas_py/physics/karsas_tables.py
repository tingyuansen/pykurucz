"""Karsas cross-section tables extracted from atlas7v.for."""

from __future__ import annotations

import hashlib

import numpy as np

from numba import jit

from .karsas_tables_data import load_karsas_tables as _load_karsas

_KT = _load_karsas()
FREQ_LOG = _KT["FREQ_LOG"]
XN_LOG = _KT["XN_LOG"]
XL_LOG_ARRAY = _KT["XL_LOG_ARRAY"]
EKARSAS = _KT["EKARSAS"]


LN10 = np.log(10.0)

# Convert XL_LOG dictionary to 3D numpy array for Numba compatibility
# Shape: (max_ell+1, max_n, table_size) = (6, 6, 29)
# Invalid entries (ell >= n or n > 6) are filled with NaN

@jit(nopython=True)
def _xkarsas_jit(
    freq: float,
    zeff_squared: float,
    n: int,
    ell: int,
    freq_log_table: np.ndarray,
    xn_log_table: np.ndarray,
    xl_log_array: np.ndarray,
    ekarsas_table: np.ndarray,
    ln10: float,
) -> float:
    """Return the Karsas hydrogen cross-section coefficient - JIT-compiled."""
    if freq <= 0.0 or zeff_squared <= 0.0 or n <= 0:
        return 0.0
    if ell < 0:
        ell = 0
    freq_log = np.log10(freq / zeff_squared)  # Table lookup uses log10

    if n <= 15:
        column = freq_log_table[:, n - 1]
        if freq_log < column[-1]:
            return 0.0
        if ell >= n or n > 6:
            values = xn_log_table[:, n - 1]
        else:
            # Check if XL_LOG entry exists (not NaN)
            values_slice = xl_log_array[ell, n - 1, :]
            if np.isnan(values_slice[0]):
                return 0.0
            values = values_slice

        # Binary search: column is descending, find first idx >= 1 where freq_log > column[idx]
        # Binary search on descending array: find rightmost idx where column[idx] < freq_log
        left, right = 1, column.size - 1
        idx = column.size  # Default: not found

        while left <= right:
            mid = (left + right) // 2
            if freq_log > column[mid]:
                idx = mid
                right = mid - 1  # Search left for earlier match
            else:
                left = mid + 1  # Search right

        if idx >= column.size:
            return np.exp(values[-1] * ln10) / zeff_squared

        # Linear interpolation between idx-1 and idx
        denom = column[idx - 1] - column[idx]
        if abs(denom) < 1e-15:
            return np.exp(values[idx - 1] * ln10) / zeff_squared
        weight = (freq_log - column[idx]) / denom
        x_prev = values[idx - 1]
        x_curr = values[idx]
        x_val = (x_prev - x_curr) * weight + x_curr
        return np.exp(x_val * ln10) / zeff_squared

    freqn15 = np.empty(29, dtype=np.float64)
    inv_n2 = 1.0 / (n * n)
    ryd_c = 109677.576 * 2.99792458e10
    freqn15[-1] = np.log10(ryd_c * inv_n2)  # Table lookup uses log10
    if freq_log < freqn15[-1]:
        return 0.0
    for idx in range(1, 28):
        freqn15[idx] = np.log10(
            (ekarsas_table[idx] + inv_n2) * ryd_c
        )  # Table lookup uses log10
        if freq_log > freqn15[idx]:
            denom = freqn15[idx - 1] - freqn15[idx]
            if denom == 0.0:
                return 0.0
            weight = (freq_log - freqn15[idx]) / denom
            x_prev = xn_log_table[idx - 1, 14]
            x_curr = xn_log_table[idx, 14]
            x_val = (x_prev - x_curr) * weight + x_curr
            return np.exp(x_val * ln10) / zeff_squared
    x_val = xn_log_table[28, 14]
    return np.exp(x_val * ln10) / zeff_squared


# Pure Python fallback (for when Numba is not available)
def _xkarsas_python(freq: float, zeff_squared: float, n: int, ell: int) -> float:
    """Pure Python fallback for xkarsas when Numba is not available."""
    if freq <= 0.0 or zeff_squared <= 0.0 or n <= 0:
        return 0.0
    if ell < 0:
        ell = 0
    freq_log = np.log10(freq / zeff_squared)

    if n <= 15:
        column = FREQ_LOG[:, n - 1]
        if freq_log < column[-1]:
            return 0.0
        if ell >= n or n > 6:
            values = XN_LOG[:, n - 1]
        else:
            try:
                values = XL_LOG[(ell, n)]
            except KeyError:
                return 0.0

        # Binary search
        left, right = 1, column.size - 1
        idx = column.size

        while left <= right:
            mid = (left + right) // 2
            if freq_log > column[mid]:
                idx = mid
                right = mid - 1
            else:
                left = mid + 1

        if idx >= column.size:
            return float(np.exp(values[-1] * LN10) / zeff_squared)

        denom = column[idx - 1] - column[idx]
        if abs(denom) < 1e-15:
            return float(np.exp(values[idx - 1] * LN10) / zeff_squared)
        weight = (freq_log - column[idx]) / denom
        x_prev = values[idx - 1]
        x_curr = values[idx]
        x_val = (x_prev - x_curr) * weight + x_curr
        return float(np.exp(x_val * LN10) / zeff_squared)

    freqn15 = np.empty(29, dtype=np.float64)
    inv_n2 = 1.0 / (n * n)
    ryd_c = 109677.576 * 2.99792458e10
    freqn15[-1] = np.log10(ryd_c * inv_n2)
    if freq_log < freqn15[-1]:
        return 0.0
    for idx in range(1, 28):
        freqn15[idx] = np.log10((EKARSAS[idx] + inv_n2) * ryd_c)
        if freq_log > freqn15[idx]:
            denom = freqn15[idx - 1] - freqn15[idx]
            if denom == 0.0:
                return 0.0
            weight = (freq_log - freqn15[idx]) / denom
            x_prev = XN_LOG[idx - 1, 14]
            x_curr = XN_LOG[idx, 14]
            x_val = (x_prev - x_curr) * weight + x_curr
            return float(np.exp(x_val * LN10) / zeff_squared)
    x_val = XN_LOG[28, 14]
    return float(np.exp(x_val * LN10) / zeff_squared)


# Public API: wrapper function for backward compatibility
def xkarsas(freq: float, zeff_squared: float, n: int, ell: int) -> float:
    """Return the Karsas hydrogen cross-section coefficient."""
    return _xkarsas_jit(
        freq, zeff_squared, n, ell, FREQ_LOG, XN_LOG, XL_LOG_ARRAY, EKARSAS, LN10
    )


@jit(nopython=True)
def _xkarsas_grid_jit(
    freq: np.ndarray,
    zeff_squared: float,
    n: int,
    ell: int,
    freq_log_table: np.ndarray,
    xn_log_table: np.ndarray,
    xl_log_array: np.ndarray,
    ekarsas_table: np.ndarray,
    ln10: float,
) -> np.ndarray:
    out = np.empty(freq.size, dtype=np.float64)
    for i in range(freq.size):
        out[i] = _xkarsas_jit(
            freq[i],
            zeff_squared,
            n,
            ell,
            freq_log_table,
            xn_log_table,
            xl_log_array,
            ekarsas_table,
            ln10,
        )
    return out


_XKARSAS_GRID_CACHE: dict[tuple[tuple[int, str, bytes], float, int, int], np.ndarray] = {}


def _freq_grid_cache_key(freq: np.ndarray) -> tuple[np.ndarray, tuple[int, str, bytes]]:
    freq_arr = np.ascontiguousarray(freq, dtype=np.float64)
    digest = hashlib.blake2b(freq_arr.view(np.uint8), digest_size=16).digest()
    return freq_arr, (int(freq_arr.size), freq_arr.dtype.str, digest)


def xkarsas_grid(freq: np.ndarray, zeff_squared: float, n: int, ell: int) -> np.ndarray:
    """Return Karsas cross-sections for a frequency grid."""
    freq_arr, freq_key = _freq_grid_cache_key(freq)
    key = (freq_key, float(zeff_squared), int(n), int(ell))
    cached = _XKARSAS_GRID_CACHE.get(key)
    if cached is not None:
        return cached.copy()
    out = _xkarsas_grid_jit(
        freq_arr,
        zeff_squared,
        n,
        ell,
        FREQ_LOG,
        XN_LOG,
        XL_LOG_ARRAY,
        EKARSAS,
        LN10,
    )
    _XKARSAS_GRID_CACHE[key] = out
    return out.copy()
