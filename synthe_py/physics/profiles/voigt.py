"""Voigt profile implementation faithful to the legacy SYNTHE code."""

from __future__ import annotations

import numpy as np

from .. import tables


def voigt_profile(v: np.ndarray | float, a: np.ndarray | float) -> np.ndarray | float:
    """Evaluate the Voigt function H(a, v) using the Fortran approximation."""

    scalar_input = np.isscalar(v) and np.isscalar(a)
    v_arr = np.atleast_1d(np.asarray(v, dtype=np.float64))
    a_arr = np.atleast_1d(np.asarray(a, dtype=np.float64))

    voigt_tables = tables.voigt_tables()
    n = v_arr.size
    result = np.zeros(n, dtype=np.float64)

    for idx in range(n):
        vi = v_arr[idx]
        ai = a_arr[idx] if a_arr.size > 1 else a_arr[0]

        # CRITICAL FIX: Voigt function is symmetric in v, so use abs(vi) for table lookup
        # Bug was: negative vi -> negative index -> clamped to 0 -> returned center value!
        # FIX: Use +0.5 for proper rounding, not +1.5 which caused systematic 1-2% errors
        # Table index i corresponds to x = i/200, so for x=v we want i = round(v*200) = int(v*200 + 0.5)
        iv = int(abs(vi) * 200.0 + 0.5)
        iv = max(0, min(iv, voigt_tables.h0tab.size - 1))

        if ai < 0.2:
            if abs(vi) > 10.0:
                result[idx] = 0.5642 * ai / (vi ** 2)
            else:
                # Fortran VOIGT line 1711: (H2TAB*A + H1TAB)*A + H0TAB
                result[idx] = (
                    (voigt_tables.h2tab[iv] * ai + voigt_tables.h1tab[iv]) * ai
                    + voigt_tables.h0tab[iv]
                )
        elif ai > 1.4 or (ai + abs(vi)) > 3.2:
            aa = ai * ai
            vv = vi * vi
            u = (aa + vv) * 1.4142
            voigt_val = ai * 0.79788 / u
            if ai <= 100.0:
                aau = aa / u
                vvu = vv / u
                uu = u * u
                voigt_val = (
                    (((aau - 10.0 * vvu) * aau * 3.0 + 15.0 * vvu * vvu) + 3.0 * vv - aa)
                    / uu
                    + 1.0
                ) * voigt_val
            result[idx] = voigt_val
        else:
            vv = vi * vi
            h0 = voigt_tables.h0tab[iv]
            h1 = voigt_tables.h1tab[iv] + h0 * 1.12838
            h2 = voigt_tables.h2tab[iv] + h1 * 1.12838 - h0
            h3 = (1.0 - voigt_tables.h2tab[iv]) * 0.37613 - h1 * 0.66667 * vv + h2 * 1.12838
            h4 = (3.0 * h3 - h1) * 0.37613 + h0 * 0.66667 * vv * vv
            poly_a = (((h4 * ai + h3) * ai + h2) * ai + h1) * ai + h0
            poly_b = (
                ((-0.122727278 * ai + 0.532770573) * ai - 0.96284325) * ai
                + 0.979895032
            )
            result[idx] = poly_a * poly_b

    return result[0] if scalar_input else result
