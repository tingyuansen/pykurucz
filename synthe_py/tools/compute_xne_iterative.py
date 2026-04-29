#!/usr/bin/env python3
"""Iterative computation of XNE (electron density) using POPS, matching Fortran exactly.

This implements the iterative algorithm from atlas7v.for lines 2945-2974.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Callable

# Constants
K_BOLTZ_FORTRAN = 1.38054e-16  # erg/K (from atlas7v.for line 1954)


def compute_xne_iterative(
    temperature: np.ndarray,
    tk: np.ndarray,
    gas_pressure: np.ndarray,
    xabund: np.ndarray,  # Abundances for elements 1-99 (shape: 99)
    pfsaha_func: Callable,  # Function: pfsaha_func(j, iz, nion, mode, current_xne=None, current_xnatom=None) -> electron_contribution
    max_iterations: int = 200,
    tolerance: float = 0.0005,
) -> Tuple[np.ndarray, np.ndarray]:
    """Iteratively compute XNE and XNATOM matching Fortran algorithm.

    From atlas7v.for lines 2945-2974:
    - Initialize XNE(1) = P(1)/TK(1)/2
    - For each layer, iterate until convergence
    - XNATOM = P/TK - XNE

    Args:
        temperature: Temperature array (n_layers,)
        tk: k_B * T array (n_layers,)
        gas_pressure: Gas pressure array (n_layers,)
        xabund: Element abundances (99,)
        pfsaha_func: Function to call PFSAHA(iz, nion, mode=4) returning electron contribution
        max_iterations: Maximum iterations per layer (default 200)
        tolerance: Convergence tolerance (default 0.0005)

    Returns:
        (xne, xnatom) arrays with shape (n_layers,)
    """
    n_layers = len(temperature)
    xne = np.zeros(n_layers, dtype=np.float64)
    xnatom = np.zeros(n_layers, dtype=np.float64)

    # Initialize first layer (from atlas7v.for line 2958: XNE(1)=P(1)/TK(1)/2.)
    # Use exact Fortran formula - don't try to "improve" it
    xne[0] = gas_pressure[0] / tk[0] / 2.0

    # Process each layer
    for j in range(n_layers):
        # Initialize XNE for this layer (from atlas7v.for line 2947)
        if j > 0:
            xne[j] = xne[j - 1] * gas_pressure[j] / gas_pressure[j - 1]

        # XNTOT = P/TK (from atlas7v.for line 2948)
        xntot = gas_pressure[j] / tk[j]

        # Initial XNATOM (from atlas7v.for line 2949)
        xnatom[j] = xntot - xne[j]

        # Element list - only the 10 elements that Fortran processes
        # From atlas7v.for lines 2950-2951:
        # DATA NELEMZ/1,2,6,11,12,13,14,19,20,26/  (H, He, C, Na, Mg, Al, Si, K, Ca, Fe)
        # DATA NIONZ/1,2,2,2,2,2,2,2,2,2/,NZ/10/
        nz = 10
        nelemz = np.array([1, 2, 6, 11, 12, 13, 14, 19, 20, 26], dtype=np.int32)
        nionz = np.array([1, 2, 2, 2, 2, 2, 2, 2, 2, 2], dtype=np.int32)

        # MASK array: 1 = include, 0 = exclude (from atlas7v.for line 2951)
        mask = np.ones(nz, dtype=np.int32)

        # Iterate until convergence (from atlas7v.for line 2965: DO 20 L=1,200)
        for iteration in range(max_iterations):
            xnenew = 0.0
            x_contrib = np.zeros(
                nz, dtype=np.float64
            )  # Store X(I) for masking (from Fortran line 2973)

            # Sum contributions from all elements (from atlas7v.for lines 2967-2974)
            for i in range(nz):
                if mask[i] == 0:
                    continue

                iz = nelemz[i]  # Element number (1-99)
                nion = nionz[i]  # Number of ions for this element (from Fortran NIONZ)

                # Call PFSAHA in mode 4 (electrons) - this returns the average electron count
                # From atlas7v.for line 2971: CALL PFSAHA(J,IZ,NION,4,ELEC)
                # Mode 4 computes sum(F[ion-1] * (ion-1)) over all ions
                # CRITICAL: Pass current XNE and XNATOM values to PFSAHA for correct Saha equation
                try:
                    # Try calling with current XNE/XNATOM if the function supports it
                    try:
                        elec = pfsaha_func(
                            j,
                            iz,
                            nion=nion,
                            mode=4,
                            current_xne=xne[j],
                            current_xnatom=xnatom[j],
                        )
                    except TypeError:
                        # Fallback: function doesn't support current_xne parameter
                        elec = pfsaha_func(j, iz, nion=nion, mode=4)
                    # X(I) = ELEC(J) * XNATOM(J) * XABUND(IZ) (from atlas7v.for line 2973)
                    # ELEC(J) is already the average electron count per atom
                    # CRITICAL: Include ALL non-NaN values, even if very small (Fortran doesn't filter)
                    if elec is not None and not np.isnan(elec):
                        x_contrib[i] = elec * xnatom[j] * xabund[iz - 1]
                        xnenew += x_contrib[i]
                    else:
                        x_contrib[i] = 0.0

                        # DEBUG: Log first few iterations for surface layer
                        if j == 0 and iteration < 3:
                            element_names = [
                                "H",
                                "He",
                                "C",
                                "Na",
                                "Mg",
                                "Al",
                                "Si",
                                "K",
                                "Ca",
                                "Fe",
                            ]
                            print(
                                f"DEBUG iteration {iteration}, layer {j}, {element_names[i]} (iz={iz}):"
                            )
                            print(f"  XNE = {xne[j]:.6e}, XNATOM = {xnatom[j]:.6e}")
                            print(f"  ELEC = {elec:.6e}, XABUND = {xabund[iz-1]:.6e}")
                            print(
                                f"  X(I) = ELEC * XNATOM * XABUND = {x_contrib[i]:.6e}"
                            )
                except (IndexError, ValueError, KeyError, RuntimeError):
                    # Element/ion doesn't exist, skip
                    x_contrib[i] = 0.0
                    continue

            # Average with previous value (from atlas7v.for line 2976)
            # CRITICAL: This averaging is what allows Fortran to converge even with bad initial guess
            xnenew_before_avg = xnenew
            xnenew = (xnenew + xne[j]) / 2.0

            # DEBUG: Log iteration progress for surface layer (every 10 iterations or first 5, or when XNE is close to target)
            target_xne = 5.314  # Fortran's expected value for surface layer
            xne_close_to_target = (
                abs(xne[j] - target_xne) < target_xne * 0.5
            )  # Within 50% of target
            should_log = (j == 0 and (iteration < 5 or iteration % 10 == 0)) or (
                j == 0 and xne_close_to_target and iteration < 50
            )

            if should_log:
                print(f"DEBUG iteration {iteration}, layer {j}:")
                print(f"  XNENEW (sum) = {xnenew_before_avg:.6e}")
                print(f"  XNE (current) = {xne[j]:.6e}")
                print(
                    f"  XNENEW (averaged) = ({xnenew_before_avg:.6e} + {xne[j]:.6e}) / 2 = {xnenew:.6e}"
                )
                if xne_close_to_target:
                    print(f"  *** XNE is close to target {target_xne} ***")
                    # Show individual element contributions when close to target
                    print(f"  Element contributions:")
                    element_names = [
                        "H",
                        "He",
                        "C",
                        "Na",
                        "Mg",
                        "Al",
                        "Si",
                        "K",
                        "Ca",
                        "Fe",
                    ]
                    for i in range(nz):
                        if mask[i] != 0:
                            print(f"    {element_names[i]}: X(I) = {x_contrib[i]:.6e}")

            # Check convergence (from atlas7v.for line 2977)
            if xnenew > 0:
                error = abs((xne[j] - xnenew) / xnenew)
            else:
                error = abs(xne[j] - xnenew)

            # DEBUG: Log convergence check (every 10 iterations or first 5)
            if j == 0 and (iteration < 5 or iteration % 10 == 0):
                print(f"  ERROR = {error:.6e} (tolerance = {tolerance:.6e})")
                if iteration % 10 == 0 and iteration > 0:
                    print()

            # Update XNE (from atlas7v.for line 2978)
            xne[j] = xnenew

            # Update XNATOM (from atlas7v.for line 2979)
            xnatom[j] = xntot - xne[j]

            # DEBUG: Check for negative XNATOM (unphysical but Fortran allows it during iteration)
            if xnatom[j] < 0:
                # This can happen with huge initial XNE, but should correct as iteration proceeds
                # Fortran doesn't clamp this - it allows negative values during iteration
                # The averaging will eventually bring XNE down to reasonable values
                if (
                    j == 0 and iteration < 5
                ):  # Only log first few iterations of first layer
                    print(
                        f"DEBUG iteration {iteration}: XNATOM[{j}] = {xnatom[j]:.6e} (negative, will correct)"
                    )

            # Check if converged (from atlas7v.for line 2980)
            if error < tolerance:
                break

            # Mask out elements with very small contributions (from atlas7v.for lines 2981-2986)
            # Only for layers after the first (J > 1)
            if j > 0:
                x1 = 0.00001 * xne[j]  # From atlas7v.for line 2982
                if error < 0.05:  # From atlas7v.for line 2983
                    x1 = x1 * 10.0

                # Mask if contribution is too small (from atlas7v.for lines 2984-2986)
                for i in range(nz):
                    if x_contrib[i] < x1:
                        mask[i] = 0

        # DEBUG DO1516: Print layer values matching Fortran format
        # Format: DEBUG DO1516: J=  1 T=  1.1091E+03 TK=  1.5312E-13 P=  1.8560E-01 XNE=  5.3140E+00
        # Print for layers 1-5 and layer 80 (matching Fortran debug output)
        if j in [0, 1, 2, 3, 4, 79]:  # j is 0-based, Fortran J is 1-based
            j_fortran = j + 1  # Convert to 1-based for Fortran comparison
            print(
                f"DEBUG DO1516: J={j_fortran:3d} T={temperature[j]:11.4E} TK={tk[j]:13.5E} "
                f"P={gas_pressure[j]:11.4E} XNE={xne[j]:11.4E}"
            )
            # DEBUG DO1516 POST: Print XNATOM and TK_FROM_PN
            tk_from_pn = gas_pressure[j] / (xnatom[j] + xne[j]) if (xnatom[j] + xne[j]) > 0 else tk[j]
            print(
                f"DEBUG DO1516 POST: J={j_fortran:3d} TK={tk[j]:13.5E} XNATOM={xnatom[j]:11.4E} "
                f"TK_FROM_PN={tk_from_pn:13.5E}"
            )

    return xne, xnatom
