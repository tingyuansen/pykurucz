#!/usr/bin/env python3
"""
Gibbs free-energy minimization for molecular equilibrium.

This solver uses the method of Lagrange multipliers (element potentials) which is
the standard approach used by NASA-CEA, Cantera, and other thermodynamic codes.

The algorithm:
1. Define Gibbs free energy: G = Σ n_i * (μ°_i/kT + ln(n_i/n_tot))
2. Element conservation: Σ a_ij * n_i = b_j for each element j
3. At equilibrium with Lagrange multipliers λ_j:
   μ°_i/kT + ln(n_i/n_tot) = Σ a_ij * λ_j
4. Solve: n_i = n_tot * exp(Σ a_ij * λ_j - μ°_i/kT)
5. Iterate to satisfy element conservation

CRITICAL: All variables are scaled to avoid numerical issues:
- Concentrations scaled by total nuclei density
- Element potentials scaled by max|μ°|
"""

from __future__ import annotations

import warnings
import numpy as np
from typing import Tuple, Optional

# Physical constants
k_BOLTZ = 1.380649e-16  # erg/K


def _gibbs_objective(n: np.ndarray, mu0: np.ndarray) -> float:
    """Gibbs free energy G/kT = Σ n_i * (μ°_i/kT + ln(n_i/n_tot))"""
    ntot = np.sum(n) + 1e-300
    # Avoid log(0) by clipping
    n_safe = np.maximum(n, 1e-300)
    return float(np.sum(n * (mu0 + np.log(n_safe / ntot))))


def _compute_n_from_lambda(
    lam: np.ndarray,
    mu0: np.ndarray,
    stoich: np.ndarray,
    n_scale: float,
) -> np.ndarray:
    """
    Compute species concentrations from element potentials (Lagrange multipliers).
    
    At equilibrium: n_i = n_scale * exp(Σ_j a_ij λ_j - μ°_i)
    """
    # Compute exponent: Σ_j a_ij λ_j - μ°_i
    exponent = stoich @ lam - mu0
    
    # Clip exponent to avoid overflow/underflow
    exponent = np.clip(exponent, -700, 700)
    
    n = n_scale * np.exp(exponent)
    return n


def _element_residual(
    lam: np.ndarray,
    mu0: np.ndarray,
    stoich: np.ndarray,
    elem_totals: np.ndarray,
    n_scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute element conservation residual and Jacobian.
    
    Residual: r_j = Σ_i a_ij n_i - b_j
    Jacobian: J_jk = ∂r_j/∂λ_k = Σ_i a_ij a_ik n_i
    """
    n = _compute_n_from_lambda(lam, mu0, stoich, n_scale)
    
    # Residual: stoich.T @ n - elem_totals
    computed = stoich.T @ n
    residual = computed - elem_totals
    
    # Jacobian: J_jk = Σ_i a_ij a_ik n_i = (A^T diag(n) A)_jk
    # More efficiently: J = stoich.T @ (n[:, None] * stoich)
    n_diag = n[:, np.newaxis] * stoich  # (n_species, n_elements)
    jacobian = stoich.T @ n_diag  # (n_elements, n_elements)
    
    return residual, jacobian


def minimize_gibbs(
    temperature: float,
    pressure: float,
    mu0: np.ndarray,
    stoich: np.ndarray,
    elem_totals: np.ndarray,
    charges: np.ndarray | None = None,
    charge_total: float = 0.0,
    max_iter: int = 100,
    tol: float = 1e-8,
    logn_min: float = -300.0,
    logn_max: float = 300.0,
) -> np.ndarray:
    """
    Minimize Gibbs free energy using the Lagrange multiplier (element potential) method.
    
    This is the standard approach used by thermodynamic equilibrium codes.
    It's more robust than direct constrained optimization because:
    1. The problem is reformulated as unconstrained root-finding
    2. Newton iteration with analytical Jacobian converges quickly
    3. Natural handling of very different concentration scales
    
    Parameters
    ----------
    temperature : float
        Temperature in K
    pressure : float
        Pressure in dyn/cm²
    mu0 : np.ndarray
        Reduced chemical potentials μ°/kT for each species
    stoich : np.ndarray
        Stoichiometry matrix (n_species, n_elements)
    elem_totals : np.ndarray
        Total number density for each element (conservation constraints)
    charges : np.ndarray | None
        Not used in current implementation (charge neutrality handled separately)
    charge_total : float
        Not used in current implementation
    max_iter : int
        Maximum Newton iterations
    tol : float
        Convergence tolerance (relative residual)
        
    Returns
    -------
    n : np.ndarray
        Equilibrium number densities for each species
    """
    stoich = np.asarray(stoich, dtype=np.float64)
    mu0 = np.asarray(mu0, dtype=np.float64)
    elem_totals = np.asarray(elem_totals, dtype=np.float64)
    
    n_species, n_elements = stoich.shape
    
    # Validate inputs
    if mu0.shape[0] != n_species:
        raise ValueError(f"mu0 length {mu0.shape[0]} != n_species {n_species}")
    if elem_totals.shape[0] != n_elements:
        raise ValueError(f"elem_totals length {elem_totals.shape[0]} != n_elements {n_elements}")
    
    # Handle edge cases
    if n_species == 0 or n_elements == 0:
        return np.zeros(n_species)
    
    # Scale factor for concentrations (use total nuclei)
    n_scale = np.sum(elem_totals)
    if n_scale <= 0 or not np.isfinite(n_scale):
        n_scale = 1.0
    
    # Scale elem_totals for numerical stability
    elem_scaled = elem_totals / n_scale
    
    # Initial guess for λ (element potentials)
    # Start with λ = 0, which gives n_i = n_scale * exp(-μ°_i)
    lam = np.zeros(n_elements, dtype=np.float64)
    
    # Newton iteration to find λ that satisfies element conservation
    for iteration in range(max_iter):
        # Compute residual and Jacobian (using scaled concentrations)
        n_current = _compute_n_from_lambda(lam, mu0, stoich, 1.0)  # n_scale=1 for scaled
        
        computed = stoich.T @ n_current
        residual = computed - elem_scaled
        
        # Check convergence
        rel_residual = np.abs(residual) / (elem_scaled + 1e-300)
        max_rel_error = np.max(rel_residual)
        
        if max_rel_error < tol:
            # Converged! Return unscaled concentrations
            return n_current * n_scale
        
        # Compute Jacobian for Newton step
        n_diag = n_current[:, np.newaxis] * stoich
        jacobian = stoich.T @ n_diag
        
        # Regularize Jacobian if near-singular
        diag_min = 1e-20 * np.max(np.abs(np.diag(jacobian)))
        jacobian += np.eye(n_elements) * diag_min
        
        # Newton step: J * Δλ = -residual
        try:
            delta_lam = np.linalg.solve(jacobian, -residual)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            delta_lam = np.linalg.lstsq(jacobian, -residual, rcond=None)[0]
        
        # Line search to ensure we don't overshoot
        alpha = 1.0
        for ls_iter in range(20):
            lam_new = lam + alpha * delta_lam
            n_new = _compute_n_from_lambda(lam_new, mu0, stoich, 1.0)
            computed_new = stoich.T @ n_new
            residual_new = computed_new - elem_scaled
            
            # Accept if residual decreased
            if np.linalg.norm(residual_new) < np.linalg.norm(residual) * 1.1:
                lam = lam_new
                break
            alpha *= 0.5
        else:
            # Line search failed, take small step anyway
            lam = lam + 0.1 * delta_lam
    
    # Max iterations reached - return best result
    n_result = _compute_n_from_lambda(lam, mu0, stoich, n_scale)
    
    # Warn if not converged
    computed_final = stoich.T @ n_result
    rel_error_final = np.max(np.abs(computed_final - elem_totals) / (elem_totals + 1e-300))
    if rel_error_final > 0.01:  # > 1% error
        warnings.warn(
            f"Gibbs minimization did not converge: max relative error = {rel_error_final:.2e}"
        )
    
    return n_result


def minimize_gibbs_simple(
    mu0: np.ndarray,
    stoich: np.ndarray,
    elem_totals: np.ndarray,
) -> np.ndarray:
    """
    Simplified Gibbs minimization for quick testing.
    
    Uses the same Lagrange multiplier method but with default parameters.
    """
    return minimize_gibbs(
        temperature=5000.0,  # Dummy
        pressure=1e6,  # Dummy
        mu0=mu0,
        stoich=stoich,
        elem_totals=elem_totals,
    )


__all__ = ["minimize_gibbs", "minimize_gibbs_simple"]
