"""
Computation of departure coefficient tables (BHYD, BC1, BC2, BSI1, BSI2).

These are statistical equilibrium factors used in NLTE calculations.
For LTE (the default case), all values should be 1.0.

This module provides functions to compute these values from atmosphere
properties, ensuring consistency across different atmospheric models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple
import numpy as np

if TYPE_CHECKING:
    from ..io.atmosphere import AtmosphereModel


def compute_bhyd(
    atmosphere: "AtmosphereModel",
    nlte: bool = False,
) -> np.ndarray:
    """
    Compute BHYD (hydrogen departure coefficients).
    
    Parameters
    ----------
    atmosphere:
        The atmosphere model
    nlte:
        If True, compute NLTE departure coefficients (not yet implemented).
        If False (default), return LTE values (all 1.0).
    
    Returns
    -------
    bhyd:
        Array of shape (layers, 8) containing hydrogen departure coefficients.
        For LTE, all values are 1.0.
    """
    n_layers = atmosphere.layers
    
    if not nlte:
        # LTE case: all departure coefficients are 1.0
        # This matches Fortran behavior in atlas7v.for line 1910:
        # DO 1201 I=1,6; BHYD(J,I)=1.
        return np.ones((n_layers, 8), dtype=np.float64)
    
    # TODO: Implement NLTE computation when needed
    # For now, NLTE also returns 1.0
    return np.ones((n_layers, 8), dtype=np.float64)


def compute_bc1(
    atmosphere: "AtmosphereModel",
    nlte: bool = False,
) -> np.ndarray:
    """
    Compute BC1 (carbon level 1 departure coefficients).
    
    Parameters
    ----------
    atmosphere:
        The atmosphere model
    nlte:
        If True, compute NLTE departure coefficients (not yet implemented).
        If False (default), return LTE values (all 1.0).
    
    Returns
    -------
    bc1:
        Array of shape (layers, 14) containing carbon level 1 departure coefficients.
        For LTE, all values are 1.0.
    """
    n_layers = atmosphere.layers
    
    if not nlte:
        # LTE case: all departure coefficients are 1.0
        return np.ones((n_layers, 14), dtype=np.float64)
    
    # TODO: Implement NLTE computation when needed
    return np.ones((n_layers, 14), dtype=np.float64)


def compute_bc2(
    atmosphere: "AtmosphereModel",
    nlte: bool = False,
) -> np.ndarray:
    """
    Compute BC2 (carbon level 2 departure coefficients).
    
    Parameters
    ----------
    atmosphere:
        The atmosphere model
    nlte:
        If True, compute NLTE departure coefficients (not yet implemented).
        If False (default), return LTE values (all 1.0).
    
    Returns
    -------
    bc2:
        Array of shape (layers, 6) containing carbon level 2 departure coefficients.
        For LTE, all values are 1.0.
    """
    n_layers = atmosphere.layers
    
    if not nlte:
        # LTE case: all departure coefficients are 1.0
        return np.ones((n_layers, 6), dtype=np.float64)
    
    # TODO: Implement NLTE computation when needed
    return np.ones((n_layers, 6), dtype=np.float64)


def compute_bsi1(
    atmosphere: "AtmosphereModel",
    nlte: bool = False,
) -> np.ndarray:
    """
    Compute BSI1 (silicon level 1 departure coefficients).
    
    Parameters
    ----------
    atmosphere:
        The atmosphere model
    nlte:
        If True, compute NLTE departure coefficients (not yet implemented).
        If False (default), return LTE values (all 1.0).
    
    Returns
    -------
    bsi1:
        Array of shape (layers, 11) containing silicon level 1 departure coefficients.
        For LTE, all values are 1.0.
    """
    n_layers = atmosphere.layers
    
    if not nlte:
        # LTE case: all departure coefficients are 1.0
        return np.ones((n_layers, 11), dtype=np.float64)
    
    # TODO: Implement NLTE computation when needed
    return np.ones((n_layers, 11), dtype=np.float64)


def compute_bsi2(
    atmosphere: "AtmosphereModel",
    nlte: bool = False,
) -> np.ndarray:
    """
    Compute BSI2 (silicon level 2 departure coefficients).
    
    Parameters
    ----------
    atmosphere:
        The atmosphere model
    nlte:
        If True, compute NLTE departure coefficients (not yet implemented).
        If False (default), return LTE values (all 1.0).
    
    Returns
    -------
    bsi2:
        Array of shape (layers, 10) containing silicon level 2 departure coefficients.
        For LTE, all values are 1.0.
    """
    n_layers = atmosphere.layers
    
    if not nlte:
        # LTE case: all departure coefficients are 1.0
        return np.ones((n_layers, 10), dtype=np.float64)
    
    # TODO: Implement NLTE computation when needed
    return np.ones((n_layers, 10), dtype=np.float64)


def compute_all_b_tables(
    atmosphere: "AtmosphereModel",
    nlte: bool = False,
    verify_against: "AtmosphereModel | None" = None,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute all B tables (BHYD, BC1, BC2, BSI1, BSI2) from atmosphere properties.
    
    Parameters
    ----------
    atmosphere:
        The atmosphere model to compute B tables for
    nlte:
        If True, compute NLTE departure coefficients (not yet implemented).
        If False (default), return LTE values (all 1.0).
    verify_against:
        Optional atmosphere model with pre-computed B tables to verify against.
        If provided, will log differences if any are found.
    rtol, atol:
        Relative and absolute tolerances for verification.
    
    Returns
    -------
    bhyd, bc1, bc2, bsi1, bsi2:
        All B tables as numpy arrays with appropriate shapes.
    """
    import logging
    
    logger = logging.getLogger(__name__)
    
    bhyd = compute_bhyd(atmosphere, nlte=nlte)
    bc1 = compute_bc1(atmosphere, nlte=nlte)
    bc2 = compute_bc2(atmosphere, nlte=nlte)
    bsi1 = compute_bsi1(atmosphere, nlte=nlte)
    bsi2 = compute_bsi2(atmosphere, nlte=nlte)
    
    # Verification against loaded values if provided
    if verify_against is not None:
        if verify_against.bhyd is not None:
            if not np.allclose(bhyd, verify_against.bhyd, rtol=rtol, atol=atol):
                max_diff = np.max(np.abs(bhyd - verify_against.bhyd))
                logger.warning(
                    f"BHYD computed vs loaded differ: max_diff={max_diff:.2e}, "
                    f"rtol={rtol}, atol={atol}"
                )
            else:
                logger.debug("BHYD computed values match loaded values")
        
        if verify_against.bc1 is not None:
            if not np.allclose(bc1, verify_against.bc1, rtol=rtol, atol=atol):
                max_diff = np.max(np.abs(bc1 - verify_against.bc1))
                logger.warning(
                    f"BC1 computed vs loaded differ: max_diff={max_diff:.2e}, "
                    f"rtol={rtol}, atol={atol}"
                )
            else:
                logger.debug("BC1 computed values match loaded values")
        
        if verify_against.bc2 is not None:
            if not np.allclose(bc2, verify_against.bc2, rtol=rtol, atol=atol):
                max_diff = np.max(np.abs(bc2 - verify_against.bc2))
                logger.warning(
                    f"BC2 computed vs loaded differ: max_diff={max_diff:.2e}, "
                    f"rtol={rtol}, atol={atol}"
                )
            else:
                logger.debug("BC2 computed values match loaded values")
        
        if verify_against.bsi1 is not None:
            if not np.allclose(bsi1, verify_against.bsi1, rtol=rtol, atol=atol):
                max_diff = np.max(np.abs(bsi1 - verify_against.bsi1))
                logger.warning(
                    f"BSI1 computed vs loaded differ: max_diff={max_diff:.2e}, "
                    f"rtol={rtol}, atol={atol}"
                )
            else:
                logger.debug("BSI1 computed values match loaded values")
        
        if verify_against.bsi2 is not None:
            if not np.allclose(bsi2, verify_against.bsi2, rtol=rtol, atol=atol):
                max_diff = np.max(np.abs(bsi2 - verify_against.bsi2))
                logger.warning(
                    f"BSI2 computed vs loaded differ: max_diff={max_diff:.2e}, "
                    f"rtol={rtol}, atol={atol}"
                )
            else:
                logger.debug("BSI2 computed values match loaded values")
    
    return bhyd, bc1, bc2, bsi1, bsi2

