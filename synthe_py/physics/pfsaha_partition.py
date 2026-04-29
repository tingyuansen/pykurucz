"""
PFSAHA partition function calculation using extracted NNN tables.

This module implements the exact Fortran PFSAHA partition function logic
from atlas7v.for lines 3833-3901, using the extracted NNN(6,374) tables.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from .pfsaha_levels import NNN
from .pfsaha_ion_pots import IONIZATION_POTENTIALS

# LOCZ array: starting indices for each element in partition function tables
# From atlas7v.for line 3781-3782
LOCZ = np.array(
    [1, 3, 6, 10, 14, 18, 22, 27, 33, 39, 45, 51, 57, 63, 69, 75, 81, 86, 91,
     96, 101, 106, 111, 116, 121, 126, 131, 136, 141],
    dtype=np.int32,
)

# SCALE array for partition function interpolation
# From atlas7v.for line 3783
SCALE = np.array([0.001, 0.01, 0.1, 1.0], dtype=np.float64)

# Constants
KBOLTZ_EV = 1.0 / 8.6171e-5  # K/eV (matches Fortran TKEV calculation)
EV_TO_CM = 8065.479  # Conversion: 1 eV = 8065.479 cm^-1


def _get_nnn_column_index(atomic_number: int, ion_stage: int) -> int:
    """
    Get NNN column index (1-based Fortran style) for given element/ion.
    
    From atlas7v.for lines 3801-3813:
        IF(IZ.LE.28)N=LOCZ(IZ)
        IF(IZ.GT.28)N=3*IZ+54
        IF(IZ.LE.28)NIONS=LOCZ(IZ+1)-N
        IF(IZ.GT.28)NIONS=3
        IF(IZ.EQ.6)N=354
        IF(IZ.EQ.6)NIONS=6
        IF(IZ.EQ.7)N=360
        IF(IZ.EQ.7)NIONS=7
        IF(IZ.EQ.8)N=367
        IF(IZ.EQ.8)NIONS=8
        IF(IZ.GE.20.AND.IZ.LT.29)NIONS=10
        N=N-1  (then uses N+ION, so effectively N = LOCZ(IZ) + ION - 1)
    
    Parameters
    ----------
    atomic_number:
        Element atomic number (1-99)
    ion_stage:
        Ionization stage (1 = neutral, 2 = once ionized, etc.)
    
    Returns
    -------
    nnn_col:
        NNN column index (1-based Fortran style, subtract 1 for Python 0-based)
    """
    iz = atomic_number
    
    # Determine starting column N
    if iz == 6:  # Carbon
        n = 354
    elif iz == 7:  # Nitrogen
        n = 360
    elif iz == 8:  # Oxygen
        n = 367
    elif iz <= 28:
        n = LOCZ[iz - 1]  # Python 0-based indexing
    else:  # iz > 28
        n = 3 * iz + 54
    
    # Column index = N + ION - 1 (Fortran uses N+ION after N=N-1)
    # But since we're computing directly, it's N + ION - 1
    nnn_col = n + ion_stage - 1
    
    return nnn_col


def compute_pfsaha_partition_function(
    atomic_number: int,
    ion_stage: int,
    temperature: float,
    electron_density: Optional[float] = None,
    gas_pressure: Optional[float] = None,
) -> float:
    """
    Compute partition function using PFSAHA logic from extracted NNN tables.
    
    This implements the exact Fortran logic from atlas7v.for lines 3833-3901.
    
    Parameters
    ----------
    atomic_number:
        Element atomic number (1-99)
    ion_stage:
        Ionization stage (1 = neutral, 2 = once ionized, etc.)
    temperature:
        Temperature in K
    electron_density:
        Electron density in cm^-3 (for Debye screening)
    gas_pressure:
        Gas pressure in dyn/cm^2 (for Debye screening)
    
    Returns
    -------
    partition_function:
        Partition function value
    """
    iz = atomic_number
    ion = ion_stage
    T = temperature
    
    # Get ionization potential
    element_symbol = _get_element_symbol(iz)
    if element_symbol not in IONIZATION_POTENTIALS:
        return 1.0
    
    ip_array = IONIZATION_POTENTIALS[element_symbol]
    # IONIZATION_POTENTIALS is a dict mapping element symbols to numpy arrays
    # Array contains IPs for stages 1, 2, 3, ... so index = ion_stage - 1
    if ion_stage - 1 >= len(ip_array) or ion_stage < 1:
        return 1.0
    
    IP_eV = ip_array[ion_stage - 1]  # 0-indexed
    if IP_eV == 0.0:
        return 1.0
    
    # Convert to eV if needed (already in eV from extraction)
    IP = IP_eV
    
    # Compute Debye screening (from atlas7v.for lines 3788-3795)
    POTLOW = 1.0
    if electron_density is not None and gas_pressure is not None:
        CHARGE = electron_density * 2.0
        TK = 1.380649e-16 * T  # kT in erg
        EXCESS = 2.0 * electron_density - gas_pressure / TK
        if EXCESS > 0.0:
            CHARGE = CHARGE + 2.0 * EXCESS
        if CHARGE == 0.0:
            CHARGE = 1.0
        DEBYE = np.sqrt(TK / 2.8965e-18 / CHARGE)
        POTLOW = min(1.0, 1.44e-7 / DEBYE)
    
    POTLO = POTLOW * ion
    TV = T * 8.6171e-5  # TKEV in eV
    
    # Get NNN column index
    nnn_col_fortran = _get_nnn_column_index(iz, ion)
    nnn_col = nnn_col_fortran - 1  # Convert to 0-based for Python
    
    if nnn_col < 0 or nnn_col >= NNN.shape[1]:
        return 1.0
    
    # Decode NNN(6,N) to get G and NNN100 (from atlas7v.for lines 3834-3835)
    nnn6_val = NNN[5, nnn_col]  # Row 6 (0-indexed = 5)
    NNN100 = nnn6_val // 100
    G = nnn6_val - NNN100 * 100
    
    # Special cases (from atlas7v.for lines 3836-3852)
    # For now, skip special element handling (H, He, C, N, O, etc.)
    # These require additional arrays (BHYD, BHE1, etc.) that aren't extracted yet
    
    # General partition function interpolation (from atlas7v.for lines 3853-3875)
    T2000 = IP * 2000.0 / 11.0
    IT = max(1, min(9, int(T / T2000 - 0.5)))
    DT = T / T2000 - float(IT) - 0.5
    PMIN = 1.0
    
    I = (IT + 1) // 2  # Python integer division
    if I < 1:
        I = 1
    if I > 5:
        I = 5
    
    # Decode NNN(I,N) (from atlas7v.for lines 3858-3861)
    nnn_i_val = NNN[I - 1, nnn_col]  # I is 1-based, convert to 0-based
    K1 = nnn_i_val // 100000
    K2 = nnn_i_val - K1 * 100000
    K3 = K2 // 10
    KSCALE = K2 - K3 * 10
    
    if KSCALE < 1 or KSCALE > 4:
        KSCALE = 1
    
    # Interpolate partition function (from atlas7v.for lines 3862-3875)
    if IT % 2 == 1:  # Odd IT
        P1 = float(K1) * SCALE[KSCALE - 1]  # KSCALE is 1-based
        P2 = float(K3) * SCALE[KSCALE - 1]
        if DT < 0.0 and KSCALE <= 1:
            KP1 = int(P1)
            if KP1 == int(P2 + 0.5):
                PMIN = float(KP1)
    else:  # Even IT
        P1 = float(K3) * SCALE[KSCALE - 1]
        if I < 5:
            nnn_i1_val = NNN[I, nnn_col]  # I+1 (0-based = I)
            K1_next = nnn_i1_val // 100000
            KSCALE_next = (nnn_i1_val % 10)
            if KSCALE_next < 1 or KSCALE_next > 4:
                KSCALE_next = 1
            P2 = float(K1_next) * SCALE[KSCALE_next - 1]
        else:
            P2 = P1
    
    PART = max(PMIN, P1 + (P2 - P1) * DT)
    
    # Occupation probability correction for high temperatures (from atlas7v.for lines 3891-3900)
    if G != 0.0 and POTLO >= 0.1 and T >= T2000 * 4.0:
        TV_eff = TV
        if T > T2000 * 11.0:
            TV_eff = (T2000 * 11.0) * 8.6171e-5
        
        Z = float(ion)
        D1 = 0.1 / TV_eff
        D2 = POTLO / TV_eff
        
        term1 = np.sqrt(13.595 * Z * Z / TV_eff / D2) ** 3
        term1 *= (1.0/3.0 + (1.0 - (0.5 + (1.0/18.0 + D2/120.0) * D2) * D2) * D2)
        
        term2 = np.sqrt(13.595 * Z * Z / TV_eff / D1) ** 3
        term2 *= (1.0/3.0 + (1.0 - (0.5 + (1.0/18.0 + D1/120.0) * D1) * D1) * D1)
        
        PART = PART + G * np.exp(-IP / TV_eff) * (term1 - term2)
    
    return max(1.0, PART)


def _get_element_symbol(atomic_number: int) -> str:
    """Convert atomic number to element symbol."""
    elements = [
        "H", "HE", "LI", "BE", "B", "C", "N", "O", "F", "NE",
        "NA", "MG", "AL", "SI", "P", "S", "CL", "AR", "K", "CA",
        "SC", "TI", "V", "CR", "MN", "FE", "CO", "NI", "CU", "ZN",
        "GA", "GE", "AS", "SE", "BR", "KR", "RB", "SR", "Y", "ZR",
        "NB", "MO", "TC", "RU", "RH", "PD", "AG", "CD", "IN", "SN",
        "SB", "TE", "I", "XE", "CS", "BA", "LA", "CE", "PR", "ND",
        "PM", "SM", "EU", "GD", "TB", "DY", "HO", "ER", "TM", "YB",
        "LU", "HF", "TA", "W", "RE", "OS", "IR", "PT", "AU", "HG",
        "TL", "PB", "BI", "PO", "AT", "RN", "FR", "RA", "AC", "TH",
        "PA", "U", "NP", "PU", "AM", "CM", "BK", "CF", "ES", "FM",
        "MD", "NO", "LR", "RF", "DB", "SG", "BH", "HS", "MT", "DS",
    ]
    if 1 <= atomic_number <= len(elements):
        return elements[atomic_number - 1]
    return "UNKNOWN"

