"""Molecular line opacity accumulation for the Python SYNTHE pipeline.

Corresponds to the molecular-line portion of Fortran synthe.for's XLINOP loop:

    KAPPA0 = CGF * XNFDOP(NELION) * BOLT
    BUFFER(NBUFF) += KAPCEN            [center bin, no STIM in molecular loop]
    BUFFER(NBUFF±k) += KAPPA0 * G(k)  [wing bins, Gaussian profile]
    ASYNTH(J) = TRANSP(J,I) * STIM(J) [applied globally in synthe.for line 94]

Fortran synthe.for applies STIM globally when converting BUFFER → ASYNTH (line 94),
not inside the per-line molecular loop (lines 250–330).  The net result is that each
line's contribution to ASYNTH equals KAPCEN × STIM(bin).

Profile spreading replicates Fortran's N10DOP Gaussian:
    N10DOP = int(10 * DOPPLE * RESOLU)
    PROFILE(k) = KAPPA0 * exp(-(k * DVOIGT)^2)  where DVOIGT = 1/(DOPPLE*RESOLU)

This is a PEAK-NORMALISED Gaussian (value = KAPPA0 at k=0) spread over ±N10DOP bins.
It is implemented as a convolution of the per-NELION accumulated center-bin opacity
with the unnormalised Gaussian kernel G[k] = exp(-k^2 / (DOPPLE*RESOLU)^2).

To use scipy.ndimage.gaussian_filter1d (area-normalised, σ in bins) for speed, the
output must be rescaled by (DOPPLE*RESOLU*sqrt(π)) so the peak matches KAPPA0.

Public API
----------
compute_mol_asynth(
    mol_arrays, mol_xnfdop, atm, wavelength, hckt, stim, ratiolg, ixwlbeg
) -> np.ndarray  shape (n_layers, n_wavelengths)
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)

_C_LIGHT_CMS = 2.99792458e10   # cm/s
_SQRT_PI = math.sqrt(math.pi)

# Molecular masses (amu) keyed by NELION — matches xnfpelsyn.for MOMASS array.
# Used to compute per-species Doppler width (DOPPLE) for profile spreading.
_MOL_MASS: Dict[int, float] = {
    240: 2.016,    # H2
    246: 13.019,   # CH
    252: 15.015,   # NH
    258: 17.003,   # OH
    264: 24.023,   # C2
    270: 26.018,   # CN
    276: 28.011,   # CO
    300: 25.313,   # MgH
    306: 40.305,   # MgO
    312: 29.093,   # SiH
    324: 42.980,   # AlO
    330: 44.085,   # SiO
    342: 41.086,   # CaH
    366: 64.900,   # TiO
    372: 66.940,   # VO
    432: 53.007,   # CrH
    444: 57.005,   # FeH
    492: 24.000,   # NaH
    498: 40.000,   # KH
    534: 18.011,   # H2O
}
_MOL_MASS_DEFAULT = 20.0  # fallback amu if NELION not in table


def _peak_gaussian_convolve(
    layer: np.ndarray,
    dopple: float,
    resolu: float,
) -> np.ndarray:
    """Spread line-center opacity over ±N10DOP bins using Fortran's Gaussian kernel.

    Fortran synthe.for (lines 286-299):
        N10DOP = int(10 * DOPPLE * RESOLU)
        PROFILE(k) = KAPPA0 * exp(-(k / (DOPPLE * RESOLU))^2)

    The kernel is PEAK-NORMALISED (value = 1 at k=0).  scipy.ndimage.gaussian_filter1d
    uses an AREA-NORMALISED Gaussian, so the output is rescaled by
        scale = DOPPLE * RESOLU * sqrt(π)
    to recover the correct peak value.

    Parameters
    ----------
    layer:  1-D array of center-bin accumulated opacity for one depth layer
    dopple: dimensionless Doppler width (v/c) for this species at this depth
    resolu: spectral resolving power R (= 1 / (ratio - 1) for the log-λ grid)

    Returns
    -------
    1-D array with opacity spread according to Fortran's Gaussian profile
    """
    sigma_bins = dopple * resolu  # = N10DOP / 10  (standard deviation of G_Fortran)
    if sigma_bins < 0.05:
        # Width < 0.05 bin → negligible spreading, return as-is
        return layer

    # Convert to scipy sigma (scipy uses exp(-k²/(2σ²))):
    #   G_Fortran(k) = exp(-k²/σ_bins²)  ←→  scipy sigma = σ_bins/√2
    sigma_scipy = sigma_bins / math.sqrt(2.0)

    # Apply scipy's area-normalised Gaussian then rescale to peak-normalised
    # so that for a delta-function input at center, output[center] = input[center].
    scale = sigma_bins * _SQRT_PI  # = σ_bins * √π  (= σ_scipy * √(2π))

    return gaussian_filter1d(layer, sigma=sigma_scipy, mode="constant") * scale


def compute_mol_asynth(
    mol_compiled,
    mol_xnfdop: Dict[int, np.ndarray],
    atm,
    wavelength: np.ndarray,
    hckt: np.ndarray,
    stim: np.ndarray,
    ratiolg: float,
    ixwlbeg: int,
    continuum: Optional[np.ndarray] = None,
    cutoff: float = 1e-4,
) -> np.ndarray:
    """Compute molecular line opacity contribution to ASYNTH.

    Parameters
    ----------
    mol_compiled:   dict with keys nbuff, cgf, nelion, elo_cm, gamma_rad, gamma_stark, gamma_vdw, limb
    mol_xnfdop:     {nelion: xnfdop_array[n_layers]}  from mol_populations.compute_mol_xnfdop()
    atm:            AtmosphereModel
    wavelength:     (n_wl,) wavelength grid in nm
    hckt:           (n_layers,) hc/kT in cm (= 1.4388/T)
    stim:           (n_layers, n_wl) stimulated emission factor 1 - exp(-hν/kT)
                    Applied globally (matching Fortran synthe.for line 94) via
                    ASYNTH(J) = TRANSP(J,I) * STIM(J).
    ratiolg:        log(1 + 1/R) where R is resolving power
    ixwlbeg:        first grid index (computed from wlbeg/ratiolg)
    continuum:      (n_layers, n_wl) continuum opacity array (cont_abs + cont_scat).
                    If provided, enables the Fortran KAPMIN check
                    (Fortran synthe.for lines 255-259):
                        KAPMIN = CONTINUUM(NBUFF) * CUTOFF
                        skip line if CGF*XNFDOP < KAPMIN (pre-BOLT)
                        skip line if CGF*XNFDOP*BOLT < KAPMIN (post-BOLT)
    cutoff:         threshold fraction of continuum below which a line is skipped (default 1e-4)

    Returns
    -------
    asynth_mol: (n_layers, n_wl) molecular contribution to line opacity
    """
    n_layers = atm.layers
    n_wl = len(wavelength)
    asynth_mol = np.zeros((n_layers, n_wl), dtype=np.float64)

    nbuff_arr   = np.asarray(mol_compiled["nbuff"],       dtype=np.int32)
    cgf_arr     = np.asarray(mol_compiled["cgf"],         dtype=np.float64)
    nelion_arr  = np.asarray(mol_compiled["nelion"],      dtype=np.int32)
    elo_arr     = np.asarray(mol_compiled["elo_cm"],      dtype=np.float64)
    gamrf_arr   = np.asarray(mol_compiled["gamma_rad"],   dtype=np.float64)
    gamsf_arr   = np.asarray(mol_compiled["gamma_stark"], dtype=np.float64)
    gamwf_arr   = np.asarray(mol_compiled["gamma_vdw"],  dtype=np.float64)

    n_lines = len(nbuff_arr)
    if n_lines == 0 or not mol_xnfdop:
        return asynth_mol

    logger.info("Computing molecular ASYNTH: %d lines × %d layers × %d wl", n_lines, n_layers, n_wl)

    # Pre-group lines by NELION for batch processing
    unique_nelions = np.unique(nelion_arr)
    nelion_to_mask: Dict[int, np.ndarray] = {}
    for nel in unique_nelions:
        nelion_to_mask[int(nel)] = np.where(nelion_arr == nel)[0]

    # Electron density and TXNXN for broadening
    electron_density = np.maximum(np.asarray(atm.electron_density, dtype=np.float64), 1e-30)
    if atm.txnxn is not None:
        txnxn = np.asarray(atm.txnxn, dtype=np.float64)
    else:
        xnf_h  = np.asarray(atm.xnf_h,  dtype=np.float64) if atm.xnf_h  is not None else np.zeros(n_layers)
        xnf_he = np.asarray(atm.xnf_he1, dtype=np.float64) if atm.xnf_he1 is not None else np.zeros(n_layers)
        xnf_h2 = np.asarray(atm.xnf_h2,  dtype=np.float64) if atm.xnf_h2  is not None else np.zeros(n_layers)
        txnxn  = (xnf_h + 0.42 * xnf_he + 0.85 * xnf_h2) * (atm.temperature / 1e4) ** 0.3

    temperature = np.asarray(atm.temperature, dtype=np.float64)

    # Turbulent velocity: atm.turbulent_velocity is in cm/s (ATLAS12 convention)
    # Convert to dimensionless v/c for Doppler-width calculation
    vturb_cms = np.zeros(n_layers, dtype=np.float64)
    if atm.turbulent_velocity is not None:
        vturb_cms = np.asarray(atm.turbulent_velocity, dtype=np.float64)

    # Resolving power R from the log-λ grid: ratio = exp(ratiolg), R = 1/(ratio-1)
    resolu = 300000.0  # default fallback
    if n_wl > 1 and ratiolg > 0.0:
        resolu = 1.0 / (math.exp(ratiolg) - 1.0)

    # Physical constants for Doppler width
    _KBOLTZ = 1.380649e-16   # erg/K
    _AMU    = 1.66053906660e-24  # g

    n_processed = 0
    for nel_int, line_mask in nelion_to_mask.items():
        if nel_int not in mol_xnfdop:
            continue

        xnfdop = mol_xnfdop[nel_int]    # (n_layers,)
        nbuffs  = nbuff_arr[line_mask]
        cgfs    = cgf_arr[line_mask]
        elos    = elo_arr[line_mask]
        gamrfs  = gamrf_arr[line_mask]
        gamsfs  = gamsf_arr[line_mask]
        gamwfs  = gamwf_arr[line_mask]

        # Filter to wavelength grid: 1 <= nbuff <= n_wl
        valid = (nbuffs >= 1) & (nbuffs <= n_wl)
        if not np.any(valid):
            continue
        nbuffs = nbuffs[valid]
        cgfs   = cgfs[valid]
        elos   = elos[valid]
        gamrfs = gamrfs[valid]
        gamsfs = gamsfs[valid]
        gamwfs = gamwfs[valid]

        wl_indices = nbuffs - 1  # 0-based

        use_kapmin = (continuum is not None) and (cutoff > 0.0)

        # Molecular mass for this NELION (matches Fortran MOMASS array)
        mol_mass_amu = _MOL_MASS.get(nel_int, _MOL_MASS_DEFAULT)

        # Per-NELION accumulation buffer: accumulate center-bin opacity for all
        # layers, then convolve each layer with the Gaussian profile.
        nelion_buf = np.zeros((n_layers, n_wl), dtype=np.float64)

        for depth_idx in range(n_layers):
            hckt_val   = hckt[depth_idx]
            xnfdop_val = xnfdop[depth_idx]
            if xnfdop_val <= 0.0:
                continue

            # KAPPA0 = CGF * XNFDOP  (Fortran synthe.for line 251)
            kappa0_pre = cgfs * xnfdop_val

            # KAPMIN check 1: before Boltzmann factor
            if use_kapmin:
                cont_at_lines = continuum[depth_idx, wl_indices]
                kapmin = cont_at_lines * cutoff
                pre_mask = kappa0_pre >= kapmin
                if not np.any(pre_mask):
                    continue
            else:
                pre_mask = np.ones(len(cgfs), dtype=bool)

            bolt = np.exp(-elos * hckt_val)
            kappa0 = kappa0_pre * bolt

            # KAPMIN check 2: after Boltzmann factor
            if use_kapmin:
                post_mask = pre_mask & (kappa0 >= kapmin)
            else:
                post_mask = pre_mask

            if not np.any(post_mask):
                continue

            # Damping parameter: ADAMP = (GAMMAR + GAMMAS*XNE + GAMMAW*TXNXN) / DOPPLE
            # Doppler width for this species at this depth (Fortran xnfpelsyn.for line 354):
            #   DOPPLE = sqrt(2*kT/m + vturb^2) / c
            T_j       = temperature[depth_idx]
            vt_cms    = vturb_cms[depth_idx]
            dopple    = math.sqrt(
                max(2.0 * _KBOLTZ * T_j / (mol_mass_amu * _AMU) + vt_cms * vt_cms, 0.0)
            ) / _C_LIGHT_CMS

            xne_val   = electron_density[depth_idx]
            txnxn_val = txnxn[depth_idx]

            adamp = (gamrfs + gamsfs * xne_val + gamwfs * txnxn_val)
            if dopple > 0:
                adamp = adamp / dopple
            else:
                adamp[:] = 0.0

            # Voigt profile at line center: H(0, a) ≈ 1 - 1.128*a for small a
            a_clamp = np.minimum(adamp, 0.5)
            h_center = np.where(
                a_clamp < 0.2,
                1.0 - 1.128 * a_clamp,
                np.exp(-a_clamp**2),
            )

            kapcen = kappa0 * h_center  # Peak opacity (no STIM yet)

            # STIM: applied here to match Fortran's global ASYNTH = TRANSP * STIM
            # (synthe.for line 94).  For wing bins, stim varies negligibly over
            # ±N10DOP ≈ ±0.05 nm, so using center-bin stim is an excellent approx.
            stim_vals = stim[depth_idx, wl_indices]

            contrib = kapcen * stim_vals  # (n_lines,)
            np.add.at(nelion_buf[depth_idx], wl_indices[post_mask], contrib[post_mask])

        # --- Gaussian profile spreading (Fortran N10DOP convolution) ---
        # Apply per-layer, because DOPPLE depends on T(layer).
        for depth_idx in range(n_layers):
            if not np.any(nelion_buf[depth_idx] > 0.0):
                continue

            T_j    = temperature[depth_idx]
            vt_cms = vturb_cms[depth_idx]
            dopple = math.sqrt(
                max(2.0 * _KBOLTZ * T_j / (mol_mass_amu * _AMU) + vt_cms * vt_cms, 0.0)
            ) / _C_LIGHT_CMS

            nelion_buf[depth_idx] = _peak_gaussian_convolve(
                nelion_buf[depth_idx], dopple, resolu
            )

        asynth_mol += nelion_buf
        n_processed += len(cgfs)

    logger.info("Molecular opacity: accumulated %d line-weight contributions", n_processed)
    return asynth_mol
