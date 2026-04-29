"""Molecular number-density computation for molecular line opacity.

Calls nmolec_exact (already used in convert_atm_to_npz.py) on the
AtmosphereModel data at synthesis time to obtain molecular number densities
XNMOL[depth, imol].  Those are then converted into the Fortran XNFDOP
quantity (density / (rho * doppler_width)) that drives line opacity.

Public API
----------
compute_mol_xnfdop(atm, nelion_set, molecules_path=None)
    -> dict {nelion: np.ndarray shape (n_layers,)}

Each entry is: xnfdop[depth] = n_mol[depth] / (rho[depth] * doppler_width_mol[depth])
which directly replaces XNFDOP(NELION) in the Fortran SYNTHE inner loop.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Boltzmann constant and AMU in cgs
_KBOLTZ = 1.380649e-16  # erg/K
_AMU = 1.66053906660e-24  # g
_C_LIGHT_CMS = 2.99792458e10  # cm/s

# ---------------------------------------------------------------------------
# Internal molecular partition functions
# ---------------------------------------------------------------------------
# XNFPMOL is computed by _compute_xnfpmol (from pops_exact.py), which directly
# implements the Fortran atlas7v.for Path-1 formula (lines 4325-4368):
#   XNFPMOL(J,JMOL) = exp(D_0/kT) × ∏[n_atom / (Z_atom × C_atom)] × C_mol
# This matches XNMOL / Z_int_mol using only atomic partition functions (PFSAHA)
# and dissociation energies from molecules.dat — no external spectroscopic data.
# XNFDOP(NELION) = XNFPMOL / (RHO × DOPPLE)   [Fortran xnfpelsyn.for]
# KAPPA0 = CGF × XNFDOP × BOLT                  [Fortran synthe.for]
# ---------------------------------------------------------------------------

# Molecular masses (amu) keyed by NELION for Doppler-width calculation.
# Values taken directly from Fortran xnfpelsyn.for MOMASS DATA statement.
# NELION = NELEM*6, IDMOL index = NELEM-39, MOMASS index = NELEM-39.
_NELION_TO_MASS: Dict[int, float] = {
    240:   2.0,   # H2
    246:  13.0,   # CH
    252:  15.0,   # NH
    258:  17.0,   # OH
    264:  24.0,   # C2
    270:  26.0,   # CN
    276:  28.0,   # CO
    282:  28.0,   # N2
    288:  30.0,   # NO
    294:  32.0,   # O2
    300:  25.0,   # MgH
    306:  28.0,   # AlH  (Fortran IDMOL(12)=113, MOMASS(12)=28)
    312:  29.0,   # SiH
    318:  40.0,   # MgO  (Fortran IDMOL(14)=812, MOMASS(14)=40)
    324:  43.0,   # AlO
    330:  44.0,   # SiO
    336:  33.0,   # SH
    342:  41.0,   # CaH
    348:  48.0,   # SO
    354:  56.0,   # CaO
    360:  61.0,   # ScO
    366:  64.0,   # TiO
    372:  67.0,   # VO
    378:   8.0,   # HeH
    384:  10.0,   # LiH
    390:  12.0,   # BeH
    396:  20.0,   # FH
    402:  32.0,   # PH
    408:  36.0,   # ClH
    414:  46.0,   # ScH
    420:  49.0,   # TiH
    426:  52.0,   # VH
    432:  53.0,   # CrH
    438:  56.0,   # MnH
    444:  57.0,   # FeH
    450:  13.0,   # 13CH (isotopologue)
    456:  15.0,   # 15NH (isotopologue)
    462:  17.0,   # 18OH (isotopologue)
    468:  25.0,   # MgH (isotopologue)
    474:  28.0,   # AlH (isotopologue)
    480:  29.0,   # SiH (isotopologue)
    486:  41.0,   # CaH (isotopologue)
    492:  24.0,   # NaH
    498:  40.0,   # KH
    504:   3.0,   # H3+
    510:  51.0,   # ClO
    516:  68.0,   # CrO
    522:  71.0,   # MnO
    528:  72.0,   # FeO
    534:  18.0,   # H2O
    540:  44.0,   # CO2
    546:  14.0,   # HCN
    552:  36.0,   # C3
    558:  60.0,   # CoH
    564:  59.0,   # NiH
    570:  64.0,   # CuH
    576:  75.0,   # CoO
    582:  74.0,   # NiO
    588:  79.0,   # CuO
    594:  28.0,   # 13CO (isotopologue)
    780: 104.0,   # YO
    786: 107.0,   # ZrO
    792: 155.0,   # LaO
}

# Mapping from NELION to molecule code(s) in molecules.dat.
# Derived directly from Fortran xnfpelsyn.for IDMOL DATA statement (lines 111-126).
# NELION = NELEM*6 for molecular ION=6; IDMOL(NELEM-39) = molecule code.
# Integer codes match code_mol from readmol_exact; .01 suffixes are isotopologues.
_NELION_TO_CODES: Dict[int, List[int]] = {
    240: [101],       # H2          IDMOL(1)=101
    246: [106],       # CH          IDMOL(2)=106
    252: [107],       # NH          IDMOL(3)=107
    258: [108],       # OH          IDMOL(4)=108
    264: [606],       # C2          IDMOL(5)=606
    270: [607],       # CN          IDMOL(6)=607
    276: [608],       # CO          IDMOL(7)=608
    282: [707],       # N2          IDMOL(8)=707
    288: [708],       # NO          IDMOL(9)=708
    294: [808],       # O2          IDMOL(10)=808
    300: [112],       # MgH         IDMOL(11)=112
    306: [113],       # AlH         IDMOL(12)=113  (NOT MgO!)
    312: [114],       # SiH         IDMOL(13)=114
    318: [812],       # MgO         IDMOL(14)=812
    324: [813],       # AlO         IDMOL(15)=813
    330: [814],       # SiO         IDMOL(16)=814
    336: [116],       # SH          IDMOL(17)=116
    342: [120],       # CaH         IDMOL(18)=120
    348: [816],       # SO          IDMOL(19)=816
    354: [820],       # CaO         IDMOL(20)=820
    360: [821],       # ScO         IDMOL(21)=821
    366: [822],       # TiO         IDMOL(22)=822  (NOT 816!)
    372: [823],       # VO          IDMOL(23)=823  (NOT 851!)
    378: [103],       # HeH         IDMOL(24)=103
    384: [104],       # LiH         IDMOL(25)=104
    390: [105],       # BeH         IDMOL(26)=105
    396: [109],       # FH          IDMOL(27)=109
    402: [115],       # PH          IDMOL(28)=115
    408: [117],       # ClH         IDMOL(29)=117
    414: [121],       # ScH         IDMOL(30)=121
    420: [122],       # TiH         IDMOL(31)=122
    426: [123],       # VH          IDMOL(32)=123
    432: [124],       # CrH         IDMOL(33)=124
    438: [125],       # MnH         IDMOL(34)=125
    444: [126],       # FeH         IDMOL(35)=126
    492: [111],       # NaH         IDMOL(43)=111  (NOT 123!)
    498: [119],       # KH          IDMOL(44)=119
    510: [817],       # ClO         IDMOL(46)=817
    516: [824],       # CrO         IDMOL(47)=824
    522: [825],       # MnO         IDMOL(48)=825
    528: [826],       # FeO         IDMOL(49)=826
    534: [10108],     # H2O         IDMOL(50)=10108  (NOT 108!)
    540: [60808],     # CO2         IDMOL(51)=60808
    546: [10106],     # HCN         IDMOL(52)=10106
    552: [60606],     # C3          IDMOL(53)=60606
    558: [127],       # CoH         IDMOL(54)=127
    564: [128],       # NiH         IDMOL(55)=128
    570: [129],       # CuH         IDMOL(56)=129
    576: [827],       # CoO         IDMOL(57)=827
    582: [828],       # NiO         IDMOL(58)=828
    588: [829],       # CuO         IDMOL(59)=829
    780: [839],       # YO          IDMOL(91)=839
    786: [840],       # ZrO         IDMOL(92)=840
    792: [857],       # LaO         IDMOL(93)=857
}


def _build_xabund_from_atm(atm) -> Optional[np.ndarray]:
    """Reconstruct 99-element abundance array from AtmosphereModel metadata.

    Returns array in *linear* scale (not log) matching the Fortran XABUND array.
    Returns None if abundances are unavailable.
    """
    # Prefer atm.xabund — it was already converted to linear scale by convert_atm_to_npz.py
    # (via 10**log_val for metals, kept as-is for H/He number fractions).
    # Fallback: reconstruct from meta["abundances"], but that dict stores raw .atm values
    # where H/He are linear fractions and metals are log10 values, so we must convert.
    if atm.xabund is not None and len(atm.xabund) >= 2:
        arr = np.asarray(atm.xabund, dtype=np.float64)
        if np.any(arr > 0):
            return arr

    meta = getattr(atm, "metadata", {}) or {}
    xabund_data = meta.get("abundances", {})
    if not xabund_data:
        return None

    # Build linear-scale xabund from raw .atm metadata.
    # Element 1 (H) and 2 (He) are stored as linear number fractions.
    # Elements 3+ are stored as log10 values (relative to total or to H).
    xabund = np.zeros(100, dtype=np.float64)  # 1-indexed, slot 0 unused
    for elem_num, val in xabund_data.items():
        idx = int(elem_num)
        if 1 <= idx <= 99:
            if idx <= 2:
                xabund[idx] = float(val)  # H, He: already linear fractions
            else:
                log_val = float(val)
                xabund[idx] = 10.0 ** log_val if log_val > -50.0 else 0.0

    # Normalize so the array sums to 1 (number fractions)
    total = np.sum(xabund[1:])
    if total > 0:
        xabund[1:] /= total
    else:
        return None

    return xabund[1:]  # Return 1-based as 0-based (index 0 = H, index 1 = He, ...)


def compute_mol_xnfdop(
    atm,
    nelion_set: Set[int],
    molecules_path: Optional[Path] = None,
) -> Dict[int, np.ndarray]:
    """Compute molecular XNFDOP for a set of NELION species.

    Parameters
    ----------
    atm:         AtmosphereModel (must have temperature, gas_pressure, electron_density,
                 mass_density, hckt, and ideally metadata["abundances"])
    nelion_set:  set of NELION codes that appear in the compiled molecular lines
    molecules_path: optional path to molecules.dat / fort.2

    Returns
    -------
    dict {nelion: xnfdop_array[n_layers]}  (empty dict if computation fails)
    """
    try:
        from ..tools.nmolec_exact import nmolec_exact, MAXMOL
        from ..tools.readmol_exact import readmol_exact
        from ..tools.departure_tables import initialize_departure_tables
        from ..tools.pops_exact import pfsaha_exact, _compute_xnfpmol, load_fortran_data, POTION
    except ImportError as exc:
        logger.warning("Molecular populations unavailable (import error: %s)", exc)
        return {}

    # Ensure POTION (atomic partition function tables) are loaded — required by pfsaha_exact.
    # load_fortran_data() reads synthe_py/data/fortran_data.npz; safe to call multiple times.
    if POTION is None:
        try:
            load_fortran_data()
        except Exception as exc:
            logger.warning("Could not load fortran_data.npz: %s; molecular populations may be wrong", exc)

    # Locate molecules.dat
    if molecules_path is None:
        candidates = [
            Path(__file__).parent.parent.parent.parent / "lines" / "molecules.dat",
            Path(__file__).parent.parent.parent / "lines" / "molecules.dat",
            Path("lines") / "molecules.dat",
        ]
        for c in candidates:
            if c.exists():
                molecules_path = c
                break
    if molecules_path is None or not molecules_path.exists():
        logger.warning("molecules.dat not found; cannot compute molecular populations")
        return {}

    # Build xabund array
    xabund_full = _build_xabund_from_atm(atm)
    if xabund_full is None:
        logger.warning("No abundance data in AtmosphereModel; cannot compute molecular populations")
        return {}

    # Pad / trim to 99 elements
    xabund99 = np.zeros(99, dtype=np.float64)
    n_copy = min(len(xabund_full), 99)
    xabund99[:n_copy] = xabund_full[:n_copy]

    n_layers = atm.layers

    # Atmospheric state arrays
    temperature = np.asarray(atm.temperature, dtype=np.float64)
    electron_density = np.maximum(np.asarray(atm.electron_density, dtype=np.float64), 1e-30)
    gas_pressure = np.asarray(atm.gas_pressure, dtype=np.float64)
    mass_density = np.asarray(atm.mass_density, dtype=np.float64)

    # Derived thermodynamic quantities
    kboltz_ev = 8.617333262e-5  # eV/K
    tkev = temperature * kboltz_ev
    tk   = temperature * 1.380649e-16  # k_B*T in erg
    tlog = np.log(temperature)
    # xnatom = P/(k_B*T) - n_e (approximate)
    xnatom_atomic = gas_pressure / tk - electron_density

    # hckt for Boltzmann factor
    hckt = np.zeros(n_layers, dtype=np.float64)
    if atm.hckt is not None and len(atm.hckt) == n_layers:
        hckt = np.asarray(atm.hckt, dtype=np.float64)
    else:
        hckt = 1.4388 / temperature  # h*c/k in cm (c in cm/s, kT in cm^-1)

    # Read molecular data
    try:
        nummol, code_mol, equil, locj, kcomps, idequa, nequa, nloc = readmol_exact(
            molecules_path
        )
    except Exception as exc:
        logger.warning("Failed to read molecules.dat: %s", exc)
        return {}

    # Departure tables (initialized to LTE = 1)
    try:
        departure_tables = initialize_departure_tables(n_layers)
    except Exception as exc:
        logger.warning("Failed to initialize departure tables: %s", exc)
        return {}

    bhyd = departure_tables["bhyd"]
    bc1  = departure_tables["bc1"]
    bo1  = departure_tables["bo1"]
    bmg1 = departure_tables["bmg1"]
    bal1 = departure_tables["bal1"]
    bsi1 = departure_tables["bsi1"]
    bca1 = departure_tables["bca1"]

    electron_work = electron_density.copy()
    answer_full = np.zeros((n_layers, 31), dtype=np.float64)

    def pfsaha_wrapper(j, iz, nion, mode, frac, nlte_on):
        try:
            pfsaha_exact(
                j=0,
                iz=iz,
                nion=nion,
                mode=mode,
                temperature=temperature,
                tkev=tkev,
                tk=tk,
                hkt=6.6256e-27 / tk,
                hckt=hckt,
                tlog=tlog,
                gas_pressure=gas_pressure,
                electron_density=electron_work,
                xnatom=xnatom_atomic,
                answer=answer_full,
                departure_tables=departure_tables,
                nlte_on=nlte_on,
            )
            frac[j, :] = answer_full[j, :]
        except Exception:
            pass

    hkt = 6.6256e-27 / tk  # h / (k_B × T) in s

    # Run nmolec_exact; capture xnatom_out (converged XNATOM = XN(1) from Newton iteration,
    # accounting for molecular depletion of atoms) and xnz_out (full converged XN vector).
    # Fortran atlas12.for line 4335: XNATOM(J)=XN(1) after convergence.
    # xnatom_out MUST be used in _compute_xnfpmol (not the uncorrected xnatom_atomic).
    xnmol = np.zeros((n_layers, MAXMOL), dtype=np.float64)
    try:
        xnatom_out, xnmol_out, xnz_out = nmolec_exact(
            n_layers=n_layers,
            temperature=temperature,
            tkev=tkev,
            tk=tk,
            tlog=tlog,
            gas_pressure=gas_pressure,
            electron_density=electron_work,
            xabund=xabund99,
            xnatom_atomic=xnatom_atomic,
            nummol=nummol,
            code_mol=code_mol,
            equil=equil,
            locj=locj,
            kcomps=kcomps,
            idequa=idequa,
            nequa=nequa,
            bhyd=bhyd,
            bc1=bc1,
            bo1=bo1,
            bmg1=bmg1,
            bal1=bal1,
            bsi1=bsi1,
            bca1=bca1,
            pfsaha_func=pfsaha_wrapper,
            xnmol=xnmol,
            use_gibbs=False,   # Gibbs path returns electron_density, not xnmol — use Newton
            auto_gibbs=False,
        )
        if (
            xnmol_out is not None
            and hasattr(xnmol_out, "ndim")
            and xnmol_out.ndim == 2
            and xnmol_out.shape == (n_layers, MAXMOL)
        ):
            xnmol = xnmol_out
        # Use converged xnatom_out for _compute_xnfpmol; fall back to xnatom_atomic if
        # xnatom_out is missing or degenerate (should not happen with Newton path).
        xnatom_for_xnfpmol = (
            xnatom_out
            if (xnatom_out is not None and np.all(np.isfinite(xnatom_out)) and np.any(xnatom_out > 0))
            else xnatom_atomic
        )
    except Exception as exc:
        logger.warning("nmolec_exact failed: %s; molecular opacity will be zero", exc)
        return {}

    # Compute XNFPMOL using the exact Fortran atlas7v.for Path-1 formula:
    #   XNFPMOL(J,JMOL) = exp(D_0/kT) × ∏[XNZ_atoms / (Z_atom × C_atom)] × C_mol
    # which equals n_mol / Z_int_mol, but uses atomic partition functions from PFSAHA
    # (already tabulated in the Fortran code) rather than any externally-derived formula.
    # atlas7v.for lines 4325-4368; Python port in pops_exact._compute_xnfpmol.
    # CRITICAL: pass xnatom_for_xnfpmol (the converged NMOLEC value, = XN(1) after
    # convergence) — NOT the raw xnatom_atomic (= P/(kT) - n_e). Fortran NMOLEC updates
    # XNATOM(J)=XN(1) before exiting, and the XNFPMOL computation uses that updated value.
    try:
        xnfpmol = _compute_xnfpmol(
            temperature=temperature,
            tkev=tkev,
            tk=tk,
            hkt=hkt,
            hckt=hckt,
            tlog=tlog,
            gas_pressure=gas_pressure,
            electron_density=electron_work,
            xnatom=xnatom_for_xnfpmol,
            xnz=xnz_out,
            xnmol=xnmol,
            code_mol=code_mol,
            equil=equil,
            locj=locj,
            kcomps=kcomps,
            idequa=idequa,
            nequa=nequa,
            bhyd=bhyd,
            bc1=bc1,
            bo1=bo1,
            bmg1=bmg1,
            bal1=bal1,
            bsi1=bsi1,
            bca1=bca1,
        )
    except Exception as exc:
        logger.warning("_compute_xnfpmol failed: %s; falling back to xnmol (no Z_int)", exc)
        # Fallback: use raw xnmol (this will be wrong but won't crash)
        xnfpmol = xnmol[:, :nummol].copy() if nummol <= xnmol.shape[1] else xnmol.copy()

    # Build NELION → code_mol index mapping
    # code_mol[imol] holds the molecule code (e.g. 106 for CH, 607 for CN, ...)
    code_mol_arr = np.asarray(code_mol, dtype=np.float64)

    # Map NELION → sum of XNFPMOL over matching isotopologues
    # XNFPMOL = XNMOL / Z_int_mol (Fortran convention) — already computed above.
    result: Dict[int, np.ndarray] = {}
    rho = np.maximum(mass_density, 1e-40)

    for nelion in nelion_set:
        codes = _NELION_TO_CODES.get(nelion)
        if codes is None:
            continue

        # Sum XNFPMOL over all isotopologues matching this NELION.
        # Fortran xnfpelsyn.for line 328: CALL POPS(IDMOL(NELEM-39),1,XNFP(1,6,NELEM))
        # uses the EXACT float IDMOL value (e.g. 114.00 not 114.01).
        # We must NOT truncate with int() — that conflates isotopologues.
        mol_density_normalized = np.zeros(n_layers, dtype=np.float64)
        for code in codes:
            code_float = float(code)
            for imol in range(nummol):
                cm = code_mol_arr[imol]
                if abs(cm - code_float) < 0.005 and imol < xnfpmol.shape[1]:
                    mol_density_normalized += xnfpmol[:, imol]

        mol_density_normalized = np.maximum(mol_density_normalized, 0.0)

        # Compute doppler width for this molecule (Fortran DOPPLE = sqrt(2kT/m)/c + vturb/c)
        mol_mass = _NELION_TO_MASS.get(nelion, 28.0)  # default: CO mass
        thermal_vel = np.sqrt(2.0 * _KBOLTZ * temperature / (mol_mass * _AMU)) / _C_LIGHT_CMS

        # Doppler width (in velocity units, fractional): sqrt(thermal^2 + turbulent^2)
        vturb = np.zeros(n_layers, dtype=np.float64)
        if atm.turbulent_velocity is not None:
            vturb = np.asarray(atm.turbulent_velocity, dtype=np.float64) / _C_LIGHT_CMS

        doppler_frac = np.sqrt(thermal_vel**2 + vturb**2)

        # XNFDOP(NELION) = XNFPMOL / (RHO × DOPPLE)
        # Fortran xnfpelsyn.for: XNFDOP(NELION) = AMOL / DOPPLE / RHO
        #   where AMOL = XNFPMOL (returned by POPS with MODE=1 for molecular NELION)
        with np.errstate(divide="ignore", invalid="ignore"):
            xnfdop = np.where(
                (rho > 0) & (doppler_frac > 0),
                mol_density_normalized / (rho * doppler_frac),
                0.0,
            )

        result[nelion] = xnfdop
        logger.debug(
            "NELION=%d: max(xnfpmol)=%.3e, max(xnfdop)=%.3e",
            nelion,
            np.max(mol_density_normalized),
            np.max(xnfdop),
        )

    return result


def compute_mol_xnfpmol_dopple(
    atm,
    nelion_set: Set[int],
    molecules_path: Optional[Path] = None,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Compute raw molecular XNFPMOL and DOPPLE for injection into atmosphere arrays.

    Follows Fortran xnfpelsyn.for:
      XNFPEL(6, NELEM) = XNFPMOL via POPS(IDMOL, 1, ...)
      DOPPLE(6, NELEM) = sqrt(2*k*T / (MOMASS * 1.660e-24) + VTURB**2) / c

    Returns
    -------
    (xnfpmol_dict, dopple_dict): both keyed by NELION, values shape (n_layers,)
    """
    xnfdop_result = compute_mol_xnfdop(atm, nelion_set, molecules_path)
    if not xnfdop_result:
        return {}, {}

    n_layers = atm.layers
    temperature = np.asarray(atm.temperature, dtype=np.float64)
    mass_density = np.maximum(np.asarray(atm.mass_density, dtype=np.float64), 1e-40)

    vturb = np.zeros(n_layers, dtype=np.float64)
    if atm.turbulent_velocity is not None:
        vturb = np.asarray(atm.turbulent_velocity, dtype=np.float64) / _C_LIGHT_CMS

    xnfpmol_dict: Dict[int, np.ndarray] = {}
    dopple_dict: Dict[int, np.ndarray] = {}

    for nelion in nelion_set:
        if nelion not in xnfdop_result:
            continue

        # Reconstruct XNFPMOL and DOPPLE from XNFDOP = XNFPMOL / (RHO * DOPPLE)
        mol_mass = _NELION_TO_MASS.get(nelion, 28.0)
        thermal_vel = np.sqrt(2.0 * _KBOLTZ * temperature / (mol_mass * _AMU)) / _C_LIGHT_CMS
        dopple = np.sqrt(thermal_vel**2 + vturb**2)

        xnfpmol = xnfdop_result[nelion] * mass_density * dopple

        xnfpmol_dict[nelion] = np.maximum(xnfpmol, 0.0)
        dopple_dict[nelion] = dopple

    return xnfpmol_dict, dopple_dict
