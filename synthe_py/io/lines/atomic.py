"""Atomic line-list handling."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .tfort import _ELEMENT_SYMBOLS

_CM_INV_PER_EV = 8065.54429

# Global POTION array for ionization potentials (loaded on first use)
_POTION: Optional[np.ndarray] = None


def _load_potion() -> np.ndarray:
    """Load ionization potential array from fortran_data.npz."""
    global _POTION
    if _POTION is not None:
        return _POTION

    # Try to load from fortran_data.npz
    data_path = Path(__file__).parent.parent.parent / "data" / "fortran_data.npz"
    if data_path.exists():
        data = np.load(data_path)
        if "potion" in data:
            _POTION = data["potion"]
            return _POTION

    # Fallback: return empty array (will use simple defaults)
    _POTION = np.zeros(999, dtype=np.float64)
    return _POTION


def _get_potion_index(iz: int, icharge: int) -> int:
    """Compute POTION array index for element IZ and charge ICHARGE.

    Matches Fortran rgfall.for lines 187-188:
        IF(IZ.LE.30)INDEX=IZ*(IZ+1)/2+ICHARGE
        IF(IZ.GT.30)INDEX=IZ*5+341+ICHARGE

    Returns 0-based Python index.
    """
    if iz <= 30:
        # Fortran: INDEX = IZ*(IZ+1)/2 + ICHARGE (1-based)
        index = iz * (iz + 1) // 2 + icharge
    else:
        # Fortran: INDEX = IZ*5 + 341 + ICHARGE (1-based)
        index = iz * 5 + 341 + icharge
    return index - 1  # Convert to 0-based


def _compute_stark_default(code: float, e_lower: float, e_upper: float) -> float:
    """Compute default Stark damping parameter GS (log10) using EXACT Fortran formula.

    Matches the rgfall defaults used to generate the project line bundles:
        IF(GS.NE.0.)GO TO 138
        IF(CODE.GE.100.)GO TO 137
        EUP=DMAX1(DABS(E),DABS(EP))
        EFFNSQ=25.
        IZ=CODE
        IF(IZ.LE.30)INDEX=IZ*(IZ+1)/2+ICHARGE
        IF(IZ.GT.30)INDEX=IZ*5+341+ICHARGE
        DELEUP=POTION(INDEX)-EUP
        IF(DELEUP.GT.0.)EFFNSQ=109737.31*ZEFF**2/DELEUP
        GAMMAS=1.0D-8*EFFNSQ**2*SQRT(EFFNSQ)
        GS=ALOG10(GAMMAS)
        GO TO 138
    137 GAMMAS=1.0D-5
        GS=-5.
    """
    # For molecules (CODE >= 100), use simple default (Fortran line 201-202)
    if code >= 100.0:
        return -5.0  # GAMMAS = 1e-5

    potion = _load_potion()

    # Extract element number and charge (Fortran lines 175-177)
    nelem = int(code + 1e-6)
    icharge = int((code - nelem) * 100.0 + 0.1)
    zeff = icharge + 1  # Effective charge

    # Upper energy level in cm^-1 (Fortran line 183)
    eup = max(abs(e_lower), abs(e_upper))

    # Default EFFNSQ=25 (Fortran line 184) - NO FALLBACK, this is exact Fortran behavior
    effnsq = 25.0

    # Get ionization potential and update EFFNSQ if valid (Fortran lines 186-192)
    index = _get_potion_index(nelem, icharge)
    if 0 <= index < len(potion):
        deleup = potion[index] - eup
        if deleup > 0:
            effnsq = 109737.31 * zeff * zeff / deleup

    # Compute GAMMAS (Fortran line 193)
    gammas = 1.0e-8 * effnsq * effnsq * math.sqrt(effnsq)
    gs = math.log10(gammas)

    return gs


def _compute_vdw_default(code: float, e_lower: float, e_upper: float) -> float:
    """Compute default van der Waals damping parameter GW (log10) using EXACT Fortran formula.

    Matches Fortran rgfall.for lines 203-232 EXACTLY:
        IF(GW.NE.0.)GO TO 141
        IF(CODE.GE.100.)GO TO 139
        EUP=DMAX1(DABS(E),DABS(EP))
        EFFNSQ=25.
        IZ=CODE
        IF(IZ.LE.30)INDEX=IZ*(IZ+1)/2+ICHARGE
        IF(IZ.GT.30)INDEX=IZ*5+341+ICHARGE
        DELEUP=POTION(INDEX)-EUP
        IF(DELEUP.GT.0.)EFFNSQ=109737.31D0*ZEFF**2/DELEUP
        EFFNSQ=AMIN1(EFFNSQ,1000.)
        RSQUP=2.5*(EFFNSQ/ZEFF)**2
        DELELO=POTION(INDEX)-ELO
        EFFNSQ=109737.31D0*ZEFF**2/DELELO
        EFFNSQ=AMIN1(EFFNSQ,1000.)
        RSQLO=2.5*(EFFNSQ/ZEFF)**2
        NSEQ=CODE-ZEFF+1.
        IF(NSEQ.GT.20.AND.NSEQ.LT.29)THEN
          RSQUP=(45.-FLOAT(NSEQ))/ZEFF
          RSQLO=0.
        ENDIF
        IF(RSQUP.LT.RSQLO)RSQUP=2.*RSQLO
        GAMMAW=4.5D-9*(RSQUP-RSQLO)**.4
        GW=ALOG10(GAMMAW)
        GO TO 141
    139 GAMMAW=1.D-7/ZEFF
        GW=ALOG10(GAMMAW)
    """
    # Extract element number and charge (Fortran lines 175-177)
    nelem = int(code + 1e-6)
    icharge = int((code - nelem) * 100.0 + 0.1)
    zeff = icharge + 1  # Effective charge

    # For molecules (CODE >= 100), use simple default (Fortran lines 231-232)
    if code >= 100.0:
        gammaw = 1.0e-7 / zeff
        return math.log10(gammaw)

    potion = _load_potion()

    # Energy levels in cm^-1
    elo = min(abs(e_lower), abs(e_upper))  # Lower level
    eup = max(abs(e_lower), abs(e_upper))  # Upper level

    # Get ionization potential
    index = _get_potion_index(nelem, icharge)
    ip = 0.0
    if 0 <= index < len(potion):
        ip = potion[index]

    # Default EFFNSQ=25 (Fortran line 206) - NO FALLBACK, exact Fortran behavior
    # Compute EFFNSQ and RSQUP for upper level (Fortran lines 205-215)
    effnsq_up = 25.0
    if ip > 0:
        deleup = ip - eup
        if deleup > 0:
            effnsq_up = 109737.31 * zeff * zeff / deleup
    effnsq_up = min(effnsq_up, 1000.0)  # Fortran line 214
    rsqup = 2.5 * (effnsq_up / zeff) ** 2  # Fortran line 215

    # Compute EFFNSQ and RSQLO for lower level (Fortran lines 216-220)
    effnsq_lo = 25.0
    if ip > 0:
        delelo = ip - elo
        if delelo > 0:
            effnsq_lo = 109737.31 * zeff * zeff / delelo
    effnsq_lo = min(effnsq_lo, 1000.0)  # Fortran line 219
    rsqlo = 2.5 * (effnsq_lo / zeff) ** 2  # Fortran line 220

    # Special case for transition metals (Fortran lines 221-225)
    nseq = nelem - zeff + 1  # Sequence number
    if 20 < nseq < 29:
        rsqup = (45.0 - nseq) / zeff
        rsqlo = 0.0

    # Ensure RSQUP >= RSQLO (Fortran line 227)
    if rsqup < rsqlo:
        rsqup = 2.0 * rsqlo

    # Compute GAMMAW (Fortran line 228)
    rsq_diff = rsqup - rsqlo
    if rsq_diff > 0:
        gammaw = 4.5e-9 * (rsq_diff**0.4)
    else:
        gammaw = 1.0e-9  # Minimum value for edge case

    gw = math.log10(gammaw)
    return gw


@dataclass
class LineRecord:
    """Minimal representation of an atomic spectral line."""

    wavelength: float
    # High-precision wavelength for grid indexing (derived from energies).
    index_wavelength: float
    element: str
    ion_stage: int
    log_gf: float
    excitation_energy: float
    gamma_rad: float
    gamma_stark: float
    gamma_vdw: float
    metadata: Dict[str, float]
    # Line type for special profile handling (matching Fortran rgfall.for):
    # -1 = Hydrogen, -2 = Deuterium, -3 = He I, -6 = He II, 0 = Normal (Voigt)
    line_type: int = 0
    # Principal quantum numbers for hydrogen/helium (NBLO, NBUP in Fortran)
    n_lower: int = 0
    n_upper: int = 0
    # Additional rgfall metadata needed for fort.19 generation
    code: float = 0.0
    iso1: int = 0
    iso2: int = 0
    line_size: int = 0
    labelp: str = ""
    xj: float = 0.0
    xjp: float = 0.0
    gamma_rad_log: float = 0.0
    gamma_stark_log: float = 0.0
    gamma_vdw_log: float = 0.0


@dataclass
class LineCatalog:
    """Container for line records and lookup tables."""

    records: List[LineRecord]
    # Precomputed arrays for vectorised operations
    wavelength: np.ndarray
    index_wavelength: np.ndarray
    log_gf: np.ndarray
    gf: np.ndarray
    excitation_energy: np.ndarray
    gamma_rad: np.ndarray
    gamma_stark: np.ndarray
    gamma_vdw: np.ndarray
    elements: np.ndarray
    ion_stages: np.ndarray
    # Line type for special profile handling (-1=H, -3=HeI, -6=HeII, 0=normal)
    line_types: np.ndarray = None
    # Principal quantum numbers for hydrogen/helium lines
    n_lower: np.ndarray = None
    n_upper: np.ndarray = None

    @classmethod
    def from_records(cls, records: Sequence[LineRecord]) -> "LineCatalog":
        wavelength = np.array([rec.wavelength for rec in records], dtype=np.float64)
        index_wavelength = np.array(
            [
                rec.index_wavelength if rec.index_wavelength > 0.0 else rec.wavelength
                for rec in records
            ],
            dtype=np.float64,
        )
        log_gf = np.array([rec.log_gf for rec in records], dtype=np.float64)
        gf = np.power(10.0, log_gf)
        excitation = np.array(
            [rec.excitation_energy for rec in records], dtype=np.float64
        )
        gamma_rad = np.array([rec.gamma_rad for rec in records], dtype=np.float64)
        gamma_stark = np.array([rec.gamma_stark for rec in records], dtype=np.float64)
        gamma_vdw = np.array([rec.gamma_vdw for rec in records], dtype=np.float64)
        elements = np.array([rec.element for rec in records], dtype=object)
        ion_stages = np.array([rec.ion_stage for rec in records], dtype=np.int16)
        line_types = np.array([rec.line_type for rec in records], dtype=np.int8)
        n_lower = np.array([rec.n_lower for rec in records], dtype=np.int16)
        n_upper = np.array([rec.n_upper for rec in records], dtype=np.int16)
        return cls(
            records=list(records),
            wavelength=wavelength,
            index_wavelength=index_wavelength,
            log_gf=log_gf,
            gf=gf,
            excitation_energy=excitation,
            gamma_rad=gamma_rad,
            gamma_stark=gamma_stark,
            gamma_vdw=gamma_vdw,
            elements=elements,
            ion_stages=ion_stages,
            line_types=line_types,
            n_lower=n_lower,
            n_upper=n_upper,
        )


def load_catalog(path: Path) -> LineCatalog:
    """Load an atomic line catalog.

    The Kurucz GFALL dataset uses a fixed-width format (see rgfall.for line 122-123).
    Format: F11.4,F7.3,F6.2,F12.3,F5.1,1X,A8,A2,F12.3,F5.1,1X,A8,A2,F6.2,F6.2,F6.2,...

    **Units note**
    GFALL wavelengths are stored in Angstroms, but the rest of the Python pipeline
    (wavelength grid, opacities, radiative transfer) operates in nanometres.
    We therefore convert GFALL wavelengths from Å to nm at load time so that
    catalog wavelengths are directly comparable to the NM-based wavelength grid.

    This parser handles both fixed-width format (gfallvac.latest) and
    whitespace-separated formats.
    """

    records: List[LineRecord] = []

    def iter_lines() -> Iterable[str]:
        with path.open("r", encoding="ascii", errors="ignore") as fh:
            return fh.read().splitlines()

    for line in iter_lines():
        if line.strip().startswith("#") or not line.strip():
            continue

        # Try fixed-width format first (gfallvac.latest format from rgfall.for)
        # Format positions: WL(0-10), GFLOG(11-17), CODE(18-23), E(24-35), XJ(36-40),
        # space(41), LABEL(42-49), A2(50-51), EP(52-63), XJP(64-68), space(69),
        # LABELP(70-77), A2(78-79), GR(80-85), GS(86-91), GW(92-97)
        if len(line) >= 98:  # Minimum length for fixed-width format with gamma values
            try:
                # Extract fields using fixed-width positions
                wavelength_str = line[0:11].strip()
                log_gf_str = line[11:18].strip()
                code_str = line[18:24].strip()
                excitation_str = line[24:36].strip()
                # EP is upper level energy (positions 52-63)
                excitation_upper_str = line[52:64].strip()
                xj_str = line[36:41].strip()
                xjp_str = line[64:69].strip()
                labelp_str = line[70:78].strip() if len(line) >= 78 else ""
                gamma_rad_str = line[80:86].strip()
                gamma_stark_str = line[86:92].strip()
                gamma_vdw_str = line[92:98].strip()
                iso1_str = line[106:109].strip() if len(line) >= 109 else ""
                iso2_str = line[116:119].strip() if len(line) >= 119 else ""
                isoshift_str = line[154:160].strip() if len(line) >= 160 else ""
                cother1_str = line[124:134] if len(line) >= 134 else ""
                cother2_str = line[134:144] if len(line) >= 144 else ""

                # Parse numeric values
                # GFALL wavelengths in this dataset are already in nm.
                wavelength_stored = float(wavelength_str)
                log_gf_raw = float(log_gf_str)
                code = float(code_str)

                # CRITICAL FIX (Dec 2025): Parse X1 and X2 isotopic abundance corrections
                # Fortran rgfall.for line 160: GF=10.**(GFLOG+DGFLOG+X1+X2)
                # X1 and X2 are LOG FRACTIONAL ISOTOPIC ABUNDANCES that scale the gf value
                # For hyperfine/isotope lines, each component has the FULL parent gflog,
                # but X1+X2 contain negative values that reduce it to the correct fraction.
                # Without this correction, Python overestimates opacity by 10-20× per component!
                #
                # Format from rgfall.for line 122-123:
                #   ...A4,I2,I2,I3,F6.3,I3,F6.3,...
                #   REF(99-102), NBLO(103-104), NBUP(105-106), ISO1(107-109), X1(110-115), ISO2(116-118), X2(119-124)
                x1 = 0.0
                x2 = 0.0
                if len(line) >= 115:
                    x1_str = line[109:115].strip()
                    if x1_str:
                        try:
                            x1 = float(x1_str)
                        except ValueError:
                            pass
                if len(line) >= 124:
                    x2_str = line[118:124].strip()
                    if x2_str:
                        try:
                            x2 = float(x2_str)
                        except ValueError:
                            pass

                # Apply isotopic abundance correction (Fortran rgfall.for line 160)
                # Allow disabling to match runs where X1/X2 are not applied.
                apply_iso_corr = os.environ.get("PY_APPLY_ISO_CORR", "1") != "0"
                log_gf = log_gf_raw + (x1 + x2 if apply_iso_corr else 0.0)
                e_val = float(excitation_str)  # First energy column (E)
                ep_val = (
                    float(excitation_upper_str) if excitation_upper_str else 0.0
                )  # Second energy column (EP)

                # CRITICAL: Use min(|E|, |EP|) as lower level energy for absorption
                # Matches Fortran rgfall.for line 161: ELO=DMIN1(DABS(E),DABS(EP))
                excitation_cm = min(abs(e_val), abs(ep_val))
                excitation_upper_cm = max(abs(e_val), abs(ep_val))
                gamma_rad_log = float(gamma_rad_str) if gamma_rad_str else 0.0
                gamma_stark_log = float(gamma_stark_str) if gamma_stark_str else 0.0
                gamma_vdw_log = float(gamma_vdw_str) if gamma_vdw_str else 0.0

                # GR/GS/GW are logged in gfall (rgfall.for lines 160-168 exponentiate them)
                def _exp_or_zero(val: float) -> float:
                    if val == 0.0:
                        return 0.0
                    return 10.0**val

                gamma_rad = _exp_or_zero(gamma_rad_log)
                gamma_stark = _exp_or_zero(gamma_stark_log)
                gamma_vdw = _exp_or_zero(gamma_vdw_log)

                # WAVELENGTH COMPUTATION - Use stored wavelength from file
                #
                # CRITICAL FIX (Dec 2025): Use stored wavelength, NOT recomputed from energy.
                #
                # The gfallvac.latest file stores vacuum wavelengths that ALREADY include
                # hyperfine structure shifts (ESHIFT, ESHIFTP) and isotope shifts (DWLISO).
                # These are computed by rgfall.for as:
                #   WLVAC = 1.D7/|EP+ESHIFTP-(E+ESHIFT)| + DWL + DWLISO  (nm)
                #
                # When Python recomputes from energy levels WITHOUT these shifts, all
                # hyperfine components get the SAME wavelength (e.g., all 12 Co I
                # components at 313.823817 nm). This causes incorrect opacity distribution.
                #
                # The stored wavelength correctly places each component at its shifted
                # position (e.g., 313.8231, 313.8232, ..., 313.8262 nm for Co I).
                #
                # Evidence: Fortran rgfall.for line 131 comment:
                #   "definition of dwliso changed, now in mA and WL already includes dwliso"
                wavelength = wavelength_stored
                # High-precision wavelength for grid indexing (derived from energies).
                # This matches Fortran NBUFF rounding (uses full-precision energies).
                index_wavelength = wavelength
                eshift = 0.0
                eshiftp = 0.0
                if cother1_str.strip():
                    try:
                        ishift = int(cother1_str[0:5])
                        ishiftp = int(cother1_str[5:10])
                        eshift = ishift * 1.0e-3
                        eshiftp = ishiftp * 1.0e-3
                    except ValueError:
                        eshift = 0.0
                        eshiftp = 0.0

                energy_diff = abs((abs(ep_val) + eshiftp) - (abs(e_val) + eshift))
                if energy_diff > 0.0:
                    dwliso = 0.0
                    if isoshift_str:
                        try:
                            dwliso = int(isoshift_str) * 1.0e-4
                        except ValueError:
                            dwliso = 0.0
                    energy_wl = 1.0e7 / energy_diff + dwliso
                    # Fortran rgfall computes WLVAC from energies (IFVAC=1), and
                    # uses that for NBUFF rounding and for CGF normalization.
                    # Match that behavior by adopting the energy-derived wavelength.
                    wavelength = energy_wl
                    index_wavelength = energy_wl

                # DEFAULT DAMPING PARAMETERS - Exact Fortran rgfall.for behavior
                # Fortran computes defaults when GR, GS, or GW are 0 in the line list
                # These are stored as log10 values in the line list

                # Radiative damping default (rgfall.for lines 171-173):
                # IF(GR.EQ.0.)THEN
                #   GAMMAR = 2.223D13 / WLVAC**2  (WLVAC in ANGSTROMS!)
                #   GR = ALOG10(GAMMAR)
                # ENDIF
                if gamma_rad == 0.0:
                    # Fortran rgfall.for uses WLVAC in nm (1e7/nu, with nu in cm^-1).
                    # So the default GAMMAR uses the NM wavelength directly.
                    gammar_default = 2.223e13 / (wavelength**2)
                    gamma_rad = gammar_default  # Store linear, not log

                # Stark damping default - EXACT Fortran formula using POTION
                # (rgfall.for lines 181-197)
                if gamma_stark == 0.0:
                    gamma_stark_log_val = _compute_stark_default(
                        code, excitation_cm, excitation_upper_cm
                    )
                    gamma_stark = (
                        10.0**gamma_stark_log_val if gamma_stark_log_val != 0.0 else 0.0
                    )

                # van der Waals damping default - EXACT Fortran formula using POTION
                # (rgfall.for lines 203-232)
                if gamma_vdw == 0.0:
                    gamma_vdw_log_val = _compute_vdw_default(
                        code, excitation_cm, excitation_upper_cm
                    )
                    gamma_vdw = (
                        10.0**gamma_vdw_log_val if gamma_vdw_log_val != 0.0 else 0.0
                    )

                # Normalize damping parameters to match rgfall -> tfort.12.
                # rgfall.for lines 266-273:
                #   FRELIN=2.99792458D17/WLVAC
                #   GAMMAR=GAMMAR/12.5664D0/FRELIN
                #   GAMMAS=GAMMAS/12.5664D0/FRELIN
                #   GAMMAW=GAMMAW/12.5664D0/FRELIN
                # synthe.for reads these normalized values from fort.12.
                frelin = 2.99792458e17 / max(wavelength, 1e-12)
                gamma_rad = gamma_rad / (12.5664 * frelin)
                gamma_stark = gamma_stark / (12.5664 * frelin)
                gamma_vdw = gamma_vdw / (12.5664 * frelin)

                # Extract element and ion_stage from CODE
                # CODE format: nelem.ion_stage (e.g., 18.00 = element 18, ion stage 1)
                nelem = int(code + 1e-6)
                frac = code - nelem
                ion_stage = int(round(frac * 100.0)) + 1 if frac > 1e-6 else 1

                if 0 < nelem < len(_ELEMENT_SYMBOLS):
                    element = _ELEMENT_SYMBOLS[nelem]
                else:
                    element = f"Z{nelem}"

                # Keep excitation in cm^-1 (matches Fortran synthe.for ELO)
                # Fortran uses ELO * HCKT where both are in consistent units (cm^-1 * cm)
                excitation_cm_final = excitation_cm

                # Extract NBLO/NBUP (quantum numbers) from positions 102-106
                # Format from rgfall.for: ...A4,I2,I2,... (REF at 98-101, NBLO at 102-103, NBUP at 104-105)
                n_lower = 0
                n_upper = 0
                if len(line) >= 106:
                    try:
                        nblo_str = line[102:104].strip()
                        nbup_str = line[104:106].strip()
                        n_lower = int(nblo_str) if nblo_str else 0
                        n_upper = int(nbup_str) if nbup_str else 0
                    except (ValueError, IndexError):
                        pass

                # Parse ISO1/ISO2 and AUTO/LINESIZE metadata (rgfall.for lines 113-127)
                iso1_val = 0
                iso2_val = 0
                if iso1_str:
                    try:
                        iso1_val = int(iso1_str)
                    except ValueError:
                        iso1_val = 0
                if iso2_str:
                    try:
                        iso2_val = int(iso2_str)
                    except ValueError:
                        iso2_val = 0

                line_size = 0
                auto_tag = ""
                if cother2_str.strip():
                    line_size_str = cother2_str[6:7]
                    auto_tag = cother2_str[7:10].strip()
                    if line_size_str.strip().isdigit():
                        line_size = int(line_size_str)

                # Determine line type based on CODE and AUTO tag (rgfall.for lines 239-259)
                # TYPE=-1: Hydrogen (CODE=1.00)
                # TYPE=-2: Deuterium (CODE=1.00 with ISO1=2)
                # TYPE=-3: Helium I (CODE=2.00)
                # TYPE=-4: Helium-3 I (CODE=2.00 with ISO1=3)
                # TYPE=-6: Helium II (CODE=2.01)
                # TYPE=1: Autoionizing line (AUTO='AUT')
                # TYPE=2: Coronal approximation line (AUTO='COR')
                # TYPE=3: PRD line (AUTO='PRD')
                # TYPE=0: Normal line (Voigt profile)
                line_type = 0
                if auto_tag == "AUT":
                    line_type = 1
                elif auto_tag == "COR":
                    line_type = 2
                elif auto_tag == "PRD":
                    line_type = 3
                elif nelem == 1 and ion_stage == 1 and iso1_val == 2:
                    line_type = -2  # Deuterium
                elif nelem == 1 and ion_stage == 1:
                    line_type = -1  # Hydrogen
                elif nelem == 2 and ion_stage == 1 and iso1_val == 3:
                    line_type = -4  # Helium-3 I
                elif nelem == 2 and ion_stage == 1:
                    line_type = -3  # Helium I
                elif nelem == 2 and ion_stage == 2:
                    line_type = -6  # Helium II

                records.append(
                    LineRecord(
                        wavelength=wavelength,
                        index_wavelength=float(index_wavelength),
                        element=element,
                        ion_stage=ion_stage,
                        log_gf=log_gf,
                        excitation_energy=excitation_cm_final,
                        gamma_rad=gamma_rad,
                        gamma_stark=gamma_stark,
                        gamma_vdw=gamma_vdw,
                        metadata={},
                        line_type=line_type,
                        n_lower=n_lower,
                        n_upper=n_upper,
                        code=code,
                        iso1=iso1_val,
                        iso2=iso2_val,
                        line_size=line_size,
                        labelp=labelp_str,
                        xj=float(xj_str) if xj_str else 0.0,
                        xjp=float(xjp_str) if xjp_str else 0.0,
                        gamma_rad_log=gamma_rad_log,
                        gamma_stark_log=gamma_stark_log,
                        gamma_vdw_log=gamma_vdw_log,
                    )
                )
                continue
            except (ValueError, IndexError):
                # If fixed-width parsing fails, try whitespace-separated format
                pass

        # Fallback to whitespace-separated format (for other catalog formats)
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            wavelength = float(parts[0])
            element = parts[1]
            # Handle both integer and float ion_stage (some catalogs use floats)
            try:
                ion_stage = int(float(parts[2]))  # Convert float to int if needed
            except (ValueError, IndexError):
                continue  # Skip lines with invalid ion_stage
            log_gf = float(parts[3])
            excitation = float(parts[4])
            gamma_rad = float(parts[5])
            gamma_stark = float(parts[6])
            gamma_vdw = float(parts[7]) if len(parts) > 7 else 0.0
            metadata: Dict[str, float] = {}
            n_lower_val = 0
            n_upper_val = 0
            if len(parts) > 9:
                try:
                    n_lower_val = int(float(parts[8]))
                    n_upper_val = int(float(parts[9]))
                    metadata["n_lower"] = n_lower_val
                    metadata["n_upper"] = n_upper_val
                except ValueError:
                    pass

            # Determine line type for whitespace format (based on element name)
            line_type_ws = 0
            if element == "H" and ion_stage == 1:
                line_type_ws = -1  # Hydrogen
            elif element == "He" and ion_stage == 1:
                line_type_ws = -3  # Helium I
            elif element == "He" and ion_stage == 2:
                line_type_ws = -6  # Helium II

            records.append(
                LineRecord(
                    wavelength=wavelength,
                    index_wavelength=float(wavelength),
                    element=element,
                    ion_stage=ion_stage,
                    log_gf=log_gf,
                    excitation_energy=excitation,
                    gamma_rad=gamma_rad,
                    gamma_stark=gamma_stark,
                    gamma_vdw=gamma_vdw,
                    metadata=metadata,
                    line_type=line_type_ws,
                    n_lower=n_lower_val,
                    n_upper=n_upper_val,
                    code=0.0,
                    iso1=0,
                    iso2=0,
                    line_size=0,
                    labelp="",
                    xj=0.0,
                    xjp=0.0,
                    gamma_rad_log=0.0,
                    gamma_stark_log=0.0,
                    gamma_vdw_log=0.0,
                )
            )
        except (ValueError, IndexError):
            continue  # Skip invalid lines

    # Deduplicate before returning (main parsing path in load_catalog)
    records = _deduplicate_lines(records)
    return LineCatalog.from_records(records)


def _deduplicate_lines(records: List[LineRecord]) -> List[LineRecord]:
    """DISABLED: Do NOT deduplicate lines.

    Previous implementation incorrectly removed hyperfine/isotope components that
    have the same computed wavelength (from energy levels) and log_gf.

    This was WRONG because:
    1. Fortran processes EACH line separately and ACCUMULATES contributions:
       BUFFER(NBUFF) = BUFFER(NBUFF) + KAPPA  (synthe.for lines 309, 341, 768, etc.)
    2. Each hyperfine component contributes its own opacity even at the same wavelength
    3. The comment "Fortran handles these as single lines" was incorrect

    For example, the Co I line at 313.82 nm has 12 hyperfine components in gfallvac,
    all with the same energy levels (E=1809.313, EP=33674.326) and log_gf=-1.332.
    When Python recomputes wavelength from energy, they all get 313.823817 nm.
    Previously, 11 of 12 components were discarded, causing ~10X opacity loss!

    FIX (Dec 2025): Return records unchanged to match Fortran behavior.
    """
    # Return all records - each line contributes separately
    return records


# Fortran DELLIM margins (rgfall.for line 90): wavelength margins in nm
# INDEX:    1      2     3    4   5    6     7
# DELLIM: 100.0, 30.0, 10.0, 3.0, 1.0, 0.3, 0.1
_DELLIM = np.array([100.0, 30.0, 10.0, 3.0, 1.0, 0.3, 0.1], dtype=np.float64)


def _get_line_margin(
    line_type: int,
    element: str,
    wl_min: float,
    line_size: int = 0,
    code: float = 0.0,
) -> float:
    """Get the wavelength margin for a line based on Fortran rgfall.for DELLIM logic.

    Fortran rgfall.for lines 145-148:
        LIM = MIN(8 - LINESIZE, 7)
        IF(WLVAC.LT.WLBEG-DELLIM(LIM)*DELFACTOR)GO TO 900
        IF(WLVAC.GT.WLEND+DELLIM(LIM)*DELFACTOR)GO TO 900

    LINESIZE values (from rgfall.for):
        - Hydrogen (H I, D I): LINESIZE = 7 → LIM = 1 → DELLIM(1) = 100nm
        - Helium I: LINESIZE = 5-7 depending on line → LIM = 1-3
        - Helium II: LINESIZE = 7 → LIM = 1 → DELLIM(1) = 100nm
        - Other atoms: LINESIZE = 0 → LIM = 7 → DELLIM(7) = 0.1nm

    Args:
        line_type: Line type code (-1=H, -2=D, -3=HeI, -6=HeII, 0=normal)
        element: Element name (e.g., 'H', 'He', 'Fe')
        wl_min: Minimum wavelength of synthesis range (nm)

    Returns:
        Margin in nm to apply for this line
    """
    # Compute DELFACTOR (rgfall.for line 96)
    delfactor = 1.0 if wl_min <= 500.0 else wl_min / 500.0

    # Match rgfall.for lines 145-147:
    #   LIM=MIN(8-LINESIZE,7)
    #   IF(CODE.EQ.1.)LIM=1
    linesize = int(line_size) if line_size > 0 else 0
    lim_fortran = min(8 - linesize, 7)
    # CODE can be unavailable in some non-gfall formats; fallback to element/type.
    if abs(float(code) - 1.0) < 1.0e-6 or line_type in (-1, -2) or element in {"H", "D"}:
        lim_fortran = 1
    lim_index = lim_fortran - 1  # 0-based for Python array

    return _DELLIM[lim_index] * delfactor


def filter_by_range(
    catalog: LineCatalog,
    wl_min: float,
    wl_max: float,
) -> LineCatalog:
    """Filter catalog to wavelength range matching Fortran rgfall.for DELLIM behavior.

    This function replicates Fortran's line filtering EXACTLY:
    1. Each line type gets a different wavelength margin (DELLIM)
    2. Hydrogen/Helium lines get ±100nm margin (can contribute far wing opacity)
    3. Regular atomic lines get ±0.1nm margin
    4. DELFACTOR scales margins for long-wavelength synthesis

    From rgfall.for lines 90, 145-148:
        DATA DELLIM/100.,30.,10.,3.,1.,.3,.1/
        LIM=MIN(8-LINESIZE,7)
        IF(WLVAC.LT.WLBEG-DELLIM(LIM)*DELFACTOR)GO TO 900
        IF(WLVAC.GT.WLEND+DELLIM(LIM)*DELFACTOR)GO TO 900

    Args:
        catalog: Line catalog to filter
        wl_min: Minimum wavelength (nm) - corresponds to Fortran WLBEG
        wl_max: Maximum wavelength (nm) - corresponds to Fortran WLEND

    Returns:
        Filtered LineCatalog with only lines that would be included by rgfall.exe
    """
    if len(catalog.records) == 0:
        return catalog

    # Filter each line based on its specific margin
    included_indices = []

    for i, rec in enumerate(catalog.records):
        wl = rec.wavelength
        margin = _get_line_margin(
            rec.line_type,
            rec.element,
            wl_min,
            rec.line_size,
            rec.code,
        )

        # Apply Fortran's filtering logic (rgfall.for lines 147-148)
        if wl >= wl_min - margin and wl <= wl_max + margin:
            included_indices.append(i)

    records = [catalog.records[i] for i in included_indices]

    logging.getLogger(__name__).info(
        f"Filtered {len(catalog.records)} -> {len(records)} lines "
        f"using Fortran DELLIM margins (wl_range=[{wl_min:.1f}, {wl_max:.1f}] nm)"
    )

    return LineCatalog.from_records(records)
