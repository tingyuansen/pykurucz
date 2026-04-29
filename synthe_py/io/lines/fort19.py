"""Parser for SYNTHE fort.19 wing metadata tapes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import struct
from typing import Iterator, Iterable, Sequence
from enum import IntEnum

import numpy as np

from . import atomic

_RECORD_STRUCT = struct.Struct("<dffiiiiii fffii".replace(" ", ""))

_C_LIGHT_NM = 2.99792458e17
_CGF_FACTOR = 0.026538 / 1.77245
FORT19_BUILD_LOGIC_VERSION = 2
_CODEX = np.array(
    [1.0, 2.0, 2.01, 6.0, 6.01, 12.0, 12.01, 13.0, 13.01, 14.0, 14.01, 20.0, 20.01, 8.0, 11.0, 5.0, 19.0],
    dtype=np.float64,
)
_DELLIM = np.array([100.0, 30.0, 10.0, 3.0, 1.0, 0.3, 0.1], dtype=np.float64)


class Fort19WingType(IntEnum):
    """Semantic categorisation of fort.19 line types."""

    HYDROGEN = -1
    DEUTERIUM = -2
    HELIUM_4 = -3
    HELIUM_3 = -4
    HELIUM_3_II = -6
    NORMAL = 0
    AUTOIONIZING = 1
    CORONAL = 2
    PRD = 3
    CONTINUUM = 100
    UNKNOWN = 101

    @classmethod
    def from_code(cls, code: int) -> "Fort19WingType":
        if code in {-6, -4, -3, -2, -1, 0, 1, 2, 3}:
            return cls(code)  # type: ignore[arg-type]
        if code > 3:
            return cls.CONTINUUM
        return cls.UNKNOWN


def _iter_records(handle) -> Iterator[tuple[float, ...]]:
    """Yield unpacked fort.19 records from a binary handle."""

    while True:
        header = handle.read(4)
        if not header:
            break
        (size,) = struct.unpack("<i", header)
        payload = handle.read(size)
        trailer = handle.read(4)
        if len(payload) != size or len(trailer) != 4:
            raise ValueError("Truncated fort.19 record")
        (check,) = struct.unpack("<i", trailer)
        if check != size:
            raise ValueError("fort.19 record length mismatch")
        if size != _RECORD_STRUCT.size:
            raise ValueError(f"Unexpected fort.19 record size {size}")
        yield _RECORD_STRUCT.unpack(payload)


@dataclass(frozen=True)
class Fort19Data:
    """Structured access to fort.19 wing records."""

    wavelength_vacuum: np.ndarray
    energy_lower: np.ndarray
    oscillator_strength: np.ndarray
    n_lower: np.ndarray
    n_upper: np.ndarray
    ion_index: np.ndarray
    line_type: np.ndarray
    continuum_index: np.ndarray
    element_index: np.ndarray
    gamma_rad: np.ndarray
    gamma_stark: np.ndarray
    gamma_vdw: np.ndarray
    nbuff: np.ndarray
    limb: np.ndarray
    wing_type: np.ndarray

    def indices_for(self, wing_type: Fort19WingType) -> np.ndarray:
        """Return the indices of records matching the requested wing type."""
        return np.nonzero(self.wing_type == wing_type)[0]

    def iter_indices(self, wing_types: Iterable[Fort19WingType]) -> np.ndarray:
        """Return indices matching any of the supplied wing types."""
        mask = np.zeros_like(self.wing_type, dtype=bool)
        for wtype in wing_types:
            mask |= self.wing_type == wtype
        return np.nonzero(mask)[0]

    def subset(self, indices: Sequence[int]) -> "Fort19Data":
        """Return a new Fort19Data limited to the specified indices."""
        idx = np.asarray(indices, dtype=int)
        return Fort19Data(
            wavelength_vacuum=self.wavelength_vacuum[idx],
            energy_lower=self.energy_lower[idx],
            oscillator_strength=self.oscillator_strength[idx],
            n_lower=self.n_lower[idx],
            n_upper=self.n_upper[idx],
            ion_index=self.ion_index[idx],
            line_type=self.line_type[idx],
            continuum_index=self.continuum_index[idx],
            element_index=self.element_index[idx],
            gamma_rad=self.gamma_rad[idx],
            gamma_stark=self.gamma_stark[idx],
            gamma_vdw=self.gamma_vdw[idx],
            nbuff=self.nbuff[idx],
            limb=self.limb[idx],
            wing_type=self.wing_type[idx],
        )


def _classify_line_types(line_type: np.ndarray) -> np.ndarray:
    """Vectorised helper returning Fort19WingType per record."""

    vectorized = np.vectorize(lambda value: Fort19WingType.from_code(int(value)), otypes=[object])
    return vectorized(line_type)

def load(path: Path) -> Fort19Data:
    """Load a fort.19 file into NumPy arrays."""

    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            line_type = np.asarray(data["line_type"], dtype=np.int16)
            stored_wing = data.get("wing_type")
            if stored_wing is not None:
                wing_type = np.asarray(
                    [Fort19WingType.from_code(int(code)) for code in stored_wing],
                    dtype=object,
                )
            else:
                wing_type = _classify_line_types(line_type)
            return Fort19Data(
                wavelength_vacuum=np.asarray(data["wavelength_vacuum"], dtype=np.float64),
                energy_lower=np.asarray(data["energy_lower"], dtype=np.float32),
                oscillator_strength=np.asarray(data["oscillator_strength"], dtype=np.float32),
                n_lower=np.asarray(data["n_lower"], dtype=np.int16),
                n_upper=np.asarray(data["n_upper"], dtype=np.int16),
                ion_index=np.asarray(data["ion_index"], dtype=np.int16),
                line_type=line_type,
                continuum_index=np.asarray(data["continuum_index"], dtype=np.int16),
                element_index=np.asarray(data["element_index"], dtype=np.int16),
                gamma_rad=np.asarray(data["gamma_rad"], dtype=np.float32),
                gamma_stark=np.asarray(data["gamma_stark"], dtype=np.float32),
                gamma_vdw=np.asarray(data["gamma_vdw"], dtype=np.float32),
                nbuff=np.asarray(data["nbuff"], dtype=np.int32),
                limb=np.asarray(data["limb"], dtype=np.int32),
                wing_type=wing_type,
            )

    wavelengths: list[float] = []
    energies: list[float] = []
    gfs: list[float] = []
    nblo: list[int] = []
    nbup: list[int] = []
    nelion: list[int] = []
    linetype: list[int] = []
    ncon: list[int] = []
    nelionx: list[int] = []
    gamma_r: list[float] = []
    gamma_s: list[float] = []
    gamma_w: list[float] = []
    nbuff_vals: list[int] = []
    limb_vals: list[int] = []

    with path.open("rb") as fh:
        for record in _iter_records(fh):
            (
                wl_vac,
                elo,
                gf,
                n_lower,
                n_upper,
                ion,
                line_type,
                continuum_idx,
                elem_idx,
                gamma_rad,
                gamma_stark,
                gamma_vdw,
                nbuff_val,
                limb_val,
            ) = record

            wavelengths.append(wl_vac)
            energies.append(elo)
            gfs.append(gf)
            nblo.append(n_lower)
            nbup.append(n_upper)
            nelion.append(ion)
            linetype.append(line_type)
            ncon.append(continuum_idx)
            nelionx.append(elem_idx)
            gamma_r.append(gamma_rad)
            gamma_s.append(gamma_stark)
            gamma_w.append(gamma_vdw)
            nbuff_vals.append(nbuff_val)
            limb_vals.append(limb_val)

    line_type_array = np.asarray(linetype, dtype=np.int16)
    return Fort19Data(
        wavelength_vacuum=np.asarray(wavelengths, dtype=np.float64),
        energy_lower=np.asarray(energies, dtype=np.float32),
        oscillator_strength=np.asarray(gfs, dtype=np.float32),
        n_lower=np.asarray(nblo, dtype=np.int16),
        n_upper=np.asarray(nbup, dtype=np.int16),
        ion_index=np.asarray(nelion, dtype=np.int16),
        line_type=line_type_array,
        continuum_index=np.asarray(ncon, dtype=np.int16),
        element_index=np.asarray(nelionx, dtype=np.int16),
        gamma_rad=np.asarray(gamma_r, dtype=np.float32),
        gamma_stark=np.asarray(gamma_s, dtype=np.float32),
        gamma_vdw=np.asarray(gamma_w, dtype=np.float32),
        nbuff=np.asarray(nbuff_vals, dtype=np.int32),
        limb=np.asarray(limb_vals, dtype=np.int32),
        wing_type=_classify_line_types(line_type_array),
    )


def build_from_catalog(
    catalog: atomic.LineCatalog,
    wlbeg: float,
    wlend: float,
    resolution: float,
) -> Fort19Data:
    """Generate fort.19-equivalent metadata from the atomic catalog (rgfall.for logic)."""

    if len(catalog.records) == 0:
        return Fort19Data(
            wavelength_vacuum=np.array([], dtype=np.float64),
            energy_lower=np.array([], dtype=np.float32),
            oscillator_strength=np.array([], dtype=np.float32),
            n_lower=np.array([], dtype=np.int16),
            n_upper=np.array([], dtype=np.int16),
            ion_index=np.array([], dtype=np.int16),
            line_type=np.array([], dtype=np.int16),
            continuum_index=np.array([], dtype=np.int16),
            element_index=np.array([], dtype=np.int16),
            gamma_rad=np.array([], dtype=np.float32),
            gamma_stark=np.array([], dtype=np.float32),
            gamma_vdw=np.array([], dtype=np.float32),
            nbuff=np.array([], dtype=np.int32),
            limb=np.array([], dtype=np.int32),
            wing_type=np.array([], dtype=object),
        )

    ratio = 1.0 + 1.0 / resolution
    ratiolg = math.log(ratio)
    # Match rgfall.for integer assignment:
    #   IXWLBEG=DLOG(WLBEG)/RATIOLG
    # where IXWLBEG is INTEGER (floor for positive values).
    ixwlbeg = math.floor(math.log(wlbeg) / ratiolg)
    if math.exp(ixwlbeg * ratiolg) < wlbeg:
        ixwlbeg += 1

    delfactor = 1.0 if wlbeg <= 500.0 else wlbeg / 500.0

    wavelengths: list[float] = []
    energies: list[float] = []
    gfs: list[float] = []
    nblo: list[int] = []
    nbup: list[int] = []
    nelion: list[int] = []
    linetype: list[int] = []
    ncon: list[int] = []
    nelionx: list[int] = []
    gamma_r: list[float] = []
    gamma_s: list[float] = []
    gamma_w: list[float] = []
    nbuff_vals: list[int] = []
    limb_vals: list[int] = []

    for rec in catalog.records:
        wlvac = float(rec.wavelength)

        # Match rgfall.for lines 145-147:
        #   LIM=MIN(8-LINESIZE,7)
        #   IF(CODE.EQ.1.)LIM=1
        linesize = rec.line_size if rec.line_size > 0 else 0
        lim = min(8 - linesize, 7)
        code_for_lim = float(rec.code) if rec.code > 0.0 else 0.0
        if abs(code_for_lim - 1.0) < 1.0e-6:
            lim = 1
        margin = _DELLIM[lim - 1] * delfactor
        if wlvac < wlbeg - margin or wlvac > wlend + margin:
            continue

        line_type = int(rec.line_type)
        gf_linear = 10.0 ** float(rec.log_gf)

        if rec.labelp.strip().upper().startswith("CONTINUU"):
            nlast = int(rec.xjp) if rec.xjp > 0.0 else int(rec.n_upper)
            line_type = nlast
            gf_linear *= (2.0 * float(rec.xj) + 1.0)

        # rgfall.for line 150: coronal approximation lines are skipped.
        if line_type == 2:
            continue

        code = float(rec.code) if rec.code > 0.0 else 0.0
        if code <= 0.0:
            # Fallback: derive CODE from element + ion stage
            try:
                nelem = atomic._ELEMENT_SYMBOLS.index(rec.element)
            except ValueError:
                nelem = 0
            if nelem > 0:
                code = nelem + (rec.ion_stage - 1) * 0.01

        nelem = int(code + 1.0e-6) if code > 0.0 else 0
        icharge = int((code - nelem) * 100.0 + 0.1) if code > 0.0 else 0
        zeff = icharge + 1
        nelion_val = nelem * 6 - 6 + int(zeff)
        if nelem > 19 and nelem < 29 and icharge > 5:
            nelion_val = 6 * (nelem + icharge * 10 - 30) - 1

        nelionx_val = 0
        if code > 0.0:
            match = np.nonzero(np.isclose(_CODEX, code, rtol=0.0, atol=1e-3))[0]
            if match.size > 0:
                nelionx_val = int(match[0]) + 1

        nblo_val = abs(int(rec.n_lower))
        nbup_val = abs(int(rec.n_upper))
        ncon_val = rec.iso2 if rec.iso1 == 0 and rec.iso2 > 0 else 0

        ixwl = math.log(wlvac) / ratiolg + 0.5
        nbuff = int(ixwl) - int(ixwlbeg) + 1

        # rgfall routing:
        # - TYPE=1 or TYPE>3 => fort.19
        # - otherwise lines with NBLO+NBUP != 0 => fort.19
        include = (line_type == 1 or line_type > 3 or (nblo_val + nbup_val) != 0)

        if not include:
            continue

        freq_hz = _C_LIGHT_NM / max(wlvac, 1.0e-30)
        denom = 12.5664 * freq_hz
        if line_type == 1 or line_type > 3:
            # fort.19 TYPE=1 / TYPE>3 keep GAMMA* on the original (unnormalized)
            # rgfall scale. The atomic parser stores normalized gamma values to
            # match fort.12 usage, so recover raw fort.19 scale here.
            gammar = (
                10.0 ** rec.gamma_rad_log
                if rec.gamma_rad_log != 0.0
                else rec.gamma_rad * denom
            )
            if rec.gamma_stark_log > 0.0:
                # rgfall.for line 170: AUTO with positive GS encodes negative ASHORE.
                gammas = -10.0 ** (-rec.gamma_stark_log)
            elif rec.gamma_stark_log < 0.0:
                gammas = 10.0 ** rec.gamma_stark_log
            else:
                gammas = rec.gamma_stark * denom
            gammaw = (
                10.0 ** rec.gamma_vdw_log
                if rec.gamma_vdw_log != 0.0
                else rec.gamma_vdw * denom
            )
        else:
            # fort.19 TYPE<=3 (except TYPE=1) uses normalized damping
            # constants, same scale as fort.12.
            gammar = rec.gamma_rad
            gammas = rec.gamma_stark
            gammaw = rec.gamma_vdw

        # rgfall.for EQUIVALENCE (GF,G,CGF): line 267 assignment to CGF
        # aliases GF for TYPE<=3 (except TYPE=1), before WRITE(19).
        gf_for_write = gf_linear
        if line_type != 1 and line_type <= 3:
            freq_hz = _C_LIGHT_NM / max(wlvac, 1.0e-30)
            gf_for_write = _CGF_FACTOR * gf_linear / freq_hz

        wavelengths.append(wlvac)
        energies.append(float(rec.excitation_energy))
        gfs.append(float(gf_for_write))
        nblo.append(int(nblo_val))
        nbup.append(int(nbup_val))
        nelion.append(int(nelion_val))
        linetype.append(int(line_type))
        ncon.append(int(ncon_val))
        nelionx.append(int(nelionx_val))
        gamma_r.append(float(gammar))
        gamma_s.append(float(gammas))
        gamma_w.append(float(gammaw))
        nbuff_vals.append(int(nbuff))
        limb_vals.append(int(lim))

    line_type_array = np.asarray(linetype, dtype=np.int16)
    return Fort19Data(
        wavelength_vacuum=np.asarray(wavelengths, dtype=np.float64),
        energy_lower=np.asarray(energies, dtype=np.float32),
        oscillator_strength=np.asarray(gfs, dtype=np.float32),
        n_lower=np.asarray(nblo, dtype=np.int16),
        n_upper=np.asarray(nbup, dtype=np.int16),
        ion_index=np.asarray(nelion, dtype=np.int16),
        line_type=line_type_array,
        continuum_index=np.asarray(ncon, dtype=np.int16),
        element_index=np.asarray(nelionx, dtype=np.int16),
        gamma_rad=np.asarray(gamma_r, dtype=np.float32),
        gamma_stark=np.asarray(gamma_s, dtype=np.float32),
        gamma_vdw=np.asarray(gamma_w, dtype=np.float32),
        nbuff=np.asarray(nbuff_vals, dtype=np.int32),
        limb=np.asarray(limb_vals, dtype=np.int32),
        wing_type=_classify_line_types(line_type_array),
    )


__all__ = ["FORT19_BUILD_LOGIC_VERSION", "Fort19Data", "Fort19WingType", "load"]
