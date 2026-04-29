"""Optional writers for legacy SYNTHE `tfort.*` tapes."""

from __future__ import annotations

import math
from pathlib import Path
import struct

import numpy as np

from . import atomic, fort19

_C_LIGHT_NM = 2.99792458e17
_FORT19_STRUCT = struct.Struct("<dffiiiiiifffii")


def _write_unformatted_record(handle, payload: bytes) -> None:
    nbytes = len(payload)
    marker = struct.pack("<i", nbytes)
    handle.write(marker)
    handle.write(payload)
    handle.write(marker)


def _infer_nelion(rec: atomic.LineRecord) -> int:
    code = float(rec.code)
    if code <= 0.0:
        return 0
    nelem = int(code + 1.0e-6)
    icharge = int((code - nelem) * 100.0 + 0.1)
    zeff = icharge + 1
    nelion = nelem * 6 - 6 + int(zeff)
    if nelem > 19 and nelem < 29 and icharge > 5:
        nelion = 6 * (nelem + icharge * 10 - 30) - 1
    return int(nelion)


def write_tfort12(
    output_path: Path,
    nbuff: np.ndarray,
    cgf: np.ndarray,
    nelion: np.ndarray,
    elo_cm: np.ndarray,
    gamma_rad: np.ndarray,
    gamma_stark: np.ndarray,
    gamma_vdw: np.ndarray,
) -> None:
    """Write `tfort.12` records from compiled metadata arrays."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fh:
        for i in range(nbuff.size):
            payload = struct.pack(
                "<ififfff",
                int(nbuff[i]),
                float(cgf[i]),
                int(nelion[i]),
                float(elo_cm[i]),
                float(gamma_rad[i]),
                float(gamma_stark[i]),
                float(gamma_vdw[i]),
            )
            _write_unformatted_record(fh, payload)


def write_tfort93(
    output_path: Path,
    wlbeg: float,
    wlend: float,
    resolution: float,
    cutoff: float,
) -> None:
    """Write minimal `tfort.93` containing grid-defining trailing doubles."""
    ratio = 1.0 + 1.0 / resolution
    ratiolg = math.log(ratio)
    payload = struct.pack("<6d", wlbeg, wlend, resolution, ratio, ratiolg, cutoff)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fh:
        _write_unformatted_record(fh, payload)


def write_tfort19(output_path: Path, data: fort19.Fort19Data) -> None:
    """Write `tfort.19` wing metadata records."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fh:
        for i in range(data.wavelength_vacuum.size):
            payload = _FORT19_STRUCT.pack(
                float(data.wavelength_vacuum[i]),
                float(data.energy_lower[i]),
                float(data.oscillator_strength[i]),
                int(data.n_lower[i]),
                int(data.n_upper[i]),
                int(data.ion_index[i]),
                int(data.line_type[i]),
                int(data.continuum_index[i]),
                int(data.element_index[i]),
                float(data.gamma_rad[i]),
                float(data.gamma_stark[i]),
                float(data.gamma_vdw[i]),
                int(data.nbuff[i]),
                int(data.limb[i]),
            )
            _write_unformatted_record(fh, payload)


def write_tfort14(output_path: Path, catalog: atomic.LineCatalog) -> None:
    """Write approximate compatibility `tfort.14` records from catalog lines."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fh:
        for rec in catalog.records:
            doubles = [0.0] * 14
            wl = float(rec.wavelength)
            doubles[0] = wl  # wavelength_air placeholder
            doubles[11] = wl  # wavelength_vac
            payload = bytearray(14 * 8 + 28 * 4)
            struct.pack_into("<14d", payload, 0, *doubles)
            offset = 14 * 8
            gf = 10.0 ** float(rec.log_gf)
            gamma_r = float(rec.gamma_rad)
            gamma_s = float(rec.gamma_stark)
            gamma_w = float(rec.gamma_vdw)
            struct.pack_into("<i", payload, offset, _infer_nelion(rec))
            offset += 4
            struct.pack_into("<f", payload, offset, gamma_r)
            offset += 4
            struct.pack_into("<f", payload, offset, gamma_s)
            offset += 4
            struct.pack_into("<f", payload, offset, gamma_w)
            offset += 4
            struct.pack_into("<i", payload, offset, 0)  # REF
            offset += 4
            struct.pack_into("<f", payload, offset, float(rec.n_lower))
            offset += 4
            struct.pack_into("<f", payload, offset, float(rec.n_upper))
            offset += 4
            struct.pack_into("<f", payload, offset, 0.0)  # ISO1
            offset += 4
            struct.pack_into("<f", payload, offset, 0.0)  # X1
            offset += 4
            struct.pack_into("<f", payload, offset, 0.0)  # ISO2
            offset += 4
            struct.pack_into("<f", payload, offset, 0.0)  # X2
            offset += 4
            struct.pack_into("<f", payload, offset, float(rec.log_gf))
            offset += 4
            struct.pack_into("<f", payload, offset, float(rec.xj))
            offset += 4
            struct.pack_into("<f", payload, offset, float(rec.xjp))
            offset += 4
            struct.pack_into("<f", payload, offset, float(rec.code))
            offset += 4
            struct.pack_into("<f", payload, offset, float(rec.excitation_energy))
            offset += 4
            struct.pack_into("<f", payload, offset, float(gf))
            offset += 4
            struct.pack_into("<f", payload, offset, float(rec.gamma_stark_log))
            offset += 4
            struct.pack_into("<f", payload, offset, float(rec.gamma_rad_log))
            offset += 4
            struct.pack_into("<f", payload, offset, float(rec.gamma_vdw_log))
            _write_unformatted_record(fh, bytes(payload))


def write_tfort20(output_path: Path, catalog: atomic.LineCatalog) -> None:
    """Write approximate compatibility `tfort.20` records with zero ALINE terms."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fh:
        for rec in catalog.records:
            payload = bytearray(14 * 8 + 28 * 4)
            doubles = [0.0] * 14
            doubles[11] = float(rec.wavelength)
            struct.pack_into("<14d", payload, 0, *doubles)
            floats = [0.0] * 28
            floats[13] = float(rec.n_lower)
            floats[14] = float(rec.n_upper)
            floats[16] = float(_infer_nelion(rec))
            # floats[20:] ALINE left as zeros for compatibility/debug.
            struct.pack_into("<28f", payload, 14 * 8, *floats)
            _write_unformatted_record(fh, bytes(payload))

