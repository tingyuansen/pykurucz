"""Fortran `fort.12` selected-line record decoding.

`SELECTLINES` writes unit 12 records as `WRITE(12) IIIIIII`, where:
- `IIIIIII` is `INTEGER*4(4)`
- equivalenced with `(IWL, IELION, IELO, IGFLOG, IGR, IGS, IGW)`,
  where `IWL` is INTEGER*4 and the remaining six are INTEGER*2.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SelectedLineRecords:
    iwl: np.ndarray
    ielion: np.ndarray
    ielo: np.ndarray
    igflog: np.ndarray
    igr: np.ndarray
    igs: np.ndarray
    igw: np.ndarray

    @property
    def size(self) -> int:
        return int(self.iwl.size)


@dataclass
class XLineRecords:
    """Decoded `fort.19` records used by `XLINOP`."""

    wlvac: np.ndarray
    elo: np.ndarray
    gf: np.ndarray
    nblo: np.ndarray
    nbup: np.ndarray
    nelion: np.ndarray
    type_code: np.ndarray
    ncon: np.ndarray
    nelionx: np.ndarray
    gammar: np.ndarray
    gammas: np.ndarray
    gammaw: np.ndarray
    iwl: np.ndarray
    lim: np.ndarray

    @property
    def size(self) -> int:
        return int(self.iwl.size)


def _decode_six_int16(words123: np.ndarray, *, swap_pairs: bool) -> np.ndarray:
    s16 = words123.view(np.int16).reshape(-1, 6)
    if not swap_pairs:
        return s16
    # Pair-swap fallback for non-native halfword ordering.
    return s16[:, [1, 0, 3, 2, 5, 4]]


def _plausibility_score(x: np.ndarray) -> int:
    # Expected rough ranges in atlas12 line lists:
    # IGFLOG/IGR/IGS/IGW are tablog indices ~1..32768.
    score = 0
    if x.shape[1] != 6:
        return score
    igflog = x[:, 2]
    igr = x[:, 3]
    igs = x[:, 4]
    igw = x[:, 5]
    for arr in (igflog, igr, igs, igw):
        score += int(np.count_nonzero((arr > 0) & (arr < 32768)))
    return score


def read_selected_lines(path: Path) -> SelectedLineRecords:
    """Read Fortran-selected line records from `fort.12`."""

    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        return SelectedLineRecords(
            iwl=np.zeros(0, dtype=np.int32),
            ielion=np.zeros(0, dtype=np.int16),
            ielo=np.zeros(0, dtype=np.int16),
            igflog=np.zeros(0, dtype=np.int16),
            igr=np.zeros(0, dtype=np.int16),
            igs=np.zeros(0, dtype=np.int16),
            igw=np.zeros(0, dtype=np.int16),
        )
    if raw.size % 4 != 0:
        raise ValueError(
            f"Invalid fort.12 word count {raw.size}; expected a multiple of 4 int32 words."
        )

    words = raw.reshape(-1, 4)
    iwl = words[:, 0].astype(np.int32, copy=False)
    w123 = words[:, 1:4].astype(np.int32, copy=False)

    dec_native = _decode_six_int16(w123, swap_pairs=False)
    dec_swapped = _decode_six_int16(w123, swap_pairs=True)
    dec = dec_native
    if _plausibility_score(dec_swapped) > _plausibility_score(dec_native):
        dec = dec_swapped

    return SelectedLineRecords(
        iwl=iwl,
        ielion=dec[:, 0].astype(np.int16, copy=False),
        ielo=dec[:, 1].astype(np.int16, copy=False),
        igflog=dec[:, 2].astype(np.int16, copy=False),
        igr=dec[:, 3].astype(np.int16, copy=False),
        igs=dec[:, 4].astype(np.int16, copy=False),
        igw=dec[:, 5].astype(np.int16, copy=False),
    )


def _decode_xline_words_60(raw_bytes: bytes) -> XLineRecords:
    """Decode XLINOP line records from 60-byte raw data.

    XLINOP (atlas12.for line 15030) uses IMPLICIT REAL*4 (A-H,O-Z) with
    explicit REAL*8 for WLVAC.  Record layout (60 bytes per line):
      - WLVAC:  float64 LE (8 bytes)
      - ELO:    float32 LE (4 bytes)
      - GF:     float32 LE (4 bytes)
      - NBLO, NBUP, NELION, TYPE, NCON, NELIONX: int32 LE (6 x 4 bytes)
      - GAMMAR, GAMMAS, GAMMAW: float32 LE (3 x 4 bytes)
      - IWL:    int32 LE (4 bytes)
      - LIM:    int32 LE (4 bytes)
    Total = 8+4+4+24+12+4+4 = 60 bytes.
    """
    n = len(raw_bytes) // 60
    if n == 0:
        return XLineRecords(
            wlvac=np.zeros(0, dtype=np.float64),
            elo=np.zeros(0, dtype=np.float64),
            gf=np.zeros(0, dtype=np.float64),
            nblo=np.zeros(0, dtype=np.int32),
            nbup=np.zeros(0, dtype=np.int32),
            nelion=np.zeros(0, dtype=np.int32),
            type_code=np.zeros(0, dtype=np.int32),
            ncon=np.zeros(0, dtype=np.int32),
            nelionx=np.zeros(0, dtype=np.int32),
            gammar=np.zeros(0, dtype=np.float64),
            gammas=np.zeros(0, dtype=np.float64),
            gammaw=np.zeros(0, dtype=np.float64),
            iwl=np.zeros(0, dtype=np.int32),
            lim=np.zeros(0, dtype=np.int32),
        )
    dt = np.dtype([
        ("wlvac", "<f8"),
        ("elo", "<f4"),
        ("gf", "<f4"),
        ("nblo", "<i4"),
        ("nbup", "<i4"),
        ("nelion", "<i4"),
        ("type_code", "<i4"),
        ("ncon", "<i4"),
        ("nelionx", "<i4"),
        ("gammar", "<f4"),
        ("gammas", "<f4"),
        ("gammaw", "<f4"),
        ("iwl", "<i4"),
        ("lim", "<i4"),
    ])
    arr = np.frombuffer(raw_bytes[: n * 60], dtype=dt)
    return XLineRecords(
        wlvac=arr["wlvac"].astype(np.float64),
        elo=arr["elo"].astype(np.float64),
        gf=arr["gf"].astype(np.float64),
        nblo=arr["nblo"].astype(np.int32),
        nbup=arr["nbup"].astype(np.int32),
        nelion=arr["nelion"].astype(np.int32),
        type_code=arr["type_code"].astype(np.int32),
        ncon=arr["ncon"].astype(np.int32),
        nelionx=arr["nelionx"].astype(np.int32),
        gammar=arr["gammar"].astype(np.float64),
        gammas=arr["gammas"].astype(np.float64),
        gammaw=arr["gammaw"].astype(np.float64),
        iwl=arr["iwl"].astype(np.int32),
        lim=arr["lim"].astype(np.int32),
    )


def _decode_xline_words(words_i4: np.ndarray) -> XLineRecords:
    """Decode XLINOP line records from 14-word (56-byte) int32 array.

    Legacy 14-word format where WLVAC is stored as float32 (first int32 word).
    Used by older binary files and the synthetic test paths.
    """
    if words_i4.ndim != 2 or words_i4.shape[1] != 14:
        raise ValueError("XLINOP words must have shape (N,14)")
    words_f4 = words_i4.view(np.float32)
    return XLineRecords(
        wlvac=words_f4[:, 0].astype(np.float64),
        elo=words_f4[:, 1].astype(np.float64),
        gf=words_f4[:, 2].astype(np.float64),
        nblo=words_i4[:, 3].astype(np.int32),
        nbup=words_i4[:, 4].astype(np.int32),
        nelion=words_i4[:, 5].astype(np.int32),
        type_code=words_i4[:, 6].astype(np.int32),
        ncon=words_i4[:, 7].astype(np.int32),
        nelionx=words_i4[:, 8].astype(np.int32),
        gammar=words_f4[:, 9].astype(np.float64),
        gammas=words_f4[:, 10].astype(np.float64),
        gammaw=words_f4[:, 11].astype(np.float64),
        iwl=words_i4[:, 12].astype(np.int32),
        lim=words_i4[:, 13].astype(np.int32),
    )


def _read_nlteline_with_markers_60(raw: bytes) -> bytes:
    """Extract the raw data bytes from a Fortran sequential fort.19 with 60-byte records."""
    out_parts: list[bytes] = []
    pos = 0
    n = len(raw)
    while pos + 8 <= n:
        nbytes = int.from_bytes(raw[pos : pos + 4], "little", signed=True)
        pos += 4
        if nbytes != 60:
            raise ValueError(f"Unexpected fort.19 record size {nbytes}; expected 60 bytes (XLINOP REAL*4 format)")
        if pos + nbytes + 4 > n:
            raise ValueError("Record exceeds file bounds in fort.19")
        out_parts.append(raw[pos : pos + nbytes])
        pos += nbytes
        tail = int.from_bytes(raw[pos : pos + 4], "little", signed=True)
        pos += 4
        if tail != nbytes:
            raise ValueError("Mismatched Fortran record markers in fort.19")
    if pos != n:
        raise ValueError("Trailing bytes in fort.19 stream")
    return b"".join(out_parts)


def _read_nlteline_with_markers(raw: bytes, *, endian: str) -> np.ndarray:
    words: list[np.ndarray] = []
    pos = 0
    n = len(raw)
    marker_dtype = np.dtype(f"{endian}i4")
    word_dtype = np.dtype(f"{endian}i4")
    while pos + 8 <= n:
        nbytes = int(np.frombuffer(raw, dtype=marker_dtype, count=1, offset=pos)[0])
        pos += 4
        if nbytes <= 0 or pos + nbytes + 4 > n:
            raise ValueError("Invalid Fortran record marker in fort.19")
        if nbytes != 56:
            raise ValueError(f"Unexpected fort.19 record size {nbytes}; expected 56 bytes")
        rec = np.frombuffer(raw, dtype=word_dtype, count=14, offset=pos).astype(np.int32, copy=False)
        pos += nbytes
        tail = int(np.frombuffer(raw, dtype=marker_dtype, count=1, offset=pos)[0])
        pos += 4
        if tail != nbytes:
            raise ValueError("Mismatched Fortran record markers in fort.19")
        words.append(rec)
    if pos != n:
        raise ValueError("Trailing bytes in fort.19 stream")
    if not words:
        return np.zeros((0, 14), dtype=np.int32)
    return np.vstack(words)


def _try_decode_nlteline(raw: bytes) -> XLineRecords:
    # Case 1: canonical 60-byte records with Fortran markers (atlas12 XLINOP REAL*4 format).
    try:
        data_bytes = _read_nlteline_with_markers_60(raw)
        if len(data_bytes) % 60 == 0:
            rec = _decode_xline_words_60(data_bytes)
            if rec.wlvac.size == 0 or (np.isfinite(rec.wlvac).all() and float(np.nanmax(rec.wlvac)) > 0.0):
                return rec
    except Exception:
        pass
    # Case 2: fixed 60-byte records without Fortran markers.
    if len(raw) % 60 == 0:
        rec = _decode_xline_words_60(raw)
        if rec.wlvac.size == 0 or (np.isfinite(rec.wlvac).all() and float(np.nanmax(rec.wlvac)) > 0.0):
            return rec
    # Case 3: fixed records without record markers (legacy 56-byte format).
    if len(raw) % 56 == 0:
        for endian in ("<", ">"):
            words = np.frombuffer(raw, dtype=np.dtype(f"{endian}i4")).reshape(-1, 14).astype(np.int32, copy=False)
            rec = _decode_xline_words(words)
            if rec.wlvac.size == 0:
                return rec
            if np.isfinite(rec.wlvac).all() and np.nanmax(rec.wlvac) > 0.0:
                return rec
    # Case 4: sequential unformatted with 4-byte markers (legacy 56-byte format).
    for endian in ("<", ">"):
        try:
            words = _read_nlteline_with_markers(raw, endian=endian)
        except Exception:
            continue
        rec = _decode_xline_words(words)
        if rec.wlvac.size == 0:
            return rec
        if np.isfinite(rec.wlvac).all() and np.nanmax(rec.wlvac) > 0.0:
            return rec
    raise ValueError("Unable to decode fort.19 nlteline records")


def read_nlteline_records(path: Path) -> XLineRecords:
    """Read `fort.19` records for `XLINOP`.

    Supports:
    - fixed 56-byte records (14 x 4-byte words), and
    - sequential unformatted records with 4-byte Fortran markers.
    """

    raw = path.read_bytes()
    if not raw:
        return XLineRecords(
            wlvac=np.zeros(0, dtype=np.float64),
            elo=np.zeros(0, dtype=np.float64),
            gf=np.zeros(0, dtype=np.float64),
            nblo=np.zeros(0, dtype=np.int32),
            nbup=np.zeros(0, dtype=np.int32),
            nelion=np.zeros(0, dtype=np.int32),
            type_code=np.zeros(0, dtype=np.int32),
            ncon=np.zeros(0, dtype=np.int32),
            nelionx=np.zeros(0, dtype=np.int32),
            gammar=np.zeros(0, dtype=np.float64),
            gammas=np.zeros(0, dtype=np.float64),
            gammaw=np.zeros(0, dtype=np.float64),
            iwl=np.zeros(0, dtype=np.int32),
            lim=np.zeros(0, dtype=np.int32),
        )
    return _try_decode_nlteline(raw)

