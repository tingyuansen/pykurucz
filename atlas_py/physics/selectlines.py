"""Python port of `SELECTLINES` from `atlas12.for` (lines 14430-14943).

``SELECTLINES`` pre-filters the Kurucz binary line catalogs (``fort.11``,
``fort.111``, ``fort.21``, ``fort.31``, ``fort.41``, ``fort.51``, ``fort.61``)
and writes a compact ``fort.12`` containing only lines with significant
opacity contribution.

The selection criterion (per line) is:

  CENRATIO = (0.026538/1.77245) * TABLOG[IGFLOG] * XNFDOPMAX[NELION, NU] / FREQ4

  Accept if  CENRATIO >= 1.0
         AND CENRATIO * exp(-TABLOG[IELO] * HCKT[NRHOX-1]) >= 1.0

where XNFDOPMAX[NELION, NU] = max_J(XNFDOP[J, NELION] / TABCONT[J, NU]).

Molecular lists include an isotope-correction offset applied to IGFLOG.
"""

from __future__ import annotations

import io
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

# SELECTLINES constants from atlas12.for
_CGF_SCALE = 0.026538 / 1.77245  # = 0.014999...
_RATIOLG = math.log(1.0 + 1.0 / 2_000_000.0)

# MOLCODES and ISOX (atlas12.for lines 14487-14733)
_MOLCODES: tuple[int, ...] = (
    8410, 8411, 8460, 8461, 8470, 8471, 8480, 8481, 8482, 8510, 8511,
    8512, 8530, 8531, 8532, 8580, 8581, 8582, 8583, 8584, 8620, 8621,
    8622, 8623, 8640, 8641, 8642, 8643, 8680, 8681, 8682, 8690, 8691,
    8692, 8693, 8700, 8701, 8702, 8703, 8704, 8705, 8890, 8891, 8892,
    8896, 8960,
)

# Isotope log(gf) offsets for diatomic molecules (atlas12.for lines 14643-14733)
_ISOX: tuple[int, ...] = (
    0,       # H2
    -4469,   # HDD
    -5,      # 12CH
    -1955,   # 13CH
    -2,      # 14NH
    -2444,   # 15NH
    -1,      # 16OH
    -3398,   # 17OH
    -2690,   # 18OH
    -105,    # 24MgH
    -996,    # 25MgH
    -947,    # 26MgH
    -35,     # 28SiH
    -1331,   # 29SiH
    -1516,   # 30SiH
    -13,     # 40CaH
    -2189,   # 42CaH
    -2870,   # 43CaH
    -1681,   # 44CaH
    -4398,   # 46CaH
    -1362,   # 50CrH
    -77,     # 52CrH
    -1022,   # 53CrH
    -1626,   # 54CrH
    -1237,   # 54FeH
    -38,     # 56FeH
    -1658,   # 57FeH
    -2553,   # 58FeH
    -5 + -5,         # 12C12C
    -5 + -1955,      # 12C13C
    -1955 + -1955,   # 13C13C
    -5 + -2,         # 12C14N
    -1955 + -2,      # 13C14N
    -5 + -2444,      # 12C15N
    -1955 + -2444,   # 13C15N
    -5 + -1,         # 12C16O
    -1955 + -1,      # 13C16O
    -5 + -3398,      # 12C17O
    -1955 + -3398,   # 13C17O
    -5 + -2690,      # 12C18O
    -1955 + -2690,   # 13C18O
    -1 + -35,        # 28Si16O
    -1 + -1331,      # 29Si16O
    -1 + -1516,      # 30Si16O
    -2690 + -35,     # 28Si18O
    -1 + -1,         # 51V16O (note: ISOX(46) in Fortran uses -001-001 which is -(1)-(1)=-2)
)


def _build_tablog() -> np.ndarray:
    """TABLOG(32768) = 10^((i - 16384) * 0.001), i = 1..32768."""
    i = np.arange(1, 32769, dtype=np.float64)
    return 10.0 ** ((i - 16384.0) * 0.001)


@dataclass
class SelectionCounts:
    """Number of lines selected from each catalog."""
    lowlines: int = 0
    lowlines_observed: int = 0
    hilines: int = 0
    diatomics: int = 0
    tio: int = 0
    h2o: int = 0
    h3plus: int = 0

    @property
    def total(self) -> int:
        return (self.lowlines + self.lowlines_observed + self.hilines
                + self.diatomics + self.tio + self.h2o + self.h3plus)


def compute_xnfdopmax(
    xnfdop: np.ndarray,
    tabcont: np.ndarray,
) -> np.ndarray:
    """Compute XNFDOPMAX(NELION, NU) = max_J(XNFDOP[J, NELION] / TABCONT[J, NU]).

    Parameters
    ----------
    xnfdop:
        Shape ``(nrhox, mion)``, REAL*4 Doppler populations.
    tabcont:
        Shape ``(nrhox, 344)``, REAL*4 continuum opacity table.

    Returns
    -------
    xnfdopmax:
        Shape ``(mion, 344)`` float32 array.
    """
    nrhox, mion = xnfdop.shape
    _, n344 = tabcont.shape
    tabcont_safe = np.where(tabcont > 0.0, tabcont, np.inf)
    xnfdopmax = np.empty((mion, n344), dtype=np.float32)
    for nu in range(n344):
        tc = tabcont_safe[:, nu].reshape(-1, 1)  # (nrhox, 1)
        ratio = xnfdop / tc  # (nrhox, mion)
        xnfdopmax[:, nu] = ratio.max(axis=0)
    return xnfdopmax


def _read_fixed_records(path: Path) -> np.ndarray:
    """Read a fixed-record RECORDTYPE='FIXED' RECL=4 Fortran file.

    Each record is exactly 4 × 4-byte words = 16 bytes, with no Fortran
    record-length markers. Returns shape (N, 4) int32.
    """
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size % 4 != 0:
        raise ValueError(
            f"{path}: word count {raw.size} not a multiple of 4."
        )
    return raw.reshape(-1, 4)


def _read_sequential_16byte_records(path: Path) -> np.ndarray:
    """Read sequential unformatted records containing IIIIIII payloads.

    Some Kurucz catalogs (notably `diatomicspacksrt.bin`) are stored as
    sequential unformatted records with 4-byte record-length markers around
    each 16-byte payload. This decoder enforces marker=16 and returns
    shape (N, 4) int32 payload words.
    """
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size % 6 != 0:
        raise ValueError(
            f"{path}: int32 word count {raw.size} not a multiple of 6 for sequential records."
        )
    recs = raw.reshape(-1, 6)
    if not np.all((recs[:, 0] == 16) & (recs[:, 5] == 16)):
        bad = int(np.count_nonzero((recs[:, 0] != 16) | (recs[:, 5] != 16)))
        raise ValueError(
            f"{path}: {bad} records do not have 16-byte Fortran markers."
        )
    return recs[:, 1:5].astype(np.int32, copy=False)


def _read_h2o_records(path: Path) -> np.ndarray:
    """Read H2O line list file as Fortran `READ(51) IWL, IELO, IGFLOG` records.

    atlas12.for opens unit 51 with RECORDTYPE='FIXED', RECL=2 and reads
    ``IWL, IELO, IGFLOG`` (line 14993). In this dataset, each fixed record is
    8 bytes: int32 ``IWL`` followed by packed int16 ``IELO`` and int16
    ``IGFLOG``.

    Returns shape (N, 3) int32 columns: [IWL, IELO, IGFLOG].
    """
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size % 2 != 0:
        raise ValueError(
            f"{path}: int32 word count {raw.size} not a multiple of 2 for H2O records."
        )
    words = raw.reshape(-1, 2)
    iwl = words[:, 0].astype(np.int32)
    pair16 = np.ascontiguousarray(words[:, 1]).view(np.int16).reshape(-1, 2)
    ielo = pair16[:, 0].astype(np.int32)
    igflog = pair16[:, 1].astype(np.int32)
    return np.column_stack((iwl, ielo, igflog))


def _unpack_iiiiiii(words: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unpack IIIIIII(4) int32 array into (IWL, IELION, IELO, IGFLOG, IGR, IGS, IGW).

    Layout:
    - words[:, 0] = IWL (int32)
    - words[:, 1:4] viewed as int16: [IELION, IELO, IGFLOG, IGR, IGS, IGW]
    """
    iwl = words[:, 0]
    s16 = words[:, 1:4].view(np.int16).reshape(-1, 6)
    return (iwl, s16[:, 0], s16[:, 1], s16[:, 2], s16[:, 3], s16[:, 4], s16[:, 5])


def _process_standard_catalog(
    words: np.ndarray,
    *,
    iwavetab: np.ndarray,
    tablog: np.ndarray,
    xnfdopmax: np.ndarray,
    hckt_deepest: float,
    nelion_override: int | None = None,
    igflog_offset_array: np.ndarray | None = None,
    nu_floor: int = 0,
    xnf_min: float = 0.0,
    igs_override: int | None = None,
    igw_override: int | None = None,
    output: io.RawIOBase,
) -> tuple[int, int]:
    """Process a single standard-format line catalog and write selected records.

    Parameters
    ----------
    words:
        Shape (N, 4) int32, raw file records.
    iwavetab:
        Shape (344,) int64, IWAVETAB integer-encoded wavelengths.
    tablog:
        Shape (32768,) float64, precomputed TABLOG.
    xnfdopmax:
        Shape (mion, 344) float32.
    hckt_deepest:
        HCKT at deepest layer (NRHOX).
    nelion_override:
        If not None, use this NELION for all lines (e.g., TIO uses 895).
    igflog_offset_array:
        If not None, shape (N,) int32 offsets added to IGFLOG (for molecules).
    output:
        Open binary write handle for fort.12.

    Returns
    -------
    count:
        Number of lines written.
    """
    n344 = iwavetab.shape[0]
    n_lines = words.shape[0]
    iwl, ielion, ielo, igflog, igr, igs, igw = _unpack_iiiiiii(words)
    iwl = iwl.astype(np.int64)
    ielion_i32 = ielion.astype(np.int32)
    ielo_i32 = ielo.astype(np.int32)
    igflog_i32 = igflog.astype(np.int32)

    if igflog_offset_array is not None:
        igflog_i32 = np.maximum(igflog_i32 + igflog_offset_array, 1)

    # Build NELION for each line (1-based index into xnfdopmax)
    if nelion_override is not None:
        nelion_arr = np.full(n_lines, nelion_override - 1, dtype=np.int32)  # 0-based
    else:
        nelion_arr = (np.abs(ielion_i32) // 10 - 1).clip(0, xnfdopmax.shape[0] - 1)

    # Determine frequency bin NU for each line using sorted iwavetab
    # IWL < IWAVETAB(NU) means the line is in bin NU-1 (0-based: bin nu_0-based)
    # We need to find nu such that IWL < iwavetab[nu]
    # atlas12.for starts NU=1 and advances while IWL >= IWAVETAB(NU).
    # In Python (0-indexed): advance while IWL >= iwavetab[nu]; result bin is nu.
    # This is equivalent to: nu = np.searchsorted(iwavetab, iwl, side='right') - 1
    # IWAVETAB includes the sentinel at index 344 in Fortran (0-based: 343).
    # Keep that terminal bin reachable for exact parity.
    nu_arr = np.searchsorted(iwavetab, iwl, side='right')  # 0-based NU index
    nu_arr = np.clip(nu_arr, 0, n344 - 1)
    if nu_floor > 0:
        nu_arr = np.maximum(nu_arr, int(nu_floor))
    # Fortran advances NU monotonically and never decreases within a catalog pass.
    nu_arr = np.maximum.accumulate(nu_arr)

    # Clip table indices
    igflog_clipped = np.clip(igflog_i32 - 1, 0, len(tablog) - 1)
    ielo_clipped = np.clip(ielo_i32 - 1, 0, len(tablog) - 1)

    # FREQ4 from WAVETAB is not stored; we use precomputed from SELECTLINES inline.
    # In Fortran: FREQ4 = 2.99792458e17 / WAVETAB(NU)
    # We reconstruct from iwavetab: WAVETAB = EXP(IWAVETAB * RATIOLG)
    # But WAVETAB is already in the Python waveset. We need FREQ4 per bin.
    # For efficiency, precompute freq4 per bin from iwavetab.
    wavetab_from_i = np.exp(iwavetab * _RATIOLG).astype(np.float64)
    wavetab_from_i = np.where(wavetab_from_i > 0, wavetab_from_i, 1e-300)
    freq4_per_bin = (2.99792458e17 / wavetab_from_i).astype(np.float32)

    # Gather per-line arrays
    freq4 = freq4_per_bin[nu_arr]
    tlog_gf = tablog[igflog_clipped].astype(np.float32)
    tlog_elo = tablog[ielo_clipped].astype(np.float32)
    xnfdopmax_line = xnfdopmax[nelion_arr, nu_arr]

    # CENRATIO computation
    denom = np.where(freq4 > 0, freq4, 1e-37)
    cenratio = _CGF_SCALE * tlog_gf * xnfdopmax_line / denom

    # Selection criteria
    # 1. xnfdopmax > threshold (skip if essentially zero)
    valid_xnf = xnfdopmax_line > np.float32(xnf_min)
    # 2. CENRATIO >= 1.0
    valid_cr1 = cenratio >= 1.0
    # 3. CENRATIO * exp(-TABLOG[IELO] * HCKT[NRHOX-1]) >= 1.0
    exp_factor = np.exp(-tlog_elo * np.float32(hckt_deepest))
    valid_cr2 = cenratio * exp_factor >= 1.0

    selected = valid_xnf & valid_cr1 & valid_cr2
    sel_idx = np.where(selected)[0]

    need_repack = (
        igflog_offset_array is not None
        or igs_override is not None
        or igw_override is not None
    )

    if not need_repack:
        for i in sel_idx:
            output.write(words[i].tobytes())
    else:
        iwl_i32 = iwl.astype(np.int32, copy=False)
        igr_i32 = igr.astype(np.int32, copy=False)
        igs_i32 = np.full(n_lines, igs_override if igs_override is not None else 0, dtype=np.int32)
        if igs_override is None:
            igs_i32 = igs.astype(np.int32, copy=False)
        igw_i32 = np.full(n_lines, igw_override if igw_override is not None else 0, dtype=np.int32)
        if igw_override is None:
            igw_i32 = igw.astype(np.int32, copy=False)

        for i in sel_idx:
            buf = np.zeros(4, dtype=np.int32)
            buf[0] = int(iwl_i32[i])
            s16 = buf[1:4].view(np.int16)
            s16[0] = np.int16(int(ielion_i32[i]))
            s16[1] = np.int16(int(ielo_i32[i]))
            s16[2] = np.int16(int(igflog_i32[i]))
            s16[3] = np.int16(int(igr_i32[i]))
            s16[4] = np.int16(int(igs_i32[i]))
            s16[5] = np.int16(int(igw_i32[i]))
            output.write(buf.tobytes())

    nu_end = int(nu_arr.max()) if nu_arr.size > 0 else int(nu_floor)
    return int(sel_idx.size), nu_end


def _process_h2o_catalog(
    recs: np.ndarray,
    *,
    iwavetab: np.ndarray,
    tablog: np.ndarray,
    xnfdopmax: np.ndarray,
    hckt_deepest: float,
    igs_fixed: int,
    igw_fixed: int,
    gammar_coeff: float,
    output: io.RawIOBase,
) -> int:
    """Process the H2O line catalog (RECL=2 format, 3 int16 per record).

    Fortran atlas12.for lines 14831-14886.
    """
    n344 = iwavetab.shape[0]
    n_lines = recs.shape[0]
    iwl_raw = recs[:, 0].astype(np.int32)
    ielo_raw = recs[:, 1].astype(np.int32)
    igflog_raw = recs[:, 2].astype(np.int32)

    # Determine ISO from sign bits (atlas12.for lines 1843-1848)
    # ISO=1 if IELO>0 and IGFLOG>0
    # ISO=2 if IELO>0
    # ISO=3 if IGFLOG>0
    # ISO=4 otherwise
    iso = np.full(n_lines, 4, dtype=np.int32)
    iso = np.where(igflog_raw > 0, 3, iso)
    iso = np.where(ielo_raw > 0, 2, iso)
    iso = np.where((ielo_raw > 0) & (igflog_raw > 0), 1, iso)

    ielion_arr = -(9399 + iso)

    # ELO = ABS(IELO), KGFLOG = ABS(IGFLOG)
    elo = np.abs(ielo_raw).astype(np.float32)
    kgflog = np.abs(igflog_raw).astype(np.int32)

    # Isotope offsets: 1H1H16O=-1, 1H1H17O=-3398, 1H1H18O=-2690, 1H2H16O=-5000
    iso_offsets = np.array([-1, -3398, -2690, -5000], dtype=np.int32)
    gflog_adj = np.maximum(kgflog + iso_offsets[iso - 1], 1)

    # NELION = 940 (fixed for H2O)
    nelion_0based = np.full(n_lines, 939, dtype=np.int32)

    iwl_i64 = iwl_raw.astype(np.int64)
    nu_arr = np.searchsorted(iwavetab, iwl_i64, side='right')
    nu_arr = np.clip(nu_arr, 0, n344 - 1)
    # Mirror Fortran loop state: NU only increments within the H2O pass.
    nu_arr = np.maximum.accumulate(nu_arr)

    wavetab_from_i = np.exp(iwavetab * _RATIOLG).astype(np.float64)
    wavetab_from_i = np.where(wavetab_from_i > 0, wavetab_from_i, 1e-300)
    freq4_per_bin = (2.99792458e17 / wavetab_from_i).astype(np.float32)

    freq4 = freq4_per_bin[nu_arr]
    tlog_gf = tablog[np.clip(gflog_adj - 1, 0, len(tablog) - 1)].astype(np.float32)
    xnfdopmax_line = xnfdopmax[nelion_0based, nu_arr]

    denom = np.where(freq4 > 0, freq4, 1e-37)
    cenratio = _CGF_SCALE * tlog_gf * xnfdopmax_line / denom

    valid_xnf = xnfdopmax_line > 0.0
    valid_cr1 = cenratio >= 1.0
    # H2O uses ELO directly: exp(-ELO * HCKT[NRHOX-1])
    exp_factor = np.exp(-elo * np.float32(hckt_deepest))
    valid_cr2 = cenratio * exp_factor >= 1.0

    selected = valid_xnf & valid_cr1 & valid_cr2
    sel_idx = np.where(selected)[0]

    # Reconstruct IIIIIII for output: re-encode into 4 int32 words.
    # IWL as int32, then IELION/IELO/IGFLOG/IGR/IGS/IGW as int16.
    # IELO = int(log10(ELO)*1000 + 16384.5) (atlas12.for line 14871)
    # IGR computed from GAMMAR_COEFF (atlas12.for line 14838-14839)
    count = 0
    for i in sel_idx:
        nu_i = int(nu_arr[i])
        freq4_i = float(freq4[i])
        gammar_i = gammar_coeff * (freq4_i ** 2) * 0.001
        gr_i = max(1, min(32768, int(math.log10(max(gammar_i, 1e-300)) * 1000.0 + 16384.5)))
        ielo_enc = max(1, min(32768, int(math.log10(max(float(elo[i]), 1e-300)) * 1000.0 + 16384.5)))
        igf_i = int(gflog_adj[i])
        iel_i = int(ielion_arr[i])
        buf = np.zeros(4, dtype=np.int32)
        buf[0] = int(iwl_raw[i])
        s16 = buf[1:4].view(np.int16)
        s16[0] = np.int16(iel_i)
        s16[1] = np.int16(ielo_enc)
        s16[2] = np.int16(igf_i)
        s16[3] = np.int16(gr_i)
        s16[4] = np.int16(igs_fixed)
        s16[5] = np.int16(igw_fixed)
        output.write(buf.tobytes())
        count += 1

    return count


def selectlines(
    *,
    xnfdop: np.ndarray,
    tabcont: np.ndarray,
    iwavetab: np.ndarray,
    hckt: np.ndarray,
    fort11_path: Path | None = None,
    fort111_path: Path | None = None,
    fort21_path: Path | None = None,
    fort31_path: Path | None = None,
    fort41_path: Path | None = None,
    fort51_path: Path | None = None,
    fort61_path: Path | None = None,
    fort12_output: Path,
) -> SelectionCounts:
    """Port of Fortran ``SELECTLINES`` subroutine (atlas12.for 14430-14943).

    Reads Kurucz binary line catalogs, applies the CENRATIO selection
    criterion, and writes surviving records to ``fort12_output``.

    Parameters
    ----------
    xnfdop:
        Shape ``(nrhox, mion)`` float32 – Doppler-broadened populations.
    tabcont:
        Shape ``(nrhox, 344)`` float32 – continuum opacity table.
    iwavetab:
        Shape ``(344,)`` int64 – integer-encoded wavelength grid (from
        ``build_kapcont_wavetab()``).
    hckt:
        Shape ``(nrhox,)`` float64 – H/(kT) at each depth layer.
    fort11_path, fort111_path, fort21_path, fort31_path, fort41_path,
    fort51_path, fort61_path:
        Optional paths to each line-list binary file. Skipped if ``None``
        or if the file does not exist.
    fort12_output:
        Destination path for the selected-line binary output.

    Returns
    -------
    SelectionCounts
        Per-catalog selected-line counts.
    """
    tablog = _build_tablog()
    xnfdopmax = compute_xnfdopmax(
        xnfdop.astype(np.float32), tabcont.astype(np.float32)
    )
    hckt_deepest = float(hckt[-1])
    mion = xnfdop.shape[1]
    iwavetab_i64 = np.asarray(iwavetab, dtype=np.int64)

    counts = SelectionCounts()

    with open(fort12_output, "wb") as fout:
        # ----------------------------------------------------------------
        # LOWLINES (fort.11) – RECL=4 = 16 bytes/record
        # ----------------------------------------------------------------
        nu_after_lowlines = 0
        if fort11_path is not None and Path(fort11_path).exists():
            logger.info("SELECTLINES: reading lowlines from %s", fort11_path)
            words = _read_fixed_records(fort11_path)
            counts.lowlines, nu_after_lowlines = _process_standard_catalog(
                words,
                iwavetab=iwavetab_i64,
                tablog=tablog,
                xnfdopmax=xnfdopmax,
                hckt_deepest=hckt_deepest,
                nu_floor=nu_after_lowlines,
                xnf_min=1.0e-37,
                output=fout,
            )
            logger.info("SELECTLINES: %d lines from lowlines", counts.lowlines)

        # ----------------------------------------------------------------
        # LOWLINES observed (fort.111) – same format
        # ----------------------------------------------------------------
        if fort111_path is not None and Path(fort111_path).exists():
            logger.info("SELECTLINES: reading lowlines-observed from %s", fort111_path)
            words = _read_fixed_records(fort111_path)
            counts.lowlines_observed, _ = _process_standard_catalog(
                words,
                iwavetab=iwavetab_i64,
                tablog=tablog,
                xnfdopmax=xnfdopmax,
                hckt_deepest=hckt_deepest,
                xnf_min=1.0e-37,
                output=fout,
            )
            logger.info("SELECTLINES: %d lines from lowlines-obs", counts.lowlines_observed)

        # ----------------------------------------------------------------
        # HILINES (fort.21) – same format
        # ----------------------------------------------------------------
        if fort21_path is not None and Path(fort21_path).exists():
            logger.info("SELECTLINES: reading hilines from %s", fort21_path)
            words = _read_fixed_records(fort21_path)
            counts.hilines, _ = _process_standard_catalog(
                words,
                iwavetab=iwavetab_i64,
                tablog=tablog,
                xnfdopmax=xnfdopmax,
                hckt_deepest=hckt_deepest,
                output=fout,
            )
            logger.info("SELECTLINES: %d lines from hilines", counts.hilines)

        # ----------------------------------------------------------------
        # DIATOMICS (fort.31)
        # Isotope correction: IGFLOG = max(KGFLOG + ISOX[IMOL], 1)
        # NELION = ABS(IELION/10)
        # ----------------------------------------------------------------
        if fort31_path is not None and Path(fort31_path).exists():
            logger.info("SELECTLINES: reading diatomics from %s", fort31_path)
            words = _read_sequential_16byte_records(fort31_path)
            if words.shape[0] > 0:
                _, ielion_d, _, igflog_d, _, _, _ = _unpack_iiiiiii(words)
                molcodes = np.array(_MOLCODES, dtype=np.int32)
                isox_arr = np.array(_ISOX, dtype=np.int32)
                molcode_arr = np.abs(ielion_d.astype(np.int32))
                imol_arr = np.full(words.shape[0], 0, dtype=np.int32)
                for k, mc in enumerate(molcodes):
                    imol_arr[molcode_arr == mc] = k
                igflog_offset = isox_arr[imol_arr]
                counts.diatomics, _ = _process_standard_catalog(
                    words,
                    iwavetab=iwavetab_i64,
                    tablog=tablog,
                    xnfdopmax=xnfdopmax,
                    hckt_deepest=hckt_deepest,
                    igflog_offset_array=igflog_offset,
                    igs_override=1,
                    output=fout,
                )
            logger.info("SELECTLINES: %d lines from diatomics", counts.diatomics)

        # ----------------------------------------------------------------
        # TIO (fort.41)
        # NELION = 895 (fixed, 0-based: 894)
        # IGFLOG: offset by -1101, -1138, -131, -1259, or -1272 for isotopes
        # IGS=1, IGW=9384 (fixed)
        # ----------------------------------------------------------------
        if fort41_path is not None and Path(fort41_path).exists():
            logger.info("SELECTLINES: reading TIO from %s", fort41_path)
            words = _read_fixed_records(fort41_path)
            if words.shape[0] > 0:
                _, ielion_t, _, igflog_t, _, _, _ = _unpack_iiiiiii(words)
                iso_t = (np.abs(ielion_t.astype(np.int32)) - 8949).clip(1, 5)
                tio_offsets = np.array([-1101, -1138, -131, -1259, -1272], dtype=np.int32)
                igflog_offset_tio = tio_offsets[iso_t - 1]
                counts.tio, _ = _process_standard_catalog(
                    words,
                    iwavetab=iwavetab_i64,
                    tablog=tablog,
                    xnfdopmax=xnfdopmax,
                    hckt_deepest=hckt_deepest,
                    nelion_override=895,
                    igflog_offset_array=igflog_offset_tio,
                    igs_override=1,
                    igw_override=9384,
                    output=fout,
                )
            logger.info("SELECTLINES: %d lines from TIO", counts.tio)

        # ----------------------------------------------------------------
        # H2O (fort.51) – RECL=2 = 6 bytes/record, 3 int16
        # NELION = 940 (fixed)
        # IGS=1, IGW=9384 (fixed)
        # GAMMAR from freq: GAMMAR=2.474e-22*FREQ^2*0.001
        # ----------------------------------------------------------------
        if fort51_path is not None and Path(fort51_path).exists():
            logger.info("SELECTLINES: reading H2O from %s", fort51_path)
            recs = _read_h2o_records(fort51_path)
            if recs.shape[0] > 0:
                counts.h2o = _process_h2o_catalog(
                    recs,
                    iwavetab=iwavetab_i64,
                    tablog=tablog,
                    xnfdopmax=xnfdopmax,
                    hckt_deepest=hckt_deepest,
                    igs_fixed=1,
                    igw_fixed=9384,
                    gammar_coeff=2.474e-22,
                    output=fout,
                )
            logger.info("SELECTLINES: %d lines from H2O", counts.h2o)

        # ----------------------------------------------------------------
        # H3+ (fort.61) – same format as standard
        # NELION = 895 (same as TIO, 0-based: 894)
        # IGFLOG = max(KGFLOG - 1272, 1)
        # IGS=1, IGW=9384 (fixed)
        # ----------------------------------------------------------------
        if fort61_path is not None and Path(fort61_path).exists():
            logger.info("SELECTLINES: reading H3+ from %s", fort61_path)
            words = _read_fixed_records(fort61_path)
            if words.shape[0] > 0:
                igflog_offset_h3 = np.full(words.shape[0], -1272, dtype=np.int32)
                counts.h3plus, _ = _process_standard_catalog(
                    words,
                    iwavetab=iwavetab_i64,
                    tablog=tablog,
                    xnfdopmax=xnfdopmax,
                    hckt_deepest=hckt_deepest,
                    nelion_override=895,
                    igflog_offset_array=igflog_offset_h3,
                    igs_override=1,
                    igw_override=9384,
                    output=fout,
                )
            logger.info("SELECTLINES: %d lines from H3+", counts.h3plus)

    logger.info(
        "SELECTLINES complete: %d total lines -> %s",
        counts.total, fort12_output,
    )
    return counts
