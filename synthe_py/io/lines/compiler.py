"""Self-contained Python line compiler contract and implementation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np

from . import atomic, fort19, parsed_cache
from .tfort import _ELEMENT_SYMBOLS

_C_LIGHT_NM = 2.99792458e17
_CGF_CONSTANT = 0.026538 / 1.77245
_COMPILED_CACHE_SCHEMA = 4
COMPILED_CACHE_LOGIC_VERSION = 3


@dataclass(frozen=True)
class LineCompilerContract:
    """Contract for Python-compiled metadata replacing runtime `tfort.*` reads."""

    wavelength_unit: str
    excitation_unit: str
    damping_unit: str
    nbuff_indexing: str
    cgf_definition: str
    notes: tuple[str, ...]


LINE_COMPILER_CONTRACT = LineCompilerContract(
    wavelength_unit="nm (vacuum)",
    excitation_unit="cm^-1 (lower-level energy)",
    damping_unit="Normalized gamma = gamma_linear / (4*pi*nu)",
    nbuff_indexing="1-based geometric grid index (Fortran-compatible)",
    cgf_definition="CGF = (0.026538/1.77245) * GF / FREQ",
    notes=(
        "Input: gfall + run grid config (wlbeg/wlend/resolution).",
        "Output arrays match fort.12 semantics for NBUFF/CGF/NELION/gammas.",
        "fort.19 wing metadata is generated in-memory via Python compiler logic.",
        "No runtime dependency on companion tfort.19/tfort.93 files.",
    ),
)


@dataclass(frozen=True)
class CompiledLineCatalog:
    """Compiled catalog and Fortran-equivalent metadata arrays."""

    catalog: atomic.LineCatalog
    fort19_data: fort19.Fort19Data
    nbuff: np.ndarray
    cgf: np.ndarray
    nelion: np.ndarray
    elo_cm: np.ndarray
    gamma_rad: np.ndarray
    gamma_stark: np.ndarray
    gamma_vdw: np.ndarray
    limb: np.ndarray


def _default_cache_dir(catalog_path: Path) -> Path:
    return catalog_path.parent / ".py_line_cache"


def _cache_key(
    catalog_path: Path,
    wlbeg: float,
    wlend: float,
    resolution: float,
    line_filter: bool,
) -> str:
    payload = compiled_cache_key_payload(
        catalog_path=catalog_path,
        wlbeg=wlbeg,
        wlend=wlend,
        resolution=resolution,
        line_filter=line_filter,
    )
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return digest[:24]


def compiled_cache_key_payload(
    catalog_path: Path,
    wlbeg: float,
    wlend: float,
    resolution: float,
    line_filter: bool,
) -> dict[str, object]:
    st = catalog_path.stat()
    return {
        "schema": _COMPILED_CACHE_SCHEMA,
        "logic_version": COMPILED_CACHE_LOGIC_VERSION,
        "source": str(catalog_path.resolve()),
        "size": int(st.st_size),
        "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
        "wlbeg": float(wlbeg),
        "wlend": float(wlend),
        "resolution": float(resolution),
        "line_filter": bool(line_filter),
        "iso_corr_enabled": bool(os.environ.get("PY_APPLY_ISO_CORR", "1") != "0"),
        "contract": LINE_COMPILER_CONTRACT.nbuff_indexing,
        "parsed_cache_logic_version": parsed_cache.PARSED_CACHE_LOGIC_VERSION,
        "fort19_build_logic_version": fort19.FORT19_BUILD_LOGIC_VERSION,
    }


def _cache_file_path(
    catalog_path: Path,
    wlbeg: float,
    wlend: float,
    resolution: float,
    line_filter: bool,
    cache_directory: Optional[Path],
) -> Path:
    cache_dir = (
        cache_directory
        if cache_directory is not None
        else _default_cache_dir(catalog_path)
    )
    return (
        cache_dir
        / f"compiled_lines_{_cache_key(catalog_path, wlbeg, wlend, resolution, line_filter)}.npz"
    )


def _catalog_from_cached_arrays(data: np.lib.npyio.NpzFile) -> atomic.LineCatalog:
    wavelength = np.asarray(data["catalog_wavelength"], dtype=np.float64)
    index_wavelength = np.asarray(data["catalog_index_wavelength"], dtype=np.float64)
    element = np.asarray(data["catalog_element"], dtype=object)
    ion_stage = np.asarray(data["catalog_ion_stage"], dtype=np.int16)
    log_gf = np.asarray(data["catalog_log_gf"], dtype=np.float64)
    excitation_energy = np.asarray(data["catalog_excitation_energy"], dtype=np.float64)
    gamma_rad = np.asarray(data["catalog_gamma_rad"], dtype=np.float64)
    gamma_stark = np.asarray(data["catalog_gamma_stark"], dtype=np.float64)
    gamma_vdw = np.asarray(data["catalog_gamma_vdw"], dtype=np.float64)
    line_type = np.asarray(data["catalog_line_type"], dtype=np.int16)
    n_lower = np.asarray(data["catalog_n_lower"], dtype=np.int16)
    n_upper = np.asarray(data["catalog_n_upper"], dtype=np.int16)
    code = np.asarray(data["catalog_code"], dtype=np.float64)
    iso1 = np.asarray(data["catalog_iso1"], dtype=np.int16)
    iso2 = np.asarray(data["catalog_iso2"], dtype=np.int16)
    line_size = np.asarray(data["catalog_line_size"], dtype=np.int16)
    labelp = np.asarray(data["catalog_labelp"], dtype=object)
    xj = np.asarray(data["catalog_xj"], dtype=np.float64)
    xjp = np.asarray(data["catalog_xjp"], dtype=np.float64)
    gamma_rad_log = np.asarray(data["catalog_gamma_rad_log"], dtype=np.float64)
    gamma_stark_log = np.asarray(data["catalog_gamma_stark_log"], dtype=np.float64)
    gamma_vdw_log = np.asarray(data["catalog_gamma_vdw_log"], dtype=np.float64)

    n = int(wavelength.size)
    records: list[atomic.LineRecord] = []
    for i in range(n):
        records.append(
            atomic.LineRecord(
                wavelength=float(wavelength[i]),
                index_wavelength=float(index_wavelength[i]),
                element=str(element[i]),
                ion_stage=int(ion_stage[i]),
                log_gf=float(log_gf[i]),
                excitation_energy=float(excitation_energy[i]),
                gamma_rad=float(gamma_rad[i]),
                gamma_stark=float(gamma_stark[i]),
                gamma_vdw=float(gamma_vdw[i]),
                metadata={},
                line_type=int(line_type[i]),
                n_lower=int(n_lower[i]),
                n_upper=int(n_upper[i]),
                code=float(code[i]),
                iso1=int(iso1[i]),
                iso2=int(iso2[i]),
                line_size=int(line_size[i]),
                labelp=str(labelp[i]),
                xj=float(xj[i]),
                xjp=float(xjp[i]),
                gamma_rad_log=float(gamma_rad_log[i]),
                gamma_stark_log=float(gamma_stark_log[i]),
                gamma_vdw_log=float(gamma_vdw_log[i]),
            )
        )
    return atomic.LineCatalog.from_records(records)


def _compiled_from_cache(cache_path: Path) -> Optional[CompiledLineCatalog]:
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            schema = int(np.asarray(data.get("__compiled_cache_schema", 1)).item())
            if schema != _COMPILED_CACHE_SCHEMA:
                return None
            catalog = _catalog_from_cached_arrays(data)
            fort19_data = fort19.Fort19Data(
                wavelength_vacuum=np.asarray(
                    data["f19_wavelength_vacuum"], dtype=np.float64
                ),
                energy_lower=np.asarray(data["f19_energy_lower"], dtype=np.float32),
                oscillator_strength=np.asarray(
                    data["f19_oscillator_strength"], dtype=np.float32
                ),
                n_lower=np.asarray(data["f19_n_lower"], dtype=np.int16),
                n_upper=np.asarray(data["f19_n_upper"], dtype=np.int16),
                ion_index=np.asarray(data["f19_ion_index"], dtype=np.int16),
                line_type=np.asarray(data["f19_line_type"], dtype=np.int16),
                continuum_index=np.asarray(data["f19_continuum_index"], dtype=np.int16),
                element_index=np.asarray(data["f19_element_index"], dtype=np.int16),
                gamma_rad=np.asarray(data["f19_gamma_rad"], dtype=np.float32),
                gamma_stark=np.asarray(data["f19_gamma_stark"], dtype=np.float32),
                gamma_vdw=np.asarray(data["f19_gamma_vdw"], dtype=np.float32),
                nbuff=np.asarray(data["f19_nbuff"], dtype=np.int32),
                limb=np.asarray(data["f19_limb"], dtype=np.int32),
                wing_type=np.asarray(
                    [
                        fort19.Fort19WingType.from_code(int(x))
                        for x in data["f19_wing_type"]
                    ],
                    dtype=object,
                ),
            )
            return CompiledLineCatalog(
                catalog=catalog,
                fort19_data=fort19_data,
                nbuff=np.asarray(data["t12_nbuff"], dtype=np.int32),
                cgf=np.asarray(data["t12_cgf"], dtype=np.float32),
                nelion=np.asarray(data["t12_nelion"], dtype=np.int16),
                elo_cm=np.asarray(data["t12_elo_cm"], dtype=np.float32),
                gamma_rad=np.asarray(data["t12_gamma_rad"], dtype=np.float32),
                gamma_stark=np.asarray(data["t12_gamma_stark"], dtype=np.float32),
                gamma_vdw=np.asarray(data["t12_gamma_vdw"], dtype=np.float32),
                limb=np.asarray(data["t12_limb"], dtype=np.int16),
            )
    except Exception:
        return None


def compiled_cache_summary(cache_path: Path) -> Optional[dict[str, object]]:
    """Return quick metadata for a compiled cache file without record reconstruction."""
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            schema = int(np.asarray(data.get("__compiled_cache_schema", 1)).item())
            return {
                "cache_path": str(cache_path),
                "schema": schema,
                "catalog_records": int(np.asarray(data["catalog_wavelength"]).size),
                "t12_records": int(np.asarray(data["t12_nbuff"]).size),
                "f19_records": int(np.asarray(data["f19_wavelength_vacuum"]).size),
            }
    except Exception:
        return None


def _write_cache(cache_path: Path, compiled: CompiledLineCatalog) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    catalog = compiled.catalog
    recs = catalog.records
    np.savez_compressed(
        cache_path,
        __compiled_cache_schema=np.array([_COMPILED_CACHE_SCHEMA], dtype=np.int16),
        catalog_wavelength=np.asarray([r.wavelength for r in recs], dtype=np.float64),
        catalog_index_wavelength=np.asarray(
            [r.index_wavelength for r in recs], dtype=np.float64
        ),
        catalog_element=np.asarray([r.element for r in recs], dtype="<U8"),
        catalog_ion_stage=np.asarray([r.ion_stage for r in recs], dtype=np.int16),
        catalog_log_gf=np.asarray([r.log_gf for r in recs], dtype=np.float64),
        catalog_excitation_energy=np.asarray(
            [r.excitation_energy for r in recs], dtype=np.float64
        ),
        catalog_gamma_rad=np.asarray([r.gamma_rad for r in recs], dtype=np.float64),
        catalog_gamma_stark=np.asarray([r.gamma_stark for r in recs], dtype=np.float64),
        catalog_gamma_vdw=np.asarray([r.gamma_vdw for r in recs], dtype=np.float64),
        catalog_line_type=np.asarray([r.line_type for r in recs], dtype=np.int16),
        catalog_n_lower=np.asarray([r.n_lower for r in recs], dtype=np.int16),
        catalog_n_upper=np.asarray([r.n_upper for r in recs], dtype=np.int16),
        catalog_code=np.asarray([r.code for r in recs], dtype=np.float64),
        catalog_iso1=np.asarray([r.iso1 for r in recs], dtype=np.int16),
        catalog_iso2=np.asarray([r.iso2 for r in recs], dtype=np.int16),
        catalog_line_size=np.asarray([r.line_size for r in recs], dtype=np.int16),
        catalog_labelp=np.asarray([r.labelp for r in recs], dtype="<U16"),
        catalog_xj=np.asarray([r.xj for r in recs], dtype=np.float64),
        catalog_xjp=np.asarray([r.xjp for r in recs], dtype=np.float64),
        catalog_gamma_rad_log=np.asarray(
            [r.gamma_rad_log for r in recs], dtype=np.float64
        ),
        catalog_gamma_stark_log=np.asarray(
            [r.gamma_stark_log for r in recs], dtype=np.float64
        ),
        catalog_gamma_vdw_log=np.asarray(
            [r.gamma_vdw_log for r in recs], dtype=np.float64
        ),
        f19_wavelength_vacuum=compiled.fort19_data.wavelength_vacuum,
        f19_energy_lower=compiled.fort19_data.energy_lower,
        f19_oscillator_strength=compiled.fort19_data.oscillator_strength,
        f19_n_lower=compiled.fort19_data.n_lower,
        f19_n_upper=compiled.fort19_data.n_upper,
        f19_ion_index=compiled.fort19_data.ion_index,
        f19_line_type=compiled.fort19_data.line_type,
        f19_continuum_index=compiled.fort19_data.continuum_index,
        f19_element_index=compiled.fort19_data.element_index,
        f19_gamma_rad=compiled.fort19_data.gamma_rad,
        f19_gamma_stark=compiled.fort19_data.gamma_stark,
        f19_gamma_vdw=compiled.fort19_data.gamma_vdw,
        f19_nbuff=compiled.fort19_data.nbuff,
        f19_limb=compiled.fort19_data.limb,
        f19_wing_type=np.asarray(
            [wt.value for wt in compiled.fort19_data.wing_type], dtype=np.int16
        ),
        t12_nbuff=compiled.nbuff,
        t12_cgf=compiled.cgf,
        t12_nelion=compiled.nelion,
        t12_elo_cm=compiled.elo_cm,
        t12_gamma_rad=compiled.gamma_rad,
        t12_gamma_stark=compiled.gamma_stark,
        t12_gamma_vdw=compiled.gamma_vdw,
        t12_limb=compiled.limb,
    )


def _element_z_from_symbol(symbol: str) -> int:
    try:
        return _ELEMENT_SYMBOLS.index(symbol)
    except ValueError:
        return 0


def _nelion_from_record(rec: atomic.LineRecord) -> int:
    code = float(rec.code) if rec.code > 0.0 else 0.0
    if code <= 0.0:
        nelem = _element_z_from_symbol(rec.element)
        if nelem <= 0:
            return 0
        code = nelem + (max(int(rec.ion_stage), 1) - 1) * 0.01

    nelem = int(code + 1.0e-6)
    icharge = int((code - nelem) * 100.0 + 0.1)
    zeff = icharge + 1
    nelion = nelem * 6 - 6 + int(zeff)
    if nelem > 19 and nelem < 29 and icharge > 5:
        nelion = 6 * (nelem + icharge * 10 - 30) - 1
    return nelion


def _linesize_to_limb(
    line_type: int,
    element: str,
    line_size: int,
    code: float,
) -> int:
    # Match rgfall.for lines 145-147:
    #   LIM=MIN(8-LINESIZE,7)
    #   IF(CODE.EQ.1.)LIM=1
    linesize = int(line_size) if line_size > 0 else 0
    lim = min(8 - linesize, 7)
    code_for_lim = float(code) if code > 0.0 else 0.0
    if abs(code_for_lim - 1.0) < 1.0e-6 or line_type in (-1, -2) or element in {"H", "D"}:
        lim = 1
    return lim


def compile_atomic_catalog(
    catalog_path: Path,
    wlbeg: float,
    wlend: float,
    resolution: float,
    line_filter: bool = True,
    cache_directory: Optional[Path] = None,
) -> CompiledLineCatalog:
    """Compile gfall-style atomic data into runtime metadata without `tfort.*`."""

    cache_path = _cache_file_path(
        catalog_path=catalog_path,
        wlbeg=wlbeg,
        wlend=wlend,
        resolution=resolution,
        line_filter=line_filter,
        cache_directory=cache_directory,
    )
    disable_compiled_cache = os.environ.get("PY_DISABLE_COMPILED_CACHE", "0") == "1"
    if not disable_compiled_cache:
        cached = _compiled_from_cache(cache_path)
        if cached is not None:
            return cached

    disable_parsed_cache = os.environ.get("PY_DISABLE_PARSED_CACHE", "0") == "1"
    if disable_parsed_cache:
        catalog = atomic.load_catalog(catalog_path)
        if line_filter:
            catalog = atomic.filter_by_range(catalog, wlbeg, wlend)
    else:
        catalog = parsed_cache.load_or_build_parsed_catalog_window(
            catalog_path,
            wlbeg=wlbeg,
            wlend=wlend,
            cache_directory=cache_directory,
        )
        # `load_or_build_parsed_catalog_window` already applies Fortran DELLIM
        # window filtering over cached arrays; this avoids full-catalog rebuilds.
        if not line_filter:
            catalog = parsed_cache.load_or_build_parsed_catalog(
                catalog_path,
                cache_directory=cache_directory,
            )

    fort19_data = fort19.build_from_catalog(
        catalog=catalog,
        wlbeg=wlbeg,
        wlend=wlend,
        resolution=resolution,
    )

    if len(catalog.records) == 0:
        empty_i32 = np.array([], dtype=np.int32)
        empty_i16 = np.array([], dtype=np.int16)
        empty_f32 = np.array([], dtype=np.float32)
        return CompiledLineCatalog(
            catalog=catalog,
            fort19_data=fort19_data,
            nbuff=empty_i32,
            cgf=empty_f32,
            nelion=empty_i16,
            elo_cm=empty_f32,
            gamma_rad=empty_f32,
            gamma_stark=empty_f32,
            gamma_vdw=empty_f32,
            limb=empty_i16,
        )

    ratio = 1.0 + 1.0 / resolution
    ratiolg = math.log(ratio)
    ixwlbeg = math.floor(math.log(wlbeg) / ratiolg)
    if math.exp(ixwlbeg * ratiolg) < wlbeg:
        ixwlbeg += 1

    t12_nbuff: list[int] = []
    t12_cgf: list[float] = []
    t12_nelion: list[int] = []
    t12_elo_cm: list[float] = []
    t12_gamma_rad: list[float] = []
    t12_gamma_stark: list[float] = []
    t12_gamma_vdw: list[float] = []
    t12_limb: list[int] = []

    for rec in catalog.records:
        wl = (
            float(rec.index_wavelength)
            if float(rec.index_wavelength) > 0.0
            else float(rec.wavelength)
        )
        ixwl = math.log(max(wl, 1e-30)) / ratiolg + 0.5
        nbuff_val = int(ixwl) - int(ixwlbeg) + 1

        line_type = int(rec.line_type)
        gf_linear = 10.0 ** float(rec.log_gf)
        if rec.labelp.strip().upper().startswith("CONTINUU"):
            nlast = int(rec.xjp) if rec.xjp > 0.0 else int(rec.n_upper)
            line_type = nlast
            gf_linear *= 2.0 * float(rec.xj) + 1.0

        # rgfall.for line 150: coronal lines are skipped entirely.
        if line_type == 2:
            continue

        nblo = abs(int(rec.n_lower))
        nbup = abs(int(rec.n_upper))
        # rgfall.for route to fort.12 only for plain lines (NBLO+NBUP == 0),
        # excluding autoionizing and continuum records.
        if line_type == 1 or line_type > 3 or (nblo + nbup) != 0:
            continue

        freq = _C_LIGHT_NM / max(wl, 1e-30)
        limb_val = int(
            _linesize_to_limb(
                line_type=line_type,
                element=rec.element,
                line_size=rec.line_size,
                code=rec.code,
            )
        )
        nelion_val = int(_nelion_from_record(rec))

        # In rgfall, non-auto lines written to fort.12 are normalized by 4*pi*nu.
        gamma_r = float(rec.gamma_rad)
        gamma_s = float(rec.gamma_stark)
        gamma_w = float(rec.gamma_vdw)
        t12_nbuff.append(nbuff_val)
        t12_cgf.append(float(_CGF_CONSTANT * gf_linear / freq))
        t12_nelion.append(nelion_val)
        t12_elo_cm.append(float(rec.excitation_energy))
        t12_gamma_rad.append(gamma_r)
        t12_gamma_stark.append(gamma_s)
        t12_gamma_vdw.append(gamma_w)
        t12_limb.append(limb_val)

    compiled = CompiledLineCatalog(
        catalog=catalog,
        fort19_data=fort19_data,
        nbuff=np.asarray(t12_nbuff, dtype=np.int32),
        cgf=np.asarray(t12_cgf, dtype=np.float32),
        nelion=np.asarray(t12_nelion, dtype=np.int16),
        elo_cm=np.asarray(t12_elo_cm, dtype=np.float32),
        gamma_rad=np.asarray(t12_gamma_rad, dtype=np.float32),
        gamma_stark=np.asarray(t12_gamma_stark, dtype=np.float32),
        gamma_vdw=np.asarray(t12_gamma_vdw, dtype=np.float32),
        limb=np.asarray(t12_limb, dtype=np.int16),
    )
    if not disable_compiled_cache:
        _write_cache(cache_path, compiled)
    return compiled


__all__ = [
    "COMPILED_CACHE_LOGIC_VERSION",
    "CompiledLineCatalog",
    "LINE_COMPILER_CONTRACT",
    "compile_atomic_catalog",
    "compiled_cache_key_payload",
    "compiled_cache_summary",
]
