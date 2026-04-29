"""Parsed GFALL cache helpers.

This module caches the expensive text parsing stage for gfallvac.latest into
array-based NPZ files, then reconstructs LineCatalog objects from those arrays.
The cache is window-independent and can be reused across many wavelength runs.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

from . import atomic

_PARSED_CACHE_SCHEMA = 1
PARSED_CACHE_LOGIC_VERSION = 2


@dataclass(frozen=True)
class ParsedCachePaths:
    """Paths for parsed-catalog cache artefacts."""

    npz: Path
    manifest: Path


def _default_cache_dir(catalog_path: Path) -> Path:
    return catalog_path.parent / ".py_line_cache"


def _source_fingerprint(catalog_path: Path) -> dict[str, Any]:
    st = catalog_path.stat()
    return {
        "source": str(catalog_path.resolve()),
        "size": int(st.st_size),
        "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
        "iso_corr_enabled": bool(os.environ.get("PY_APPLY_ISO_CORR", "1") != "0"),
    }


def parsed_cache_key_payload(catalog_path: Path) -> dict[str, Any]:
    return {
        "schema": _PARSED_CACHE_SCHEMA,
        "logic_version": PARSED_CACHE_LOGIC_VERSION,
        **_source_fingerprint(catalog_path),
    }


def _parsed_key(catalog_path: Path) -> str:
    payload = parsed_cache_key_payload(catalog_path)
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return digest[:24]


def cache_paths(
    catalog_path: Path,
    cache_directory: Optional[Path] = None,
) -> ParsedCachePaths:
    cache_dir = (
        cache_directory
        if cache_directory is not None
        else _default_cache_dir(catalog_path)
    )
    key = _parsed_key(catalog_path)
    npz = cache_dir / f"parsed_gfall_{key}.npz"
    manifest = cache_dir / f"parsed_gfall_{key}.json"
    return ParsedCachePaths(npz=npz, manifest=manifest)


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


def _catalog_from_cached_arrays_subset(
    data: np.lib.npyio.NpzFile,
    indices: np.ndarray,
) -> atomic.LineCatalog:
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size == 0:
        return atomic.LineCatalog.from_records([])
    wavelength = np.asarray(data["catalog_wavelength"], dtype=np.float64)[idx]
    index_wavelength = np.asarray(data["catalog_index_wavelength"], dtype=np.float64)[idx]
    element = np.asarray(data["catalog_element"], dtype=object)[idx]
    ion_stage = np.asarray(data["catalog_ion_stage"], dtype=np.int16)[idx]
    log_gf = np.asarray(data["catalog_log_gf"], dtype=np.float64)[idx]
    excitation_energy = np.asarray(data["catalog_excitation_energy"], dtype=np.float64)[idx]
    gamma_rad = np.asarray(data["catalog_gamma_rad"], dtype=np.float64)[idx]
    gamma_stark = np.asarray(data["catalog_gamma_stark"], dtype=np.float64)[idx]
    gamma_vdw = np.asarray(data["catalog_gamma_vdw"], dtype=np.float64)[idx]
    line_type = np.asarray(data["catalog_line_type"], dtype=np.int16)[idx]
    n_lower = np.asarray(data["catalog_n_lower"], dtype=np.int16)[idx]
    n_upper = np.asarray(data["catalog_n_upper"], dtype=np.int16)[idx]
    code = np.asarray(data["catalog_code"], dtype=np.float64)[idx]
    iso1 = np.asarray(data["catalog_iso1"], dtype=np.int16)[idx]
    iso2 = np.asarray(data["catalog_iso2"], dtype=np.int16)[idx]
    line_size = np.asarray(data["catalog_line_size"], dtype=np.int16)[idx]
    labelp = np.asarray(data["catalog_labelp"], dtype=object)[idx]
    xj = np.asarray(data["catalog_xj"], dtype=np.float64)[idx]
    xjp = np.asarray(data["catalog_xjp"], dtype=np.float64)[idx]
    gamma_rad_log = np.asarray(data["catalog_gamma_rad_log"], dtype=np.float64)[idx]
    gamma_stark_log = np.asarray(data["catalog_gamma_stark_log"], dtype=np.float64)[idx]
    gamma_vdw_log = np.asarray(data["catalog_gamma_vdw_log"], dtype=np.float64)[idx]

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


def _subset_indices_from_cache(
    data: np.lib.npyio.NpzFile,
    wlbeg: float,
    wlend: float,
) -> np.ndarray:
    wavelength = np.asarray(data["catalog_wavelength"], dtype=np.float64)
    line_type = np.asarray(data["catalog_line_type"], dtype=np.int16)
    element = np.asarray(data["catalog_element"], dtype=object)
    line_size = np.asarray(data["catalog_line_size"], dtype=np.int16)
    code = np.asarray(data["catalog_code"], dtype=np.float64)

    # Fortran DELLIM margins (rgfall.for line 90)
    dellim = np.array([100.0, 30.0, 10.0, 3.0, 1.0, 0.3, 0.1], dtype=np.float64)
    delfactor = 1.0 if wlbeg <= 500.0 else wlbeg / 500.0

    linesize = np.where(line_size > 0, line_size, 0).astype(np.int16)
    lim = np.minimum(8 - linesize, 7).astype(np.int16)
    # rgfall special-case: hydrogen/deuterium lines always LIM=1
    hmask = (
        np.isclose(code, 1.0, rtol=0.0, atol=1e-6)
        | np.isin(line_type, np.array([-1, -2], dtype=np.int16))
        | np.isin(element, np.array(["H", "D"], dtype=object))
    )
    lim = np.where(hmask, 1, lim)
    lim = np.clip(lim, 1, 7)
    margin = dellim[lim - 1] * delfactor
    mask = (wavelength >= (wlbeg - margin)) & (wavelength <= (wlend + margin))
    return np.nonzero(mask)[0]


def _write_catalog_cache(npz_path: Path, catalog: atomic.LineCatalog) -> None:
    recs = catalog.records
    np.savez_compressed(
        npz_path,
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
    )


def _is_manifest_valid(manifest_path: Path, catalog_path: Path) -> bool:
    if not manifest_path.exists():
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    expected = {
        **parsed_cache_key_payload(catalog_path),
    }
    return payload == expected


def _write_manifest(manifest_path: Path, catalog_path: Path) -> None:
    payload = {
        **parsed_cache_key_payload(catalog_path),
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_or_build_parsed_catalog(
    catalog_path: Path,
    cache_directory: Optional[Path] = None,
) -> atomic.LineCatalog:
    """Load parsed GFALL cache or build it from source if missing/stale."""
    paths = cache_paths(catalog_path, cache_directory=cache_directory)
    if paths.npz.exists() and _is_manifest_valid(paths.manifest, catalog_path):
        try:
            with np.load(paths.npz, allow_pickle=False) as data:
                return _catalog_from_cached_arrays(data)
        except Exception:
            pass

    catalog = atomic.load_catalog(catalog_path)
    paths.npz.parent.mkdir(parents=True, exist_ok=True)
    _write_catalog_cache(paths.npz, catalog)
    _write_manifest(paths.manifest, catalog_path)
    return catalog


def load_or_build_parsed_catalog_window(
    catalog_path: Path,
    wlbeg: float,
    wlend: float,
    cache_directory: Optional[Path] = None,
) -> atomic.LineCatalog:
    """Load parsed GFALL cache and return only Fortran-margin-relevant records."""
    paths = cache_paths(catalog_path, cache_directory=cache_directory)
    if paths.npz.exists() and _is_manifest_valid(paths.manifest, catalog_path):
        try:
            with np.load(paths.npz, allow_pickle=False) as data:
                idx = _subset_indices_from_cache(data, wlbeg=wlbeg, wlend=wlend)
                return _catalog_from_cached_arrays_subset(data, idx)
        except Exception:
            pass
    full = load_or_build_parsed_catalog(catalog_path, cache_directory=cache_directory)
    if len(full.records) == 0:
        return full
    return atomic.filter_by_range(full, wlbeg, wlend)


def load_manifest(
    catalog_path: Path,
    cache_directory: Optional[Path] = None,
) -> Optional[dict[str, Any]]:
    """Return parsed cache manifest payload, if present."""
    paths = cache_paths(catalog_path, cache_directory=cache_directory)
    if not paths.manifest.exists():
        return None
    try:
        return json.loads(paths.manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


__all__ = [
    "PARSED_CACHE_LOGIC_VERSION",
    "ParsedCachePaths",
    "cache_paths",
    "load_manifest",
    "load_or_build_parsed_catalog",
    "load_or_build_parsed_catalog_window",
    "parsed_cache_key_payload",
]


