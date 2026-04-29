"""Validate continuum opacity against Fortran KAPCONT stdout tables.

This compares Python `kapcont_table()` output to the KAPCONT table printed by
atlas12 stdout (wavelength headers + 80 depth rows).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from atlas_py.engine.driver import _runtime_from_atm
from atlas_py.io.atmosphere import load_atm
from atlas_py.io.molecules import readmol_atlas12
from atlas_py.physics.atlas_tables import load_atlas_tables
from atlas_py.physics.kapcont import kapcont_table
from atlas_py.physics.kapp import KappAtmosphereAdapter
from atlas_py.physics.nmolec import clear_nmolec_context, set_nmolec_context
from atlas_py.physics.populations import pops
from atlas_py.physics.popsall import popsall


def _parse_fortran_kapcont(log_path: Path) -> tuple[np.ndarray, np.ndarray]:
    all_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    # Only parse KAPCONT from pass-2 stdout so we use the post-iteration state.
    pass2_start = 0
    for li, ll in enumerate(all_lines):
        if "=== PASS 2 STDOUT ===" in ll:
            pass2_start = li + 1
            break
    lines = all_lines[pass2_start:] if pass2_start else all_lines

    # The KAPCONT section always begins with the 9.09 nm wavelength header.
    # NMOLEC uses the identical print format (5X,10F12.2 / I5,1P10E12.3), so
    # we must skip past those molecular blocks and start at KAPCONT's first line.
    kapcont_start = 0
    for li, ll in enumerate(lines):
        stripped = ll.strip()
        parts = stripped.split()
        if not parts:
            continue
        # Look for a header line that begins with 9.09 (first KAPCONT wavelength)
        try:
            if abs(float(parts[0]) - 9.09) < 0.01:
                kapcont_start = li
                break
        except ValueError:
            pass
    lines = lines[kapcont_start:]

    waves: list[float] = []
    blocks: list[np.ndarray] = []
    i = 0
    n = len(lines)
    nrhox: int | None = None
    while i < n:
        parts = lines[i].strip().split()
        if not parts:
            i += 1
            continue
        try:
            vals = [float(tok.replace("D", "E").replace("d", "e")) for tok in parts]
        except ValueError:
            i += 1
            continue
        # Header row: up to 10 plain wavelengths (no leading depth index).
        if len(vals) <= 10 and "." in parts[0]:
            nw = len(vals)
            rows: list[list[float]] = []
            j = i + 1
            while j < n and (nrhox is None or len(rows) < nrhox):
                p = lines[j].strip().split()
                if len(p) < nw + 1:
                    break
                try:
                    idx = int(p[0])
                    row = [float(tok.replace("D", "E").replace("d", "e")) for tok in p[1 : 1 + nw]]
                except ValueError:
                    break
                if idx != len(rows) + 1:
                    break
                rows.append(row)
                j += 1
            if len(rows) > 0 and (nrhox is None or len(rows) == nrhox):
                if nrhox is None:
                    nrhox = len(rows)
                waves.extend(vals)
                blocks.append(np.asarray(rows, dtype=np.float64))
                i = j
                continue
        i += 1
    if not blocks:
        raise ValueError(f"No KAPCONT blocks parsed from {log_path}")
    tab = np.hstack(blocks)
    return np.asarray(waves, dtype=np.float64), tab


def _build_adapter_and_state(
    fortran_atm: Path,
    molecules_new: Path,
    *,
    recompute_populations: bool,
) -> tuple[KappAtmosphereAdapter, np.ndarray]:
    atm = load_atm(fortran_atm)
    state = _runtime_from_atm(atm)
    teff_meta = float(atm.metadata.get("teff", "0.0"))
    itemp_cache: dict[str, int] = {}
    dummy = np.zeros((atm.layers, 1), dtype=np.float64)
    mol = readmol_atlas12(molecules_new)
    if recompute_populations:
        set_nmolec_context(
            temperature_k=atm.temperature,
            tk_erg=atm.tk,
            tlog=atm.tlog,
            gas_pressure=state.p,
            state=state,
            nummol=mol.nummol,
            code_mol=mol.code_mol,
            equil=mol.equil,
            locj=mol.locj,
            kcomps=mol.kcomps,
            idequa=mol.idequa,
            nequa=mol.nequa,
        )
        try:
            pops(
                code=0.0,
                mode=1,
                out=dummy,
                ifmol=True,
                ifpres=True,
                temperature_k=atm.temperature,
                tk_erg=atm.tk,
                state=state,
                itemp=1,
                itemp_cache=itemp_cache,
            )
            popsall(
                temperature_k=atm.temperature,
                tk_erg=atm.tk,
                state=state,
                ifmol=True,
                ifpres=True,
                itemp=1,
                itemp_cache=itemp_cache,
            )
        finally:
            clear_nmolec_context()

    n = len(atm.temperature)
    # bhyd/bmin always 1.0 in LTE (Fortran DATA statement).
    if state.bhyd is None:
        state.bhyd = np.ones((n, 6), dtype=np.float64)
    if state.bmin is None:
        state.bmin = np.ones(n, dtype=np.float64)
    adapter = KappAtmosphereAdapter(
        temperature=np.asarray(atm.temperature, dtype=np.float64),
        mass_density=np.asarray(state.rho, dtype=np.float64),
        electron_density=np.asarray(state.xne, dtype=np.float64),
        gas_pressure=np.asarray(state.p, dtype=np.float64),
        xnfph=np.column_stack([state.xnfp[:, 0], state.xnfp[:, 1]]),
        xnf_h=np.asarray(state.xnf[:, 0], dtype=np.float64),
        xnf_h_ionized=np.asarray(state.xnf[:, 1], dtype=np.float64),
        xnf_he1=np.asarray(state.xnfp[:, 2], dtype=np.float64),
        xnf_he2=np.asarray(state.xnfp[:, 3], dtype=np.float64),
        xabund=np.asarray(state.xabund, dtype=np.float64),
        bhyd=np.asarray(state.bhyd, dtype=np.float64),
        turbulent_velocity=np.asarray(atm.vturb, dtype=np.float64),
        xnf_all=np.asarray(state.xnf, dtype=np.float64),
        xnfp_all=np.asarray(state.xnfp, dtype=np.float64),
        xnfpch=np.asarray(state.xnfp[:, 845], dtype=np.float64),
        xnfpoh=np.asarray(state.xnfp[:, 847], dtype=np.float64),
    )
    return adapter, np.asarray(atm.temperature, dtype=np.float64), teff_meta


def main() -> int:
    p = argparse.ArgumentParser(description="Compare Python KAPCONT vs Fortran KAPCONT log table")
    p.add_argument("--fortran-log", type=Path, required=True, help="Fortran stdout log containing KAPCONT table")
    p.add_argument("--fortran-atm", type=Path, required=True, help="Fortran atmosphere deck/stdout model used for state")
    p.add_argument("--molecules-new", type=Path, required=True, help="Path to molecules.new")
    p.add_argument("--output", type=Path, required=True, help="Text report output path")
    p.add_argument(
        "--skip-population-recompute",
        action="store_true",
        help=(
            "Use atmosphere-state populations as-is (do not rerun POPS/POPSALL). "
            "Useful when comparing logs from Fortran runs where pressure/correction "
            "updates are disabled."
        ),
    )
    args = p.parse_args()

    import re as _re

    fwave, ftab = _parse_fortran_kapcont(args.fortran_log)
    adapter, temp, teff = _build_adapter_and_state(
        args.fortran_atm,
        args.molecules_new,
        recompute_populations=not args.skip_population_recompute,
    )
    # Parse IFOP from the atm file so KAPP uses same flags as Fortran.
    from atlas_py.io.atmosphere import load_atm as _load_atm

    _atm_meta = _load_atm(args.fortran_atm)
    _raw_ifop = _atm_meta.metadata.get("ifop", "")
    _vals = [int(x) for x in _re.findall(r"-?\d+", _raw_ifop)]
    ifop = _vals[-20:] if len(_vals) >= 20 else [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0]

    tables = load_atlas_tables()
    ptab, pwave, _ = kapcont_table(
        adapter=adapter,
        temperature_k=temp,
        teff=teff,
        atlas_tables=tables,
        ifop=ifop,
    )

    nfreq = min(ftab.shape[1], ptab.shape[1], fwave.size, pwave.size)
    ftab = ftab[:, :nfreq]
    ptab = ptab[:, :nfreq]
    waves = pwave[:nfreq]
    rel = np.abs(ptab - ftab) / np.maximum(np.abs(ftab), 1e-30)

    wave_idx = [0, 20, 40, 80, 120, 160, 200, 240, 280, 320]
    wave_idx = [i for i in wave_idx if i < nfreq]
    depth_idx = [0, 19, 25, 39, 59, min(72, ptab.shape[0] - 1)]

    lines: list[str] = []
    lines.append(f"Fortran KAPCONT parsed: depths={ftab.shape[0]} waves={ftab.shape[1]}")
    lines.append(f"Python KAPCONT built: depths={ptab.shape[0]} waves={ptab.shape[1]}")
    lines.append(
        "Global relative error: "
        f"mean={float(np.mean(rel)):.6e} "
        f"rms={float(np.sqrt(np.mean(rel**2))):.6e} "
        f"max={float(np.max(rel)):.6e}"
    )
    lines.append("Sample points (wave_nm, depth, fortran, python, rel):")
    for iw in wave_idx:
        for jd in depth_idx:
            lines.append(
                f"{waves[iw]:10.2f} d={jd+1:2d} "
                f"f={ftab[jd, iw]:.6e} p={float(ptab[jd, iw]):.6e} r={float(rel[jd, iw]):.6e}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Add layer-26 detailed breakdown near Rosseland peak (900-2000 nm)
    target_layer = 25  # 0-indexed (Fortran layer 26)
    if target_layer < ptab.shape[0] and target_layer < ftab.shape[0]:
        lines.append(f"\nLayer 26 (Fortran J=26) TABCONT near Rosseland peak (900-2000 nm):")
        lines.append(f"{'wave_nm':>10}  {'fortran':>12}  {'python':>12}  {'rel_err':>10}")
        for iw, wv in enumerate(waves):
            if 900.0 <= wv <= 2100.0 and iw < ftab.shape[1] and iw < ptab.shape[1]:
                fv = ftab[target_layer, iw]
                pv = float(ptab[target_layer, iw])
                rv = abs(pv - fv) / max(abs(fv), 1e-30)
                lines.append(f"{wv:10.1f}  {fv:12.4e}  {pv:12.4e}  {rv:10.4f}")

    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] Wrote opacity report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

