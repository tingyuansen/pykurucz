"""Single-line synthesis harness for debugging Python vs. Fortran.

This script computes TRANSP and ASYNTH for one selected line using the
existing Python pipeline pieces. It is intended for line-by-line
comparison against Fortran debug output (e.g., around 300–310 nm).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

from ..io.atmosphere import load_cached, AtmosphereModel
from ..io.lines import atomic
from ..physics import line_opacity, populations, tables
from ..engine.opacity import _element_atomic_number, C_LIGHT_NM


def _build_wavelength_grid(start: float, end: float, resolution: float) -> np.ndarray:
    """Geometric wavelength grid matching run_synthesis spacing."""
    if resolution <= 0.0:
        raise ValueError("resolution must be > 0")
    ratio = 1.0 + 1.0 / resolution
    rlog = math.log(ratio)
    ix_start = math.log(start) / rlog
    ix_floor = math.floor(ix_start)
    if math.exp(ix_floor * rlog) < start:
        ix_floor += 1
    wbegin = math.exp(ix_floor * rlog)
    wavelengths = []
    wl = wbegin
    while wl <= end * (1.0 + 1e-9):
        wavelengths.append(wl)
        wl *= ratio
    return np.array(wavelengths, dtype=np.float64)


def _select_line(
    catalog: atomic.LineCatalog,
    wavelength_nm: float,
    window_nm: float,
    element: Optional[str],
    ion_stage: Optional[int],
) -> atomic.LineCatalog:
    """Return a LineCatalog with exactly one selected line."""
    deltas = np.abs(catalog.wavelength - wavelength_nm)
    mask = deltas <= window_nm
    if element:
        mask &= np.char.upper(catalog.elements.astype(str)) == element.upper()
    if ion_stage:
        mask &= catalog.ion_stages == ion_stage

    candidates = np.where(mask)[0]
    if candidates.size == 0:
        raise ValueError(
            f"No line found within ±{window_nm} nm of {wavelength_nm} "
            f"for element={element or 'ANY'} ion={ion_stage or 'ANY'}"
        )
    best_idx = candidates[int(np.argmin(deltas[candidates]))]
    record = catalog.records[int(best_idx)]
    return atomic.LineCatalog.from_records([record])


def _center_summary(
    catalog: atomic.LineCatalog,
    atmos: AtmosphereModel,
    pops: populations.Populations,
    transp: np.ndarray,
) -> dict:
    """Compute a small set of scalar diagnostics at depth 0."""
    rec = catalog.records[0]
    atomic_num = _element_atomic_number(rec.element)
    ion_stage = int(rec.ion_stage)

    pop_val = None
    dop_val = None
    rho0 = float(atmos.mass_density[0]) if atmos.mass_density is not None else None
    if (
        atomic_num is not None
        and atmos.population_per_ion is not None
        and atmos.population_per_ion.ndim == 3
        and atmos.doppler_per_ion is not None
        and atmos.doppler_per_ion.ndim == 3
        and atmos.population_per_ion.shape[2] > (atomic_num - 1)
        and atmos.doppler_per_ion.shape[2] > (atomic_num - 1)
    ):
        pop_val = float(atmos.population_per_ion[0, ion_stage - 1, atomic_num - 1])
        dop_val = float(atmos.doppler_per_ion[0, ion_stage - 1, atomic_num - 1])

    hkt0 = pops.layers[0].hckt
    freq = C_LIGHT_NM / rec.wavelength
    stim0 = 1.0 - math.exp(-freq * hkt0)
    transp0 = float(transp[0, 0])
    asynth0 = transp0 * stim0

    kappa0 = None
    adamp = None
    if pop_val and dop_val and rho0 and rho0 > 0.0:
        xnfdop = pop_val / (rho0 * dop_val)
        gf = 10.0 ** rec.log_gf
        kappa0 = gf * xnfdop
        gamma_rad = rec.gamma_rad
        gamma_stark = rec.gamma_stark
        gamma_vdw = rec.gamma_vdw
        txnxn0 = float(atmos.txnxn[0]) if atmos.txnxn is not None else 0.0
        xne0 = float(atmos.electron_density[0])
        # Doppler width in frequency units
        delta_nu_doppler = (C_LIGHT_NM / rec.wavelength) * dop_val
        if delta_nu_doppler > 0.0:
            gamma_total = gamma_rad + gamma_stark * xne0 + gamma_vdw * txnxn0
            # Fortran: ADAMP = gamma_total / DOPPLE (gamma is pre-normalized by 4πν)
            adamp = gamma_total / dop_val if dop_val > 0 else 0.0

    return {
        "transp_depth0": transp0,
        "asynth_center_depth0": asynth0,
        "stim_depth0": stim0,
        "hkt_depth0": hkt0,
        "kappa0_est_depth0": kappa0,
        "adamp_est_depth0": adamp,
        "pop_depth0": pop_val,
        "doppler_depth0": dop_val,
        "rho_depth0": rho0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Single-line ASYNTH/TRANSP harness")
    parser.add_argument("--atmosphere", required=True, type=Path, help="NPZ atmosphere")
    parser.add_argument("--catalog", required=True, type=Path, help="Atomic line catalog (e.g., gfallvac)")
    parser.add_argument("--wavelength", type=float, default=300.9015, help="Target wavelength (nm)")
    parser.add_argument("--window", type=float, default=0.5, help="Half-width window around wavelength (nm)")
    parser.add_argument("--element", type=str, default="FE", help="Element symbol filter (e.g., FE)")
    parser.add_argument("--ion-stage", type=int, default=1, help="Ion stage filter (1=neutral)")
    parser.add_argument("--resolution", type=float, default=300_000.0, help="Resolving power lambda/dlambda for grid")
    parser.add_argument("--cutoff", type=float, default=1e-3, help="Opacity cutoff factor (CUTOFF)")
    parser.add_argument("--output", type=Path, default=None, help="Optional NPZ output to save arrays")
    args = parser.parse_args()

    atmos = load_cached(args.atmosphere)
    catalog_full = atomic.load_catalog(args.catalog)
    catalog = _select_line(catalog_full, args.wavelength, args.window, args.element, args.ion_stage)
    rec = catalog.records[0]

    print(f"Selected line: wl={rec.wavelength:.6f} nm element={rec.element} ion={rec.ion_stage} loggf={rec.log_gf:.3f}")

    pops = populations.compute_depth_state(
        atmosphere=atmos,
        line_wavelengths=catalog.wavelength,
        excitation_energy=catalog.excitation_energy,
        microturb_kms=float(args.microturb),
    )

    # Build wavelength grid and compute continuum BEFORE compute_transp
    # (required for KAPMIN = CONTINUUM * CUTOFF matching Fortran)
    wl_start = max(rec.wavelength - args.window, 0.001)
    wl_end = rec.wavelength + args.window
    wavelength_grid = _build_wavelength_grid(wl_start, wl_end, float(args.resolution))
    
    # Compute continuum absorption for KAPMIN calculation
    from ..physics.continuum import build_depth_continuum
    cont_abs, cont_scat, _, _ = build_depth_continuum(atmos, wavelength_grid)

    transp, valid_mask, _ = line_opacity.compute_transp(
        catalog=catalog,
        populations=pops,
        atmosphere=atmos,
        cutoff=float(args.cutoff),
        continuum_absorption=cont_abs,
        wavelength_grid=wavelength_grid,
    )

    asynth = line_opacity.compute_asynth_from_transp(
        transp=transp,
        catalog=catalog,
        atmosphere=atmos,
        wavelength_grid=wavelength_grid,
        valid_mask=valid_mask,
        populations=pops,
        cutoff=float(args.cutoff),
        continuum_absorption=cont_abs,
        metal_tables=tables.metal_wing_tables(),
    )

    summary = _center_summary(catalog, atmos, pops, transp)
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.output:
        meta = {
            "wavelength_selected_nm": float(rec.wavelength),
            "element": rec.element,
            "ion_stage": int(rec.ion_stage),
            "log_gf": float(rec.log_gf),
            "gamma_rad": float(rec.gamma_rad),
            "gamma_stark": float(rec.gamma_stark),
            "gamma_vdw": float(rec.gamma_vdw),
            "window_nm": float(args.window),
            "resolution": float(args.resolution),
            "cutoff": float(args.cutoff),
            "microturb_kms": float(args.microturb),
        }
        np.savez(
            args.output,
            wavelength=wavelength_grid,
            asynth=asynth,
            transp=transp,
            valid_mask=valid_mask,
            summary=json.dumps(summary),
            metadata=json.dumps(meta),
        )
        print(f"Saved results to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

