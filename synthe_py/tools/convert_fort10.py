#!/usr/bin/env python3
"""Converter for SYNTHE fort.10 atmosphere tapes."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

try:
    from .embed_atlas_tables import embed_atlas_tables
except ImportError:
    from embed_atlas_tables import embed_atlas_tables


def _read_record(handle) -> bytes:
    header = handle.read(4)
    if not header:
        raise EOFError
    (size,) = struct.unpack("<i", header)
    payload = handle.read(size)
    trailer = handle.read(4)
    if len(payload) != size or len(trailer) != 4:
        raise ValueError("Truncated fort.10 record")
    return payload


def convert_fort10(
    input_path: Path,
    output_path: Path,
    *,
    atlas_tables: Path | None = None,
    fort5_path: Path | None = None,
) -> None:
    with input_path.open("rb") as fh:
        records: list[bytes] = []
        while True:
            try:
                records.append(_read_record(fh))
            except EOFError:
                break

    if len(records) < 4:
        raise ValueError("Unexpected fort.10 structure: too few records")

    # Header
    header = records[0]
    nrhox = struct.unpack_from("<i", header, 0)[0]
    teff = struct.unpack_from("<d", header, 4)[0]
    glog = struct.unpack_from("<d", header, 12)[0]
    title = header[20:].decode("ascii", "ignore").strip()

    # Continuum edges
    # CRITICAL FIX: Fortran writes interleaved: (FRQEDG(I),WLEDGE(I),CMEDGE(I),I=1,IN)
    # This means: FRQEDG[1], WLEDGE[1], CMEDGE[1], FRQEDG[2], WLEDGE[2], CMEDGE[2], ...
    # We must read them interleaved, not sequentially!
    rec_edges = records[1]
    idx = 0
    (nedge,) = struct.unpack_from("<i", rec_edges, idx)
    idx += 4

    # Read interleaved: FRQEDG[1], WLEDGE[1], CMEDGE[1], FRQEDG[2], ...
    frqedg = np.zeros(nedge, dtype=np.float64)
    wledge = np.zeros(nedge, dtype=np.float64)
    cmedge = np.zeros(nedge, dtype=np.float64)
    for i in range(nedge):
        frqedg[i] = struct.unpack_from("<d", rec_edges, idx)[0]
        idx += 8
        wledge[i] = struct.unpack_from("<d", rec_edges, idx)[0]
        idx += 8
        cmedge[i] = struct.unpack_from("<d", rec_edges, idx)[0]
        idx += 8

    # Fortran sorts edge table by ABS(WLEDGE) when writing (xnfpelsyn.for lines 185-202)
    # After reading interleaved, edges should already be sorted
    wledge_abs = np.abs(wledge)
    is_sorted = np.all(np.diff(wledge_abs) >= 0)
    if not is_sorted:
        import warnings

        warnings.warn(
            f"Edges are not sorted after interleaved reading! This shouldn't happen.",
            UserWarning,
        )
        # Fallback: sort them
        sort_idx = np.argsort(wledge_abs)
        wledge = wledge[sort_idx]
        frqedg = frqedg[sort_idx]
        cmedge = cmedge[sort_idx]
    else:
        # Edges are already sorted - no need to sort again
        sort_idx = None

    # Remaining entries are ID/Mass pairs written as doubles.
    leftover = rec_edges[idx:]
    idmol = np.frombuffer(leftover, dtype="<f8", count=len(leftover) // 16, offset=0)
    momass = np.frombuffer(leftover, dtype="<f8", count=len(leftover) // 16, offset=8)

    # Frequency grid
    rec_freq = records[2]
    (num_freq,) = struct.unpack_from("<i", rec_freq, 0)
    freqset = np.frombuffer(rec_freq, dtype="<f8", count=num_freq, offset=4)

    # Atmospheric state block (record 3)
    state_block = records[3]
    offset = 0
    layer = nrhox

    # CRITICAL FIX: Arrays are dimensioned as kw=99 in Fortran, but only NRHOX=80 layers are used
    # Fortran writes ALL kw elements, so we must read kw elements, then take first nrhox
    # This matches how synthe.exe reads (it reads full arrays then uses first NRHOX elements)
    kw = 99  # Array dimension from xnfpelsyn.for PARAMETER (kw=99,...)

    def _take(count: int) -> np.ndarray:
        nonlocal offset
        arr = np.frombuffer(state_block, dtype="<f8", count=count, offset=offset)
        offset += count * 8
        return arr.copy()

    # Read kw elements, then take first nrhox (actual number of layers)
    temperature_full = _take(kw)
    tkev_full = _take(kw)
    tk_full = _take(kw)
    temperature = temperature_full[:layer]
    tkev = tkev_full[:layer]
    tk = tk_full[:layer]
    hkt_full = _take(kw)
    tlog_full = _take(kw)
    hckt_full = _take(kw)
    gas_pressure_full = _take(kw)
    electron_density_full = _take(kw)
    xnatm_full = _take(kw)
    mass_density_full = _take(kw)
    fort10_depth_full = _take(kw)  # This is NOT RHOX! Skip it.
    turbulent_velocity_full = _take(kw)

    # Take first nrhox elements (actual number of layers)
    hkt = hkt_full[:layer]
    tlog = tlog_full[:layer]
    hckt = hckt_full[:layer]
    gas_pressure = gas_pressure_full[:layer]
    electron_density = electron_density_full[:layer]
    xnatm = xnatm_full[:layer]
    mass_density = mass_density_full[:layer]
    fort10_depth = fort10_depth_full[:layer]
    turbulent_velocity = turbulent_velocity_full[:layer]

    # CRITICAL FIX: Read RHOX and XNE from fort.5, not fort.10!
    # Fort.10's "depth" field is NOT column mass - it's ~10^17 times too large.
    # Fort.10's XNE may also be wrong (we've seen 15 trillion× mismatch).
    # Fortran spectrv.for reads RHOX and XNE from fort.5 (atmosphere model).
    if fort5_path is not None:
        try:
            from .parse_fort5 import parse_fort5
        except ImportError:
            from parse_fort5 import parse_fort5
        fort5_atm = parse_fort5(fort5_path)
        if len(fort5_atm.rhox) != layer:
            raise ValueError(
                f"Fort.5 has {len(fort5_atm.rhox)} layers but fort.10 has {layer} layers"
            )
        depth = fort5_atm.rhox  # Use correct RHOX from fort.5
        electron_density = fort5_atm.electron_density  # Use correct XNE from fort.5
        print(f"Using RHOX and XNE from fort.5: {fort5_path}")
        print(f"  RHOX[0] = {depth[0]:.6E} (correct)")
        print(f"  fort.10 depth[0] = {fort10_depth[0]:.6E} (wrong - NOT column mass)")
        print(f"  XNE[0] = {electron_density[0]:.6E} (correct)")
        print(f"  fort.10 XNE[0] = {electron_density[0]:.6E} (may be wrong)")
    else:
        # Fallback: use fort.10 value with warning
        depth = fort10_depth
        print("WARNING: No fort.5 provided, using fort.10 'depth' and XNE fields")
        print("  This may be WRONG - fort.10 depth is NOT RHOX!")
        print("  Provide --fort5 argument to use correct RHOX and XNE values")
    xnf_h_full = _take(kw)
    xnf_h = xnf_h_full[:layer]

    # CRITICAL FIX: Fortran writes XNFHE in column-major order:
    # XNFHE(1,1), XNFHE(2,1), ..., XNFHE(kw,1), XNFHE(1,2), ..., XNFHE(kw,2)
    # NOT interleaved! Read as column-major, then store as separate arrays
    xnf_he_all = np.frombuffer(state_block, dtype="<f8", count=kw * 2, offset=offset)
    xnf_he1_full = xnf_he_all[0:kw].copy()  # First column (all kw elements)
    xnf_he2_full = xnf_he_all[kw : kw * 2].copy()  # Second column (all kw elements)
    offset += kw * 16
    xnf_h2_full = _take(kw)

    # Take first nrhox elements
    xnf_he1 = xnf_he1_full[:layer]
    xnf_he2 = xnf_he2_full[:layer]
    xnf_h2 = xnf_h2_full[:layer]

    # Preallocate per-layer arrays
    # Note: cont_total is written by Fortran but not used in Python, so we skip it
    cont_abs = np.zeros((layer, num_freq), dtype=np.float64)
    cont_scat = np.zeros((layer, num_freq), dtype=np.float64)
    population = np.zeros((layer, 6, 139), dtype=np.float64)
    doppler = np.zeros_like(population)

    # Fortran fort.10 structure per layer (from xnfpelsyn.for lines 514-517):
    #   WRITE(10)(CONTINALL(NU,J),NU=1,1131)  ! Record 1: log10(ABTOT), total opacity
    #   WRITE(10)(CONTABS(NU,J),NU=1,1131)    ! Record 2: log10(ACONT), absorption
    #   WRITE(10)(CONTSCAT(NU,J),NU=1,1131)   ! Record 3: log10(SIGMAC), scattering
    #   WRITE(10)XNFPEL,DOPPLE                 ! Record 4: populations
    #
    # Fortran spectrv.for reads these (lines 122-125):
    #   READ(10)           ! Skip CONTINALL
    #   READ(10)QCONTABS   ! Read absorption (→ ACONT)
    #   READ(10)QCONTSCAT  ! Read scattering (→ SIGMAC)
    #   READ(10)           ! Skip populations
    #
    # Then uses (lines 254-257):
    #   ACONT(J) = 10**(CONTABS)    ! Absorption opacity
    #   SIGMAC(J) = 10**(CONTSCAT)  ! Scattering opacity
    #
    # So the correct mapping is:
    #   rec_index+0: CONTINALL (total, skip)
    #   rec_index+1: CONTABS → cont_abs (absorption = ACONT)
    #   rec_index+2: CONTSCAT → cont_scat (scattering = SIGMAC)
    #   rec_index+3: Populations
    #
    # mw6 = 6 * 139 = 834 (6 ion stages × 139 elements)
    mw6 = 6 * 139  # 834

    rec_index = 4
    for depth_idx in range(layer):
        # Skip first record: CONTINALL (total opacity, not used separately)
        rec_index += 1
        
        # Read second record: CONTABS → use as absorption (ACONT)
        cont_abs[depth_idx] = np.frombuffer(
            records[rec_index], dtype="<f8", count=num_freq
        )
        rec_index += 1
        
        # Read third record: CONTSCAT → use as scattering (SIGMAC)
        cont_scat[depth_idx] = np.frombuffer(
            records[rec_index], dtype="<f8", count=num_freq
        )
        rec_index += 1
        
        # Read QXNFPEL and QDOPPLE (population and Doppler data)
        # Format: QXNFPEL(mw6), QDOPPLE(mw6) = 834*2 doubles = 13344 bytes
        pop_data = records[rec_index]
        if len(pop_data) >= mw6 * 8 * 2:
            qxnfpel = np.frombuffer(pop_data, dtype="<f8", count=mw6, offset=0)
            qdopple = np.frombuffer(pop_data, dtype="<f8", count=mw6, offset=mw6 * 8)
            
            # Reshape from flat (mw6,) to (6, 139) for each layer
            # CRITICAL FIX: Fortran indexing from rgfall.for line 178:
            #   NELION = NELEM*6 - 6 + ZEFF = (NELEM-1)*6 + ZEFF
            #   where ZEFF = ICHARGE + 1 (ZEFF=1 for neutral, ZEFF=2 for singly ionized, etc.)
            # So Fortran's NELION for neutral = (NELEM-1)*6 + 1, NOT (NELEM-1)*6 + 0!
            # Fortran array is 1-indexed, Python's is 0-indexed after frombuffer.
            # So Python's qxnfpel[k] = Fortran's QXNFPEL(k+1)
            # For element NELEM (1-based), ion ZEFF (1-based):
            #   Fortran NELION = (NELEM-1)*6 + ZEFF
            #   Python index = NELION - 1 = (NELEM-1)*6 + ZEFF - 1 = (NELEM-1)*6 + (ZEFF-1)
            #                = elem_0based*6 + ion_0based where ion_0based = ZEFF-1
            # This matches our original formula! The indexing IS correct.
            for elem_idx in range(139):
                for ion_idx in range(6):
                    flat_idx = elem_idx * 6 + ion_idx
                    if flat_idx < mw6:
                        population[depth_idx, ion_idx, elem_idx] = qxnfpel[flat_idx]
                        doppler[depth_idx, ion_idx, elem_idx] = qdopple[flat_idx]
        rec_index += 1

    # CRITICAL: Fortran sorts edges BEFORE computing coefficients (xnfpelsyn.for lines 185-202)
    # Then writes sorted edges to fort.10 (line 204) and coefficients in sorted order (lines 361-363)
    # So coefficients in fort.10 are ALREADY in sorted edge order!
    #
    # If edges read from fort.10 are NOT sorted, that means:
    #   - Either Fortran wrote them unsorted (unlikely - bug check needed)
    #   - Or Python reads them incorrectly
    #   - But coefficients are still in sorted order
    #
    # Solution: If we sort edges, coefficients are already in sorted order, so NO REORDERING needed!
    # The coefficients we read are already indexed by sorted edge intervals.
    #
    # However, if edges in fort.10 are unsorted, we need to:
    #   1. Sort edges to match Fortran's internal sorted order
    #   2. Keep coefficients as-is (they're already in sorted order)
    #   3. But we need to map: when we look up an edge index in sorted order,
    #      we need to find which coefficient interval corresponds to it
    #
    # Actually, wait - if coefficients are in sorted order and we sort edges,
    # then coefficient[i] corresponds to sorted_edge_interval[i], which is what we want!
    # So NO reordering needed - coefficients are already correct!

    # CRITICAL INSIGHT: After reading edges interleaved, they are already sorted!
    # Fortran computes coefficients using sorted edges, and writes them in frequency order
    # Coefficients[i] corresponds to sorted_edge_interval[i]
    # Since edges are already sorted, coefficients are already correctly aligned!
    # NO REORDERING NEEDED!

    extra_tables: dict[str, np.ndarray] = {}
    tail_records = len(records) - rec_index

    def _extract_table(label: str, width: int) -> np.ndarray:
        nonlocal rec_index
        table = np.frombuffer(
            records[rec_index], dtype="<f8", count=nrhox * width
        ).reshape(nrhox, width)
        extra_tables[label] = table
        rec_index += 1
        return table

    if tail_records >= 5:
        extra_tables["bhyd"] = _extract_table("bhyd", 8)
        extra_tables["bc1"] = _extract_table("bc1", 14)
        extra_tables["bc2"] = _extract_table("bc2", 6)
        extra_tables["bsi1"] = _extract_table("bsi1", 11)
        extra_tables["bsi2"] = _extract_table("bsi2", 10)

    half_edge = np.zeros(nedge - 1, dtype=np.float64)
    delta_edge = np.zeros(nedge - 1, dtype=np.float64)
    for edge_idx in range(1, nedge):
        wl_left = abs(wledge[edge_idx - 1])
        wl_right = abs(wledge[edge_idx])
        half_edge[edge_idx - 1] = 0.5 * (wl_left + wl_right)
        delta = wl_right - wl_left
        delta_edge[edge_idx - 1] = 0.5 * delta * delta

    cont_abs_coeff = cont_abs.reshape(layer, nedge - 1, 3)
    cont_scat_coeff = cont_scat.reshape(layer, nedge - 1, 3)

    arrays = dict(
        depth=depth,
        temperature=temperature,
        tkev=tkev,
        tk=tk,
        hkt=hkt,
        tlog=tlog,
        hckt=hckt,
        gas_pressure=gas_pressure,
        electron_density=electron_density,
        xnatm=xnatm,
        mass_density=mass_density,
        turbulent_velocity=turbulent_velocity,
        xnf_h=xnf_h,
        xnf_he1=xnf_he1,
        xnf_he2=xnf_he2,
        xnf_h2=xnf_h2,
        cont_abs=cont_abs,
        cont_scat=cont_scat,
        population_per_ion=population,
        doppler_per_ion=doppler,
        frqedg=frqedg,
        wledge=wledge,
        cmedge=cmedge,
        idmol=idmol,
        momass=momass,
        freqset=freqset,
        half_edge=half_edge,
        delta_edge=delta_edge,
        cont_abs_coeff=cont_abs_coeff,
        cont_scat_coeff=cont_scat_coeff,
        cont_coeff_log10=np.array(True, dtype=np.bool_),
        teff=teff,
        glog=glog,
        title=title,
        **extra_tables,
    )

    np.savez(output_path, **arrays)

    if atlas_tables is not None:
        embed_atlas_tables(atlas_tables, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert fort.10 to NPZ atmosphere data",
    )
    parser.add_argument("fort10", type=Path, help="Path to fort.10 binary")
    parser.add_argument("output", type=Path, help="Destination NPZ file")
    parser.add_argument(
        "--atlas",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "atlas_tables.npz",
        help="Path to atlas_tables.npz (defaults to packaged data)",
    )
    parser.add_argument(
        "--fort5",
        type=Path,
        default=None,
        help="Path to fort.5 atmosphere model file (contains correct RHOX values)",
    )
    args = parser.parse_args()
    convert_fort10(
        args.fort10, args.output, atlas_tables=args.atlas, fort5_path=args.fort5
    )


if __name__ == "__main__":
    main()
