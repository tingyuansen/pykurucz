"""Atmosphere file handling for the Python SYNTHE pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

from ..physics.tables import ContinuumTables, build_continuum_tables
from ..physics import atlas_tables as atlas_loader
from ..physics import bfudge_tables


@dataclass
class AtmosphereModel:
    """Structured representation of a 1D stellar atmosphere."""

    depth: np.ndarray
    temperature: np.ndarray
    gas_pressure: np.ndarray
    electron_density: np.ndarray
    mass_density: np.ndarray
    turbulent_velocity: np.ndarray
    metadata: Dict[str, str]
    tkev: Optional[np.ndarray] = None
    tk: Optional[np.ndarray] = None
    tlog: Optional[np.ndarray] = None
    hkt: Optional[np.ndarray] = None
    continuum_tables: Optional[ContinuumTables] = None
    continuum_frequency: Optional[np.ndarray] = None
    continuum_wledge: Optional[np.ndarray] = None
    continuum_half_edge: Optional[np.ndarray] = None
    continuum_delta_edge: Optional[np.ndarray] = None
    continuum_abs_coeff: Optional[np.ndarray] = None
    continuum_scat_coeff: Optional[np.ndarray] = None
    continuum_coeff_log10: bool = False
    hckt: Optional[np.ndarray] = None
    txnxn: Optional[np.ndarray] = None
    xnf_h: Optional[np.ndarray] = None
    xnf_he1: Optional[np.ndarray] = None
    xnf_he2: Optional[np.ndarray] = None
    xnf_h2: Optional[np.ndarray] = None
    xnfph: Optional[np.ndarray] = None
    dopph: Optional[np.ndarray] = None
    cont_absorption: Optional[np.ndarray] = None
    cont_scattering: Optional[np.ndarray] = None
    line_source_lower: Optional[np.ndarray] = None
    line_source_upper: Optional[np.ndarray] = None
    population_per_ion: Optional[np.ndarray] = None
    doppler_per_ion: Optional[np.ndarray] = None
    xabund: Optional[np.ndarray] = (
        None  # CRITICAL: Elemental abundances array (99 elements, linear scale)
    )
    bhyd: Optional[np.ndarray] = None
    bc1: Optional[np.ndarray] = None
    bc2: Optional[np.ndarray] = None
    bsi1: Optional[np.ndarray] = None
    bsi2: Optional[np.ndarray] = None
    xnatm: Optional[np.ndarray] = (
        None  # CRITICAL: Total number density of atoms (from NMOLEC)
    )
    atlas_tables: Optional[Dict[str, np.ndarray]] = None
    # Metal populations for UV opacities
    xnfpc: Optional[np.ndarray] = None  # Carbon I populations (n_layers, nion) for C1OP
    xnfpmg: Optional[np.ndarray] = (
        None  # Magnesium I populations (n_layers, nion) for MG1OP
    )
    xnfpsi: Optional[np.ndarray] = None  # Silicon I populations for SI1OP
    xnfpal: Optional[np.ndarray] = None  # Aluminum I populations for AL1OP
    xnfpfe: Optional[np.ndarray] = None  # Iron I populations for FE1OP
    # Molecular populations for COOLOP (from NMOLEC)
    xnfpch: Optional[np.ndarray] = None  # CH molecular population (n_layers,) for CHOP
    xnfpoh: Optional[np.ndarray] = None  # OH molecular population (n_layers,) for OHOP

    @property
    def layers(self) -> int:
        return self.temperature.size


def load_cached(path: Path) -> AtmosphereModel:
    """Load a cached numpy `.npz` representation of the atmosphere."""

    with np.load(path, allow_pickle=True) as data:
        # Load metadata (keys starting with "meta_")
        metadata = {}
        abundances_dict = {}
        for key in data:
            if key.startswith("meta_"):
                key_suffix = key[5:]  # Remove "meta_" prefix
                if key_suffix.startswith("abundances_"):
                    # Reconstruct abundances dict: meta_abundances_<elem_num> -> abundances[<elem_num>]
                    elem_num = int(key_suffix.split("_")[1])
                    abundances_dict[elem_num] = float(data[key].item())
                elif key_suffix == "abundance_scale":
                    metadata["abundance_scale"] = float(data[key].item())
                else:
                    metadata[key_suffix] = data[key].item()

        # Store abundances dict in metadata
        if abundances_dict:
            metadata["abundances"] = abundances_dict

        # CRITICAL FIX: Load xabund array if present
        xabund = None
        if "xabund" in data:
            xabund = np.asarray(data["xabund"], dtype=np.float64)

        def _pick(*names: str, required: bool = True) -> np.ndarray:
            for name in names:
                if name in data:
                    return np.asarray(data[name])
            if required:
                missing = " / ".join(names)
                raise KeyError(f"Required atmosphere field missing: {missing}")
            return np.array([], dtype=np.float64)

        depth = _pick("depth", "rhox", "qrhox")
        temperature = _pick("temperature", "qt")
        tkev = data.get("tkev")
        tk = data.get("tk")
        tlog = data.get("tlog")
        hkt = data.get("hkt")
        gas_pressure = _pick("gas_pressure", "qp")
        electron_density = _pick("electron_density", "qxne")
        mass_density = _pick("mass_density", "qrho")
        turbulent_velocity = _pick("turbulent_velocity", "qvturb")
        n_layers = depth.size

        hckt = data.get("hckt")
        if hckt is None and "qhckt" in data:
            hckt = np.asarray(data["qhckt"])

        txnxn = data.get("txnxn")
        xnf_h = data.get("xnf_h")
        if xnf_h is None and "qxnf_h" in data:
            xnf_h = np.asarray(data["qxnf_h"])

        xnf_he1 = data.get("xnf_he1")
        xnf_he2 = data.get("xnf_he2")
        if (xnf_he1 is None or xnf_he2 is None) and "qxnf_he" in data:
            he = np.asarray(data["qxnf_he"])
            if he.ndim == 2 and he.shape[1] >= 2:
                xnf_he1 = he[:, 0]
                xnf_he2 = he[:, 1]

        xnf_h2 = data.get("xnf_h2")
        if xnf_h2 is None and "qxnf_h2" in data:
            xnf_h2 = np.asarray(data["qxnf_h2"])

        # CRITICAL: Load xnatm (total number density of atoms, includes molecular contributions)
        xnatm = data.get("xnatm")
        if xnatm is not None:
            xnatm = np.asarray(xnatm, dtype=np.float64)

        xnfph = data.get("xnfph")
        if xnfph is None and "qxnfpel" in data:
            qx = np.asarray(data["qxnfpel"])
            if qx.ndim == 2 and qx.shape[1] >= 2:
                xnfph = qx[:, :2]

        dopph = data.get("dopph")
        if dopph is None and "qdopple" in data:
            qd = np.asarray(data["qdopple"])
            if qd.ndim == 2:
                dopph = qd[:, 0]

        line_source_lower = data.get("line_source_lower")
        line_source_upper = data.get("line_source_upper")
        # Populations are now always computed from Saha-Boltzmann equations (no fort.10 dependency)
        # Still load population_per_ion if present for backward compatibility, but it won't be used
        population_per_ion = data.get("population_per_ion")
        if population_per_ion is not None:
            population_per_ion = np.asarray(population_per_ion, dtype=np.float64)
            # Note: population_per_ion is loaded for backward compatibility only
            # It is not used for computation - populations are computed from Saha-Boltzmann
        elif "qxnfpel" in data:
            # Legacy format - convert if present
            qx = np.asarray(data["qxnfpel"], dtype=np.float64)
            if qx.ndim == 2 and qx.shape[1] % 6 == 0:
                population_per_ion = qx.reshape(qx.shape[0], 6, qx.shape[1] // 6)
                # Note: qxnfpel is loaded for backward compatibility only

        # CRITICAL FIX: xnfph must store ground-state H I (mode=11) and H II.
        # Fortran uses XNFPH(:,1) for H I ground state and XNFPH(:,2) for H II.
        # The NPZ stores total neutral H as xnf_h (mode=12), so we convert using U(T).
        xnf_h_ion = data.get("xnf_h_ion")
        if xnf_h_ion is not None:
            xnf_h_ion = np.asarray(xnf_h_ion, dtype=np.float64)

        if xnfph is None and xnf_h is not None:
            from ..physics.kapp import compute_ground_state_hydrogen

            ground_h = compute_ground_state_hydrogen(xnf_h, temperature)
            if xnf_h_ion is None:
                # Fallback: use zeros if H II is unavailable
                xnf_h_ion = np.zeros_like(xnf_h)
            xnfph = np.column_stack([ground_h, xnf_h_ion])

        # Fallback to population_per_ion only if xnf_h is not available
        if (
            xnfph is None
            and population_per_ion is not None
            and population_per_ion.size > 0
        ):
            # Identify the species with the largest combined population in stages 0 and 1
            hydrogen_candidate = population_per_ion[:, :2, :]
            species_totals = hydrogen_candidate.sum(axis=(0, 1))
            hydrogen_index = int(np.argmax(species_totals))
            if species_totals[hydrogen_index] > 0.0:
                xnfph = hydrogen_candidate[:, :, hydrogen_index]

        doppler_per_ion = data.get("doppler_per_ion")
        if doppler_per_ion is None and "qdopple" in data:
            qd = np.asarray(data["qdopple"], dtype=np.float64)
            if qd.ndim == 2 and qd.shape[1] % 6 == 0:
                doppler_per_ion = qd.reshape(qd.shape[0], 6, qd.shape[1] // 6)

        continuum_tables = None
        continuum_frequency = None
        continuum_wledge_arr: Optional[np.ndarray] = None
        continuum_half_edge = None
        continuum_delta_edge = None
        # CRITICAL FIX: Load coefficients directly from NPZ if present
        # These are stored as log10 values in convert_atm_to_npz.py
        if "cont_abs_coeff" in data:
            cont_abs_coeff = np.asarray(data["cont_abs_coeff"], dtype=np.float64)
        else:
            cont_abs_coeff = None
        if "cont_scat_coeff" in data:
            cont_scat_coeff = np.asarray(data["cont_scat_coeff"], dtype=np.float64)
        else:
            cont_scat_coeff = None
        coeff_is_log = bool(data.get("cont_coeff_log10", False))
        if "wledge" in data:
            continuum_wledge_arr = np.asarray(data["wledge"], dtype=np.float64)
            wledge = tuple(float(x) for x in continuum_wledge_arr.tolist())
            ablog_source = None
            if "ablog" in data:
                ablog_source = np.asarray(data["ablog"]).ravel()
            elif "qablog" in data:
                qab = np.asarray(data["qablog"])
                if qab.ndim == 2 and qab.size > 0:
                    ablog_source = qab[0, :]
                    expected = 3 * (continuum_wledge_arr.size - 1)
                    if expected > 0 and ablog_source.size >= expected:
                        ablog_source = ablog_source[:expected]
            if ablog_source is not None:
                continuum_tables = build_continuum_tables(
                    wledge,
                    tuple(float(x) for x in ablog_source.tolist()),
                )
            if "half_edge" in data:
                continuum_half_edge = np.asarray(data["half_edge"], dtype=np.float64)
            if "delta_edge" in data:
                continuum_delta_edge = np.asarray(data["delta_edge"], dtype=np.float64)
        if "freqset" in data:
            continuum_frequency = np.asarray(data["freqset"], dtype=np.float64)
        if "qcontabs" in data:
            cont_absorption = np.asarray(data["qcontabs"], dtype=np.float64)
        elif "cont_abs" in data:
            cont_absorption = np.asarray(data["cont_abs"], dtype=np.float64)

        if "qcontscat" in data:
            cont_scattering = np.asarray(data["qcontscat"], dtype=np.float64)
        elif "cont_scat" in data:
            cont_scattering = np.asarray(data["cont_scat"], dtype=np.float64)

        # Fallback: If coefficients weren't loaded directly, try to derive from cont_abs/cont_scat
        # NOTE: cont_abs/cont_scat are LINEAR values (10^log10), so this fallback should only
        # be used for backward compatibility with old NPZ files that don't have cont_abs_coeff
        if (
            cont_abs_coeff is None
            and cont_absorption is not None
            and continuum_wledge_arr is not None
        ):
            expected = 3 * (continuum_wledge_arr.size - 1)
            if cont_absorption.ndim == 2 and cont_absorption.shape[1] == expected:
                # CRITICAL: cont_absorption is LINEAR, but we need log10 for coefficients
                # Convert to log10 before reshaping
                cont_abs_log_fallback = np.log10(np.maximum(cont_absorption, 1e-30))
                cont_abs_coeff = cont_abs_log_fallback.reshape(
                    n_layers, continuum_wledge_arr.size - 1, 3
                )
                coeff_is_log = True
        if (
            cont_scat_coeff is None
            and cont_scattering is not None
            and continuum_wledge_arr is not None
        ):
            expected = 3 * (continuum_wledge_arr.size - 1)
            if cont_scattering.ndim == 2 and cont_scattering.shape[1] == expected:
                # CRITICAL: cont_scattering is LINEAR, but we need log10 for coefficients
                # Convert to log10 before reshaping
                cont_scat_log_fallback = np.log10(np.maximum(cont_scattering, 1e-30))
                cont_scat_coeff = cont_scat_log_fallback.reshape(
                    n_layers, continuum_wledge_arr.size - 1, 3
                )
                coeff_is_log = True

        # Load B tables (departure coefficients) from file if present
        # Otherwise compute them from atmosphere properties
        bhyd_loaded = data.get("bhyd")
        bc1_loaded = data.get("bc1")
        bc2_loaded = data.get("bc2")
        bsi1_loaded = data.get("bsi1")
        bsi2_loaded = data.get("bsi2")

        # Create a temporary atmosphere object to compute B tables
        # We'll create a minimal one just for computation
        temp_atm = AtmosphereModel(
            depth=depth,
            temperature=temperature,
            gas_pressure=gas_pressure,
            electron_density=electron_density,
            mass_density=mass_density,
            turbulent_velocity=turbulent_velocity,
            metadata=metadata,
        )

        # Compute B tables from atmosphere properties
        # For LTE (default), these will all be 1.0
        bhyd_computed, bc1_computed, bc2_computed, bsi1_computed, bsi2_computed = (
            bfudge_tables.compute_all_b_tables(temp_atm, nlte=False)
        )

        # Use loaded values if available, otherwise use computed
        # Also verify computed matches loaded when both exist
        import logging

        logger = logging.getLogger(__name__)

        if bhyd_loaded is not None:
            bhyd_loaded = np.asarray(bhyd_loaded, dtype=np.float64)
            # Verify computed matches loaded
            if not np.allclose(bhyd_computed, bhyd_loaded, rtol=1e-5, atol=1e-8):
                max_diff = np.max(np.abs(bhyd_computed - bhyd_loaded))
                logger.warning(
                    f"BHYD: computed vs loaded differ (max_diff={max_diff:.2e}). "
                    f"Using loaded values. This may indicate NLTE or other effects."
                )
            bhyd = bhyd_loaded
        else:
            bhyd = bhyd_computed
            logger.debug("BHYD: computed from atmosphere properties (LTE)")

        if bc1_loaded is not None:
            bc1_loaded = np.asarray(bc1_loaded, dtype=np.float64)
            if not np.allclose(bc1_computed, bc1_loaded, rtol=1e-5, atol=1e-8):
                max_diff = np.max(np.abs(bc1_computed - bc1_loaded))
                logger.warning(
                    f"BC1: computed vs loaded differ (max_diff={max_diff:.2e}). "
                    f"Using loaded values."
                )
            bc1 = bc1_loaded
        else:
            bc1 = bc1_computed
            logger.debug("BC1: computed from atmosphere properties (LTE)")

        if bc2_loaded is not None:
            bc2_loaded = np.asarray(bc2_loaded, dtype=np.float64)
            if not np.allclose(bc2_computed, bc2_loaded, rtol=1e-5, atol=1e-8):
                max_diff = np.max(np.abs(bc2_computed - bc2_loaded))
                logger.warning(
                    f"BC2: computed vs loaded differ (max_diff={max_diff:.2e}). "
                    f"Using loaded values."
                )
            bc2 = bc2_loaded
        else:
            bc2 = bc2_computed
            logger.debug("BC2: computed from atmosphere properties (LTE)")

        if bsi1_loaded is not None:
            bsi1_loaded = np.asarray(bsi1_loaded, dtype=np.float64)
            if not np.allclose(bsi1_computed, bsi1_loaded, rtol=1e-5, atol=1e-8):
                max_diff = np.max(np.abs(bsi1_computed - bsi1_loaded))
                logger.warning(
                    f"BSI1: computed vs loaded differ (max_diff={max_diff:.2e}). "
                    f"Using loaded values."
                )
            bsi1 = bsi1_loaded
        else:
            bsi1 = bsi1_computed
            logger.debug("BSI1: computed from atmosphere properties (LTE)")

        if bsi2_loaded is not None:
            bsi2_loaded = np.asarray(bsi2_loaded, dtype=np.float64)
            if not np.allclose(bsi2_computed, bsi2_loaded, rtol=1e-5, atol=1e-8):
                max_diff = np.max(np.abs(bsi2_computed - bsi2_loaded))
                logger.warning(
                    f"BSI2: computed vs loaded differ (max_diff={max_diff:.2e}). "
                    f"Using loaded values."
                )
            bsi2 = bsi2_loaded
        else:
            bsi2 = bsi2_computed
            logger.debug("BSI2: computed from atmosphere properties (LTE)")

        atlas_tables_data = {}
        atlas_key_index = data.get("atlas_tables_keys")
        if atlas_key_index is not None:
            key_list = [
                str(entry).upper() for entry in np.asarray(atlas_key_index).tolist()
            ]
            for raw_key in key_list:
                store_key = f"atlas_{raw_key.lower()}"
                if store_key in data:
                    atlas_tables_data[raw_key] = np.asarray(
                        data[store_key], dtype=np.float64
                    )

        if not atlas_tables_data:
            try:
                atlas_tables_data = atlas_loader.load_atlas_tables()
            except atlas_loader.AtlasTablesMissing:
                atlas_tables_data = {}

        # Extract metal populations for UV opacities
        # Priority: direct keys (xnfpmg, xnfpsi, xnfpfe) > population_per_ion extraction
        # Direct keys are computed by pops_exact in convert_atm_to_npz.py (mode=11)
        # population_per_ion has shape (n_layers, n_ionization_stages=6, n_elements=139)
        # Element indices are 0-based: C=5, Mg=11, Al=12, Si=13, Fe=25
        
        # First, try to load direct metal population keys (PREFERRED)
        xnfpmg = data.get("xnfpmg", None)  # Mg I ground state from pops_exact
        xnfpsi = data.get("xnfpsi", None)  # Si I ground state from pops_exact
        xnfpfe = data.get("xnfpfe", None)  # Fe I ground state from pops_exact
        xnfpc = data.get("xnfpc", None)    # C I ground state (if available)
        xnfpal = data.get("xnfpal", None)  # Al I ground state (if available)
        
        # If any direct keys are present, use them (they're computed correctly from pops_exact)
        has_direct_metal_pops = xnfpmg is not None or xnfpsi is not None or xnfpfe is not None
        
        if has_direct_metal_pops:
            # Use direct keys - these are already ground state populations
            logger.info(f"Using direct metal populations: xnfpmg={xnfpmg is not None}, "
                       f"xnfpsi={xnfpsi is not None}, xnfpfe={xnfpfe is not None}")
        elif population_per_ion is not None and population_per_ion.ndim == 3:
            pop = population_per_ion
            # C I, C II (Z=6, index=5) - need first 2 ionization stages
            if pop.shape[2] > 5:
                xnfpc = pop[:, :2, 5]  # (n_layers, 2)
            # Mg I, Mg II (Z=12, index=11)
            if pop.shape[2] > 11:
                xnfpmg = pop[:, :2, 11]  # (n_layers, 2)
            # Al I, Al II (Z=13, index=12)
            if pop.shape[2] > 12:
                xnfpal = pop[:, :2, 12]  # (n_layers, 2)
            # Si I, Si II (Z=14, index=13)
            if pop.shape[2] > 13:
                xnfpsi = pop[:, :2, 13]  # (n_layers, 2)
            # Fe I (Z=26, index=25)
            if pop.shape[2] > 25:
                xnfpfe = pop[:, :1, 25]  # (n_layers, 1)

        atmosphere = AtmosphereModel(
            depth=depth,
            temperature=temperature,
            tkev=np.asarray(tkev, dtype=np.float64) if tkev is not None else None,
            tk=np.asarray(tk, dtype=np.float64) if tk is not None else None,
            tlog=np.asarray(tlog, dtype=np.float64) if tlog is not None else None,
            hkt=np.asarray(hkt, dtype=np.float64) if hkt is not None else None,
            gas_pressure=gas_pressure,
            electron_density=electron_density,
            mass_density=mass_density,
            turbulent_velocity=turbulent_velocity,
            metadata=metadata,
            continuum_tables=continuum_tables,
            continuum_frequency=continuum_frequency,
            continuum_wledge=continuum_wledge_arr,
            continuum_half_edge=continuum_half_edge,
            continuum_delta_edge=continuum_delta_edge,
            continuum_abs_coeff=cont_abs_coeff,
            continuum_scat_coeff=cont_scat_coeff,
            continuum_coeff_log10=coeff_is_log,
            hckt=hckt,
            txnxn=txnxn,
            xnf_h=xnf_h,
            xnf_he1=xnf_he1,
            xnf_he2=xnf_he2,
            xabund=xabund,  # CRITICAL: Store xabund array for population computation
            xnf_h2=xnf_h2,
            xnfph=xnfph,
            dopph=dopph,
            cont_absorption=cont_absorption,
            cont_scattering=cont_scattering,
            line_source_lower=line_source_lower,
            line_source_upper=line_source_upper,
            population_per_ion=population_per_ion,
            doppler_per_ion=doppler_per_ion,
            bhyd=bhyd,
            bc1=bc1,
            bc2=bc2,
            bsi1=bsi1,
            bsi2=bsi2,
            xnatm=xnatm,  # CRITICAL: Pass xnatm for population computation
            atlas_tables=atlas_tables_data or None,
            xnfpc=xnfpc,
            xnfpmg=xnfpmg,
            xnfpal=xnfpal,
            xnfpsi=xnfpsi,
            xnfpfe=xnfpfe,
        )

    return atmosphere


def save_cached(model: AtmosphereModel, path: Path) -> None:
    """Persist the atmosphere to an `.npz` archive."""

    arrays = {
        "depth": model.depth,
        "temperature": model.temperature,
        "gas_pressure": model.gas_pressure,
        "electron_density": model.electron_density,
        "mass_density": model.mass_density,
        "turbulent_velocity": model.turbulent_velocity,
    }
    if model.continuum_tables is not None:
        arrays.update(
            {
                "wledge": model.continuum_tables.wledge,
                "ablog": model.continuum_tables.ablog,
            }
        )
    if model.continuum_wledge is not None and "wledge" not in arrays:
        arrays["wledge"] = model.continuum_wledge
    if model.continuum_half_edge is not None:
        arrays["half_edge"] = model.continuum_half_edge
    if model.continuum_delta_edge is not None:
        arrays["delta_edge"] = model.continuum_delta_edge
    if model.continuum_abs_coeff is not None:
        arrays["cont_abs_coeff"] = model.continuum_abs_coeff
    if model.continuum_scat_coeff is not None:
        arrays["cont_scat_coeff"] = model.continuum_scat_coeff
    if model.continuum_coeff_log10:
        arrays["cont_coeff_log10"] = np.array(
            model.continuum_coeff_log10, dtype=np.bool_
        )
    optional_fields = {
        "hckt": model.hckt,
        "txnxn": model.txnxn,
        "xnf_h": model.xnf_h,
        "xnf_he1": model.xnf_he1,
        "xnf_he2": model.xnf_he2,
        "xnf_h2": model.xnf_h2,
        "xnfph": model.xnfph,
        "dopph": model.dopph,
        "qcontabs": model.cont_absorption,
        "qcontscat": model.cont_scattering,
        "line_source_lower": model.line_source_lower,
        "line_source_upper": model.line_source_upper,
        "population_per_ion": model.population_per_ion,
        "doppler_per_ion": model.doppler_per_ion,
        "bhyd": model.bhyd,
        "bc1": model.bc1,
        "bc2": model.bc2,
        "bsi1": model.bsi1,
        "bsi2": model.bsi2,
    }
    for key, value in optional_fields.items():
        if value is not None:
            arrays[key] = value
    meta = {f"meta_{k}": v for k, v in model.metadata.items()}
    np.savez(path, **arrays, **meta)


def iterate_layers(model: AtmosphereModel) -> Iterable[int]:
    """Yield layer indices in order of increasing optical depth."""

    return range(model.layers)
