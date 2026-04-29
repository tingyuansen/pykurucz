#!/usr/bin/env python3
"""Helper function to compute XNATOM with molecular equilibrium (NMOLEC equivalent).

This function can be called from anywhere in convert_atm_to_npz.py to ensure
XNATOM includes molecular contributions when molecules are enabled.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional, Tuple

try:
    from synthe_py.tools.readmol_exact import readmol_exact
    from synthe_py.tools.nmolec_exact import nmolec_exact
    from synthe_py.tools.pops_exact import pfsaha_exact
    from synthe_py.tools.departure_tables import initialize_departure_tables

    NMOLEC_AVAILABLE = True
except ImportError:
    NMOLEC_AVAILABLE = False


def compute_xnatom_with_molecules(
    xnatom_atomic: np.ndarray,
    temperature: np.ndarray,
    tkev: np.ndarray,
    tk: np.ndarray,
    tlog: np.ndarray,
    gas_pressure: np.ndarray,
    electron_density: np.ndarray,
    xabund: np.ndarray,  # Element abundances (99,)
    wtmole: float,
    molecules_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Compute XNATOM with molecular equilibrium (NMOLEC equivalent).

    Args:
        xnatom_atomic: Atomic-only XNATOM = P/TK - XNE
        temperature: Temperature array (n_layers,)
        tkev: k_B*T in eV array (n_layers,)
        tk: k_B*T array (n_layers,)
        tlog: log(T) array (n_layers,)
        gas_pressure: Gas pressure array (n_layers,)
        electron_density: Electron density array (n_layers,)
        xabund: Element abundances (99,)
        wtmole: Mean molecular weight (amu)
        molecules_path: Path to molecules.dat file (optional, will search if None)

    Returns:
        (xnatom_molecular, success) where:
        - xnatom_molecular: XNATOM with molecular contributions (or atomic if failed)
        - success: True if NMOLEC succeeded, False otherwise
    """
    if not NMOLEC_AVAILABLE:
        return xnatom_atomic, False

    # Find molecules.dat file if not provided
    if molecules_path is None:
        script_dir = Path(__file__).parent
        possible_paths = [
            script_dir.parent.parent / "lines" / "molecules.dat",
            script_dir.parent.parent / "synthe" / "stmp_at12_aaaaa" / "fort.2",
            Path("lines") / "molecules.dat",
            Path("synthe") / "stmp_at12_aaaaa" / "fort.2",
        ]

        for path in possible_paths:
            if path.exists():
                molecules_path = path
                break

    if molecules_path is None or not molecules_path.exists():
        return xnatom_atomic, False

    try:
        # Read molecular data
        nummol, code_mol, equil, locj, kcomps, idequa, nequa, nloc = readmol_exact(
            molecules_path
        )

        # Prepare PFSAHA wrapper
        n_layers = len(temperature)
        answer_full = np.zeros((n_layers, 31), dtype=np.float64)
        departure_tables = initialize_departure_tables(n_layers)
        electron_work = electron_density.copy()

        def pfsaha_wrapper(j, iz, nion, mode, frac, nlte_on):
            """Wrapper for PFSAHA to match nmolec_exact interface.

            CRITICAL: Use electron_work (not a stale copy) so PFSAHA sees the
            updated electron density from NMOLEC seeding (XNE = XNTOT/20).
            """
            pfsaha_exact(
                j=0,  # Process all layers
                iz=iz,
                nion=nion,
                mode=mode,
                temperature=temperature,
                tkev=tkev,
                tk=tk,
                hkt=6.6256e-27 / tk,  # H_PLANCK / TK
                hckt=6.6256e-27 * 2.99792458e10 / tk,  # H_PLANCK * C / TK
                tlog=tlog,
                gas_pressure=gas_pressure,
                electron_density=electron_work,  # Use updated value from NMOLEC seeding
                xnatom=xnatom_atomic,
                answer=answer_full,
                departure_tables=departure_tables,
                nlte_on=nlte_on,
            )
            frac[j, :] = answer_full[j, :]

        bhyd = departure_tables["bhyd"]
        bc1 = departure_tables["bc1"]
        bo1 = departure_tables["bo1"]
        bmg1 = departure_tables["bmg1"]
        bal1 = departure_tables["bal1"]
        bsi1 = departure_tables["bsi1"]
        bca1 = departure_tables["bca1"]

        # Call NMOLEC with auto_gibbs enabled for cool atmospheres
        # This uses Gibbs free energy minimization instead of Newton iteration
        # for temperatures below 5000K, avoiding convergence issues
        xnatom_molecular, _, _ = nmolec_exact(
            n_layers=n_layers,
            temperature=temperature,
            tkev=tkev,
            tk=tk,
            tlog=tlog,
            gas_pressure=gas_pressure,
            electron_density=electron_work,  # NMOLEC modifies in-place
            xabund=xabund,
            xnatom_atomic=xnatom_atomic,
            nummol=nummol,
            code_mol=code_mol,
            equil=equil,
            locj=locj,
            kcomps=kcomps,
            idequa=idequa,
            nequa=nequa,
            bhyd=bhyd,
            bc1=bc1,
            bo1=bo1,
            bmg1=bmg1,
            bal1=bal1,
            bsi1=bsi1,
            bca1=bca1,
            pfsaha_func=pfsaha_wrapper,
            auto_gibbs=True,  # Auto-enable Gibbs for T < 5000K
        )

        return xnatom_molecular, electron_work, True

    except Exception as e:
        print(f"  WARNING: NMOLEC failed: {e}")
        import traceback

        traceback.print_exc()
        return xnatom_atomic, electron_density, False
