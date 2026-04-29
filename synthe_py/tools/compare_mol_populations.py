"""Compare Fortran fort.10 molecular populations/Doppler with Python's.

Usage:
  python -m synthe_py.tools.compare_mol_populations \
      --fort10 tmp_atlas_debug/t04500_g+2.50_mh-1.00/fortran/fort10.bin \
      --npz    tmp_atlas_debug/t04500_g+2.50_mh-1.00/python/python_iter1_synthe.npz \
      --nelem 61 62 19
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from synthe_py.tools.read_fort10 import read_fort10, Fort10Data
from synthe_py.physics.mol_populations import compute_mol_xnfpmol_dopple


def _load_atm_from_npz(npz_path: str | Path):
    """Load an AtmosphereModel-compatible namespace from the synthe NPZ."""
    from types import SimpleNamespace
    npz = np.load(npz_path, allow_pickle=True)

    atm = SimpleNamespace()
    atm.layers = len(npz["temperature"])
    atm.temperature = npz["temperature"]
    atm.tk = npz["tk"]
    atm.tkev = npz["tkev"]
    atm.hkt = npz["hkt"]
    atm.hckt = npz["hckt"]
    atm.tlog = npz["tlog"]
    atm.mass_density = npz["mass_density"]
    atm.electron_density = npz["electron_density"]
    atm.gas_pressure = npz["gas_pressure"]
    atm.turbulent_velocity = npz["turbulent_velocity"] if "turbulent_velocity" in npz else None
    atm.xabund = npz["xabund"]
    atm.population_per_ion = npz["population_per_ion"]
    atm.doppler_per_ion = npz["doppler_per_ion"]
    atm.depth = npz["depth"]
    atm.xnf_h = npz.get("xnf_h")
    atm.xnf_h2 = npz.get("xnf_h2")
    atm.xnf_h_ion = npz.get("xnf_h_ion")
    atm.xnf_he1 = npz.get("xnf_he1")
    atm.xnf_he2 = npz.get("xnf_he2")
    atm.bhyd = npz.get("bhyd")
    atm.bc1 = npz.get("bc1")
    atm.bo1 = npz.get("bo1")
    atm.bmg1 = npz.get("bmg1")
    atm.bal1 = npz.get("bal1")
    atm.bsi1 = npz.get("bsi1")
    atm.bca1 = npz.get("bca1")

    # idmol and momass for molecular equilibrium
    atm.idmol = npz.get("idmol")
    atm.momass = npz.get("momass")

    # Additional arrays needed by nmolec_exact
    atm.xnatm = npz.get("xnatm")
    atm.xnfpal = npz.get("xnfpal")
    atm.xnfpc = npz.get("xnfpc")
    atm.xnfpfe = npz.get("xnfpfe")
    atm.xnfpmg = npz.get("xnfpmg")
    atm.xnfpsi = npz.get("xnfpsi")
    atm.xnfph = npz.get("xnfph")
    atm.bk1 = npz.get("bk1")
    atm.bna1 = npz.get("bna1")
    atm.bb1 = npz.get("bb1")

    return atm


def compare(
    fort10_path: str,
    npz_path: str,
    nelem_list: list[int],
    molecules_path: str | None = None,
) -> None:
    """Run the comparison."""
    # Load Fortran data
    print("Loading Fortran fort.10 ...")
    fd = read_fort10(fort10_path, skip_continua=True)
    print(f"  NT={fd.nt}, TEFF={fd.teff:.1f}, GLOG={fd.glog:.2f}")

    # Load Python atmosphere
    print(f"Loading Python NPZ: {npz_path} ...")
    atm = _load_atm_from_npz(npz_path)
    print(f"  layers={atm.layers}")

    # Compute Python molecular populations
    nelion_set = {ne * 6 for ne in nelem_list if ne >= 40}
    print(f"Computing Python molecular populations for NELION = {sorted(nelion_set)} ...")
    mol_path = Path(molecules_path) if molecules_path else None
    if mol_path is None:
        candidates = [
            Path("/Users/ElliotKim/Desktop/Research/all_kurucz/kurucz/lines/molecules.dat"),
        ]
        for c in candidates:
            if c.exists():
                mol_path = c
                break
    print(f"  molecules.dat: {mol_path}")

    xnfpmol_dict, dopple_dict = compute_mol_xnfpmol_dopple(
        atm=atm,
        nelion_set=nelion_set,
        molecules_path=mol_path,
    )
    print(f"  Python computed {len(xnfpmol_dict)} NELION species")

    # Also compute Python atomic populations for comparison
    # For atomic species, population_per_ion[:, ion-1, nelem-1] is already in the NPZ

    for ne in nelem_list:
        nelion = ne * 6
        label = {19: "K I", 61: "TiO", 62: "VO"}.get(ne, f"NELEM{ne}")
        print()
        print(f"{'='*100}")
        print(f"Species: {label}  NELEM={ne}  NELION={nelion}")
        print(f"{'='*100}")

        if ne >= 40:
            # Molecular: compare XNFPEL(6, ne) and DOPPLE(6, ne)
            ei = ne - 1  # 0-based
            fort_xnfpel6 = fd.xnfpel[:, 5, ei]
            fort_dopple6 = fd.dopple[:, 5, ei]

            py_xnfpmol = xnfpmol_dict.get(nelion)
            py_dopple = dopple_dict.get(nelion)

            if py_xnfpmol is None:
                print(f"  WARNING: Python has no data for NELION={nelion}")
                continue

            print(f"{'Depth':>5}  {'Fort XNFPEL(6)':>14}  {'Py XNFPMOL':>14}  "
                  f"{'Ratio(Py/F)':>12}  {'Fort DOP(6)':>12}  {'Py DOP':>12}  "
                  f"{'Dop Ratio':>10}  {'Fort RHO':>12}")
            print("-" * 100)

            max_pop_ratio = 0.0
            max_pop_depth = 0
            max_dop_ratio = 0.0
            max_dop_depth = 0
            nt = min(fd.nt, atm.layers)

            for j in range(nt):
                fxp = fort_xnfpel6[j]
                pxp = py_xnfpmol[j]
                fd6 = fort_dopple6[j]
                pd6 = py_dopple[j]
                frho = fd.rho[j]

                if fxp > 0 and pxp > 0:
                    ratio_pop = pxp / fxp
                    ratio_dop = pd6 / fd6 if fd6 > 0 else float("nan")
                    dev = abs(ratio_pop - 1.0)
                    if dev > max_pop_ratio:
                        max_pop_ratio = dev
                        max_pop_depth = j + 1
                    dev_d = abs(ratio_dop - 1.0)
                    if dev_d > max_dop_ratio:
                        max_dop_ratio = dev_d
                        max_dop_depth = j + 1
                else:
                    ratio_pop = float("nan")
                    ratio_dop = float("nan")

                print(f"{j+1:5d}  {fxp:14.6e}  {pxp:14.6e}  {ratio_pop:12.6f}  "
                      f"{fd6:12.6e}  {pd6:12.6e}  {ratio_dop:10.6f}  {frho:12.4e}")

            print(f"\nMax |Py/Fort - 1| for XNFPEL(6): {max_pop_ratio:.4e} at depth {max_pop_depth}")
            print(f"Max |Py/Fort - 1| for DOPPLE(6):  {max_dop_ratio:.4e} at depth {max_dop_depth}")

        else:
            # Atomic species: compare XNFPEL(1, ne) from fort.10 vs population_per_ion[:, 0, ne-1]
            ei = ne - 1
            fort_xnfpel1 = fd.xnfpel[:, 0, ei]
            fort_dopple1 = fd.dopple[:, 0, ei]

            py_pop1 = atm.population_per_ion[:, 0, ei] if ei < atm.population_per_ion.shape[2] else None
            py_dop1 = atm.doppler_per_ion[:, 0, ei] if ei < atm.doppler_per_ion.shape[2] else None

            if py_pop1 is None:
                print(f"  WARNING: Python has no population_per_ion for NELEM={ne}")
                continue

            print(f"{'Depth':>5}  {'Fort XNFPEL(1)':>14}  {'Py pop(0)':>14}  "
                  f"{'Ratio(Py/F)':>12}  {'Fort DOP(1)':>12}  {'Py DOP':>12}  "
                  f"{'Dop Ratio':>10}")
            print("-" * 100)

            max_pop_ratio = 0.0
            max_pop_depth = 0
            nt = min(fd.nt, atm.layers)

            for j in range(nt):
                fxp = fort_xnfpel1[j]
                pxp = py_pop1[j]
                fd1 = fort_dopple1[j]
                pd1 = py_dop1[j] if py_dop1 is not None else 0.0

                if fxp > 0 and pxp > 0:
                    ratio = pxp / fxp
                    ratio_d = pd1 / fd1 if fd1 > 0 else float("nan")
                    dev = abs(ratio - 1.0)
                    if dev > max_pop_ratio:
                        max_pop_ratio = dev
                        max_pop_depth = j + 1
                else:
                    ratio = float("nan")
                    ratio_d = float("nan")

                print(f"{j+1:5d}  {fxp:14.6e}  {pxp:14.6e}  {ratio:12.6f}  "
                      f"{fd1:12.6e}  {pd1:12.6e}  {ratio_d:10.6f}")

            print(f"\nMax |Py/Fort - 1| for XNFPEL(1): {max_pop_ratio:.4e} at depth {max_pop_depth}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Fortran vs Python molecular populations")
    parser.add_argument("--fort10", required=True, help="Path to fort10.bin")
    parser.add_argument("--npz", required=True, help="Path to python_iter1_synthe.npz")
    parser.add_argument("--nelem", type=int, nargs="+", default=[61, 62, 19],
                        help="NELEM values (default: 61=TiO, 62=VO, 19=KI)")
    parser.add_argument("--molecules", default=None, help="Path to molecules.dat")
    args = parser.parse_args()

    compare(args.fort10, args.npz, args.nelem, args.molecules)
