#!/usr/bin/env python3
"""Parser for Kurucz ATLAS9/ATLAS12 fort.5 model atmosphere files.

Based on the format from atlas7v.for line 554:
  FORMAT(10HREAD DECK6I3,33H RHOX,T,P,XNE,ABROSS,ACCRAD,VTURB/
         (1PD15.8,0PF9.1,1P5E10.3))

Fort.5 format:
- Line 1: TEFF, GRAV, etc.
- Line 2: TITLE
- ...
- Line with "READ DECK6": marks start of atmosphere table
- Following lines: RHOX, T, P, XNE, ABROSS, ACCRAD, VTURB
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

import numpy as np


class AtmosphereData(NamedTuple):
    """Parsed atmosphere data from fort.5."""
    rhox: np.ndarray  # Column mass (g/cm²)
    temperature: np.ndarray  # Temperature (K)
    pressure: np.ndarray  # Gas pressure (dyne/cm²)
    electron_density: np.ndarray  # Electron number density (cm⁻³)
    abross: np.ndarray  # Rosseland mean absorption
    accrad: np.ndarray  # Radiative acceleration
    vturb: np.ndarray  # Turbulent velocity (cm/s)
    teff: float
    glog: float
    title: str


def parse_fort5(path: Path) -> AtmosphereData:
    """Parse a Kurucz fort.5 model atmosphere file.

    Parameters
    ----------
    path : Path
        Path to fort.5 file

    Returns
    -------
    AtmosphereData
        Parsed atmosphere structure with RHOX in g/cm²
    """
    with open(path, 'r') as f:
        lines = f.readlines()

    # Parse header
    teff = None
    glog = None
    title = None

    for i, line in enumerate(lines):
        # Look for TEFF
        if 'TEFF' in line and teff is None:
            match = re.search(r'TEFF\s*[A-Z]?\s*([\d.]+)', line)
            if match:
                teff = float(match.group(1))

        # Look for GRAVITY or GRAV
        if ('GRAVITY' in line or 'GRAV' in line) and glog is None:
            match = re.search(r'GRAV(?:ITY)?\s*[A-Z]?\s*([\d.]+)', line)
            if match:
                glog = float(match.group(1))

        # Look for TITLE
        if 'TITLE' in line and title is None:
            # Title is typically on the next line or rest of current line
            title_match = re.search(r'TITLE\s+(.+)', line)
            if title_match:
                title = title_match.group(1).strip()
            elif i + 1 < len(lines):
                title = lines[i + 1].strip()

        # Found atmosphere table
        if 'READ DECK' in line and 'RHOX' in line:
            # Next lines contain the atmosphere data
            atm_start = i + 1
            break
    else:
        raise ValueError("Could not find 'READ DECK' atmosphere table marker")

    # Parse atmosphere table
    rhox_list = []
    temp_list = []
    pres_list = []
    elec_list = []
    abro_list = []
    accr_list = []
    vtur_list = []

    # Format: (1PD15.8,0PF9.1,1P5E10.3)
    # RHOX (D15.8), T (F9.1), P (E10.3), XNE (E10.3), ABROSS (E10.3), ACCRAD (E10.3), VTURB (E10.3)
    for line in lines[atm_start:]:
        # Skip empty lines or lines with non-numeric content
        line = line.strip()
        if not line or line.startswith('READ') or line.startswith('PRADK'):
            continue

        # Try to parse the atmosphere data line
        # Format can vary but typically: RHOX T P XNE ABROSS ACCRAD VTURB [optional extra values]
        parts = line.split()
        if len(parts) < 7:
            # Not enough values, might be end of atmosphere data
            continue

        try:
            # Parse each field
            # RHOX might be in D format (e.g., 1.234D-04) or E format
            rhox_str = parts[0].replace('D', 'E').replace('d', 'e')
            rhox = float(rhox_str)

            temp = float(parts[1])
            pres = float(parts[2].replace('D', 'E').replace('d', 'e'))
            elec = float(parts[3].replace('D', 'E').replace('d', 'e'))
            abro = float(parts[4].replace('D', 'E').replace('d', 'e'))
            accr = float(parts[5].replace('D', 'E').replace('d', 'e'))
            vtur = float(parts[6].replace('D', 'E').replace('d', 'e'))

            # Sanity checks
            if rhox <= 0 or temp <= 0:
                # Likely hit end of valid data
                break

            rhox_list.append(rhox)
            temp_list.append(temp)
            pres_list.append(pres)
            elec_list.append(elec)
            abro_list.append(abro)
            accr_list.append(accr)
            vtur_list.append(vtur)

        except (ValueError, IndexError):
            # Not a valid data line, might be end of atmosphere table
            if len(rhox_list) > 0:
                # We've already read some data, so stop here
                break
            else:
                # Haven't found data yet, keep looking
                continue

    if len(rhox_list) == 0:
        raise ValueError("No atmosphere data found in fort.5 file")

    # Use defaults if header values not found
    if teff is None:
        teff = 5770.0
        print(f"Warning: TEFF not found in fort.5, using default {teff}")
    if glog is None:
        glog = 4.44
        print(f"Warning: GLOG not found in fort.5, using default {glog}")
    if title is None:
        title = "Atmosphere Model"

    return AtmosphereData(
        rhox=np.array(rhox_list, dtype=np.float64),
        temperature=np.array(temp_list, dtype=np.float64),
        pressure=np.array(pres_list, dtype=np.float64),
        electron_density=np.array(elec_list, dtype=np.float64),
        abross=np.array(abro_list, dtype=np.float64),
        accrad=np.array(accr_list, dtype=np.float64),
        vturb=np.array(vtur_list, dtype=np.float64),
        teff=teff,
        glog=glog,
        title=title,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python parse_fort5.py <fort.5>")
        sys.exit(1)

    fort5_path = Path(sys.argv[1])
    atm = parse_fort5(fort5_path)

    print(f"TEFF: {atm.teff:.1f} K")
    print(f"GLOG: {atm.glog:.3f}")
    print(f"TITLE: {atm.title}")
    print(f"Layers: {len(atm.rhox)}")
    print(f"\nFirst 5 layers:")
    print(f"{'RHOX':>15s} {'T':>10s} {'P':>12s} {'XNE':>12s}")
    for i in range(min(5, len(atm.rhox))):
        print(f"{atm.rhox[i]:15.6E} {atm.temperature[i]:10.1f} "
              f"{atm.pressure[i]:12.3E} {atm.electron_density[i]:12.3E}")
