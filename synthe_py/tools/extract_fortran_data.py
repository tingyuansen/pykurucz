#!/usr/bin/env python3
"""Extract data tables from Fortran source files for exact POPS implementation.

This script parses atlas7v.for to extract:
- POTION array (ionization potentials)
- NNN arrays (partition function data)
- Other constants needed for exact matching
"""

from __future__ import annotations

import re
from pathlib import Path
import numpy as np


def _parse_fortran_real_values(values_content: str) -> list[float]:
    """Parse Fortran DATA numeric payload, skipping continuation labels."""
    values: list[float] = []
    for raw_line in values_content.splitlines():
        # Drop trailing comments and leading continuation counters (e.g. "     2 ").
        line = raw_line.split("!")[0]
        line = re.sub(r"^\s*\d+\s+", "", line)
        if not line.strip():
            continue
        for token in line.replace(",", " ").split():
            cleaned = token.strip().rstrip("/")
            if not cleaned:
                continue
            # Avoid parsing continuation counters as data values.
            if "." not in cleaned and "D" not in cleaned.upper() and "E" not in cleaned.upper():
                continue
            try:
                values.append(float(cleaned.replace("D", "E").replace("d", "e")))
            except ValueError:
                continue
    return values


def extract_potion_data(atlas7v_path: Path, rgfall_path: Path = None) -> np.ndarray:
    """Extract POTION array from atlas7v.for and rgfall.for exactly matching Fortran."""
    potion = np.zeros(999, dtype=np.float64)
    
    # POTHe is defined in rgfall.for, not atlas7v.for
    if rgfall_path is None:
        rgfall_path = atlas7v_path.parent / "rgfall.for"
    
    # EQUIVALENCE mappings from atlas7v.for lines 16342-16440
    equivalence_map = {
        "POTH": 1,      # POTION(1)
        "POTHe": 3,     # POTION(3)
        "POTLi": 6,     # POTION(6)
        "POTBe": 10,    # POTION(10)
        "POTB": 15,     # POTION(15)
        "POTC": 21,     # POTION(21)
        "POTN": 28,     # POTION(28)
        "POTO": 36,     # POTION(36)
        "POTF": 45,     # POTION(45)
        "POTNe": 55,    # POTION(55)
        "POTNa": 66,    # POTION(66)
        "POTMg": 78,    # POTION(78)
        "POTAl": 91,    # POTION(91)
        "POTSi": 105,   # POTION(105)
        "POTP": 120,    # POTION(120)
        "POTS": 136,    # POTION(136)
        "POTCl": 153,   # POTION(153)
        "POTAr": 171,   # POTION(171)
        "POTK": 190,    # POTION(190)
        "POTCa": 210,   # POTION(210)
        "POTSc": 231,   # POTION(231)
        "POTTi": 253,   # POTION(253)
        "POTV": 276,    # POTION(276)
        "POTCr": 300,   # POTION(300)
        "POTMn": 325,   # POTION(325)
        "POTFe": 351,   # POTION(351)
        "POTCo": 378,   # POTION(378)
        "POTNi": 406,   # POTION(406)
        "POTCu": 435,   # POTION(435)
        "POTZn": 465,   # POTION(465)
        # Elements 31-99 use index = IZ*5 + 341 + ION - 1
        "POTGa": 496,   # POTION(496) = 31*5 + 341
        "POTGe": 501,   # POTION(501)
        "POTAs": 506,   # POTION(506)
        "POTSe": 511,   # POTION(511)
        "POTBr": 516,   # POTION(516)
        "POTKr": 521,   # POTION(521)
        "POTRb": 526,   # POTION(526)
        "POTSr": 531,   # POTION(531)
        "POTY": 536,    # POTION(536)
        "POTZr": 541,   # POTION(541)
        "POTNb": 546,   # POTION(546)
        "POTMo": 551,   # POTION(551)
        "POTTc": 556,   # POTION(556)
        "POTRu": 561,   # POTION(561)
        "POTRh": 566,   # POTION(566)
        "POTPd": 571,   # POTION(571)
        "POTAg": 576,   # POTION(576)
        "POTCd": 581,   # POTION(581)
        "POTIn": 586,   # POTION(586)
        "POTSn": 591,   # POTION(591)
        "POTSb": 596,   # POTION(596)
        "POTTe": 601,   # POTION(601)
        "POTI": 606,    # POTION(606)
        "POTXe": 611,   # POTION(611)
        "POTCs": 616,   # POTION(616)
        "POTBa": 621,   # POTION(621)
        "POTLa": 626,   # POTION(626)
        "POTCe": 631,   # POTION(631)
        "POTPr": 636,   # POTION(636)
        "POTNd": 641,   # POTION(641)
        "POTPm": 646,   # POTION(646)
        "POTSm": 651,   # POTION(651)
        "POTEu": 656,   # POTION(656)
        "POTGd": 661,   # POTION(661)
        "POTTb": 666,   # POTION(666)
        "POTDy": 671,   # POTION(671)
        "POTHo": 676,   # POTION(676)
        "POTEr": 681,   # POTION(681)
        "POTTm": 686,   # POTION(686)
        "POTYb": 691,   # POTION(691)
        "POTLu": 696,   # POTION(696)
        "POTHf": 701,   # POTION(701)
        "POTTa": 706,   # POTION(706)
        "POTW": 711,    # POTION(711)
        "POTRe": 716,   # POTION(716)
        "POTOs": 721,   # POTION(721)
        "POTIr": 726,   # POTION(726)
        "POTPt": 731,   # POTION(731)
        "POTAu": 736,   # POTION(736)
        "POTHg": 741,   # POTION(741)
        "POTTl": 746,   # POTION(746)
        "POTPb": 751,   # POTION(751)
        "POTBi": 756,   # POTION(756)
        "POTPo": 761,   # POTION(761)
        "POTAt": 766,   # POTION(766)
        "POTRn": 771,   # POTION(771)
        "POTFr": 776,   # POTION(776)
        "POTRa": 781,   # POTION(781)
        "POTAc": 786,   # POTION(786)
        "POTTh": 791,   # POTION(791)
        "POTPa": 796,   # POTION(796)
        "POTU": 801,    # POTION(801)
        "POTNp": 806,   # POTION(806)
        "POTPu": 811,   # POTION(811)
        "POTAm": 816,   # POTION(816)
        "POTCm": 821,   # POTION(821)
        "POTBk": 826,   # POTION(826)
        "POTCf": 831,   # POTION(831)
        "POTEs": 836,   # POTION(836)
    }
    
    # First extract from atlas7v.for
    with atlas7v_path.open("r") as f:
        lines = f.readlines()
    
    i = 0
    lines_processed = 0
    while i < len(lines):
        line = lines[i]
        lines_processed += 1
        
        # Look for DATA POT* statement (case-insensitive to catch POTHe)
        if "DATA POT" in line.upper():
            # Extract array name - handle both "DATA POTXX/" and "DATA POTXX /"
            match = re.search(r"DATA\s+(POT\w+)\s*/", line)
            if not match:
                # Try without requiring / on same line
                match = re.search(r"DATA\s+(POT\w+)", line)
            if not match:
                # Try case-insensitive
                match = re.search(r"DATA\s+(POT\w+)\s*/", line, re.IGNORECASE)
                if not match:
                    match = re.search(r"DATA\s+(POT\w+)", line, re.IGNORECASE)
            if match:
                array_name = match.group(1)
                
                # Normalize POTHe (case might vary)
                array_name_normalized = array_name
                if array_name.upper() == "POTHE":
                    array_name_normalized = "POTHe"
                
                # Debug POTHe
                if array_name_normalized == "POTHe" or "POTHe" in line or "POTHE" in line.upper():
                    print(f"  DEBUG atlas7v line {i+1}: Found POTHe, array_name='{array_name}', normalized='{array_name_normalized}'")
                
                if array_name_normalized not in equivalence_map:
                    if array_name_normalized == "POTHe":
                        print(f"  ERROR: POTHe not in equivalence_map!")
                    i += 1
                    continue
                
                array_name = array_name_normalized
                
                start_idx = equivalence_map[array_name] - 1  # Convert to 0-based
                
                # Debug POTHe extraction
                if array_name == "POTHe":
                    print(f"  DEBUG: Extracting POTHe at index {start_idx}")
                
                # Collect values across continuation lines
                # The DATA statement may span multiple lines with continuation character
                values_str = ""
                j = i
                while j < len(lines):
                    values_str += lines[j]
                    line_j_stripped = lines[j].strip()
                    # Check if this line has the closing / (at the end, after values)
                    if "/" in lines[j] and line_j_stripped.endswith("/"):
                        # Found closing /, done collecting
                        break
                    j += 1
                    # Safety: don't collect more than 10 continuation lines
                    if j - i > 10:
                        break
                
                # Extract values from the collected string
                # Remove DATA and array name, get content between / ... /
                match_full = re.search(r"DATA\s+POT\w+\s*/\s*(.+?)\s*/", values_str, re.DOTALL)
                if match_full:
                    values_content = match_full.group(1)
                    
                    values = _parse_fortran_real_values(values_content)
                    
                    # Store in POTION array
                    for idx, val in enumerate(values):
                        if start_idx + idx < len(potion):
                            potion[start_idx + idx] = val
                    
                    print(f"Extracted {array_name}: {len(values)} values starting at index {start_idx}")
                    if array_name == "POTHe":
                        print(f"  ✅ POTHe extraction successful: POTION[{start_idx}] = {potion[start_idx]:.6e}, POTION[{start_idx+1}] = {potion[start_idx+1]:.6e}")
                    # Debug POTF extraction
                    if array_name == "POTF":
                        print(f"  ✅ POTF extraction successful: POTION[{start_idx}] = {potion[start_idx]:.6e}, IP = {potion[start_idx] / 8065.479:.6e} eV")
                else:
                    # Debug: Log when extraction fails
                    if array_name in ["POTC", "POTN", "POTO", "POTF", "POTNe"]:
                        print(f"  ⚠️  WARNING: Failed to extract {array_name} - regex match_full failed")
                        print(f"     values_str length: {len(values_str)}, first 200 chars: {values_str[:200]}")
                
                i = j + 1
                continue
        
        i += 1
    
    print(f"Processed {lines_processed} lines from atlas7v.for (total: {len(lines)})")
    
    # Then extract from rgfall.for (for POTHe and other elements)
    if rgfall_path.exists():
        with rgfall_path.open("r") as f:
            rgfall_lines = f.readlines()
        
        i = 0
        while i < len(rgfall_lines):
            line = rgfall_lines[i]
            
            # Look for DATA POT* statement (case-insensitive to catch POTHe)
            if "DATA POT" in line.upper():
                # Try exact match first (case-sensitive)
                match = re.search(r"DATA\s+(POT\w+)\s*/", line)
                if not match:
                    # Try without / on same line
                    match = re.search(r"DATA\s+(POT\w+)", line)
                if not match:
                    # Try case-insensitive
                    match = re.search(r"DATA\s+(POT\w+)\s*/", line, re.IGNORECASE)
                    if not match:
                        match = re.search(r"DATA\s+(POT\w+)", line, re.IGNORECASE)
                if match:
                    array_name = match.group(1)
                    # Debug: Check if this is POTHe
                    if "POTHe" in line or "POTHE" in line.upper():
                        print(f"  DEBUG line {i+1}: Found POTHe, matched array_name='{array_name}'")
                    
                    # Normalize array name (POTHe -> POTHe, but check both)
                    array_name_normalized = array_name
                    if array_name.upper() == "POTHE":
                        array_name_normalized = "POTHe"
                    
                    if array_name_normalized not in equivalence_map:
                        if array_name.upper().startswith("POT"):
                            print(f"  DEBUG: Found {array_name} but not in equivalence_map")
                        i += 1
                        continue
                    
                    array_name = array_name_normalized
                    
                    start_idx = equivalence_map[array_name] - 1  # Convert to 0-based
                    
                    # Collect values across continuation lines
                    values_str = ""
                    j = i
                    found_closing = "/" in line
                    while j < len(rgfall_lines):
                        values_str += rgfall_lines[j]
                        if "/" in rgfall_lines[j]:
                            found_closing = True
                            if j > i:
                                break
                        j += 1
                        if found_closing and j > i:
                            break
                    
                    # Extract values
                    match_full = re.search(r"DATA\s+POT\w+\s*/\s*(.+?)\s*/", values_str, re.DOTALL)
                    if match_full:
                        values_content = match_full.group(1)
                        
                        values = _parse_fortran_real_values(values_content)
                        
                        # Store in POTION array (overwrite if already set from atlas7v.for)
                        for idx, val in enumerate(values):
                            if start_idx + idx < len(potion):
                                potion[start_idx + idx] = val
                        
                        print(f"Extracted {array_name} from rgfall.for: {len(values)} values starting at index {start_idx}")
                        if array_name == "POTHe":
                            print(f"  POTHe values: {values[:3]}")
                            print(f"  Stored at POTION[{start_idx}] = {potion[start_idx]}, POTION[{start_idx+1}] = {potion[start_idx+1]}")
                    
                    # Advance to line after the DATA statement (j is the last line with /)
                    i = j + 1
                    continue
            
            i += 1
    else:
        print(f"Warning: rgfall.for not found at {rgfall_path}, some POTION values may be missing")
    
    # CRITICAL FIX: Ensure POTHe is extracted (it's in both atlas7v and rgfall)
    # If still missing, extract directly from rgfall
    if potion[2] == 0.0 or potion[3] == 0.0:
        if rgfall_path.exists():
            with rgfall_path.open("r") as f:
                rgfall_lines = f.readlines()
            for i, line in enumerate(rgfall_lines):
                if "DATA POTHe" in line or "DATA POTHE" in line.upper():
                    match_full = re.search(r"DATA\s+POTHe\s*/\s*(.+?)\s*/", line, re.DOTALL | re.IGNORECASE)
                    if match_full:
                        values_content = match_full.group(1)
                        values = _parse_fortran_real_values(values_content)
                        if len(values) >= 2:
                            potion[2] = values[0]  # He I -> He II
                            potion[3] = values[1]  # He II -> He III
                            print(f"✅ Force-extracted POTHe from rgfall.for: POTION[2] = {potion[2]:.6e}, POTION[3] = {potion[3]:.6e}")
                    break
    
    return potion


def extract_nnn_data(atlas7v_path: Path) -> np.ndarray:
    """Extract NNN partition function arrays from atlas7v.for.
    
    NNN is a 2D array (6, 374) stored as multiple 1D arrays:
    - NNN01 through NNN40 (each 54 values, except NNN40 has 12)
    - NNN67 (78 values)
    - NNN88 (48 values)
    
    Total: 39*54 + 12 + 78 + 48 = 2106 + 12 + 78 + 48 = 2244 values
    But stored as (6, 374) = 2244 values, so that matches.
    """
    nnn = np.zeros((6, 374), dtype=np.int32)
    
    # EQUIVALENCE mappings from atlas7v.for lines 3032-3053
    # NNN01 starts at NNN(1), NNN02 at NNN(55), etc.
    equivalence_starts = [
        1, 55, 109, 163, 217, 271, 325, 379, 433, 487,
        541, 595, 649, 703, 757, 811, 865, 919, 973, 1027,
        1081, 1135, 1189, 1243, 1297, 1351, 1405, 1459, 1513, 1567,
        1621, 1675, 1729, 1783, 1837, 1891, 1945, 1999, 2053, 2107,
        2119,  # NNN67
        2197,  # NNN88
    ]
    
    with atlas7v_path.open("r") as f:
        lines = f.readlines()
    
    i = 0
    nnn_arrays = {}
    
    while i < len(lines):
        line = lines[i]
        
        # Look for DATA NNN* statement (skip commented lines)
        if line.lstrip().startswith(("c", "C")):
            i += 1
            continue
        if "DATA NNN" in line:
            # Extract array name (NNN01, NNN02, etc.)
            match = re.search(r"DATA\s+(NNN\d+)\s*/", line)
            if match:
                array_name = match.group(1)
                
                # Collect values across continuation lines
                values_str = ""
                j = i
                found_opening = False
                found_closing = False
                while j < len(lines):
                    current_line = lines[j]
                    values_str += current_line
                    
                    # Check for opening / (after DATA NNNXX)
                    if not found_opening and "/" in current_line:
                        found_opening = True
                    
                    # Check for closing / (end of data)
                    if found_opening and "/" in current_line and j > i:
                        # Check if this / is after data (not the opening /)
                        # Look for pattern like " ... /" at end of line or before comment
                        if re.search(r"[,\d]\s*/\s*$|[,\d]\s*/\s*[A-Z]", current_line):
                            found_closing = True
                            break
                    
                    j += 1
                
                # Extract values - NNN contains encoded integers
                # Format: "DATA NNN01/ 1 200020001, 200020011, ... D+F 1.00 /"
                values = []
                
                # Find the data section between the first / and last /
                # Pattern: DATA NNNXX/ ...data... /
                match_data = re.search(r"DATA\s+NNN\d+\s*/\s*(.+?)\s*/", values_str, re.DOTALL)
                if match_data:
                    data_section = match_data.group(1)
                    
                    # Split into lines and process each
                    for data_line in data_section.split("\n"):
                        if data_line.lstrip().startswith(("c", "C")):
                            continue
                        # Remove continuation line markers (digits at column 1-6: "     1 ", "     2 ")
                        data_line = re.sub(r"^\s{0,5}\d+\s+", "", data_line)
                        # Remove comments (everything after capital letters like "D+F", "G", "AEL")
                        data_line = re.sub(r"\s+[A-Z][A-Z0-9.\s]*$", "", data_line)
                        # Extract encoded integers (NNN entries are 6-9 digits).
                        # Avoid shorter tokens to prevent picking up line numbers or comments.
                        for match in re.finditer(r"\b(\d{6,9})\b", data_line):
                            val_str = match.group(1)
                            try:
                                val = int(val_str)
                                values.append(val)
                            except ValueError:
                                pass
                
                # Remove last value if it's a comment/reference (like "D+F 1.00")
                # Actually, the last value before the comment is the IP value
                # Let's keep all numeric values
                nnn_arrays[array_name] = values
                print(f"Extracted {array_name}: {len(values)} values")
                
                i = j + 1
                continue
        
        i += 1
    
    # Map to NNN array based on EQUIVALENCE
    # NNN01 -> NNN(1) through NNN(54)
    # NNN02 -> NNN(55) through NNN(108)
    # etc.
    for idx, start_pos in enumerate(equivalence_starts):
        array_name = f"NNN{idx+1:02d}" if idx < 40 else ("NNN67" if idx == 40 else "NNN88")
        
        if array_name in nnn_arrays:
            values = nnn_arrays[array_name]
            # Convert 1-based Fortran linear index to 0-based Python.
            # Fortran is column-major, with the FIRST index (I=1..6) varying fastest:
            #   linear_idx -> I = (linear_idx % 6), N = (linear_idx // 6)
            # This maps to NNN[I, N] in Python as row=I, col=N.
            start_idx = start_pos - 1
            
            for val_idx, val in enumerate(values):
                linear_idx = start_idx + val_idx
                if linear_idx < nnn.size:
                    row = linear_idx % 6
                    col = linear_idx // 6
                    nnn[row, col] = val
    
    return nnn


def extract_atmass_data() -> np.ndarray:
    """Extract atomic masses - use standard values since not found in Fortran source.
    
    Returns array of 99 atomic masses in atomic mass units (amu).
    These are standard values from chemistry.
    
    NOTE: Fortran stores ATMASS in amu, NOT grams. The conversion to grams
    happens later when computing RHO: RHO = XNATOM * WTMOLE * 1.660D-24
    """
    # Standard atomic masses (in amu, then convert to g)
    # Using IUPAC 2019 standard values
    standard_masses_amu = [
        1.008,      # H
        4.002602,   # He
        6.94,       # Li
        9.0121831,  # Be
        10.81,      # B
        12.011,     # C
        14.007,     # N
        15.999,     # O
        18.998,     # F
        20.180,     # Ne
        22.98976928, # Na
        24.305,     # Mg
        26.9815385, # Al
        28.085,     # Si
        30.973761998, # P
        32.06,      # S
        35.45,      # Cl
        39.948,     # Ar
        39.0983,    # K
        40.078,     # Ca
        44.955908,  # Sc
        47.867,     # Ti
        50.9415,    # V
        51.9961,    # Cr
        54.938044,  # Mn
        55.845,     # Fe
        58.933194,  # Co
        58.6934,    # Ni
        63.546,     # Cu
        65.38,      # Zn
        69.723,     # Ga
        72.630,     # Ge
        74.921595,  # As
        78.971,     # Se
        79.904,     # Br
        83.798,     # Kr
        85.4678,    # Rb
        87.62,      # Sr
        88.90584,   # Y
        91.224,     # Zr
        92.90637,   # Nb
        95.95,      # Mo
        97.90721,   # Tc (most stable isotope)
        101.07,     # Ru
        102.90550,  # Rh
        106.42,     # Pd
        107.8682,   # Ag
        112.414,    # Cd
        114.818,    # In
        118.710,    # Sn
        121.760,    # Sb
        127.60,     # Te
        126.90447,  # I
        131.293,    # Xe
        132.90545196, # Cs
        137.327,    # Ba
        138.90547,  # La
        140.116,    # Ce
        140.90766,  # Pr
        144.242,    # Nd
        145.0,      # Pm (most stable)
        150.36,     # Sm
        151.964,    # Eu
        157.25,     # Gd
        158.92535,  # Tb
        162.500,    # Dy
        164.93033,  # Ho
        167.259,    # Er
        168.93422,  # Tm
        173.045,    # Yb
        174.9668,   # Lu
        178.49,     # Hf
        180.94788,  # Ta
        183.84,     # W
        186.207,    # Re
        190.23,     # Os
        192.217,    # Ir
        195.084,    # Pt
        196.966569, # Au
        200.592,    # Hg
        204.38,     # Tl
        207.2,      # Pb
        208.98040,  # Bi
        209.0,      # Po (most stable)
        210.0,      # At (most stable)
        222.0,      # Rn (most stable)
        223.0,      # Fr (most stable)
        226.0,      # Ra (most stable)
        227.0,      # Ac (most stable)
        232.0377,   # Th
        231.03588,  # Pa
        238.02891,  # U
        237.0,      # Np (most stable)
        244.0,      # Pu (most stable)
        243.0,      # Am (most stable)
        247.0,      # Cm (most stable)
        247.0,      # Bk (most stable)
        251.0,      # Cf (most stable)
        252.0,      # Es (most stable)
    ]
    
    # Fortran stores ATMASS in amu (atomic mass units), NOT grams!
    # See atlas7v.for line 2730: DATA ATMASS/ 1.008,4.003,...
    # The conversion to grams happens later when computing RHO:
    # RHO(J) = XNATOM(J) * WTMOLE * 1.660D-24
    # So keep ATMASS in amu here
    atmass = np.array(standard_masses_amu, dtype=np.float64)
    
    # Pad to 99 elements if needed
    if len(atmass) < 99:
        atmass = np.pad(atmass, (0, 99 - len(atmass)), constant_values=0.0)
    
    return atmass


if __name__ == "__main__":
    atlas7v_path = Path(__file__).parent.parent.parent / "src" / "atlas7v.for"
    
    if not atlas7v_path.exists():
        print(f"ERROR: {atlas7v_path} not found")
        exit(1)
    
    print("Extracting POTION data...")
    potion = extract_potion_data(atlas7v_path)
    
    print("Extracting NNN data...")
    nnn = extract_nnn_data(atlas7v_path)
    print(f"  Extracted {np.count_nonzero(nnn)} non-zero values")
    
    print("Extracting ATMASS data...")
    atmass = extract_atmass_data()
    print(f"  Extracted {len(atmass)} atomic masses")
    
    # Save to NPZ in data directory (to match atlas_tables.npz location)
    output_path = Path(__file__).parent.parent / "data" / "fortran_data.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, potion=potion, nnn=nnn, atmass=atmass)
    print(f"Saved to {output_path}")
    print("  NOTE: This file should be committed to the repository.")

