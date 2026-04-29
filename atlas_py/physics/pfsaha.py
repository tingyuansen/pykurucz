# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnusedParameter=false, reportDeprecated=false
"""PFSAHA-style ionization and partition calculations.

Fortran reference: `kurucz/src/atlas12.for`, `SUBROUTINE PFSAHA` (line ~3137).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from .pfground import pfiron

_ELEMENTS: list[str] = [
    "",
    "H",
    "HE",
    "LI",
    "BE",
    "B",
    "C",
    "N",
    "O",
    "F",
    "NE",
    "NA",
    "MG",
    "AL",
    "SI",
    "P",
    "S",
    "CL",
    "AR",
    "K",
    "CA",
    "SC",
    "TI",
    "V",
    "CR",
    "MN",
    "FE",
    "CO",
    "NI",
    "CU",
    "ZN",
    "GA",
    "GE",
    "AS",
    "SE",
    "BR",
    "KR",
    "RB",
    "SR",
    "Y",
    "ZR",
    "NB",
    "MO",
    "TC",
    "RU",
    "RH",
    "PD",
    "AG",
    "CD",
    "IN",
    "SN",
    "SB",
    "TE",
    "I",
    "XE",
    "CS",
    "BA",
    "LA",
    "CE",
    "PR",
    "ND",
    "PM",
    "SM",
    "EU",
    "GD",
    "TB",
    "DY",
    "HO",
    "ER",
    "TM",
    "YB",
    "LU",
    "HF",
    "TA",
    "W",
    "RE",
    "OS",
    "IR",
    "PT",
    "AU",
    "HG",
    "TL",
    "PB",
    "BI",
    "PO",
    "AT",
    "RN",
    "FR",
    "RA",
    "AC",
    "TH",
    "PA",
    "U",
    "NP",
    "PU",
    "AM",
    "CM",
    "BK",
    "CF",
    "ES",
]


_K_BOLTZ = 1.38054e-16
_KEV_FACTOR = 8.6171e-5
_NNN_PATH = Path(__file__).resolve().parents[1] / "data" / "pfsaha_levels_atlas12.npz"
_POTION_PATH = Path(__file__).resolve().parents[1] / "data" / "ionpots_atlas12.npz"


def _load_nnn() -> np.ndarray:
    if not _NNN_PATH.exists():
        raise FileNotFoundError(
            f"Missing ATLAS12 PFSAHA table {_NNN_PATH}. "
            "Run atlas_py/tools/extract_pfsaha_levels_atlas12.py first."
        )
    data = np.load(_NNN_PATH, allow_pickle=False)
    nnn = np.asarray(data["NNN"], dtype=np.int64)
    if nnn.shape != (6, 365):
        raise ValueError(f"Unexpected NNN shape {nnn.shape}, expected (6, 365)")
    return nnn


NNN = _load_nnn()


def _load_potion() -> np.ndarray | None:
    if not _POTION_PATH.exists():
        return None
    data = np.load(_POTION_PATH, allow_pickle=False)
    if "POTION" not in data.files:
        return None
    arr = np.asarray(data["POTION"], dtype=np.float64)
    if arr.ndim != 1 or arr.size < 999:
        return None
    return arr


POTION = _load_potion()

# atlas12.for line ~3651
LOCZ = np.array(
    [
        1,
        3,
        6,
        10,
        14,
        18,
        22,
        27,
        33,
        39,
        45,
        51,
        57,
        63,
        69,
        75,
        81,
        86,
        91,
        96,
        101,
        106,
        111,
        116,
        121,
        126,
        131,
        136,
        141,
    ],
    dtype=np.int32,
)
SCALE = np.array([0.001, 0.01, 0.1, 1.0], dtype=np.float64)

# atlas12.for line ~3601 onward
EHYD = np.array([0.0, 82259.105, 97492.302, 102823.893, 105291.651, 106632.160])
GHYD = np.array([2.0, 8.0, 18.0, 32.0, 50.0, 72.0])
EHE1 = np.array(
    [
        0.0,
        159856.069,
        166277.546,
        169087.007,
        171135.000,
        183236.892,
        184864.936,
        185564.694,
        186101.654,
        186105.065,
        186209.471,
        190298.210,
        190940.331,
        191217.14,
        191444.588,
        191446.559,
        191451.80,
        191452.08,
        191492.817,
        193347.089,
        193663.627,
        193800.78,
        193917.245,
        193918.391,
        193921.31,
        193921.37,
        193922.5,
        193922.5,
        193942.57,
    ]
)
GHE1 = np.array(
    [1.0, 3.0, 1.0, 9.0, 3.0, 3.0, 1.0, 9.0, 15.0, 5.0, 3.0, 3.0, 1.0, 9.0, 15.0, 5.0, 21.0, 7.0, 3.0, 3.0, 1.0, 9.0, 15.0, 5.0, 21.0, 7.0, 27.0, 9.0, 3.0]
)
EHE2 = np.array([0.0, 329182.321, 390142.359, 411477.925, 421353.135, 426717.413])
GHE2 = np.array([2.0, 8.0, 18.0, 32.0, 50.0, 72.0])
EC1 = np.array([29.60, 10192.66, 21648.02, 33735.20, 60373.00, 61981.82, 64088.85, 68856.33, 69722.00, 70743.95, 71374.90, 72610.72, 73975.91, 75254.93])
GC1 = np.array([9.0, 5.0, 1.0, 5.0, 9.0, 3.0, 15.0, 3.0, 15.0, 3.0, 9.0, 5.0, 1.0, 9.0])
EC2 = np.array([42.48, 43035.8, 74931.11, 96493.74, 110652.10, 116537.65])
GC2 = np.array([6.0, 12.0, 10.0, 2.0, 6.0, 2.0])
EMG1 = np.array([0.0, 21890.854, 35051.264, 41197.403, 43503.333, 46403.065, 47847.797, 47957.034, 49346.729, 51872.526, 52556.206])
GMG1 = np.array([1.0, 9.0, 3.0, 3.0, 1.0, 5.0, 9.0, 15.0, 3.0, 3.0, 1.0])
EMG2 = np.array([0.0, 35730.36, 69804.95, 71490.54, 80639.85, 92790.51])
GMG2 = np.array([2.0, 6.0, 2.0, 10.0, 6.0, 2.0])
EAL1 = np.array([74.707, 25347.756, 29097.11, 32436.241, 32960.363, 37689.413, 38932.139, 40275.903, 41319.377])
GAL1 = np.array([6.0, 2.0, 12.0, 10.0, 6.0, 2.0, 10.0, 6.0, 14.0])
ESI1 = np.array([149.681, 6298.850, 15394.370, 33326.053, 39859.920, 40991.884, 45303.310, 47284.061, 47351.554, 48161.459, 49128.131])
GSI1 = np.array([9.0, 5.0, 1.0, 5.0, 9.0, 3.0, 15.0, 3.0, 5.0, 15.0, 9.0])
ESI2 = np.array([191.55, 43002.27, 55319.11, 65500.73, 76665.61, 79348.67])
GSI2 = np.array([6.0, 12.0, 10.0, 2.0, 2.0, 10.0])
ENA1 = np.array([0.0, 16956.172, 16973.368, 25739.991, 29172.889, 29172.839, 30266.99, 30272.58])
GNA1 = np.array([2.0, 2.0, 4.0, 2.0, 6.0, 4.0, 2.0, 4.0])
EO1 = np.array([77.975, 15867.862, 33792.583, 73768.200, 76794.978, 86629.089, 88630.977, 95476.728, 96225.049, 97420.748, 97488.476, 99094.065, 99681.051])
GO1 = np.array([9.0, 5.0, 1.0, 5.0, 3.0, 15.0, 9.0, 5.0, 3.0, 25.0, 15.0, 15.0, 9.0])
EB1 = np.array([10.17, 28810.0, 40039.65, 47856.99, 48613.01, 54767.74, 55010.08])
GB1 = np.array([6.0, 12.0, 2.0, 10.0, 6.0, 10.0, 2.0])
EK1 = np.array([0.0, 12985.170, 13042.876, 21026.551, 21534.680, 21536.988, 24701.382, 24720.139])
GK1 = np.array([2.0, 2.0, 4.0, 2.0, 6.0, 4.0, 2.0, 4.0])


def _element_symbol(atomic_number: int) -> str:
    if atomic_number < 1 or atomic_number >= len(_ELEMENTS):
        raise ValueError(f"Atomic number out of supported range: {atomic_number}")
    return _ELEMENTS[atomic_number]


def _start_and_nions(iz: int) -> tuple[int, int]:
    if iz <= 28:
        n = int(LOCZ[iz - 1])
        nions = int(LOCZ[iz] - n)
    else:
        n = 3 * iz + 54
        nions = 3
    if iz == 6:
        n = 354
        nions = 6
    if iz == 7:
        n = 360
        nions = 6
    if 20 <= iz < 29:
        nions = 10
    return n, nions


def _occupation_correction(part: float, zion: float, g: float, ip: float, potlo: float, tv: float, d1: float) -> float:
    if tv <= 0.0 or d1 <= 0.0 or potlo <= 0.0:
        return max(part, 1.0)
    d2 = potlo / tv
    if d2 <= 0.0:
        return max(part, 1.0)

    def _term(d: float) -> float:
        x = np.sqrt(13.595 * zion * zion / (tv * d))
        x3 = x * x * x
        poly = (1.0 / 3.0) + (1.0 - (0.5 + (1.0 / 18.0 + d / 120.0) * d) * d) * d
        return x3 * poly

    corr = g * np.exp(-ip / tv) * (_term(d2) - _term(d1))
    return max(part + corr, 1.0)


def _potion_index(iz: int, ion: int) -> int:
    """Return 1-based POTION index from atlas12.for formula."""
    if iz <= 30:
        return iz * (iz + 1) // 2 + ion - 1
    return iz * 5 + 341 + ion - 1


def _special_partition(n: int, hckt: float) -> tuple[float, float, bool]:
    """Return (PART, D1, used_special_case)."""
    if n == 1:  # H I
        part = 2.0
        for i in range(1, 6):
            part += GHYD[i] * np.exp(-EHYD[i] * hckt)
        d1 = 109677.576 / (6.5 * 6.5) * hckt
        return part, d1, True
    if n == 3:  # He I
        part = 1.0
        for i in range(1, 29):
            part += GHE1[i] * np.exp(-EHE1[i] * hckt)
        d1 = 109677.576 / (5.5 * 5.5) * hckt
        return part, d1, True
    if n == 4:  # He II
        part = 2.0
        for i in range(1, 6):
            part += GHE2[i] * np.exp(-EHE2[i] * hckt)
        d1 = 4.0 * 109722.267 / (6.5 * 6.5) * hckt
        return part, d1, True
    if n == 354:  # C I
        part = 1.0 + 3.0 * np.exp(-16.42 * hckt) + 5.0 * np.exp(-43.42 * hckt)
        for i in range(1, 14):
            part += GC1[i] * np.exp(-EC1[i] * hckt)
        part += (
            108.0 * np.exp(-80000.0 * hckt)
            + 189.0 * np.exp(-84000.0 * hckt)
            + 247.0 * np.exp(-87000.0 * hckt)
            + 231.0 * np.exp(-88000.0 * hckt)
            + 190.0 * np.exp(-89000.0 * hckt)
            + 300.0 * np.exp(-90000.0 * hckt)
        )
        return part, 0.0, True
    if n == 355:  # C II
        part = 2.0 + 4.0 * np.exp(-63.42 * hckt)
        for i in range(1, 6):
            part += GC2[i] * np.exp(-EC2[i] * hckt)
        part += (
            6.0 * np.exp(-131731.80 * hckt)
            + 4.0 * np.exp(-142027.1 * hckt)
            + 10.0 * np.exp(-145550.13 * hckt)
            + 10.0 * np.exp(-150463.62 * hckt)
            + 2.0 * np.exp(-157234.07 * hckt)
            + 6.0 * np.exp(-162500.0 * hckt)
            + 42.0 * np.exp(-168000.0 * hckt)
            + 56.0 * np.exp(-178000.0 * hckt)
            + 102.0 * np.exp(-183000.0 * hckt)
            + 400.0 * np.exp(-188000.0 * hckt)
        )
        d1 = 0.0
        return part, d1, True
    if n == 51:  # Mg I
        part = 1.0
        for i in range(1, 11):
            part += GMG1[i] * np.exp(-EMG1[i] * hckt)
        part += (
            5.0 * np.exp(-53134.0 * hckt)
            + 15.0 * np.exp(-54192.0 * hckt)
            + 28.0 * np.exp(-54676.0 * hckt)
            + 9.0 * np.exp(-57853.0 * hckt)
        )
        d1 = 109734.83 / (4.5 * 4.5) * hckt
        return part, d1, True
    if n == 52:  # Mg II
        part = 2.0
        for i in range(1, 6):
            part += GMG2[i] * np.exp(-EMG2[i] * hckt)
        part += (
            10.0 * np.exp(-93310.80 * hckt)
            + 14.0 * np.exp(-93799.70 * hckt)
            + 6.0 * np.exp(-97464.32 * hckt)
            + 10.0 * np.exp(-103419.82 * hckt)
            + 14.0 * np.exp(-103689.89 * hckt)
            + 18.0 * np.exp(-103705.66 * hckt)
        )
        d1 = 4.0 * 109734.83 / (5.5 * 5.5) * hckt
        return part, d1, True
    if n == 57:  # Al I
        part = 2.0 + 4.0 * np.exp(-112.061 * hckt)
        for i in range(1, 9):
            part += GAL1[i] * np.exp(-EAL1[i] * hckt)
        part += 10.0 * np.exp(-42235.0 * hckt) + 14.0 * np.exp(-43831.0 * hckt)
        d1 = 109735.08 / (5.5 * 5.5) * hckt
        return part, d1, True
    if n == 63:  # Si I
        part = 1.0 + 3.0 * np.exp(-77.115 * hckt) + 5.0 * np.exp(-223.157 * hckt)
        for i in range(1, 11):
            part += GSI1[i] * np.exp(-ESI1[i] * hckt)
        part += (
            76.0 * np.exp(-53000.0 * hckt)
            + 71.0 * np.exp(-57000.0 * hckt)
            + 191.0 * np.exp(-60000.0 * hckt)
            + 240.0 * np.exp(-62000.0 * hckt)
            + 251.0 * np.exp(-63000.0 * hckt)
            + 300.0 * np.exp(-65000.0 * hckt)
        )
        return part, 0.0, True
    if n == 64:  # Si II
        part = 2.0 + 4.0 * np.exp(-287.32 * hckt)
        for i in range(1, 6):
            part += GSI2[i] * np.exp(-ESI2[i] * hckt)
        part += (
            6.0 * np.exp(-81231.59 * hckt)
            + 6.0 * np.exp(-83937.08 * hckt)
            + 10.0 * np.exp(-101024.09 * hckt)
            + 14.0 * np.exp(-103556.35 * hckt)
            + 10.0 * np.exp(-108800.0 * hckt)
            + 42.0 * np.exp(-115000.0 * hckt)
            + 6.0 * np.exp(-121000.0 * hckt)
            + 38.0 * np.exp(-125000.0 * hckt)
            + 34.0 * np.exp(-132000.0 * hckt)
        )
        d1 = 4.0 * 109734.83 / (4.5 * 4.5) * hckt
        return part, d1, True
    if n == 367:  # O I
        part = 5.0 + 3.0 * np.exp(-158.265 * hckt) + np.exp(-226.977 * hckt)
        for i in range(1, 13):
            part += GO1[i] * np.exp(-EO1[i] * hckt)
        part += (
            15.0 * np.exp(-101140.0 * hckt)
            + 131.0 * np.exp(-103000.0 * hckt)
            + 128.0 * np.exp(-105000.0 * hckt)
            + 600.0 * np.exp(-107000.0 * hckt)
        )
        return part, 0.0, True
    if n == 45:  # Na I
        part = 2.0
        for i in range(1, 8):
            part += GNA1[i] * np.exp(-ENA1[i] * hckt)
        part += 10.0 * np.exp(-34548.745 * hckt) + 14.0 * np.exp(-34586.96 * hckt)
        d1 = 109734.83 / (4.5 * 4.5) * hckt
        return part, d1, True
    if n == 14:  # B I
        part = 2.0 + 4.0 * np.exp(-15.25 * hckt)
        for i in range(1, 7):
            part += GB1[i] * np.exp(-EB1[i] * hckt)
        part += (
            6.0 * np.exp(-57786.80 * hckt)
            + 10.0 * np.exp(-59989.0 * hckt)
            + 14.0 * np.exp(-60031.03 * hckt)
            + 2.0 * np.exp(-63561.0 * hckt)
        )
        d1 = 109734.83 / (4.5 * 4.5) * hckt
        return part, d1, True
    if n == 91:  # K I
        part = 2.0
        for i in range(1, 8):
            part += GK1[i] * np.exp(-EK1[i] * hckt)
        part += 10.0 * np.exp(-27397.077 * hckt) + 14.0 * np.exp(-28127.85 * hckt)
        d1 = 109734.83 / (5.5 * 5.5) * hckt
        return part, d1, True
    return 1.0, 0.0, False


def _pfsaha_depth_uncached(
    temperature_k: float,
    electron_density_cm3: float,
    xnatom_cm3: float,
    xabund_linear: float,
    atomic_number: int,
    nion: int,
    mode: int,
    chargesq_cm3: float | None = None,
) -> np.ndarray:
    """Compute PFSAHA-like output for one depth and one element.

    Units:
    - temperature_k: K
    - electron_density_cm3: cm^-3
    - xnatom_cm3: cm^-3
    - xabund_linear: dimensionless abundance fraction
    """

    _ = _element_symbol(atomic_number)  # range check + explicit mapping
    _ = (xnatom_cm3, xabund_linear)
    iz = int(atomic_number)
    nion = max(1, int(nion))
    t = max(float(temperature_k), 1.0)
    ne = max(float(electron_density_cm3), 1e-40)
    tk = _K_BOLTZ * t
    tkev = _KEV_FACTOR * t
    hckt = (6.6256e-27 * 2.99792458e10) / max(tk, 1e-300)

    base_mode = int(mode)
    mode1 = base_mode if base_mode <= 10 else base_mode - 10
    return_all = base_mode >= 10

    # atlas12.for line ~3658: CHARGESQ-based Debye lowering.
    if chargesq_cm3 is None:
        # Fallback used when caller does not provide CHARGESQ.
        chargesq = max(2.0 * ne, 1e-30)
    else:
        chargesq = max(float(chargesq_cm3), 1e-30)
    debye = np.sqrt(tk / (12.5664 * (4.801e-10**2) * chargesq))
    potlow = min(1.0, 1.44e-7 / max(debye, 1e-300))

    n_start, nions = _start_and_nions(iz)
    nion2 = min(nion + 2, nions)
    n = n_start - 1

    part = np.ones(nion2, dtype=np.float64)
    ip = np.zeros(nion2, dtype=np.float64)
    potlo = np.zeros(nion2, dtype=np.float64)
    f = np.zeros(nion2, dtype=np.float64)

    for ion in range(1, nion2 + 1):
        zion = float(ion)
        n += 1
        if n < 1 or n > NNN.shape[1]:
            raise ValueError(f"PFSAHA table index out of range: N={n}")

        potlo_i = potlow * zion
        nnn6 = int(NNN[5, n - 1])
        nnn100 = nnn6 // 100
        g = float(nnn6 - nnn100 * 100)
        ip_i = float(nnn100) / 1000.0
        if POTION is not None:
            pidx = _potion_index(iz, ion) - 1  # Fortran 1-based -> Python 0-based
            if 0 <= pidx < POTION.size and POTION[pidx] > 0.0:
                ip_i = POTION[pidx] / 8065.479
            elif 0 <= pidx - 1 < POTION.size and POTION[pidx - 1] > 0.0:
                # atlas12.for fallback for blank entries
                ip_i = POTION[pidx - 1] / 8065.479
        if ip_i <= 0.0 and ion > 1:
            ip_i = ip[ion - 2]
        potlo[ion - 1] = potlo_i
        ip[ion - 1] = ip_i

        # Fortran PFSAHA uses dedicated PFIRON tables for iron-group elements
        # (IZ=20..28), bypassing the generic/special partition branches.
        if 20 <= iz < 29:
            part[ion - 1] = max(
                pfiron(
                    nelem=iz,
                    ion=ion,
                    tlog10=np.log10(t) if t > 0.0 else 0.0,
                    potlow_cm1=potlo_i * 8065.479,
                ),
                1.0,
            )
            continue

        p_special, d1_special, used_special = _special_partition(n, hckt)
        if used_special:
            p = max(p_special, 1.0)
            # For special-case blocks that jump directly to label 14 in Fortran,
            # use D1 from the block; D2 derives from POTLO/TV.
            if d1_special > 0.0:
                p = _occupation_correction(p, zion, max(g, 2.0), ip_i, potlo_i, tkev, d1_special)
            part[ion - 1] = max(p, 1.0)
            continue

        # General interpolation path (atlas12.for lines ~3710-3733).
        t2000 = max(ip_i * 2000.0 / 11.0, 1e-12)
        it = max(1, min(9, int(t / t2000 - 0.5)))
        dt = t / t2000 - float(it) - 0.5
        pmin = 1.0
        i = (it + 1) // 2
        nnn_i = int(NNN[i - 1, n - 1])
        k1 = nnn_i // 100000
        k2 = nnn_i - k1 * 100000
        k3 = k2 // 10
        kscale = max(1, min(4, k2 - k3 * 10))
        if it % 2 == 1:
            p1 = float(k1) * SCALE[kscale - 1]
            p2 = float(k3) * SCALE[kscale - 1]
            if dt < 0.0 and kscale <= 1:
                kp1 = int(p1)
                if kp1 == int(p2 + 0.5):
                    pmin = float(kp1)
        else:
            p1 = float(k3) * SCALE[kscale - 1]
            nnn_i1 = int(NNN[i, n - 1])
            k1n = nnn_i1 // 100000
            kscale_n = max(1, min(4, int(nnn_i1 % 10)))
            p2 = float(k1n) * SCALE[kscale_n - 1]
        p = max(pmin, p1 + (p2 - p1) * dt)

        # PFGROUND branch: keep non-decreasing guard (full PFGROUND pending port).
        if t < t2000 * 2.0:
            part[ion - 1] = max(p, 1.0)
            continue

        # Occupation correction branch (atlas12.for lines ~3751-3759).
        if g != 0.0 and potlo_i >= 0.1 and t >= t2000 * 4.0:
            tv_eff = tkev
            if t > (t2000 * 11.0):
                tv_eff = (t2000 * 11.0) * _KEV_FACTOR
            d1 = 0.1 / max(tv_eff, 1e-30)
            p = _occupation_correction(p, zion, g, ip_i, potlo_i, tv_eff, d1)
        part[ion - 1] = max(p, 1.0)

    if mode1 not in (3, 5):
        cf = 2.0 * 2.4148e15 * t * np.sqrt(t) / ne
        for ion in range(2, nion2 + 1):
            idx = ion - 1
            f[idx] = (
                cf
                * part[idx]
                / max(part[idx - 1], 1e-300)
                * np.exp(-(ip[idx - 1] - potlo[idx - 1]) / max(tkev, 1e-30))
            )
        f[0] = 1.0
        l = nion2 + 1
        for _ in range(2, nion2 + 1):
            l -= 1
            f[0] = 1.0 + f[l - 1] * f[0]
        f[0] = 1.0 / max(f[0], 1e-300)
        for ion in range(2, nion2 + 1):
            idx = ion - 1
            f[idx] = f[idx - 1] * f[idx]

    if return_all:
        nret = min(nion, nion2)
        if mode1 == 1:
            out = np.zeros(nion, dtype=np.float64)
            out[:nret] = f[:nret] / np.maximum(part[:nret], 1e-300)
            return out
        if mode1 == 2:
            out = np.zeros(nion, dtype=np.float64)
            out[:nret] = f[:nret]
            return out
        if mode1 == 3:
            out = np.zeros(nion, dtype=np.float64)
            out[:nret] = part[:nret]
            return out
        if mode1 == 4:
            out = np.zeros(nion, dtype=np.float64)
            out[0] = np.sum(f[1:nion2] * np.arange(1, nion2, dtype=np.float64))
            return out
        if mode1 == 5:
            out = np.zeros(61, dtype=np.float64)
            out[:nret] = part[:nret]
            # Fortran mode=5 layout (atlas12.for lines 4032-4035):
            # ANSWER(32)=0, ANSWER(33)=IP(1), ANSWER(34)=IP(1)+IP(2), ...
            out[31] = 0.0
            eacc = 0.0
            for i in range(nret):
                eacc += float(ip[i])
                if 32 + i < out.size:
                    out[32 + i] = eacc
            return out
        raise NotImplementedError(f"PFSAHA mode {mode} is not implemented")

    nidx = min(max(nion, 1), nion2) - 1
    out = np.zeros(nion, dtype=np.float64)
    if mode1 == 1:
        out[0] = f[nidx] / max(part[nidx], 1e-300)
    elif mode1 == 2:
        out[0] = f[nidx]
    elif mode1 == 3:
        out[0] = part[nidx]
    elif mode1 == 4:
        out[0] = np.sum(f[1:nion2] * np.arange(1, nion2, dtype=np.float64))
    elif mode1 == 5:
        out = np.zeros(61, dtype=np.float64)
        nret = min(nion, nion2)
        out[:nret] = part[:nret]
        out[31] = 0.0
        eacc = 0.0
        for i in range(nret):
            eacc += float(ip[i])
            if 32 + i < out.size:
                out[32 + i] = eacc
    else:
        raise NotImplementedError(f"PFSAHA mode {mode} is not implemented")
    return out


@lru_cache(maxsize=262_144)
def _pfsaha_depth_cached(
    temperature_k: float,
    electron_density_cm3: float,
    xnatom_cm3: float,
    xabund_linear: float,
    atomic_number: int,
    nion: int,
    mode: int,
    chargesq_cm3: float | None,
) -> tuple[float, ...]:
    out = _pfsaha_depth_uncached(
        temperature_k=temperature_k,
        electron_density_cm3=electron_density_cm3,
        xnatom_cm3=xnatom_cm3,
        xabund_linear=xabund_linear,
        atomic_number=atomic_number,
        nion=nion,
        mode=mode,
        chargesq_cm3=chargesq_cm3,
    )
    return tuple(float(x) for x in out)


def pfsaha_depth(
    temperature_k: float,
    electron_density_cm3: float,
    xnatom_cm3: float,
    xabund_linear: float,
    atomic_number: int,
    nion: int,
    mode: int,
    chargesq_cm3: float | None = None,
) -> np.ndarray:
    """Compute PFSAHA-like output for one depth and one element.

    The scalar PFSAHA result is reused heavily by NMOLEC Newton iterations for
    identical thermodynamic inputs. Cache immutable tuples and return a fresh
    array so callers retain the original mutable-array contract.
    """
    cached = _pfsaha_depth_cached(
        float(temperature_k),
        float(electron_density_cm3),
        float(xnatom_cm3),
        float(xabund_linear),
        int(atomic_number),
        int(nion),
        int(mode),
        None if chargesq_cm3 is None else float(chargesq_cm3),
    )
    return np.asarray(cached, dtype=np.float64)

