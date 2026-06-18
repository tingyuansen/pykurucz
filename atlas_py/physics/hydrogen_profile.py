"""Fortran-faithful hydrogen profile stack for XLINOP TYPE=-1."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import re

import numpy as np

try:
    import numba
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

from .trace_runtime import trace_emit, trace_enabled, trace_in_focus

_C_LIGHT_A = 2.99792458e18
_C_LIGHT_CM = 2.99792458e10
_RYDH = 3.2880515e15
_SQRT_PI_SCALE = 1.77245
_PI = 3.14159


def _default_hydrogen_profile_npz_path() -> Path:
    # atlas_py/physics -> atlas_py/data
    return Path(__file__).resolve().parents[1] / "data" / "hydrogen_profile_atlas12.npz"


def _default_atlas12_path() -> Path:
    # Used only when re-extracting tables from Fortran source; the normal
    # runtime path is the pre-extracted .npz cache below.
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "data" / "src" / "atlas12.for"


def _stmt_text(line: str) -> str:
    # Fixed-format Fortran: columns 1-5 label, column 6 continuation.
    return line[6:] if len(line) > 6 else line


def _parse_fortran_scalar(token: str) -> float:
    t = token.strip()
    if not t:
        raise ValueError("empty Fortran token")
    return float(t.replace("D", "E").replace("d", "e"))


def _parse_fortran_value_list(payload: str) -> list[float]:
    values: list[float] = []
    for raw in payload.replace("\n", " ").split(","):
        token = raw.strip()
        if not token:
            continue
        m = re.match(r"^(\d+)\*(.+)$", token)
        if m:
            nrep = int(m.group(1))
            val = _parse_fortran_scalar(m.group(2))
            values.extend([val] * nrep)
        else:
            values.append(_parse_fortran_scalar(token))
    return values


def _parse_named_data_blocks(lines: list[str], needed: set[str]) -> dict[str, np.ndarray]:
    data_start_pat = re.compile(r"^DATA\s+([A-Z0-9_]+)\s*/(.*)$", re.IGNORECASE)
    out: dict[str, np.ndarray] = {}
    collecting = False
    current_name: str | None = None
    payload_parts: list[str] = []

    def _flush(name: str | None, parts: list[str]) -> None:
        if name is None:
            return
        key = name.upper()
        if key not in needed:
            return
        vals = _parse_fortran_value_list(" ".join(parts))
        out[key] = np.asarray(vals, dtype=np.float64)

    for line in lines:
        if line and line[0] in {"C", "c", "*", "!"}:
            continue
        stmt = _stmt_text(line).rstrip()
        if not collecting:
            m = data_start_pat.match(stmt.strip())
            if not m:
                continue
            current_name = m.group(1).upper()
            after = m.group(2)
            if "/" in after:
                before_slash = after.split("/", 1)[0]
                _flush(current_name, [before_slash])
                current_name = None
                payload_parts = []
            else:
                payload_parts = [after]
                collecting = True
        else:
            s = stmt.strip()
            if "/" in s:
                before_slash = s.split("/", 1)[0]
                payload_parts.append(before_slash)
                _flush(current_name, payload_parts)
                collecting = False
                current_name = None
                payload_parts = []
            else:
                payload_parts.append(s)
    if collecting:
        raise RuntimeError("Unterminated DATA block while parsing hydrogen profile tables")
    return out


def _extract_function_body(lines: list[str], function_name: str) -> list[str]:
    start = None
    end = None
    needle = function_name.upper()
    for i, line in enumerate(lines):
        if start is None and needle in line.upper():
            start = i
            continue
        if start is not None and line.strip().upper() == "END":
            end = i
            break
    if start is None or end is None or end <= start:
        raise RuntimeError(f"Could not locate {function_name} body in atlas12.for")
    return lines[start : end + 1]


def _require_len(data: dict[str, np.ndarray], name: str, size: int) -> np.ndarray:
    arr = data.get(name)
    if arr is None:
        raise RuntimeError(f"Missing DATA {name} in atlas12.for")
    if arr.size != size:
        raise RuntimeError(f"DATA {name} expected {size} values, got {arr.size}")
    return arr


def _parse_inline_data(lines: list[str], name: str) -> np.ndarray | None:
    pat = re.compile(rf"\b{name}\s*/([^/]*)/", re.IGNORECASE)
    for line in lines:
        if line and line[0] in {"C", "c", "*", "!"}:
            continue
        stmt = _stmt_text(line)
        m = pat.search(stmt)
        if m is None:
            continue
        vals = _parse_fortran_value_list(m.group(1))
        return np.asarray(vals, dtype=np.float64)
    return None


@dataclass(frozen=True)
class HydrogenProfileTables:
    propbm: np.ndarray  # (5,15,7)
    c: np.ndarray  # (5,7)
    d: np.ndarray  # (5,7)
    pp: np.ndarray  # (5,)
    beta: np.ndarray  # (15,)
    stalph: np.ndarray  # (34,)
    stwtal: np.ndarray  # (34,)
    istal: np.ndarray  # (4,)
    lnghal: np.ndarray  # (4,)
    stcomp: np.ndarray  # (5,4)
    stcpwt: np.ndarray  # (5,4)
    lncomp: np.ndarray  # (4,)
    cutoff_h2_plus: np.ndarray  # (111,)
    cutoff_h2: np.ndarray  # (91,)
    asumlyman: np.ndarray  # (100,)
    asum: np.ndarray  # (100,)
    y1wtm: np.ndarray  # (2,2)
    xknmtb: np.ndarray  # (4,3)
    pf_h2: np.ndarray  # (200,)


def _load_hydrogen_profile_npz(npz_path: Path) -> HydrogenProfileTables:
    with np.load(npz_path, allow_pickle=False) as data:
        return HydrogenProfileTables(
            propbm=np.asarray(data["propbm"], dtype=np.float64),
            c=np.asarray(data["c"], dtype=np.float64),
            d=np.asarray(data["d"], dtype=np.float64),
            pp=np.asarray(data["pp"], dtype=np.float64),
            beta=np.asarray(data["beta"], dtype=np.float64),
            stalph=np.asarray(data["stalph"], dtype=np.float64),
            stwtal=np.asarray(data["stwtal"], dtype=np.float64),
            istal=np.asarray(data["istal"], dtype=np.int64),
            lnghal=np.asarray(data["lnghal"], dtype=np.int64),
            stcomp=np.asarray(data["stcomp"], dtype=np.float64),
            stcpwt=np.asarray(data["stcpwt"], dtype=np.float64),
            lncomp=np.asarray(data["lncomp"], dtype=np.int64),
            cutoff_h2_plus=np.asarray(data["cutoff_h2_plus"], dtype=np.float64),
            cutoff_h2=np.asarray(data["cutoff_h2"], dtype=np.float64),
            asumlyman=np.asarray(data["asumlyman"], dtype=np.float64),
            asum=np.asarray(data["asum"], dtype=np.float64),
            y1wtm=np.asarray(data["y1wtm"], dtype=np.float64),
            xknmtb=np.asarray(data["xknmtb"], dtype=np.float64),
            pf_h2=np.asarray(data["pf_h2"], dtype=np.float64),
        )


@lru_cache(maxsize=2)
def load_hydrogen_profile_tables_from_atlas12(atlas12_path: str | None = None) -> HydrogenProfileTables:
    if atlas12_path is None:
        npz_path = _default_hydrogen_profile_npz_path()
        if npz_path.exists():
            return _load_hydrogen_profile_npz(npz_path)

    src = Path(atlas12_path) if atlas12_path is not None else _default_atlas12_path()
    if not src.exists():
        raise FileNotFoundError(f"atlas12.for not found: {src}")
    lines = src.read_text(encoding="utf-8", errors="replace").splitlines()
    sofbet_names = {
        "PROB1",
        "PROB2",
        "PROB3",
        "PROB4",
        "PROB5",
        "PROB6",
        "PROB7",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6",
        "D7",
        "PP",
        "BETA",
    }
    hprof4_names = {
        "STALPH",
        "STWTAL",
        "ISTAL",
        "LNGHAL",
        "STCOMP",
        "STCPWT",
        "LNCOMP",
        "CUTOFFH2PLUS",
        "CUTOFFH2",
        "ASUMLYMAN",
        "ASUM",
        "Y1WTM",
        "XKNMTB",
    }
    equilh2_names = {
        "PF1",
        "PF2",
    }
    sofbet_body = _extract_function_body(lines, "FUNCTION SOFBET")
    hprof4_body = _extract_function_body(lines, "FUNCTION HPROF4")
    equilh2_body = _extract_function_body(lines, "FUNCTION EQUILH2")
    data: dict[str, np.ndarray] = {}
    data.update(_parse_named_data_blocks(sofbet_body, sofbet_names))
    data.update(_parse_named_data_blocks(hprof4_body, hprof4_names))
    data.update(_parse_named_data_blocks(equilh2_body, equilh2_names))
    for name in sofbet_names.union(hprof4_names).union(equilh2_names):
        if name in data:
            continue
        arr = _parse_inline_data(sofbet_body, name)
        if arr is None:
            arr = _parse_inline_data(hprof4_body, name)
        if arr is None:
            arr = _parse_inline_data(equilh2_body, name)
        if arr is not None:
            data[name] = arr

    propbm = np.zeros((5, 15, 7), dtype=np.float64)
    for i in range(1, 8):
        block = _require_len(data, f"PROB{i}", 75)
        propbm[:, :, i - 1] = block.reshape((5, 15), order="F")

    c = np.zeros((5, 7), dtype=np.float64)
    d = np.zeros((5, 7), dtype=np.float64)
    for i in range(1, 8):
        c[:, i - 1] = _require_len(data, f"C{i}", 5)
        d[:, i - 1] = _require_len(data, f"D{i}", 5)

    pp = _require_len(data, "PP", 5)
    beta = _require_len(data, "BETA", 15)
    stalph = _require_len(data, "STALPH", 34)
    stwtal = _require_len(data, "STWTAL", 34)
    istal = _require_len(data, "ISTAL", 4).astype(np.int64)
    lnghal = _require_len(data, "LNGHAL", 4).astype(np.int64)
    stcomp = _require_len(data, "STCOMP", 20).reshape((5, 4), order="F")
    stcpwt = _require_len(data, "STCPWT", 20).reshape((5, 4), order="F")
    lncomp = _require_len(data, "LNCOMP", 4).astype(np.int64)
    cutoff_h2_plus = _require_len(data, "CUTOFFH2PLUS", 111)
    cutoff_h2 = _require_len(data, "CUTOFFH2", 91)
    asumlyman = _require_len(data, "ASUMLYMAN", 100)
    asum = _require_len(data, "ASUM", 100)
    y1wtm = _require_len(data, "Y1WTM", 4).reshape((2, 2), order="F")
    xknmtb = _require_len(data, "XKNMTB", 12).reshape((4, 3), order="F")
    pf_h2 = np.concatenate([_require_len(data, "PF1", 100), _require_len(data, "PF2", 100)])

    return HydrogenProfileTables(
        propbm=propbm,
        c=c,
        d=d,
        pp=pp,
        beta=beta,
        stalph=stalph,
        stwtal=stwtal,
        istal=istal,
        lnghal=lnghal,
        stcomp=stcomp,
        stcpwt=stcpwt,
        lncomp=lncomp,
        cutoff_h2_plus=cutoff_h2_plus,
        cutoff_h2=cutoff_h2,
        asumlyman=asumlyman,
        asum=asum,
        y1wtm=y1wtm,
        xknmtb=xknmtb,
        pf_h2=pf_h2,
    )


@lru_cache(maxsize=256)
def _hf_nm(n: int, m: int) -> float:
    if m <= n:
        return 0.0
    xn = float(n)
    ginf = 0.2027 / (xn**0.71)
    gca = 0.124 / xn
    fkn = xn * 1.9603
    wtc = 0.45 - 2.4 / (xn**3) * (xn - 1.0)
    xm = float(m)
    xmn = xm - xn
    fk = fkn * (xm / (xmn * (xm + xn))) ** 3
    xmn12 = xmn**1.2
    wt = (xmn12 - 1.0) / (xmn12 + wtc)
    return fk * (1.0 - wt * ginf - (0.222 + gca / xm) * (1.0 - wt))


def _build_exptabs() -> tuple[np.ndarray, np.ndarray]:
    i = np.arange(1001, dtype=np.float64)
    return np.exp(-i), np.exp(-i * 0.001)


@lru_cache(maxsize=1)
def _faste1_table() -> np.ndarray:
    tbl = np.zeros(2000, dtype=np.float64)
    for i in range(1, 2001):
        tbl[i - 1] = _expi(1, float(i) * 0.01)
    return tbl


def _fastex(x: float, extab: np.ndarray, extabf: np.ndarray) -> float:
    if x < 0.0 or x >= 1001.0:
        return 0.0
    i = int(x)
    j = int((x - float(i)) * 1000.0 + 1.5)
    if j < 1:
        j = 1
    if j > 1001:
        j = 1001
    return float(extab[i] * extabf[j - 1])


def _faste1(x: float, extab: np.ndarray, extabf: np.ndarray) -> float:
    if x > 20.0:
        return 0.0
    if x >= 0.5:
        idx = int(x * 100.0 + 0.5)
        if idx < 1:
            idx = 1
        if idx > 2000:
            idx = 2000
        return float(_faste1_table()[idx - 1])
    if x <= 0.0:
        return 0.0
    return (1.0 - 0.22464 * x) * x - np.log(x) - 0.57721


def _expi(n: int, x: float) -> float:
    a0, a1, a2, a3, a4, a5 = (
        -44178.5471728217,
        57721.7247139444,
        9938.31388962037,
        1842.11088668,
        101.093806161906,
        5.03416184097568,
    )
    b0, b1, b2, b3, b4 = (
        76537.3323337614,
        32597.1881290275,
        6106.10794245759,
        635.419418378382,
        37.2298352833327,
    )
    c0, c1, c2, c3, c4, c5, c6 = (
        4.65627107975096e-7,
        0.999979577051595,
        9.04161556946329,
        24.3784088791317,
        23.0192559391333,
        6.90522522784444,
        0.430967839469389,
    )
    d1, d2, d3, d4, d5, d6 = (
        10.0411643829054,
        32.4264210695138,
        41.2807841891424,
        20.4494785013794,
        3.31909213593302,
        0.103400130404874,
    )
    e0, e1, e2, e3, e4, e5, e6 = (
        -0.999999999998447,
        -26.6271060431811,
        -241.055827097015,
        -895.927957772937,
        -1298.85688746484,
        -545.374158883133,
        -5.66575206533869,
    )
    f1, f2, f3, f4, f5, f6 = (
        28.6271060422192,
        292.310039388533,
        1332.78537748257,
        2777.61949509163,
        2404.01713225909,
        631.6574832808,
    )

    ex = float(np.exp(-x))
    if x > 4.0:
        ex1 = (
            ex
            + ex
            * (e0 + (e1 + (e2 + (e3 + (e4 + (e5 + e6 / x) / x) / x) / x) / x) / x)
            / (x + f1 + (f2 + (f3 + (f4 + (f5 + f6 / x) / x) / x) / x) / x)
        ) / x
    elif x > 1.0:
        ex1 = ex * (c6 + (c5 + (c4 + (c3 + (c2 + (c1 + c0 * x) * x) * x) * x) * x) * x) / (
            d6 + (d5 + (d4 + (d3 + (d2 + (d1 + x) * x) * x) * x) * x) * x
        )
    elif x > 0.0:
        ex1 = (a0 + (a1 + (a2 + (a3 + (a4 + a5 * x) * x) * x) * x) * x) / (
            b0 + (b1 + (b2 + (b3 + (b4 + x) * x) * x) * x) * x
        ) - np.log(x)
    else:
        ex1 = 0.0
    if n == 1:
        return ex1
    out = ex1
    for i in range(1, n):
        out = (ex - x * out) / float(i)
    return out


def _sofbet(beta: float, p: float, n: int, m: int, tables: HydrogenProfileTables) -> float:
    corr = 1.0
    b2 = beta * beta
    sb = np.sqrt(max(beta, 1e-300))
    if beta <= 500.0:
        indx = 7
        mmn = m - n
        if n <= 3 and mmn <= 2:
            indx = 2 * (n - 1) + mmn
        im = min(int(5.0 * p) + 1, 4)
        if im < 1:
            im = 1
        ip = im + 1
        wtpp = 5.0 * (p - float(tables.pp[im - 1]))
        wtpm = 1.0 - wtpp
        if beta <= 25.12:
            j = int(np.searchsorted(tables.beta, beta, side="left"))
            if j < 1:
                j = 1
            if j > 14:
                j = 14
            jm = j - 1
            jp = j
            denom = float(tables.beta[jp] - tables.beta[jm])
            wtbp = 0.0 if denom == 0.0 else (beta - float(tables.beta[jm])) / denom
            wtbm = 1.0 - wtbp
            cbp = float(tables.propbm[ip - 1, jp, indx - 1]) * wtpp + float(tables.propbm[im - 1, jp, indx - 1]) * wtpm
            cbm = float(tables.propbm[ip - 1, jm, indx - 1]) * wtpp + float(tables.propbm[im - 1, jm, indx - 1]) * wtpm
            corr = 1.0 + cbp * wtbp + cbm * wtbm
            pr1 = 0.0
            pr2 = 0.0
            wt = max(min(0.5 * (10.0 - beta), 1.0), 0.0)
            if beta <= 10.0:
                pr1 = 8.0 / (83.0 + (2.0 + 0.95 * b2) * beta)
            if beta >= 8.0:
                pr2 = (1.5 / sb + 27.0 / b2) / b2
            return (pr1 * wt + pr2 * (1.0 - wt)) * corr
        cc = float(tables.c[ip - 1, indx - 1]) * wtpp + float(tables.c[im - 1, indx - 1]) * wtpm
        dd = float(tables.d[ip - 1, indx - 1]) * wtpp + float(tables.d[im - 1, indx - 1]) * wtpm
        corr = 1.0 + dd / (cc + beta * sb)
    return (1.5 / sb + 27.0 / b2) / b2 * corr


if _NUMBA_AVAILABLE:
    @numba.njit(cache=True)
    def _fastex_nb(x: float, extab: np.ndarray, extabf: np.ndarray) -> float:
        if x < 0.0 or x >= 1001.0:
            return 0.0
        i = int(x)
        j = int((x - float(i)) * 1000.0 + 1.5)
        if j < 1:
            j = 1
        if j > 1001:
            j = 1001
        return float(extab[i] * extabf[j - 1])


    @numba.njit(cache=True)
    def _faste1_nb(x: float, faste1_tbl: np.ndarray) -> float:
        if x > 20.0:
            return 0.0
        if x >= 0.5:
            idx = int(x * 100.0 + 0.5)
            if idx < 1:
                idx = 1
            if idx > 2000:
                idx = 2000
            return float(faste1_tbl[idx - 1])
        if x <= 0.0:
            return 0.0
        return (1.0 - 0.22464 * x) * x - np.log(x) - 0.57721


    @numba.njit(cache=True)
    def _sofbet_nb(
        beta: float,
        p: float,
        n: int,
        m: int,
        propbm: np.ndarray,
        c_arr: np.ndarray,
        d_arr: np.ndarray,
        pp_arr: np.ndarray,
        beta_arr: np.ndarray,
    ) -> float:
        corr = 1.0
        b2 = beta * beta
        sb = np.sqrt(max(beta, 1.0e-300))
        if beta <= 500.0:
            indx = 7
            mmn = m - n
            if n <= 3 and mmn <= 2:
                indx = 2 * (n - 1) + mmn
            im = min(int(5.0 * p) + 1, 4)
            if im < 1:
                im = 1
            ip = im + 1
            wtpp = 5.0 * (p - float(pp_arr[im - 1]))
            wtpm = 1.0 - wtpp
            if beta <= 25.12:
                j = 1
                for idx in range(beta_arr.size):
                    if beta_arr[idx] >= beta:
                        j = idx
                        break
                if j < 1:
                    j = 1
                if j > 14:
                    j = 14
                jm = j - 1
                jp = j
                denom = float(beta_arr[jp] - beta_arr[jm])
                wtbp = 0.0 if denom == 0.0 else (beta - float(beta_arr[jm])) / denom
                wtbm = 1.0 - wtbp
                cbp = float(propbm[ip - 1, jp, indx - 1]) * wtpp + float(propbm[im - 1, jp, indx - 1]) * wtpm
                cbm = float(propbm[ip - 1, jm, indx - 1]) * wtpp + float(propbm[im - 1, jm, indx - 1]) * wtpm
                corr = 1.0 + cbp * wtbp + cbm * wtbm
                pr1 = 0.0
                pr2 = 0.0
                wt = max(min(0.5 * (10.0 - beta), 1.0), 0.0)
                if beta <= 10.0:
                    pr1 = 8.0 / (83.0 + (2.0 + 0.95 * b2) * beta)
                if beta >= 8.0:
                    pr2 = (1.5 / sb + 27.0 / b2) / b2
                return (pr1 * wt + pr2 * (1.0 - wt)) * corr
            cc = float(c_arr[ip - 1, indx - 1]) * wtpp + float(c_arr[im - 1, indx - 1]) * wtpm
            dd = float(d_arr[ip - 1, indx - 1]) * wtpp + float(d_arr[im - 1, indx - 1]) * wtpm
            corr = 1.0 + dd / (cc + beta * sb)
        return (1.5 / sb + 27.0 / b2) / b2 * corr


    @numba.njit(cache=True)
    def _stark_profile_nb(
        n: int,
        m: int,
        dbeta: float,
        c1con: float,
        c2con: float,
        y1num: float,
        y1wht: float,
        freq4: float,
        delstark: float,
        dop: float,
        fo: float,
        xne: float,
        y1s: float,
        y1b: float,
        c1d: float,
        c2d: float,
        gcon1: float,
        gcon2: float,
        pp: float,
        xnf_h2: float,
        extab: np.ndarray,
        extabf: np.ndarray,
        faste1_tbl: np.ndarray,
        propbm: np.ndarray,
        c_arr: np.ndarray,
        d_arr: np.ndarray,
        pp_arr: np.ndarray,
        beta_arr: np.ndarray,
        cutoff_h2_plus: np.ndarray,
    ) -> float:
        if fo <= 0.0:
            return 0.0
        wty1 = 1.0 / (1.0 + xne / y1wht)
        y1scal = y1num * y1s * wty1 + y1b * (1.0 - wty1)
        c1 = c1d * c1con * y1scal
        c2 = c2d * c2con
        if c1 <= 0.0:
            c1 = 0.0
        if c2 <= 0.0:
            c2 = 0.0
        g1 = 6.77 * np.sqrt(max(c1, 0.0))
        logterm = 0.0
        if c1 > 0.0 and c2 > 0.0:
            logterm = np.log(np.sqrt(c2) / c1)
        gnot = g1 * max(0.0, 0.2114 + logterm) * (1.0 - gcon1 - gcon2)
        beta = abs(delstark) / fo * dbeta
        y1 = c1 * beta
        y2 = c2 * beta * beta
        gam = gnot
        if not (y2 <= 1.0e-4 and y1 <= 1.0e-5):
            gam = g1 * (
                0.5 * _fastex_nb(min(80.0, y1), extab, extabf)
                + _faste1_nb(y1, faste1_tbl)
                - 0.5 * _faste1_nb(y2, faste1_tbl)
            ) * (
                1.0
                - gcon1 / (1.0 + (90.0 * y1) ** 3)
                - gcon2 / (1.0 + 2000.0 * y1)
            )
            if gam <= 1.0e-20:
                gam = 0.0

        prqs = _sofbet_nb(beta, pp, n, m, propbm, c_arr, d_arr, pp_arr, beta_arr)
        hprof = 0.0
        if m <= 2:
            prqs = prqs * 0.5
            if freq4 >= (82259.105 - 20000.0) * _C_LIGHT_CM:
                if freq4 <= (82259.105 - 4000.0) * _C_LIGHT_CM:
                    freq15000 = (82259.105 - 15000.0) * _C_LIGHT_CM
                    spacing = 100.0 * _C_LIGHT_CM
                    if freq4 < freq15000:
                        cutoff = (cutoff_h2_plus[1] - cutoff_h2_plus[0]) / spacing * (freq4 - freq15000) + cutoff_h2_plus[0]
                    else:
                        icut = int((freq4 - freq15000) / spacing)
                        if icut < 0:
                            icut = 0
                        if icut > cutoff_h2_plus.size - 2:
                            icut = cutoff_h2_plus.size - 2
                        cutfreq = icut * spacing + freq15000
                        cutoff = (cutoff_h2_plus[icut + 1] - cutoff_h2_plus[icut]) / spacing * (freq4 - cutfreq) + cutoff_h2_plus[icut]
                    cutoff = (10.0 ** (cutoff - 14.0)) / _C_LIGHT_CM * xnf_h2
                    hprof += cutoff * _SQRT_PI_SCALE * dop
                else:
                    beta4000 = 4000.0 * _C_LIGHT_CM / fo * dbeta
                    prqsp4000 = _sofbet_nb(beta4000, pp, n, m, propbm, c_arr, d_arr, pp_arr, beta_arr) * 0.5 / fo * dbeta
                    cutoff4000 = (10.0 ** (-11.07 - 14.0)) / _C_LIGHT_CM * xnf_h2
                    if prqsp4000 != 0.0:
                        hprof += cutoff4000 / prqsp4000 * prqs / fo * dbeta * _SQRT_PI_SCALE * dop

        f = 0.0
        if gam > 0.0:
            f = gam / _PI / (gam * gam + beta * beta)
        p1 = (0.9 * y1) ** 2
        fns = (p1 + 0.03 * np.sqrt(max(y1, 0.0))) / (p1 + 1.0)
        hprof += (prqs * (1.0 + fns) + f) / fo * dbeta * _SQRT_PI_SCALE * dop
        return hprof


    @numba.njit(cache=True)
    def _doppler_profile_nb(
        finest: np.ndarray,
        finswt: np.ndarray,
        freqnm: float,
        freq4: float,
        dop: float,
        extab: np.ndarray,
        extabf: np.ndarray,
    ) -> float:
        out = 0.0
        for i in range(finest.size):
            d = abs(freq4 - freqnm - float(finest[i])) / dop
            if d <= 7.0:
                out += _fastex_nb(d * d, extab, extabf) * float(finswt[i])
        return out


    @numba.njit(cache=True)
    def _lorentz_profile_nb(
        n: int,
        m: int,
        freqnm: float,
        wavenm: float,
        radamp: float,
        resont: float,
        vdw: float,
        freq4: float,
        delt: float,
        dop: float,
        hwres: float,
        hwvdw: float,
        hwrad: float,
        xnfp_h1: float,
        cutoff_h2: np.ndarray,
    ) -> float:
        if n == 1 and m == 2:
            hwres = hwres * 4.0
            hwlor = hwres + hwvdw + hwrad
            hhw = freqnm * hwlor
            if freq4 > (82259.105 - 4000.0) * _C_LIGHT_CM:
                hprofres = hwres * freqnm / _PI / (delt * delt + hhw * hhw) * _SQRT_PI_SCALE * dop
            else:
                cutoff = 0.0
                if freq4 >= 50000.0 * _C_LIGHT_CM:
                    spacing = 200.0 * _C_LIGHT_CM
                    freq22000 = (82259.105 - 22000.0) * _C_LIGHT_CM
                    if freq4 < freq22000:
                        cutoff = (cutoff_h2[1] - cutoff_h2[0]) / spacing * (freq4 - freq22000) + cutoff_h2[0]
                    else:
                        icut = int((freq4 - freq22000) / spacing)
                        if icut < 0:
                            icut = 0
                        if icut > cutoff_h2.size - 2:
                            icut = cutoff_h2.size - 2
                        cutfreq = icut * spacing + freq22000
                        cutoff = (cutoff_h2[icut + 1] - cutoff_h2[icut]) / spacing * (freq4 - cutfreq) + cutoff_h2[icut]
                    cutoff = (10.0 ** (cutoff - 14.0)) * xnfp_h1 * 2.0 / _C_LIGHT_CM
                hprofres = cutoff * _SQRT_PI_SCALE * dop
            hprofrad = hwrad * freqnm / _PI / (delt * delt + hhw * hhw) * _SQRT_PI_SCALE * dop
            if freq4 <= 2.463e15:
                hprofrad = 0.0
            hprofvdw = hwvdw * freqnm / _PI / (delt * delt + hhw * hhw) * _SQRT_PI_SCALE * dop
            if freq4 < 1.8e15:
                hprofvdw = 0.0
            return hprofres + hprofrad + hprofvdw

        hhw = freqnm * (hwres + hwvdw + hwrad)
        if hhw <= 0.0:
            return 0.0
        return hhw / _PI / (delt * delt + hhw * hhw) * _SQRT_PI_SCALE * dop


    @numba.njit(cache=True)
    def _profile_for_setup_nb(
        n: int,
        m: int,
        freqnm: float,
        wavenm: float,
        dbeta: float,
        c1con: float,
        c2con: float,
        radamp: float,
        resont: float,
        vdw: float,
        stark: float,
        y1num: float,
        y1wht: float,
        finest: np.ndarray,
        finswt: np.ndarray,
        j0: int,
        delw_nm: float,
        xne_arr: np.ndarray,
        xnf_h1: np.ndarray,
        xnf_h2: np.ndarray,
        xnfp_h1: np.ndarray,
        dopple_h1: np.ndarray,
        fo_arr: np.ndarray,
        pp_arr_h: np.ndarray,
        y1s_arr: np.ndarray,
        y1b_arr: np.ndarray,
        t3nhe_arr: np.ndarray,
        t3nh2_arr: np.ndarray,
        c1d_arr: np.ndarray,
        c2d_arr: np.ndarray,
        gcon1_arr: np.ndarray,
        gcon2_arr: np.ndarray,
        extab: np.ndarray,
        extabf: np.ndarray,
        faste1_tbl: np.ndarray,
        propbm: np.ndarray,
        c_arr: np.ndarray,
        d_arr: np.ndarray,
        pp_tab: np.ndarray,
        beta_arr: np.ndarray,
        cutoff_h2: np.ndarray,
        cutoff_h2_plus: np.ndarray,
    ) -> float:
        delstark = -10.0 * delw_nm / wavenm * freqnm
        wl = wavenm + delw_nm * 10.0
        if wl <= 0.0:
            return 0.0
        freq4 = _C_LIGHT_A / wl
        delt = abs(freq4 - freqnm)
        dopple = float(dopple_h1[j0])
        if dopple <= 0.0:
            return 0.0
        hwstk = stark * float(fo_arr[j0])
        hwvdw = vdw * float(t3nhe_arr[j0]) + 2.0 * vdw * float(t3nh2_arr[j0])
        hwrad = radamp
        hwres = resont * float(xnf_h1[j0])
        hwlor = hwres + hwvdw + hwrad
        nwid = 1
        if not (dopple >= hwstk and dopple >= hwlor):
            nwid = 2
            if hwlor < hwstk:
                nwid = 3
        hfwid = freqnm * max(dopple, hwlor, hwstk)
        ifcore = abs(delt) <= hfwid
        dop = freqnm * dopple
        if dop <= 0.0:
            return 0.0
        out = 0.0
        if ifcore:
            if nwid == 1:
                out = _doppler_profile_nb(finest, finswt, freqnm, freq4, dop, extab, extabf)
            elif nwid == 2:
                out = _lorentz_profile_nb(
                    n, m, freqnm, wavenm, radamp, resont, vdw, freq4, delt, dop,
                    hwres, hwvdw, hwrad, float(xnfp_h1[j0]), cutoff_h2,
                )
            else:
                out = _stark_profile_nb(
                    n, m, dbeta, c1con, c2con, y1num, y1wht, freq4, delstark, dop,
                    float(fo_arr[j0]), float(xne_arr[j0]), float(y1s_arr[j0]), float(y1b_arr[j0]),
                    float(c1d_arr[j0]), float(c2d_arr[j0]), float(gcon1_arr[j0]), float(gcon2_arr[j0]),
                    float(pp_arr_h[j0]), float(xnf_h2[j0]), extab, extabf, faste1_tbl, propbm,
                    c_arr, d_arr, pp_tab, beta_arr, cutoff_h2_plus,
                )
        else:
            out = (
                _doppler_profile_nb(finest, finswt, freqnm, freq4, dop, extab, extabf)
                + _lorentz_profile_nb(
                    n, m, freqnm, wavenm, radamp, resont, vdw, freq4, delt, dop,
                    hwres, hwvdw, hwrad, float(xnfp_h1[j0]), cutoff_h2,
                )
                + _stark_profile_nb(
                    n, m, dbeta, c1con, c2con, y1num, y1wht, freq4, delstark, dop,
                    float(fo_arr[j0]), float(xne_arr[j0]), float(y1s_arr[j0]), float(y1b_arr[j0]),
                    float(c1d_arr[j0]), float(c2d_arr[j0]), float(gcon1_arr[j0]), float(gcon2_arr[j0]),
                    float(pp_arr_h[j0]), float(xnf_h2[j0]), extab, extabf, faste1_tbl, propbm,
                    c_arr, d_arr, pp_tab, beta_arr, cutoff_h2_plus,
                )
            )
        if out < 0.0:
            return 0.0
        return out


@dataclass(frozen=True)
class _LineSetup:
    n: int
    m: int
    freqnm: float
    wavenm: float
    dbeta: float
    c1con: float
    c2con: float
    radamp: float
    resont: float
    vdw: float
    stark: float
    y1num: float
    y1wht: float
    finest: np.ndarray
    finswt: np.ndarray


@dataclass
class HydrogenProfileEvaluator:
    temperature_k: np.ndarray
    xne: np.ndarray
    xnf_h1: np.ndarray
    xnf_h2: np.ndarray
    xnfp_h1: np.ndarray
    xnfp_he1: np.ndarray
    dopple_h1: np.ndarray
    xnh2: np.ndarray
    tables: HydrogenProfileTables
    pp: np.ndarray = field(init=False)
    fo: np.ndarray = field(init=False)
    y1b: np.ndarray = field(init=False)
    y1s: np.ndarray = field(init=False)
    t3nhe: np.ndarray = field(init=False)
    t3nh2: np.ndarray = field(init=False)
    c1d: np.ndarray = field(init=False)
    c2d: np.ndarray = field(init=False)
    gcon1: np.ndarray = field(init=False)
    gcon2: np.ndarray = field(init=False)
    extab: np.ndarray = field(init=False)
    extabf: np.ndarray = field(init=False)
    faste1_tbl: np.ndarray = field(init=False)
    trace_active: bool = field(init=False)
    _line_cache: dict[tuple[int, int], _LineSetup] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        t = np.asarray(self.temperature_k, dtype=np.float64)
        xne = np.asarray(self.xne, dtype=np.float64)
        self.temperature_k = t
        self.xne = xne
        self.xnf_h1 = np.asarray(self.xnf_h1, dtype=np.float64)
        self.xnf_h2 = np.asarray(self.xnf_h2, dtype=np.float64)
        self.xnfp_h1 = np.asarray(self.xnfp_h1, dtype=np.float64)
        self.xnfp_he1 = np.asarray(self.xnfp_he1, dtype=np.float64)
        self.dopple_h1 = np.asarray(self.dopple_h1, dtype=np.float64)
        self.xnh2 = np.asarray(self.xnh2, dtype=np.float64)

        xne16 = np.maximum(xne, 1e-300) ** 0.1666667
        t4 = t / 10000.0
        t43 = np.maximum(t4, 1e-300) ** 0.3
        self.pp = xne16 * 0.08989 / np.sqrt(np.maximum(t, 1e-300))
        self.fo = xne16**4 * 1.25e-9
        self.y1b = 2.0 / (1.0 + 0.012 / np.maximum(t, 1e-300) * np.sqrt(np.maximum(xne / np.maximum(t, 1e-300), 0.0)))
        self.y1s = t43 / np.maximum(xne16, 1e-300)
        # atlas12.for HPROF4 uses T3NHE = T43 * SNGL(XNFP(K,3)).
        self.t3nhe = t43 * np.maximum(np.asarray(self.xnfp_he1, dtype=np.float64), 0.0)
        self.t3nh2 = t43 * np.maximum(np.asarray(self.xnh2, dtype=np.float64), 0.0)
        self.c1d = self.fo * 78940.0 / np.maximum(t, 1e-300)
        self.c2d = self.fo * self.fo / 5.96e-23 / np.maximum(xne, 1e-300)
        self.gcon1 = 0.2 + 0.09 * np.sqrt(np.maximum(t4, 0.0)) / (1.0 + xne / 1.0e13)
        self.gcon2 = 0.2 / (1.0 + xne / 1.0e15)
        self.extab, self.extabf = _build_exptabs()
        self.faste1_tbl = _faste1_table()
        self.trace_active = trace_enabled()
        self._line_cache: dict[tuple[int, int], _LineSetup] = {}

    def _line_setup(self, n: int, m: int) -> _LineSetup | None:
        key = (int(n), int(m))
        if key in self._line_cache:
            return self._line_cache[key]
        mmn = m - n
        if mmn <= 0:
            return None
        xn = float(n)
        xm = float(m)
        xn2 = xn * xn
        xm2 = xm * xm
        xmn2 = xm2 * xn2
        xm2mn2 = xm2 - xn2
        gnm = xm2mn2 / xmn2
        if mmn <= 3 and n <= 4:
            xknm = float(self.tables.xknmtb[n - 1, mmn - 1])
        else:
            xknm = 5.5e-5 / gnm * xmn2 / (1.0 + 0.13 / float(mmn))
        y1num = 320.0
        if m == 2:
            y1num = 550.0
        if m == 3:
            y1num = 380.0
        y1wht = 1.0e13
        if mmn <= 3:
            y1wht = 1.0e14
        if mmn <= 2 and n <= 2:
            y1wht = float(self.tables.y1wtm[n - 1, mmn - 1])
        freqnm = _RYDH * gnm
        dbeta = _C_LIGHT_A / (freqnm * freqnm) / xknm
        wavenm = _C_LIGHT_A / freqnm
        c1con = xknm / wavenm * gnm * xm2mn2
        c2con = (xknm / wavenm) ** 2
        radamp = float(self.tables.asum[n - 1] + self.tables.asum[m - 1])
        if n == 1:
            radamp = float(self.tables.asumlyman[m - 1])
        radamp = radamp / 12.5664 / freqnm
        resont = _hf_nm(1, m) / xm / (1.0 - 1.0 / xm2)
        if n != 1:
            resont += _hf_nm(1, n) / xn / (1.0 - 1.0 / xn2)
        resont = resont * 3.92e-24 / gnm
        vdw = 4.45e-26 / gnm * (xm2 * (7.0 * xm2 + 5.0)) ** 0.4
        stark = 1.6678e-18 * freqnm * xknm

        if n > 4 or m > 10:
            finest = np.asarray([0.0], dtype=np.float64)
            finswt = np.asarray([1.0], dtype=np.float64)
        elif mmn != 1:
            ifins = int(self.tables.lncomp[n - 1])
            finest = np.asarray(self.tables.stcomp[:ifins, n - 1] * 1.0e7, dtype=np.float64)
            finswt = np.asarray(self.tables.stcpwt[:ifins, n - 1] / xn2, dtype=np.float64)
        else:
            ifins = int(self.tables.lnghal[n - 1])
            ipos = int(self.tables.istal[n - 1])
            i0 = ipos - 1
            i1 = i0 + ifins
            finest = np.asarray(self.tables.stalph[i0:i1] * 1.0e7, dtype=np.float64)
            finswt = np.asarray(self.tables.stwtal[i0:i1] / xn2 / 3.0, dtype=np.float64)

        setup = _LineSetup(
            n=n,
            m=m,
            freqnm=freqnm,
            wavenm=wavenm,
            dbeta=dbeta,
            c1con=c1con,
            c2con=c2con,
            radamp=radamp,
            resont=resont,
            vdw=vdw,
            stark=stark,
            y1num=y1num,
            y1wht=y1wht,
            finest=finest,
            finswt=finswt,
        )
        self._line_cache[key] = setup
        return setup

    def profile(
        self,
        n: int,
        m: int,
        j: int,
        delw_nm: float,
        trace_line_1b: int = -1,
        trace_nu_1b: int = 0,
    ) -> float:
        setup = self._line_setup(int(n), int(m))
        if setup is None:
            return 0.0
        return self.profile_for_setup(
            setup,
            j,
            delw_nm,
            trace_line_1b=trace_line_1b,
            trace_nu_1b=trace_nu_1b,
        )

    def profile_for_setup(
        self,
        setup: _LineSetup,
        j: int,
        delw_nm: float,
        trace_line_1b: int = -1,
        trace_nu_1b: int = 0,
    ) -> float:
        j0 = int(j)
        if _NUMBA_AVAILABLE and not self.trace_active:
            return float(
                _profile_for_setup_nb(
                    setup.n,
                    setup.m,
                    setup.freqnm,
                    setup.wavenm,
                    setup.dbeta,
                    setup.c1con,
                    setup.c2con,
                    setup.radamp,
                    setup.resont,
                    setup.vdw,
                    setup.stark,
                    setup.y1num,
                    setup.y1wht,
                    setup.finest,
                    setup.finswt,
                    j0,
                    float(delw_nm),
                    self.xne,
                    self.xnf_h1,
                    self.xnf_h2,
                    self.xnfp_h1,
                    self.dopple_h1,
                    self.fo,
                    self.pp,
                    self.y1s,
                    self.y1b,
                    self.t3nhe,
                    self.t3nh2,
                    self.c1d,
                    self.c2d,
                    self.gcon1,
                    self.gcon2,
                    self.extab,
                    self.extabf,
                    self.faste1_tbl,
                    self.tables.propbm,
                    self.tables.c,
                    self.tables.d,
                    self.tables.pp,
                    self.tables.beta,
                    self.tables.cutoff_h2,
                    self.tables.cutoff_h2_plus,
                )
            )
        delstark = -10.0 * float(delw_nm) / setup.wavenm * setup.freqnm
        wl = setup.wavenm + float(delw_nm) * 10.0
        if wl <= 0.0:
            return 0.0
        freq4 = _C_LIGHT_A / wl
        delt = abs(freq4 - setup.freqnm)
        dopple = float(self.dopple_h1[j0])
        if dopple <= 0.0:
            return 0.0
        hwstk = setup.stark * float(self.fo[j0])
        hwvdw = setup.vdw * float(self.t3nhe[j0]) + 2.0 * setup.vdw * float(self.t3nh2[j0])
        hwrad = setup.radamp
        hwres = setup.resont * float(self.xnf_h1[j0])
        hwlor = hwres + hwvdw + hwrad
        nwid = 1
        if not (dopple >= hwstk and dopple >= hwlor):
            nwid = 2
            if hwlor < hwstk:
                nwid = 3
        hfwid = setup.freqnm * max(dopple, hwlor, hwstk)
        ifcore = abs(delt) <= hfwid
        dop = setup.freqnm * dopple
        if dop <= 0.0:
            return 0.0
        out = 0.0
        if ifcore:
            if nwid == 1:
                out = max(self._doppler_profile(setup, freq4, dop), 0.0)
            elif nwid == 2:
                out = max(self._lorentz_profile(setup, j0, freq4, delt, dop, hwres, hwvdw, hwrad), 0.0)
            else:
                out = max(self._stark_profile(setup, j0, freq4, delstark, dop), 0.0)
        else:
            out = max(
                self._doppler_profile(setup, freq4, dop)
                + self._lorentz_profile(setup, j0, freq4, delt, dop, hwres, hwvdw, hwrad)
                + self._stark_profile(setup, j0, freq4, delstark, dop),
                0.0,
            )
        wl_nm = wl / 10.0
        if self.trace_active and trace_in_focus(wlvac_nm=wl_nm, j0=j0):
            trace_emit(
                event="hprof4_return",
                iter_num=1,
                line_num_1b=int(trace_line_1b),
                depth_1b=j0 + 1,
                nu_1b=int(trace_nu_1b),
                type_code=-1,
                wlvac_nm=wl_nm,
                center=0.0,
                adamp=float(nwid),
                cv=float(out),
                tabcont=float(1 if ifcore else 0),
                branch="hprof4",
                reason="return",
            )
        return out

    def _doppler_profile(self, setup: _LineSetup, freq4: float, dop: float) -> float:
        if _NUMBA_AVAILABLE:
            return float(
                _doppler_profile_nb(
                    setup.finest,
                    setup.finswt,
                    setup.freqnm,
                    freq4,
                    dop,
                    self.extab,
                    self.extabf,
                )
            )
        out = 0.0
        for offset, wt in zip(setup.finest, setup.finswt, strict=False):
            d = abs(freq4 - setup.freqnm - float(offset)) / dop
            if d <= 7.0:
                out += _fastex(d * d, self.extab, self.extabf) * float(wt)
        return out

    def _lorentz_profile(
        self,
        setup: _LineSetup,
        j0: int,
        freq4: float,
        delt: float,
        dop: float,
        hwres: float,
        hwvdw: float,
        hwrad: float,
    ) -> float:
        if _NUMBA_AVAILABLE:
            return float(
                _lorentz_profile_nb(
                    setup.n,
                    setup.m,
                    setup.freqnm,
                    setup.wavenm,
                    setup.radamp,
                    setup.resont,
                    setup.vdw,
                    freq4,
                    delt,
                    dop,
                    hwres,
                    hwvdw,
                    hwrad,
                    float(self.xnfp_h1[j0]),
                    self.tables.cutoff_h2,
                )
            )
        n = setup.n
        m = setup.m
        if n == 1 and m == 2:
            hwres = hwres * 4.0
            hwlor = hwres + hwvdw + hwrad
            hhw = setup.freqnm * hwlor
            if freq4 > (82259.105 - 4000.0) * _C_LIGHT_CM:
                hprofres = hwres * setup.freqnm / _PI / (delt * delt + hhw * hhw) * _SQRT_PI_SCALE * dop
            else:
                cutoff = 0.0
                if freq4 >= 50000.0 * _C_LIGHT_CM:
                    spacing = 200.0 * _C_LIGHT_CM
                    freq22000 = (82259.105 - 22000.0) * _C_LIGHT_CM
                    if freq4 < freq22000:
                        cutoff = (self.tables.cutoff_h2[1] - self.tables.cutoff_h2[0]) / spacing * (freq4 - freq22000) + self.tables.cutoff_h2[0]
                    else:
                        icut = int((freq4 - freq22000) / spacing)
                        if icut < 0:
                            icut = 0
                        if icut > self.tables.cutoff_h2.size - 2:
                            icut = self.tables.cutoff_h2.size - 2
                        cutfreq = icut * spacing + freq22000
                        cutoff = (self.tables.cutoff_h2[icut + 1] - self.tables.cutoff_h2[icut]) / spacing * (freq4 - cutfreq) + self.tables.cutoff_h2[icut]
                    cutoff = (10.0 ** (cutoff - 14.0)) * float(self.xnfp_h1[j0]) * 2.0 / _C_LIGHT_CM
                hprofres = cutoff * _SQRT_PI_SCALE * dop
            hprofrad = hwrad * setup.freqnm / _PI / (delt * delt + hhw * hhw) * _SQRT_PI_SCALE * dop
            if freq4 <= 2.463e15:
                hprofrad = 0.0
            hprofvdw = hwvdw * setup.freqnm / _PI / (delt * delt + hhw * hhw) * _SQRT_PI_SCALE * dop
            if freq4 < 1.8e15:
                hprofvdw = 0.0
            return hprofres + hprofrad + hprofvdw

        hhw = setup.freqnm * (hwres + hwvdw + hwrad)
        if hhw <= 0.0:
            return 0.0
        return hhw / _PI / (delt * delt + hhw * hhw) * _SQRT_PI_SCALE * dop

    def _stark_profile(self, setup: _LineSetup, j0: int, freq4: float, delstark: float, dop: float) -> float:
        if _NUMBA_AVAILABLE:
            return float(
                _stark_profile_nb(
                    setup.n,
                    setup.m,
                    setup.dbeta,
                    setup.c1con,
                    setup.c2con,
                    setup.y1num,
                    setup.y1wht,
                    freq4,
                    delstark,
                    dop,
                    float(self.fo[j0]),
                    float(self.xne[j0]),
                    float(self.y1s[j0]),
                    float(self.y1b[j0]),
                    float(self.c1d[j0]),
                    float(self.c2d[j0]),
                    float(self.gcon1[j0]),
                    float(self.gcon2[j0]),
                    float(self.pp[j0]),
                    float(self.xnf_h2[j0]),
                    self.extab,
                    self.extabf,
                    self.faste1_tbl,
                    self.tables.propbm,
                    self.tables.c,
                    self.tables.d,
                    self.tables.pp,
                    self.tables.beta,
                    self.tables.cutoff_h2_plus,
                )
            )
        fo = float(self.fo[j0])
        if fo <= 0.0:
            return 0.0
        wty1 = 1.0 / (1.0 + float(self.xne[j0]) / setup.y1wht)
        y1scal = setup.y1num * float(self.y1s[j0]) * wty1 + float(self.y1b[j0]) * (1.0 - wty1)
        c1 = float(self.c1d[j0]) * setup.c1con * y1scal
        c2 = float(self.c2d[j0]) * setup.c2con
        if c1 <= 0.0:
            c1 = 0.0
        if c2 <= 0.0:
            c2 = 0.0
        g1 = 6.77 * np.sqrt(max(c1, 0.0))
        logterm = 0.0
        if c1 > 0.0 and c2 > 0.0:
            logterm = np.log(np.sqrt(c2) / c1)
        gnot = g1 * max(0.0, 0.2114 + logterm) * (1.0 - float(self.gcon1[j0]) - float(self.gcon2[j0]))
        beta = abs(delstark) / fo * setup.dbeta
        y1 = c1 * beta
        y2 = c2 * beta * beta
        gam = gnot
        if not (y2 <= 1.0e-4 and y1 <= 1.0e-5):
            gam = g1 * (
                0.5 * _fastex(min(80.0, y1), self.extab, self.extabf)
                + _faste1(y1, self.extab, self.extabf)
                - 0.5 * _faste1(y2, self.extab, self.extabf)
            ) * (
                1.0
                - float(self.gcon1[j0]) / (1.0 + (90.0 * y1) ** 3)
                - float(self.gcon2[j0]) / (1.0 + 2000.0 * y1)
            )
            if gam <= 1.0e-20:
                gam = 0.0

        prqs = _sofbet(beta, float(self.pp[j0]), setup.n, setup.m, self.tables)
        hprof = 0.0
        if setup.m <= 2:
            prqs = prqs * 0.5
            if freq4 >= (82259.105 - 20000.0) * _C_LIGHT_CM:
                if freq4 <= (82259.105 - 4000.0) * _C_LIGHT_CM:
                    freq15000 = (82259.105 - 15000.0) * _C_LIGHT_CM
                    spacing = 100.0 * _C_LIGHT_CM
                    if freq4 < freq15000:
                        cutoff = (self.tables.cutoff_h2_plus[1] - self.tables.cutoff_h2_plus[0]) / spacing * (
                            freq4 - freq15000
                        ) + self.tables.cutoff_h2_plus[0]
                    else:
                        icut = int((freq4 - freq15000) / spacing)
                        if icut < 0:
                            icut = 0
                        if icut > self.tables.cutoff_h2_plus.size - 2:
                            icut = self.tables.cutoff_h2_plus.size - 2
                        cutfreq = icut * spacing + freq15000
                        cutoff = (self.tables.cutoff_h2_plus[icut + 1] - self.tables.cutoff_h2_plus[icut]) / spacing * (
                            freq4 - cutfreq
                        ) + self.tables.cutoff_h2_plus[icut]
                    cutoff = (10.0 ** (cutoff - 14.0)) / _C_LIGHT_CM * float(self.xnf_h2[j0])
                    hprof += cutoff * _SQRT_PI_SCALE * dop
                else:
                    beta4000 = 4000.0 * _C_LIGHT_CM / fo * setup.dbeta
                    prqsp4000 = _sofbet(beta4000, float(self.pp[j0]), setup.n, setup.m, self.tables) * 0.5 / fo * setup.dbeta
                    cutoff4000 = (10.0 ** (-11.07 - 14.0)) / _C_LIGHT_CM * float(self.xnf_h2[j0])
                    if prqsp4000 != 0.0:
                        hprof += cutoff4000 / prqsp4000 * prqs / fo * setup.dbeta * _SQRT_PI_SCALE * dop

        f = 0.0
        if gam > 0.0:
            f = gam / _PI / (gam * gam + beta * beta)
        p1 = (0.9 * y1) ** 2
        fns = (p1 + 0.03 * np.sqrt(max(y1, 0.0))) / (p1 + 1.0)
        hprof += (prqs * (1.0 + fns) + f) / fo * setup.dbeta * _SQRT_PI_SCALE * dop
        return hprof


def equil_h2(temperature_k: np.ndarray | float, *, tables: HydrogenProfileTables | None = None) -> np.ndarray:
    tbl = load_hydrogen_profile_tables_from_atlas12() if tables is None else tables
    t = np.asarray(temperature_k, dtype=np.float64)
    n = np.floor(t / 100.0).astype(np.int64)
    n = np.minimum(199, np.maximum(1, n))
    pf = tbl.pf_h2
    part = pf[n - 1] + (pf[n] - pf[n - 1]) / 100.0 * (t - n * 100.0)
    out = part * (2.0**1.5) / 4.0
    out = out / (2.0 * 3.14159 * 1.008 * 1.660e-24 * 1.38054e-16 / 6.6256e-27**2 * t) ** 1.5
    out = out * np.exp(36118.11 * 6.6256e-27 * 2.997925e10 / 1.38054e-16 / t)
    return out


def compute_xnh2(
    *,
    temperature_k: np.ndarray,
    xnfp_h1: np.ndarray,
    bhyd1: np.ndarray | None = None,
    tables: HydrogenProfileTables | None = None,
) -> np.ndarray:
    t = np.asarray(temperature_k, dtype=np.float64)
    xh1 = np.asarray(xnfp_h1, dtype=np.float64)
    if bhyd1 is None:
        b1 = np.ones_like(t, dtype=np.float64)
    else:
        b1 = np.asarray(bhyd1, dtype=np.float64)
    return (xh1 * 2.0 * b1) ** 2 * equil_h2(t, tables=tables)
