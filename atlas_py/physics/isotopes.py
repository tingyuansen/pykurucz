"""Fortran-faithful ISOTOPES table reconstruction from `atlas12.for`.

Parses EQUIVALENCE/DATA declarations and executes imperative isotope
assignments in `SUBROUTINE ISOTOPES` before packing into `ISOTOPE`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

import numpy as np

_MION = 1006
_ISO_COLS = 265
_ISO_ROWS = 20


def _default_isotopes_npz_path() -> Path:
    # atlas_py/physics -> atlas_py/data
    return Path(__file__).resolve().parents[1] / "data" / "isotopes_atlas12.npz"


def _default_atlas12_path() -> Path:
    # Used only when re-extracting tables from Fortran source; the normal
    # runtime path is the pre-extracted .npz cache below.
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "data" / "src" / "atlas12.for"


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


def _extract_isotopes_body(lines: list[str]) -> list[str]:
    start = None
    end = None
    for i, line in enumerate(lines):
        if start is None and "SUBROUTINE ISOTOPES" in line.upper():
            start = i
            continue
        if start is not None and line.strip().upper() == "END":
            end = i
            break
    if start is None or end is None or end <= start:
        raise RuntimeError("Could not locate SUBROUTINE ISOTOPES in atlas12.for")
    return lines[start : end + 1]


def _stmt_text(line: str) -> str:
    # Fixed-format Fortran: columns 1-5 label, column 6 continuation.
    return line[6:] if len(line) > 6 else line


def _parse_isoion_from_isotopes_body(body: list[str]) -> np.ndarray:
    eq_pat = re.compile(
        r"EQUIVALENCE\s*\(\s*ISOION\(\s*(\d+)\s*\)\s*,\s*([A-Z0-9_]+)\s*\(1\)\s*\)",
        re.IGNORECASE,
    )
    data_start_pat = re.compile(r"^DATA\s+([A-Z0-9_]+)\s*/(.*)$", re.IGNORECASE)

    equiv_start_by_name: dict[str, int] = {}
    for line in body:
        stmt = _stmt_text(line).strip()
        m = eq_pat.search(stmt)
        if m:
            start_1based = int(m.group(1))
            name = m.group(2).upper()
            equiv_start_by_name[name] = start_1based

    arr_size = _ISO_ROWS * _ISO_COLS
    array_by_name: dict[str, np.ndarray] = {}

    dim_pat = re.compile(r"\b([A-Z0-9_]+)\s*\(\s*20\s*\)", re.IGNORECASE)
    for line in body:
        stmt = _stmt_text(line)
        if "DIMENSION" not in stmt.upper():
            continue
        for m in dim_pat.finditer(stmt):
            array_by_name.setdefault(m.group(1).upper(), np.zeros(_ISO_ROWS, dtype=np.float64))

    collecting = False
    current_name: str | None = None
    payload_parts: list[str] = []

    def _flush_data_block(name: str | None, parts: list[str]) -> None:
        if name is None:
            return
        key = name.upper()
        if key not in equiv_start_by_name:
            return
        vals = _parse_fortran_value_list(" ".join(parts))
        dst = array_by_name.setdefault(key, np.zeros(_ISO_ROWS, dtype=np.float64))
        ncopy = min(_ISO_ROWS, len(vals))
        if ncopy > 0:
            dst[:ncopy] = np.asarray(vals[:ncopy], dtype=np.float64)

    for line in body:
        stmt = _stmt_text(line).rstrip()
        if not collecting:
            m = data_start_pat.match(stmt.strip())
            if not m:
                continue
            current_name = m.group(1).upper()
            after = m.group(2)
            if "/" in after:
                before_slash = after.split("/", 1)[0]
                _flush_data_block(current_name, [before_slash])
                current_name = None
                payload_parts = []
                collecting = False
            else:
                payload_parts = [after]
                collecting = True
        else:
            s = stmt.strip()
            if "/" in s:
                before_slash = s.split("/", 1)[0]
                payload_parts.append(before_slash)
                _flush_data_block(current_name, payload_parts)
                collecting = False
                current_name = None
                payload_parts = []
            else:
                payload_parts.append(s)

    if collecting:
        raise RuntimeError("Unterminated DATA block in ISOTOPES subroutine")

    for name in equiv_start_by_name:
        array_by_name.setdefault(name, np.zeros(_ISO_ROWS, dtype=np.float64))

    _apply_isotopes_imperative_statements(body, array_by_name)

    isoion_flat = np.zeros(arr_size, dtype=np.float64)
    for name, start_1based in equiv_start_by_name.items():
        vals = array_by_name.get(name)
        if vals is None:
            continue
        start0 = start_1based - 1
        ncopy = min(_ISO_ROWS, arr_size - start0)
        if ncopy > 0:
            isoion_flat[start0 : start0 + ncopy] = vals[:ncopy]
    return isoion_flat.reshape((_ISO_ROWS, _ISO_COLS), order="F")


def _line_label(line: str) -> str | None:
    lab = line[:5].strip() if len(line) >= 5 else ""
    return lab if lab else None


def _is_comment_line(line: str) -> bool:
    return bool(line) and line[0] in {"C", "c", "*", "!"}


def _norm_exp(expr: str) -> str:
    return re.sub(r"(?<=\d)[dD](?=[+-]?\d)", "E", expr)


def _eval_scalar_expr(expr: str, scalars: dict[str, int]) -> float:
    txt = _norm_exp(expr.strip())
    env = {k: float(v) for k, v in scalars.items()}
    return float(eval(txt, {"__builtins__": {}}, env))


def _eval_iso_expr(expr: str, arrays: dict[str, np.ndarray], scalars: dict[str, int]) -> float:
    arr_pat = re.compile(r"([A-Z0-9_]+)\s*\(\s*([^)]+?)\s*\)", re.IGNORECASE)
    txt = _norm_exp(expr.strip())

    def _arr(name: str, idx: float) -> float:
        key = name.upper()
        arr = arrays.get(key)
        if arr is None:
            raise KeyError(f"Unknown isotope array: {name}")
        i = int(round(float(idx)))
        if i < 1 or i > arr.size:
            raise IndexError(f"Index out of bounds for {name}: {i}")
        return float(arr[i - 1])

    def _repl(m: re.Match[str]) -> str:
        key = m.group(1).upper()
        if key in arrays:
            return f'__arr("{key}", ({m.group(2)}))'
        return m.group(0)

    py = arr_pat.sub(_repl, txt)
    env: dict[str, float | object] = {"__arr": _arr}
    for k, v in scalars.items():
        env[k] = float(v)
        env[k.lower()] = float(v)
    return float(eval(py, {"__builtins__": {}}, env))


def _apply_iso_assignment(stmt: str, arrays: dict[str, np.ndarray], scalars: dict[str, int]) -> None:
    if "=" not in stmt:
        return
    lhs, rhs = stmt.split("=", 1)
    m = re.fullmatch(r"\s*([A-Z0-9_]+)\s*\(\s*([^)]+?)\s*\)\s*", lhs, flags=re.IGNORECASE)
    if m is None:
        return
    name = m.group(1).upper()
    if name not in arrays:
        return
    idx = int(round(_eval_scalar_expr(m.group(2), scalars)))
    if idx < 1 or idx > arrays[name].size:
        return
    arrays[name][idx - 1] = _eval_iso_expr(rhs, arrays, scalars)


def _execute_iso_statements(lines: list[str], arrays: dict[str, np.ndarray], scalars: dict[str, int]) -> None:
    do_pat = re.compile(
        r"^DO\s+(\d+)\s+([A-Z][A-Z0-9_]*)\s*=\s*([^,]+)\s*,\s*([^,]+)\s*$",
        flags=re.IGNORECASE,
    )
    i = 0
    while i < len(lines):
        raw = lines[i]
        if _is_comment_line(raw):
            i += 1
            continue
        stmt = _stmt_text(raw).strip()
        if not stmt:
            i += 1
            continue
        if stmt.upper().startswith("N=0"):
            break
        mdo = do_pat.match(stmt)
        if mdo:
            label = mdo.group(1)
            var = mdo.group(2).upper()
            lo = int(round(_eval_scalar_expr(mdo.group(3), scalars)))
            hi = int(round(_eval_scalar_expr(mdo.group(4), scalars)))
            end_idx = None
            for j in range(i + 1, len(lines)):
                if _line_label(lines[j]) == label:
                    end_idx = j
                    break
            if end_idx is None:
                raise RuntimeError(f"Unterminated DO loop with label {label} in ISOTOPES")
            block = lines[i + 1 : end_idx]
            old = scalars.get(var)
            for v in range(lo, hi + 1):
                scalars[var] = v
                scalars[var.lower()] = v
                _execute_iso_statements(block, arrays, scalars)
            if old is None:
                scalars.pop(var, None)
                scalars.pop(var.lower(), None)
            else:
                scalars[var] = old
                scalars[var.lower()] = old
            i = end_idx + 1
            continue
        _apply_iso_assignment(stmt, arrays, scalars)
        i += 1


def _apply_isotopes_imperative_statements(body: list[str], arrays: dict[str, np.ndarray]) -> None:
    # Execute imperative assignments (products/copies/looped copies) that
    # follow DATA declarations and precede ISOTOPE packing (N=0 block).
    _execute_iso_statements(body, arrays, {})


def _build_isotope_array(isoion: np.ndarray) -> np.ndarray:
    isotope = np.zeros((10, 2, _MION), dtype=np.float64)
    n = 0
    for iz in range(1, 31):
        for _ion in range(1, iz + 2):
            n += 1
            isotope[:, 0, n - 1] = isoion[:10, iz - 1]
            isotope[:, 1, n - 1] = isoion[10:20, iz - 1]
    for iz in range(31, 100):
        for _ion in range(1, 6):
            n += 1
            isotope[:, 0, n - 1] = isoion[:10, iz - 1]
            isotope[:, 1, n - 1] = isoion[10:20, iz - 1]
    for iz in range(100, 266):
        n += 1
        isotope[:, 0, n - 1] = isoion[:10, iz - 1]
        isotope[:, 1, n - 1] = isoion[10:20, iz - 1]
    if n != _MION:
        raise RuntimeError(f"ISOTOPES mapping produced N={n}, expected {_MION}")
    return isotope


def _major_isotope_mass(isotope: np.ndarray) -> np.ndarray:
    amass = np.zeros(_MION, dtype=np.float64)
    xbest = np.zeros(_MION, dtype=np.float64)
    for n in range(_MION):
        for i in range(10):
            frac = float(isotope[i, 1, n])
            if frac > xbest[n]:
                xbest[n] = frac
                amass[n] = float(isotope[i, 0, n])
    return amass


def _load_isotopes_npz(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as data:
        isotope = np.asarray(data["isotope"], dtype=np.float64)
        amassiso_major = np.asarray(data["amassiso_major"], dtype=np.float64)
    if isotope.shape != (10, 2, _MION):
        raise RuntimeError(f"Invalid isotope shape in {npz_path}: {isotope.shape}")
    if amassiso_major.shape != (_MION,):
        raise RuntimeError(f"Invalid amassiso_major shape in {npz_path}: {amassiso_major.shape}")
    return isotope, amassiso_major


@lru_cache(maxsize=2)
def load_isotopes_from_atlas12(atlas12_path: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return Fortran-ordered (`ISOTOPE`, `AMASSISO(1,:)`) arrays.

    Returns:
    - isotope: shape (10, 2, 1006)
    - amassiso_major: shape (1006,), max-abundance isotope mass per NELION

    Imperative isotope assignments in `SUBROUTINE ISOTOPES` are applied
    before the final `ISOTOPE` packing loops.
    """

    if atlas12_path is None:
        npz_path = _default_isotopes_npz_path()
        if npz_path.exists():
            return _load_isotopes_npz(npz_path)

    src = Path(atlas12_path) if atlas12_path is not None else _default_atlas12_path()
    if not src.exists():
        raise FileNotFoundError(f"atlas12.for not found: {src}")
    lines = src.read_text(encoding="utf-8", errors="replace").splitlines()
    body = _extract_isotopes_body(lines)
    isoion = _parse_isoion_from_isotopes_body(body)
    isotope = _build_isotope_array(isoion)
    amassiso_major = _major_isotope_mass(isotope)
    return isotope, amassiso_major

