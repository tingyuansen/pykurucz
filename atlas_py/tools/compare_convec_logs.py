"""Compare Fortran/Python CONVEC per-depth diagnostic logs."""

from __future__ import annotations

import argparse
from pathlib import Path


_FLOAT_COLS = [
    "EDENS1",
    "EDENS2",
    "EDENS3",
    "EDENS4",
    "RHO1",
    "RHO2",
    "RHO3",
    "RHO4",
    "DEDT",
    "DRDT",
    "DEDPG",
    "DRDPG",
    "DLTDLP",
    "GRDADB",
    "DEL",
    "HEATCP",
    "DLRDLT",
    "FLXCNV",
]


def _read_rows(path: Path) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return rows
    default_header = ["J", *_FLOAT_COLS, "REASON"]
    first_parts = [p.strip() for p in lines[0].split(",")]
    if first_parts and first_parts[0] == "J":
        header = first_parts
        data_lines = lines[1:]
    else:
        header = default_header
        data_lines = lines
    for line in data_lines:
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == len(header):
            row = {header[i]: parts[i] for i in range(len(header))}
        else:
            # Fortran writer format: "J, <space-separated 10 floats>, REASON"
            # i.e., only 3 comma-separated fields.
            if len(parts) != 3:
                continue
            mid = parts[1].split()
            if len(mid) != len(_FLOAT_COLS):
                continue
            row = {"J": parts[0], "REASON": parts[2]}
            for idx, col in enumerate(_FLOAT_COLS):
                row[col] = mid[idx]
        try:
            j = int(row["J"])
        except Exception:
            continue
        rows[j] = row
    return rows


def _to_float(text: str) -> float:
    return float(text.replace("D", "E").replace("d", "e"))


def main() -> int:
    p = argparse.ArgumentParser(description="Compare CONVEC per-depth logs")
    p.add_argument("--fortran-log", type=Path, required=True)
    p.add_argument("--python-log", type=Path, required=True)
    p.add_argument("--frac-threshold", type=float, default=0.01)
    p.add_argument("--show-worst", type=int, default=10)
    args = p.parse_args()

    f_rows = _read_rows(args.fortran_log)
    p_rows = _read_rows(args.python_log)

    all_j = sorted(set(f_rows.keys()) | set(p_rows.keys()))
    print(f"fortran_rows={len(f_rows)} python_rows={len(p_rows)} aligned_rows={len(all_j)}")
    if not all_j:
        print("status=FAIL reason=no_rows")
        return 1

    first_reason = None
    first_float = None
    worst: list[tuple[float, int, str, float, float]] = []

    for j in all_j:
        fr = f_rows.get(j)
        pr = p_rows.get(j)
        if fr is None or pr is None:
            if first_reason is None:
                first_reason = (j, "ROW_MISSING", "ROW_MISSING")
            continue

        freason = fr.get("REASON", "")
        preason = pr.get("REASON", "")
        if first_reason is None and freason != preason:
            first_reason = (j, freason, preason)

        for col in _FLOAT_COLS:
            try:
                fv = _to_float(fr[col])
                pv = _to_float(pr[col])
            except Exception:
                continue
            denom = max(abs(fv), 1e-300)
            frac = abs(pv - fv) / denom
            worst.append((frac, j, col, fv, pv))
            if first_float is None and frac > args.frac_threshold:
                first_float = (j, col, fv, pv, frac)

    if first_reason is None:
        print("first_reason_mismatch=NONE")
    else:
        j, frs, prs = first_reason
        print(f"first_reason_mismatch=J{j} fortran={frs} python={prs}")

    if first_float is None:
        print(f"first_float_mismatch_over_{args.frac_threshold}=NONE")
    else:
        j, col, fv, pv, frac = first_float
        print(
            f"first_float_mismatch_over_{args.frac_threshold}=J{j} {col} "
            f"fortran={fv:.8e} python={pv:.8e} frac={frac:.8e}"
        )

    worst.sort(reverse=True, key=lambda x: x[0])
    nshow = max(0, int(args.show_worst))
    if nshow > 0:
        print("worst_float_deltas:")
        for frac, j, col, fv, pv in worst[:nshow]:
            print(f"  J={j:3d} {col:8s} frac={frac:.8e} fortran={fv:.8e} python={pv:.8e}")

    status = "PASS"
    if first_reason is not None or first_float is not None:
        status = "FAIL"
    print(f"status={status}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

