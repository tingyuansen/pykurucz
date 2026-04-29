"""Extract JOSH/BLOCKJ/BLOCKH tables directly from atlas12.for.

Outputs:
- CH_WEIGHTS (51,)
- CK_WEIGHTS (51,)
- XTAU_GRID (51,)
- COEFJ_MATRIX (51, 51)
- COEFH_MATRIX (51, 51)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

_NUM_RE = re.compile(r"[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][+\-]?\d+)?")


def _extract_subroutine(lines: list[str], name: str) -> list[str]:
    start = None
    for i, line in enumerate(lines):
        if f"SUBROUTINE {name}" in line:
            start = i
            break
    if start is None:
        raise ValueError(f"Subroutine not found: {name}")
    end = None
    for i in range(start + 1, len(lines)):
        if lines[i].strip() == "END":
            end = i
            break
    if end is None:
        raise ValueError(f"End not found for subroutine: {name}")
    return lines[start : end + 1]


def _parse_data_blocks(sub_lines: list[str], var_name: str) -> list[float]:
    vals: list[float] = []
    i = 0
    key = f"DATA {var_name}"
    while i < len(sub_lines):
        line = sub_lines[i]
        if key not in line:
            i += 1
            continue
        # For fixed-form Fortran, columns 1-6 are labels/continuations.
        payload = line[6:] if len(line) > 6 else line
        if "/" not in payload:
            raise ValueError(f"DATA line missing '/': {line}")
        payload = payload.split("/", 1)[1]
        block_text = payload
        while "/" not in block_text:
            i += 1
            if i >= len(sub_lines):
                raise ValueError(f"Unterminated DATA block for {var_name}")
            nxt = sub_lines[i]
            block_text += " " + (nxt[6:] if len(nxt) > 6 else nxt)
        block_text = block_text.split("/", 1)[0]
        for tok in _NUM_RE.findall(block_text):
            vals.append(float(tok.replace("D", "E").replace("d", "e")))
        i += 1
    if not vals:
        raise ValueError(f"No DATA values parsed for {var_name}")
    return vals


def main() -> int:
    p = argparse.ArgumentParser(description="Extract JOSH tables from atlas12.for")
    _repo_root = Path(__file__).resolve().parents[2]
    _default_atlas12 = (
        (_repo_root / "data" / "src" / "atlas12.for")
        if (_repo_root / "data" / "src" / "atlas12.for").exists()
        else (_repo_root.parent / "kurucz" / "src" / "atlas12.for")
    )
    p.add_argument(
        "--atlas12-for",
        type=Path,
        default=_default_atlas12,
        help="Path to atlas12.for (default: data/src/atlas12.for or ../kurucz/src/atlas12.for)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "josh_tables_atlas12.npz",
        help="Output .npz path",
    )
    args = p.parse_args()

    lines = args.atlas12_for.read_text(encoding="utf-8", errors="replace").splitlines()
    josh_lines = _extract_subroutine(lines, "JOSH")
    blockj_lines = _extract_subroutine(lines, "BLOCKJ")
    blockh_lines = _extract_subroutine(lines, "BLOCKH")

    ch_weights = np.asarray(_parse_data_blocks(josh_lines, "CH"), dtype=np.float64)
    ck_weights = np.asarray(_parse_data_blocks(josh_lines, "CK"), dtype=np.float64)
    xtau_grid = np.asarray(_parse_data_blocks(josh_lines, "XTAU8"), dtype=np.float64)
    coefj_flat = np.asarray(_parse_data_blocks(blockj_lines, "CJ"), dtype=np.float64)
    coefh_flat = np.asarray(_parse_data_blocks(blockh_lines, "CH"), dtype=np.float64)

    if ch_weights.size != 51 or ck_weights.size != 51 or xtau_grid.size != 51:
        raise ValueError(
            f"Unexpected JOSH vector sizes: CH={ch_weights.size}, CK={ck_weights.size}, XTAU8={xtau_grid.size}"
        )
    if coefj_flat.size != 2601 or coefh_flat.size != 2601:
        raise ValueError(
            f"Unexpected BLOCK matrix sizes: CJ={coefj_flat.size}, CH={coefh_flat.size}"
        )

    coefj_matrix = coefj_flat.reshape((51, 51), order="F")
    coefh_matrix = coefh_flat.reshape((51, 51), order="F")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        ch_weights=ch_weights,
        ck_weights=ck_weights,
        xtau_grid=xtau_grid,
        coefj_matrix=coefj_matrix,
        coefh_matrix=coefh_matrix,
    )
    print(f"Wrote ATLAS12 JOSH tables: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
