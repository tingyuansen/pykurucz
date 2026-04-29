#!/usr/bin/env python3
"""
Extract DEQ/A9_2 traces for a specific layer / iteration / SOLVIT call.

Modes:
  - nmolec: pull the “Selected DEQ columns before SOLVIT” block (and FT/PY_DEQ_A9_2)
            from nmolec_debug*.log (only available for the first call per iter).
  - solvit: pull the `A9_2 BEFORE/AFTER eliminations` lines from solvit_debug*.log
            (these include call numbers for every SOLVIT invocation).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

NM_HEADER_RE = re.compile(
    r"(?P<prefix>FT|PY)_NMOLEC: Selected DEQ columns before SOLVIT "
    r"\(LAYER=\s*(?P<layer>\d+)\s+ITER=\s*(?P<iter>\d+)"
    r"(?:\s+CALL=\s*(?P<call>\d+))?\)"
)

A9_RE = re.compile(
    r"(?P<prefix>FT|PY)_DEQ_A9_2 iter=\s*(?P<iter>\d+)\s+layer=\s*(?P<layer>\d+)"
    r"(?:\s+call=\s*(?P<call>\d+))?\s+value=\s*(?P<value>[+-]?\d\.\d+E[+-]\d+)"
)

SOLVIT_A9_RE = re.compile(
    r"(?P<prefix>FT|PY)_MATRIX iter\s+(?P<iter>\d+):\s+A9_2\s+"
    r"(?P<stage>BEFORE|AFTER)\s+eliminations\s*=\s*(?P<value>[+-]?\d\.\d+E[+-]\d+)\s*"
    r"\(layer=\s*(?P<layer>\d+)\s+iter=\s*(?P<niter>\d+)\s+call=\s*(?P<call>\d+)\)"
)


def _find_block(
    path: Path,
    target_layer: int,
    target_iter: int,
    target_call: Optional[int],
) -> Optional[List[str]]:
    if not path.exists():
        return None

    current_buffer: List[str] = []
    capturing = False
    buffer_for_return: Optional[List[str]] = None

    with path.open() as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            header = NM_HEADER_RE.search(line)
            if header:
                if capturing and buffer_for_return is None:
                    buffer_for_return = list(current_buffer)
                layer = int(header.group("layer"))
                iteration = int(header.group("iter"))
                call_str = header.group("call")
                call = int(call_str) if call_str is not None else None

                capturing = (
                    layer == target_layer
                    and iteration == target_iter
                    and (target_call is None or call == target_call)
                )
                current_buffer = [line] if capturing else []
                continue

            if capturing:
                current_buffer.append(line)

    if capturing and buffer_for_return is None:
        buffer_for_return = current_buffer

    return buffer_for_return


def _find_a9_value(
    path: Path,
    target_layer: int,
    target_iter: int,
    target_call: Optional[int],
    *,
    solvit: bool = False,
) -> Optional[str]:
    if not path.exists():
        return None
    regex = SOLVIT_A9_RE if solvit else A9_RE
    with path.open() as fh:
        for line in fh:
            match = regex.search(line)
            if not match:
                continue
            layer = int(match.group("layer"))
            if solvit:
                iteration = int(match.group("niter"))
            else:
                iteration = int(match.group("iter"))
            call_str = match.group("call")
            call = int(call_str) if call_str is not None else None
            if layer == target_layer and iteration == target_iter:
                if target_call is None or call == target_call:
                    return line.rstrip("\n")
    return None


def _print_section(title: str, lines: Optional[Iterable[str]]) -> None:
    print(title)
    if not lines:
        print("  [no matching block found]")
        return
    for ln in lines:
        print(ln)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract DEQ tracked columns or SOLVIT A9_2 entries for a given "
            "layer/iter/SOLVIT call."
        )
    )
    parser.add_argument(
        "--layer", type=int, default=1, help="Atmosphere layer (1-based)"
    )
    parser.add_argument("--iter", type=int, default=0, help="Newton iteration index")
    parser.add_argument(
        "--call",
        type=int,
        default=None,
        help="SOLVIT call index to use for both logs (optional)",
    )
    parser.add_argument(
        "--ft-call",
        type=int,
        default=None,
        help="Override call index for the Fortran log (defaults to --call)",
    )
    parser.add_argument(
        "--py-call",
        type=int,
        default=None,
        help="Override call index for the Python log (defaults to --call)",
    )
    parser.add_argument(
        "--fortran-log",
        type=Path,
        default=Path("synthe/stmp_at12_aaaaa/nmolec_debug.log"),
        help="Path to Fortran nmolec_debug.log",
    )
    parser.add_argument(
        "--python-log",
        type=Path,
        default=Path("logs/nmolec_debug_python.log"),
        help="Path to Python logs/nmolec_debug_python.log",
    )
    parser.add_argument(
        "--fortran-solvit-log",
        type=Path,
        default=Path("synthe/stmp_at12_aaaaa/solvit_debug.log"),
        help="Path to Fortran solvit_debug.log",
    )
    parser.add_argument(
        "--python-solvit-log",
        type=Path,
        default=Path("solvit_debug_python.log"),
        help="Path to Python solvit_debug_python.log",
    )
    parser.add_argument(
        "--mode",
        choices=("nmolec", "solvit"),
        default="nmolec",
        help="Select nmolec (DEQ column blocks) or solvit (A9_2 entries)",
    )
    args = parser.parse_args()

    ft_call = args.ft_call if args.ft_call is not None else args.call
    py_call = args.py_call if args.py_call is not None else args.call

    call_caption = (
        f"ft_call={ft_call} py_call={py_call}"
        if ft_call != py_call
        else f"call={ft_call}"
    )

    if args.mode == "nmolec":
        ft_block = _find_block(args.fortran_log, args.layer, args.iter, ft_call)
        py_block = _find_block(args.python_log, args.layer, args.iter, py_call)

        print(
            f"=== Selected DEQ columns (layer={args.layer} iter={args.iter} "
            f"{call_caption}) ==="
        )
        _print_section("Fortran:", ft_block)
        _print_section("Python :", py_block)

        ft_a9 = _find_a9_value(
            args.fortran_log, args.layer, args.iter, ft_call, solvit=False
        )
        py_a9 = _find_a9_value(
            args.python_log, args.layer, args.iter, py_call, solvit=False
        )
    else:
        print(
            f"=== SOLVIT A9_2 entries (layer={args.layer} iter={args.iter} "
            f"{call_caption}) ==="
        )
        ft_a9 = _find_a9_value(
            args.fortran_solvit_log, args.layer, args.iter, ft_call, solvit=True
        )
        py_a9 = _find_a9_value(
            args.python_solvit_log, args.layer, args.iter, py_call, solvit=True
        )

    print("=== A9_2 values ===")
    print(ft_a9 or "Fortran: [not logged]")
    print(py_a9 or "Python : [not logged]")


if __name__ == "__main__":
    main()
