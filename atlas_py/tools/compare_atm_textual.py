"""Token-level `.atm` comparison with numeric outlier threshold."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


_FORTRAN_EXP_RE = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+))(?:[dD]([+-]?\d+))$")


def _split_tokens(line: str) -> list[str]:
    # Fortran fixed-format can emit touching signed fields (e.g. 4.131E+00-1.058E+07).
    fixed = re.sub(r"(?<=[0-9])([+-])(?=\d)", r" \1", line)
    return fixed.split()


def _to_float(tok: str) -> float | None:
    """Parse numeric token, including Fortran D exponents."""
    try:
        return float(tok)
    except ValueError:
        m = _FORTRAN_EXP_RE.match(tok)
        if m is None:
            return None
        return float(f"{m.group(1)}e{m.group(2)}")


def main() -> int:
    p = argparse.ArgumentParser(description="Compare two .atm files token-by-token")
    p.add_argument("ref", type=Path, help="Fortran/reference .atm")
    p.add_argument("test", type=Path, help="Python/test .atm")
    p.add_argument(
        "--frac-threshold",
        type=float,
        default=0.10,
        help="Maximum allowed per-token fractional error",
    )
    p.add_argument(
        "--show-worst",
        type=int,
        default=10,
        help="Show top-N worst numeric token differences",
    )
    args = p.parse_args()

    ref_lines = args.ref.read_text(encoding="utf-8", errors="replace").splitlines()
    test_lines = args.test.read_text(encoding="utf-8", errors="replace").splitlines()

    numeric_count = 0
    string_count = 0
    line_count_mismatch = abs(len(ref_lines) - len(test_lines))
    token_count_mismatch = 0
    string_mismatch = 0
    worst_numeric: list[tuple[float, int, int, str, str]] = []

    nlines = min(len(ref_lines), len(test_lines))
    for i in range(nlines):
        r_tokens = _split_tokens(ref_lines[i])
        t_tokens = _split_tokens(test_lines[i])
        if len(r_tokens) != len(t_tokens):
            token_count_mismatch += abs(len(r_tokens) - len(t_tokens))
        ntok = min(len(r_tokens), len(t_tokens))
        for j in range(ntok):
            rt = r_tokens[j]
            tt = t_tokens[j]
            rf = _to_float(rt)
            tf = _to_float(tt)
            if rf is not None and tf is not None:
                numeric_count += 1
                frac = abs(tf - rf) / max(abs(rf), 1e-300)
                worst_numeric.append((frac, i + 1, j + 1, rt, tt))
            else:
                string_count += 1
                if rt != tt:
                    string_mismatch += 1

    worst_numeric.sort(key=lambda x: x[0], reverse=True)
    max_frac = worst_numeric[0][0] if worst_numeric else 0.0
    outlier_count = sum(1 for x in worst_numeric if x[0] > args.frac_threshold)

    print(f"ref_lines={len(ref_lines)} test_lines={len(test_lines)}")
    print(f"line_count_mismatch={line_count_mismatch}")
    print(f"token_count_mismatch={token_count_mismatch}")
    print(f"string_tokens={string_count} string_mismatch={string_mismatch}")
    print(f"numeric_tokens={numeric_count}")
    print(f"max_numeric_frac={max_frac:.6e}")
    print(f"numeric_outliers_over_threshold={outlier_count}")
    print(f"frac_threshold={args.frac_threshold:.6e}")
    passed = (
        line_count_mismatch == 0
        and token_count_mismatch == 0
        and string_mismatch == 0
        and outlier_count == 0
    )
    print(f"status={'PASS' if passed else 'FAIL'}")

    if args.show_worst > 0 and worst_numeric:
        print("worst_numeric_tokens:")
        for frac, line_no, tok_no, rt, tt in worst_numeric[: args.show_worst]:
            print(
                f"  line={line_no} token={tok_no} frac={frac:.6e} ref={rt} test={tt}"
            )

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

