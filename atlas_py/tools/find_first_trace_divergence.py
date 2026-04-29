"""Find the first divergence between Fortran/Python trace files."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import json
from collections import defaultdict

from atlas_py.trace_events import TraceEvent, read_trace_events


def _fmt(ev: TraceEvent) -> str:
    return (
        f"key={ev.key()} src={ev.source} type={ev.type_code} "
        f"wlvac={ev.wlvac_nm:.9e} center={ev.center:.9e} adamp={ev.adamp:.9e} "
        f"cv={ev.cv:.9e} tab={ev.tabcont:.9e} branch={ev.branch} reason={ev.reason}"
    )


def _is_close(a: float, b: float, rel: float, abs_: float) -> bool:
    d = abs(a - b)
    return d <= max(abs_, rel * max(abs(a), abs(b), 1.0))


def _off_by_one_hint(ft: TraceEvent, py: TraceEvent) -> str:
    hints: list[str] = []
    if ft.key()[0] == py.key()[0] and ft.key()[1] == py.key()[1] and ft.key()[4] == py.key()[4]:
        d_depth = py.key()[2] - ft.key()[2]
        d_nu = py.key()[3] - ft.key()[3]
        if d_depth in (-1, 1):
            hints.append(f"depth off-by-one suspected (py-ft={d_depth})")
        if d_nu in (-1, 1):
            hints.append(f"nu off-by-one suspected (py-ft={d_nu})")
    d_line = py.key()[1] - ft.key()[1]
    if d_line in (-1, 1):
        hints.append(f"line off-by-one suspected (py-ft={d_line})")
    if not hints:
        return "no immediate off-by-one signature"
    return "; ".join(hints)


def _compare(ft: TraceEvent, py: TraceEvent, rel: float, abs_: float) -> list[str]:
    mismatches: list[str] = []
    if ft.type_code != py.type_code:
        mismatches.append(f"type_code: ft={ft.type_code} py={py.type_code}")
    if ft.branch != py.branch:
        mismatches.append(f"branch: ft={ft.branch} py={py.branch}")
    if ft.reason != py.reason:
        mismatches.append(f"reason: ft={ft.reason} py={py.reason}")
    for name in ("wlvac_nm", "center", "adamp", "cv", "tabcont"):
        fa = getattr(ft, name)
        pb = getattr(py, name)
        if not _is_close(float(fa), float(pb), rel=rel, abs_=abs_):
            mismatches.append(f"{name}: ft={fa:.9e} py={pb:.9e}")
    return mismatches


def main() -> int:
    ap = argparse.ArgumentParser(description="Find first Fortran/Python trace divergence")
    ap.add_argument("--fortran-trace", type=Path, required=True)
    ap.add_argument("--python-trace", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, default=None)
    ap.add_argument("--context", type=int, default=3)
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--atol", type=float, default=1e-10)
    ap.add_argument("--trace-max-events", type=int, default=None)
    args = ap.parse_args()

    ft_events = read_trace_events(args.fortran_trace)
    py_events = read_trace_events(args.python_trace)
    ft_event_names = {e.event for e in ft_events}
    py_event_names = {e.event for e in py_events}
    common_events = ft_event_names.intersection(py_event_names)
    if common_events:
        ft_events = [e for e in ft_events if e.event in common_events]
        py_events = [e for e in py_events if e.event in common_events]

    n = min(len(ft_events), len(py_events))
    report: dict[str, object] = {
        "fortran_count": len(ft_events),
        "python_count": len(py_events),
        "common_events": sorted(common_events),
        "status": "match",
    }
    trace_truncated = False
    if args.trace_max_events is not None and args.trace_max_events > 0:
        trace_truncated = (len(ft_events) >= args.trace_max_events) or (
            len(py_events) >= args.trace_max_events
        )
        report["trace_max_events"] = int(args.trace_max_events)
        report["trace_truncated"] = bool(trace_truncated)

    ft_map: dict[tuple[int, int, int, int, str], list[TraceEvent]] = defaultdict(list)
    py_map: dict[tuple[int, int, int, int, str], list[TraceEvent]] = defaultdict(list)
    for ev in ft_events:
        ft_map[ev.key()].append(ev)
    for ev in py_events:
        py_map[ev.key()].append(ev)
    all_keys = sorted(set(ft_map.keys()).union(py_map.keys()))
    dup_keys_ft = sum(1 for v in ft_map.values() if len(v) > 1)
    dup_keys_py = sum(1 for v in py_map.values() if len(v) > 1)
    if dup_keys_ft > 0 or dup_keys_py > 0:
        report["duplicate_key_count_fortran"] = dup_keys_ft
        report["duplicate_key_count_python"] = dup_keys_py
    key_miss = None
    for k in all_keys:
        in_ft = k in ft_map and len(ft_map[k]) > 0
        in_py = k in py_map and len(py_map[k]) > 0
        if in_ft != in_py:
            key_miss = k
            break
    if key_miss is not None:
        ft_list = ft_map.get(key_miss, [])
        py_list = py_map.get(key_miss, [])
        ft = ft_list[0] if ft_list else None
        py = py_list[0] if py_list else None
        hint = "check indexing and event gating"
        if ft is not None and py is not None:
            hint = _off_by_one_hint(ft, py)
        report.update(
            {
                "status": "key_mismatch",
                "key": key_miss,
                "present_in_fortran": ft is not None,
                "present_in_python": py is not None,
                "off_by_one_hint": hint,
                "fortran_event": asdict(ft) if ft is not None else None,
                "python_event": asdict(py) if py is not None else None,
            }
        )
        if trace_truncated:
            report["truncation_warning"] = (
                "at least one trace hit trace_max_events; key mismatch may be a truncation artifact"
            )
    else:
        for k in sorted(ft_map.keys()):
            ft_list = ft_map[k]
            py_list = py_map[k]
            if len(ft_list) != len(py_list):
                report.update(
                    {
                        "status": "count_mismatch",
                        "key": k,
                        "fortran_count_for_key": len(ft_list),
                        "python_count_for_key": len(py_list),
                        "off_by_one_hint": "check indexing and event gating",
                        "fortran_event": asdict(ft_list[0]) if ft_list else None,
                        "python_event": asdict(py_list[0]) if py_list else None,
                    }
                )
                if trace_truncated:
                    report["truncation_warning"] = (
                        "at least one trace hit trace_max_events; count mismatch may be a truncation artifact"
                    )
                break
            for occ_idx, (ft, py) in enumerate(zip(ft_list, py_list)):
                diffs = _compare(ft, py, rel=args.rtol, abs_=args.atol)
                if diffs:
                    report.update(
                        {
                            "status": "value_mismatch",
                            "key": k,
                            "occurrence_index": occ_idx,
                            "differences": diffs,
                            "off_by_one_hint": _off_by_one_hint(ft, py),
                            "fortran_event": asdict(ft),
                            "python_event": asdict(py),
                        }
                    )
                    break
            if report["status"] != "match":
                break

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"status: {report['status']}")
    print(f"fortran_count: {len(ft_events)}")
    print(f"python_count: {len(py_events)}")
    if report["status"] != "match":
        print("first divergence details:")
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
