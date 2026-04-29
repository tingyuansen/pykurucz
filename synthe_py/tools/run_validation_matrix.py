#!/usr/bin/env python3
"""Run the full self-contained validation matrix and archive reports."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _run_parity_gate(
    root: Path,
    outdir: Path,
    wlbeg: float,
    wlend: float,
    resolution: float,
    threshold: float,
    skip_stage_compare: bool,
    atmospheres: list[str],
    disable_parsed_cache: bool,
    disable_compiled_cache: bool,
    tfort12: Path | None,
    tfort19: Path | None,
    python_timeout: str,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "synthe_py.tools.parity_gate",
        "--root",
        str(root),
        "--outdir",
        str(outdir),
        "--wlbeg",
        str(wlbeg),
        "--wlend",
        str(wlend),
        "--resolution",
        str(resolution),
        "--threshold",
        str(threshold),
    ]
    if skip_stage_compare:
        cmd.append("--skip-stage-compare")
    if atmospheres:
        cmd.extend(["--atmospheres", *atmospheres])
    if tfort12 is not None:
        cmd.extend(["--tfort12", str(tfort12)])
    if tfort19 is not None:
        cmd.extend(["--tfort19", str(tfort19)])
    cmd.extend(["--python-timeout", python_timeout])

    env = os.environ.copy()
    if disable_parsed_cache:
        env["PY_DISABLE_PARSED_CACHE"] = "1"
    if disable_compiled_cache:
        env["PY_DISABLE_COMPILED_CACHE"] = "1"

    proc = subprocess.run(
        cmd,
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        timeout=7200,
    )

    summary_path = outdir / "parity_gate_summary.json"
    summary: dict[str, Any] = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except ValueError:
            summary = {}
    return {
        "command": cmd,
        "returncode": int(proc.returncode),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
        "summary": summary,
    }


def _collect_stage_metrics(stage_report_path: Path) -> dict[str, Any]:
    if not stage_report_path.exists():
        return {"available": False}
    data = json.loads(stage_report_path.read_text(encoding="utf-8"))
    out: dict[str, Any] = {"available": True, "atmospheres": {}}
    for atm, payload in data.items():
        flux = next(
            (s for s in payload.get("stages", []) if s.get("stage") == "rt_flux"), None
        )
        cont = next(
            (s for s in payload.get("stages", []) if s.get("stage") == "rt_continuum"),
            None,
        )
        out["atmospheres"][atm] = {
            "first_failure": payload.get("first_failure"),
            "rt_flux_rms": None if flux is None else flux.get("rms_rel_err"),
            "rt_flux_max": None if flux is None else flux.get("max_rel_err"),
            "rt_cont_rms": None if cont is None else cont.get("rms_rel_err"),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full parity validation matrix")
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--resolution", type=float, default=300000.0)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Output root directory (default: <root>/stage_diag/validation_matrix)",
    )
    parser.add_argument("--disable-parsed-cache", action="store_true", default=False)
    parser.add_argument("--disable-compiled-cache", action="store_true", default=False)
    parser.add_argument(
        "--tfort12",
        type=Path,
        default=None,
        help="Optional path to metadata ground-truth tfort.12",
    )
    parser.add_argument(
        "--tfort19",
        type=Path,
        default=None,
        help="Optional path to metadata ground-truth tfort.19",
    )
    parser.add_argument(
        "--python-timeout",
        type=str,
        default=None,
        help="Python stage timeout in seconds; use 'none' or '0' for no timeout",
    )
    parser.add_argument(
        "--atmospheres",
        nargs="+",
        default=["t05770g4.44", "t03750g3.50", "t02500g-1.0"],
    )
    args = parser.parse_args()

    root = args.root.resolve()
    out_root = (
        args.out_root.resolve()
        if args.out_root
        else (root / "stage_diag" / "validation_matrix")
    )
    out_root.mkdir(parents=True, exist_ok=True)

    narrow_windows = [(368.0, 372.0), (300.0, 304.0), (850.0, 854.0)]
    band_windows = [(300.0, 500.0), (500.0, 900.0), (900.0, 1800.0)]
    full_windows = [(300.0, 1800.0)]

    report: dict[str, Any] = {
        "config": {
            "resolution": args.resolution,
            "threshold": args.threshold,
            "atmospheres": args.atmospheres,
            "disable_parsed_cache": bool(args.disable_parsed_cache),
            "disable_compiled_cache": bool(args.disable_compiled_cache),
        },
        "runs": {"narrow": [], "bands": [], "full": []},
    }

    for wlbeg, wlend in narrow_windows:
        outdir = out_root / "narrow" / f"{wlbeg:g}_{wlend:g}"
        outdir.mkdir(parents=True, exist_ok=True)
        result = _run_parity_gate(
            root=root,
            outdir=outdir,
            wlbeg=wlbeg,
            wlend=wlend,
            resolution=args.resolution,
            threshold=args.threshold,
            skip_stage_compare=True,
            atmospheres=args.atmospheres,
            disable_parsed_cache=args.disable_parsed_cache,
            disable_compiled_cache=args.disable_compiled_cache,
            tfort12=args.tfort12.resolve() if args.tfort12 else None,
            tfort19=args.tfort19.resolve() if args.tfort19 else None,
            python_timeout=args.python_timeout,
        )
        report["runs"]["narrow"].append({"window": [wlbeg, wlend], **result})

    for wlbeg, wlend in band_windows:
        outdir = out_root / "bands" / f"{wlbeg:g}_{wlend:g}"
        outdir.mkdir(parents=True, exist_ok=True)
        result = _run_parity_gate(
            root=root,
            outdir=outdir,
            wlbeg=wlbeg,
            wlend=wlend,
            resolution=args.resolution,
            threshold=args.threshold,
            skip_stage_compare=False,
            atmospheres=args.atmospheres,
            disable_parsed_cache=args.disable_parsed_cache,
            disable_compiled_cache=args.disable_compiled_cache,
            tfort12=args.tfort12.resolve() if args.tfort12 else None,
            tfort19=args.tfort19.resolve() if args.tfort19 else None,
            python_timeout=args.python_timeout,
        )
        stage_path = outdir / "stage_compare.json"
        report["runs"]["bands"].append(
            {
                "window": [wlbeg, wlend],
                **result,
                "stage_metrics": _collect_stage_metrics(stage_path),
            }
        )

    for wlbeg, wlend in full_windows:
        outdir = out_root / "full" / f"{wlbeg:g}_{wlend:g}"
        outdir.mkdir(parents=True, exist_ok=True)
        result = _run_parity_gate(
            root=root,
            outdir=outdir,
            wlbeg=wlbeg,
            wlend=wlend,
            resolution=args.resolution,
            threshold=args.threshold,
            skip_stage_compare=False,
            atmospheres=args.atmospheres,
            disable_parsed_cache=args.disable_parsed_cache,
            disable_compiled_cache=args.disable_compiled_cache,
            tfort12=args.tfort12.resolve() if args.tfort12 else None,
            tfort19=args.tfort19.resolve() if args.tfort19 else None,
            python_timeout=args.python_timeout,
        )
        stage_path = outdir / "stage_compare.json"
        report["runs"]["full"].append(
            {
                "window": [wlbeg, wlend],
                **result,
                "stage_metrics": _collect_stage_metrics(stage_path),
            }
        )

    report_path = out_root / "validation_matrix_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report_path": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
