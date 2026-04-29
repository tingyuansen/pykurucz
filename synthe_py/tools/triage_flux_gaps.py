#!/usr/bin/env python3
"""Summarize rt_flux failures from validation matrix reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _extract_flux_metrics(run: dict[str, Any]) -> dict[str, Any]:
    metrics = {"worst_atmosphere": None, "worst_rt_flux_rms": None, "details": []}
    stage_metrics = run.get("stage_metrics", {})
    atmos = stage_metrics.get("atmospheres", {}) if isinstance(stage_metrics, dict) else {}
    worst_key = None
    worst_val = -1.0
    for atm, payload in atmos.items():
        rms = payload.get("rt_flux_rms")
        metrics["details"].append(
            {
                "atmosphere": atm,
                "first_failure": payload.get("first_failure"),
                "rt_flux_rms": rms,
                "rt_flux_max": payload.get("rt_flux_max"),
                "rt_cont_rms": payload.get("rt_cont_rms"),
            }
        )
        if isinstance(rms, (float, int)) and float(rms) > worst_val:
            worst_val = float(rms)
            worst_key = atm
    metrics["worst_atmosphere"] = worst_key
    metrics["worst_rt_flux_rms"] = None if worst_key is None else worst_val
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Triage rt_flux gaps from matrix report")
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Path to validation_matrix_report.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output JSON path (default: alongside report)",
    )
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    triage: dict[str, Any] = {"bands": [], "full": []}

    for run in report.get("runs", {}).get("bands", []):
        triage["bands"].append(
            {
                "window": run.get("window"),
                "returncode": run.get("returncode"),
                **_extract_flux_metrics(run),
            }
        )

    for run in report.get("runs", {}).get("full", []):
        triage["full"].append(
            {
                "window": run.get("window"),
                "returncode": run.get("returncode"),
                **_extract_flux_metrics(run),
            }
        )

    out_path = (
        args.out.resolve()
        if args.out is not None
        else args.report.resolve().with_name("validation_matrix_flux_triage.json")
    )
    out_path.write_text(json.dumps(triage, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"triage_report_path": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()







