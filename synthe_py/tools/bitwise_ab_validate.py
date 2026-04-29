#!/usr/bin/env python3
"""Run strict bitwise A/B validation for Python synthesis outputs.

This tool executes baseline and candidate pipelines for one or more atmospheres,
then enforces a strict output gate:
  1) SHA256 digest equality
  2) byte-for-byte file equality

It also records runtime deltas parsed from synthesis logs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


TOTAL_RE = re.compile(r"Timing: total pipeline in ([0-9.]+)s")
LINE_OPACITY_RE = re.compile(r"Timing: line opacity stage in ([0-9.]+)s")
TRANSP_RE = re.compile(r"Timing: TRANSP in ([0-9.]+)s")


def parse_env_pairs(items: List[str]) -> Dict[str, str]:
    env_map: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid env assignment '{item}'. Expected KEY=VALUE.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid env assignment '{item}'. Empty key.")
        env_map[key] = value
    return env_map


def resolve_atm_path(repo_root: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_file():
        return candidate.resolve()
    sample_candidate = repo_root / "samples" / value
    if sample_candidate.is_file():
        return sample_candidate.resolve()
    raise FileNotFoundError(f"Atmosphere file not found: {value}")


def parse_runtime_metrics(log_text: str) -> Tuple[float | None, float | None, float | None]:
    total = TOTAL_RE.search(log_text)
    line_opacity = LINE_OPACITY_RE.search(log_text)
    transp = TRANSP_RE.search(log_text)
    return (
        float(total.group(1)) if total else None,
        float(line_opacity.group(1)) if line_opacity else None,
        float(transp.group(1)) if transp else None,
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def run_pipeline(
    *,
    repo_root: Path,
    atm_path: Path,
    npz_path: Path,
    spec_path: Path,
    log_path: Path,
    cache_dir: Path,
    wl_start: int,
    wl_end: int,
    resolution: int,
    n_workers: int,
    extra_env: Dict[str, str],
) -> Tuple[float | None, float | None, float | None]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(extra_env)

    convert_cmd = [
        sys.executable,
        str(repo_root / "synthe_py" / "tools" / "convert_atm_to_npz.py"),
        str(atm_path),
        str(npz_path),
        "--atlas-tables",
        str(repo_root / "synthe_py" / "data" / "atlas_tables.npz"),
    ]
    synth_cmd = [
        sys.executable,
        "-m",
        "synthe_py.cli",
        str(atm_path),
        str(repo_root / "lines" / "gfallvac.latest"),
        "--npz",
        str(npz_path),
        "--spec",
        str(spec_path),
        "--wl-start",
        str(wl_start),
        "--wl-end",
        str(wl_end),
        "--resolution",
        str(resolution),
        "--n-workers",
        str(n_workers),
        "--cache",
        str(cache_dir),
        "--log-level",
        "INFO",
    ]

    with log_path.open("w", encoding="utf-8") as log_f:
        log_f.write("# convert command\n")
        log_f.write(shlex.join(convert_cmd) + "\n\n")
        log_f.flush()
        convert = subprocess.run(
            convert_cmd,
            cwd=repo_root,
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
        if convert.returncode != 0:
            raise RuntimeError(f"convert_atm_to_npz failed for {atm_path.name}. See {log_path}")

        log_f.write("\n# synth command\n")
        log_f.write(shlex.join(synth_cmd) + "\n\n")
        log_f.flush()
        synth = subprocess.run(
            synth_cmd,
            cwd=repo_root,
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
        if synth.returncode != 0:
            raise RuntimeError(f"synthe_py.cli failed for {atm_path.name}. See {log_path}")

    log_text = log_path.read_text(encoding="utf-8")
    return parse_runtime_metrics(log_text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict bitwise A/B validator for spectrum outputs.")
    parser.add_argument(
        "--atm",
        action="append",
        required=True,
        help="Atmosphere file (absolute path or filename under samples/). Repeatable.",
    )
    parser.add_argument("--wl-start", type=int, default=300)
    parser.add_argument("--wl-end", type=int, default=1800)
    parser.add_argument("--resolution", type=int, default=300000)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument(
        "--baseline-env",
        action="append",
        default=[],
        help="Environment assignment KEY=VALUE for baseline run. Repeatable.",
    )
    parser.add_argument(
        "--candidate-env",
        action="append",
        default=[],
        help="Environment assignment KEY=VALUE for candidate run. Repeatable.",
    )
    parser.add_argument(
        "--results-root",
        default="results/bitwise_ab",
        help="Output directory for A/B artifacts and summary CSV.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately on first mismatch.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    baseline_env = parse_env_pairs(args.baseline_env)
    candidate_env = parse_env_pairs(args.candidate_env)

    results_root = (repo_root / args.results_root).resolve()
    baseline_dir = results_root / "baseline"
    candidate_dir = results_root / "candidate"
    summary_csv = results_root / "ab_summary.csv"
    results_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []

    for atm_value in args.atm:
        atm_path = resolve_atm_path(repo_root, atm_value)
        stem = atm_path.stem
        print(f"[A/B] {stem}")

        baseline_npz = baseline_dir / "npz" / f"{stem}.npz"
        baseline_spec = baseline_dir / "spec" / f"{stem}.spec"
        baseline_log = baseline_dir / "logs" / f"{stem}.log"
        baseline_cache = baseline_dir / "cache"

        candidate_npz = candidate_dir / "npz" / f"{stem}.npz"
        candidate_spec = candidate_dir / "spec" / f"{stem}.spec"
        candidate_log = candidate_dir / "logs" / f"{stem}.log"
        candidate_cache = candidate_dir / "cache"

        base_total, base_line_op, base_transp = run_pipeline(
            repo_root=repo_root,
            atm_path=atm_path,
            npz_path=baseline_npz,
            spec_path=baseline_spec,
            log_path=baseline_log,
            cache_dir=baseline_cache,
            wl_start=args.wl_start,
            wl_end=args.wl_end,
            resolution=args.resolution,
            n_workers=args.n_workers,
            extra_env=baseline_env,
        )
        cand_total, cand_line_op, cand_transp = run_pipeline(
            repo_root=repo_root,
            atm_path=atm_path,
            npz_path=candidate_npz,
            spec_path=candidate_spec,
            log_path=candidate_log,
            cache_dir=candidate_cache,
            wl_start=args.wl_start,
            wl_end=args.wl_end,
            resolution=args.resolution,
            n_workers=args.n_workers,
            extra_env=candidate_env,
        )

        base_sha = sha256_file(baseline_spec)
        cand_sha = sha256_file(candidate_spec)
        sha_match = base_sha == cand_sha
        byte_match = baseline_spec.read_bytes() == candidate_spec.read_bytes()
        passed = sha_match and byte_match

        if not passed:
            print(f"  FAIL: bitwise mismatch for {stem}")
        else:
            print(f"  PASS: bitwise match for {stem}")

        rows.append(
            {
                "atmosphere": stem,
                "pass": int(passed),
                "sha_match": int(sha_match),
                "byte_match": int(byte_match),
                "baseline_sha256": base_sha,
                "candidate_sha256": cand_sha,
                "baseline_total_s": base_total,
                "candidate_total_s": cand_total,
                "total_speedup_x": (base_total / cand_total) if (base_total and cand_total and cand_total > 0) else "",
                "baseline_line_opacity_s": base_line_op,
                "candidate_line_opacity_s": cand_line_op,
                "line_opacity_speedup_x": (base_line_op / cand_line_op) if (base_line_op and cand_line_op and cand_line_op > 0) else "",
                "baseline_transp_s": base_transp,
                "candidate_transp_s": cand_transp,
                "transp_speedup_x": (base_transp / cand_transp) if (base_transp and cand_transp and cand_transp > 0) else "",
                "baseline_log": str(baseline_log),
                "candidate_log": str(candidate_log),
                "baseline_spec": str(baseline_spec),
                "candidate_spec": str(candidate_spec),
            }
        )

        if args.fail_fast and not passed:
            break

    fieldnames = [
        "atmosphere",
        "pass",
        "sha_match",
        "byte_match",
        "baseline_sha256",
        "candidate_sha256",
        "baseline_total_s",
        "candidate_total_s",
        "total_speedup_x",
        "baseline_line_opacity_s",
        "candidate_line_opacity_s",
        "line_opacity_speedup_x",
        "baseline_transp_s",
        "candidate_transp_s",
        "transp_speedup_x",
        "baseline_log",
        "candidate_log",
        "baseline_spec",
        "candidate_spec",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {summary_csv}")

    all_pass = all(int(row["pass"]) == 1 for row in rows) if rows else False
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
