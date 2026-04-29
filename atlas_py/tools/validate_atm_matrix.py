"""Mass Python ATLAS `.atm` validation against the Fortran ground-truth grid.

Reads `tmp_atlas_debug/fortran_atm_grid/manifest.json` (or scans case directories),
runs `atlas_py.cli` once per case using the stored `input.atm` and Fortran `fort.12`,
compares `python_iter1.atm` to the stored `fortran_iter1.atm`, and writes reports under
each case directory (default: ``<case>/py_matrix/``).

Does **not** re-run Fortran when reference files are present; use
``python -m atlas_py.tools.validate_phase1`` for a full Fortran+Python single-case run.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from atlas_py.io.atmosphere import load_atm
from atlas_py.tools.validate_phase1 import (
    _default_kurucz_root,
    _ensure_gfpred_assembled,
    _run,
    _trim_fortran_state,
)


@dataclass
class CaseRecord:
    case_dir: str
    teff: float | None
    logg: float | None
    mh: float | None
    skipped: bool
    skip_reason: str | None
    python_ok: bool
    textual_pass: bool | None
    max_numeric_frac: float | None
    textual_status_line: str | None
    error: str | None


_CASE_DIR_RE = re.compile(r"^case_\d{3}_")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_compare_atm_textual(stdout: str) -> tuple[bool | None, float | None, str | None]:
    """Parse key lines from compare_atm_textual stdout."""
    max_frac: float | None = None
    status_line: str | None = None
    passed: bool | None = None
    for line in stdout.splitlines():
        if line.startswith("max_numeric_frac="):
            try:
                max_frac = float(line.split("=", 1)[1].strip())
            except ValueError:
                pass
        if line.startswith("status="):
            status_line = line.strip()
            if "FAIL" in line:
                passed = False
            elif "PASS" in line:
                passed = True
    return passed, max_frac, status_line


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _discover_cases_from_dirs(grid_root: Path) -> list[dict[str, Any]]:
    """Build pseudo-manifest entries by scanning ``case_*`` directories."""
    samples: list[dict[str, Any]] = []
    for p in sorted(grid_root.iterdir()):
        if not p.is_dir() or not _CASE_DIR_RE.match(p.name):
            continue
        input_atm = p / "input.atm"
        fortran_atm = p / "fortran_iter1.atm"
        flog = p / "fortran_iter1.log"
        if not (input_atm.is_file() and fortran_atm.is_file()):
            continue
        samples.append(
            {
                "case_dir": str(p),
                "input_atm": str(input_atm),
                "fortran_atm": str(fortran_atm),
                "fortran_log": str(flog) if flog.is_file() else "",
                "success": True,
                "teff": None,
                "logg": None,
                "mh": None,
            }
        )
    return samples


def _build_python_cmd(
    input_atm: Path,
    python_atm: Path,
    python_state: Path,
    gfpred_bin: Path,
    lowobs_bin: Path,
    hilines_bin: Path,
    diatomics_bin: Path,
    tio_bin: Path,
    h2o_bin: Path,
    nlteline_bin: Path,
    molecules_new: Path,
    iterations: int,
    enable_molecules: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "atlas_py.cli",
        str(input_atm.resolve()),
        "--output-atm",
        str(python_atm),
        "--iterations",
        str(max(1, iterations)),
        "--debug-state",
        str(python_state),
        "--fort11",
        str(gfpred_bin),
        "--fort111",
        str(lowobs_bin),
        "--fort21",
        str(hilines_bin),
        "--fort31",
        str(diatomics_bin),
        "--fort41",
        str(tio_bin),
        "--fort51",
        str(h2o_bin),
        "--nlteline-bin",
        str(nlteline_bin),
    ]
    if enable_molecules:
        cmd.extend(["--enable-molecules", "--molecules", str(molecules_new)])
    return cmd


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Run Python ATLAS over the Fortran ground-truth grid and compare `.atm` outputs."
        )
    )
    p.add_argument(
        "--grid-root",
        type=Path,
        default=None,
        help="Root of fortran_atm_grid (default: <repo>/tmp_atlas_debug/fortran_atm_grid)",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to manifest.json (default: <grid-root>/manifest.json)",
    )
    p.add_argument(
        "--scan-dirs-if-no-manifest",
        action="store_true",
        help="If manifest.json is missing under --grid-root, discover case_* dirs instead of erroring",
    )
    p.add_argument(
        "--artifacts-subdir",
        type=str,
        default="py_matrix",
        help="Subdirectory under each case_dir for Python outputs and reports",
    )
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument(
        "--enable-molecules",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match validate_phase1 default (molecules on)",
    )
    p.add_argument("--show-worst", type=int, default=5)
    p.add_argument("--atm-frac-threshold", type=float, default=0.10)
    p.add_argument(
        "--skip-state-compare",
        action="store_true",
        help="Skip Fortran log extract and compare_state_npz",
    )
    p.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="If >0, only process the first N successful manifest entries",
    )
    p.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Write summary JSON (default: <grid-root>/<artifacts-subdir>_summary.json)",
    )
    p.add_argument("--kurucz-root", type=Path, default=_default_kurucz_root())
    p.add_argument("--molecules-new", type=Path, default=None)
    p.add_argument("--gfpred-bin", type=Path, default=None)
    p.add_argument("--lowobs-bin", type=Path, default=None)
    p.add_argument("--hilines-bin", type=Path, default=None)
    p.add_argument("--diatomics-bin", type=Path, default=None)
    p.add_argument("--tio-bin", type=Path, default=None)
    p.add_argument("--h2o-bin", type=Path, default=None)
    args = p.parse_args()

    repo = _repo_root()
    grid_root = (
        args.grid_root.resolve()
        if args.grid_root is not None
        else (repo / "tmp_atlas_debug" / "fortran_atm_grid")
    )

    manifest_path = (
        args.manifest.resolve()
        if args.manifest is not None
        else (grid_root / "manifest.json")
    )

    samples: list[dict[str, Any]]
    manifest_meta: dict[str, Any] = {}
    if manifest_path.is_file():
        data = _load_manifest(manifest_path)
        manifest_meta = {k: v for k, v in data.items() if k != "samples"}
        samples = list(data.get("samples", []))
    elif args.scan_dirs_if_no_manifest:
        print(f"[warn] No manifest at {manifest_path}; scanning directories under {grid_root}")
        samples = _discover_cases_from_dirs(grid_root)
    else:
        print(
            f"ERROR: Manifest not found: {manifest_path}\n"
            "Use --scan-dirs-if-no-manifest to discover case_* directories, or generate the grid first.",
            file=sys.stderr,
        )
        return 2

    kurucz_root = args.kurucz_root.resolve()
    lines_dir = kurucz_root / "lines"
    mol_dir = kurucz_root / "molecules"
    molecules_new = (args.molecules_new or (lines_dir / "molecules.new")).resolve()
    gfpred_bin = (args.gfpred_bin or (lines_dir / "gfpred29dec2014.bin")).resolve()
    lowobs_bin = (args.lowobs_bin or (lines_dir / "lowobsat12.bin")).resolve()
    hilines_bin = (args.hilines_bin or (lines_dir / "hilines.bin")).resolve()
    diatomics_bin = (args.diatomics_bin or (lines_dir / "diatomicspacksrt.bin")).resolve()
    tio_bin = (args.tio_bin or (mol_dir / "tio" / "schwenke.bin")).resolve()
    h2o_bin = (args.h2o_bin or (mol_dir / "h2o" / "h2ofastfix.bin")).resolve()
    nltelinobsat12_bin = (lines_dir / "nltelinobsat12.bin").resolve()
    _ensure_gfpred_assembled(gfpred_bin)

    summary_path = (
        args.summary.resolve()
        if args.summary is not None
        else (grid_root / f"{args.artifacts_subdir}_summary.json")
    )

    records: list[CaseRecord] = []
    n_processed = 0
    any_fail = False

    for sample in samples:
        if args.max_cases > 0 and n_processed >= args.max_cases:
            break
        if not sample.get("success", False):
            records.append(
                CaseRecord(
                    case_dir=sample.get("case_dir", ""),
                    teff=sample.get("teff"),
                    logg=sample.get("logg"),
                    mh=sample.get("mh"),
                    skipped=True,
                    skip_reason="manifest success=false",
                    python_ok=False,
                    textual_pass=None,
                    max_numeric_frac=None,
                    textual_status_line=None,
                    error=None,
                )
            )
            continue

        case_dir = Path(sample["case_dir"])
        input_atm = Path(sample["input_atm"])
        fortran_atm = Path(sample["fortran_atm"])
        fortran_log = Path(sample["fortran_log"]) if sample.get("fortran_log") else None

        if not case_dir.is_dir():
            records.append(
                CaseRecord(
                    case_dir=str(case_dir),
                    teff=sample.get("teff"),
                    logg=sample.get("logg"),
                    mh=sample.get("mh"),
                    skipped=True,
                    skip_reason="case_dir missing",
                    python_ok=False,
                    textual_pass=None,
                    max_numeric_frac=None,
                    textual_status_line=None,
                    error=None,
                )
            )
            continue
        if not input_atm.is_file() or not fortran_atm.is_file():
            records.append(
                CaseRecord(
                    case_dir=str(case_dir),
                    teff=sample.get("teff"),
                    logg=sample.get("logg"),
                    mh=sample.get("mh"),
                    skipped=True,
                    skip_reason="missing input.atm or fortran_iter1.atm",
                    python_ok=False,
                    textual_pass=None,
                    max_numeric_frac=None,
                    textual_status_line=None,
                    error=None,
                )
            )
            continue

        n_processed += 1
        out_dir = case_dir / args.artifacts_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        python_atm = out_dir / "python_iter1.atm"
        python_state = out_dir / "python_state.npz"
        atm_report = out_dir / "compare_atm.txt"
        atm_textual_report = out_dir / "compare_atm_textual.txt"
        state_report = out_dir / "compare_state.txt"
        fortran_state = out_dir / "fortran_state.npz"

        err: str | None = None
        py_ok = False
        textual_pass: bool | None = None
        max_frac: float | None = None
        status_line: str | None = None

        try:
            py_cmd = _build_python_cmd(
                input_atm=input_atm,
                python_atm=python_atm,
                python_state=python_state,
                gfpred_bin=gfpred_bin,
                lowobs_bin=lowobs_bin,
                hilines_bin=hilines_bin,
                diatomics_bin=diatomics_bin,
                tio_bin=tio_bin,
                h2o_bin=h2o_bin,
                nlteline_bin=nltelinobsat12_bin,
                molecules_new=molecules_new,
                iterations=args.iterations,
                enable_molecules=args.enable_molecules,
            )
            _run(py_cmd)
            py_ok = True

            compare_atm_cmd = [
                sys.executable,
                "-m",
                "atlas_py.tools.compare_atm",
                str(fortran_atm),
                str(python_atm),
                "--show-worst",
                str(max(0, args.show_worst)),
            ]
            atm_text = _run(compare_atm_cmd)
            atm_report.write_text(atm_text, encoding="utf-8")

            compare_textual_cmd = [
                sys.executable,
                "-m",
                "atlas_py.tools.compare_atm_textual",
                str(fortran_atm),
                str(python_atm),
                "--frac-threshold",
                str(args.atm_frac_threshold),
                "--show-worst",
                str(max(0, args.show_worst)),
            ]
            proc_textual = subprocess.run(
                compare_textual_cmd,
                text=True,
                capture_output=True,
            )
            textual_out = proc_textual.stdout + (
                "\n" + proc_textual.stderr if proc_textual.stderr else ""
            )
            atm_textual_report.write_text(textual_out, encoding="utf-8")
            textual_pass, max_frac, status_line = _parse_compare_atm_textual(textual_out)
            if proc_textual.returncode != 0 or textual_pass is False:
                any_fail = True

            if not args.skip_state_compare and fortran_log and fortran_log.is_file():
                layers = load_atm(input_atm).layers
                extract_cmd = [
                    sys.executable,
                    "-m",
                    "atlas_py.tools.extract_fortran_eos_table",
                    str(fortran_log),
                    "--output",
                    str(fortran_state),
                ]
                try:
                    _run(extract_cmd)
                    _trim_fortran_state(fortran_state, layers=layers)
                    compare_state_cmd = [
                        sys.executable,
                        "-m",
                        "atlas_py.tools.compare_state_npz",
                        str(fortran_state),
                        str(python_state),
                    ]
                    state_text = _run(compare_state_cmd)
                except RuntimeError as exc:
                    state_text = f"State comparison unavailable.\nReason:\n{exc}\n"
                state_report.write_text(state_text, encoding="utf-8")
            elif not args.skip_state_compare:
                state_report.write_text(
                    "State comparison skipped: no fortran_iter1.log for this case.\n",
                    encoding="utf-8",
                )
            else:
                state_report.write_text(
                    "State comparison skipped: --skip-state-compare\n", encoding="utf-8"
                )

        except RuntimeError as exc:
            err = str(exc)
            any_fail = True
            if atm_textual_report.parent.exists():
                atm_textual_report.write_text(
                    f"compare_atm_textual not run or incomplete.\n{err}\n",
                    encoding="utf-8",
                )

        records.append(
            CaseRecord(
                case_dir=str(case_dir),
                teff=sample.get("teff"),
                logg=sample.get("logg"),
                mh=sample.get("mh"),
                skipped=False,
                skip_reason=None,
                python_ok=py_ok,
                textual_pass=textual_pass,
                max_numeric_frac=max_frac,
                textual_status_line=status_line,
                error=err,
            )
        )

    summary: dict[str, Any] = {
        "grid_root": str(grid_root),
        "manifest": str(manifest_path) if manifest_path.is_file() else None,
        "artifacts_subdir": args.artifacts_subdir,
        "config": {
            "iterations": args.iterations,
            "enable_molecules": args.enable_molecules,
            "atm_frac_threshold": args.atm_frac_threshold,
            "skip_state_compare": args.skip_state_compare,
        },
        "manifest_meta": manifest_meta,
        "cases": [asdict(r) for r in records],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    n_run = sum(1 for r in records if not r.skipped)
    n_ok = sum(
        1
        for r in records
        if not r.skipped and r.python_ok and r.textual_pass is True and r.error is None
    )
    print(f"[ok] Summary written to {summary_path}")
    print(f"[ok] Cases in manifest processed (cap applied): {n_run}; textual PASS: {n_ok}")
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
