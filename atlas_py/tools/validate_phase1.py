"""Run one-shot Phase-1 ATLAS12 ground-truth validation.

Pipeline:
1) Run Fortran atlas12.exe for one iteration from an input .atm.
2) Run atlas_py for the same iteration count from the same input .atm.
3) Compare output .atm files.
4) Extract Fortran EOS table and compare against atlas_py debug-state arrays.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from atlas_py.io.atmosphere import load_atm


def _default_kurucz_root() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    # Prefer self-contained data/ tree; fall back to sibling kurucz/ repo.
    data_dir = repo_root / "data"
    if data_dir.is_dir():
        return data_dir
    return repo_root.parent / "kurucz"


def _ensure_gfpred_assembled(gfpred_bin: Path) -> None:
    """Verify that gfpred29dec2014.bin is present; raise a helpful error if not.

    The name is kept for backward compatibility with existing callers.
    The file is distributed as a single binary via scripts/download_data.py
    (pulled from Google Drive); the previous split-parts assembly is no
    longer needed.
    """
    if gfpred_bin.exists():
        return
    raise FileNotFoundError(
        f"Required atlas_py binary not found: {gfpred_bin}\n"
        "Run `python scripts/download_data.py` to populate data/, or "
        "`bash scripts/setup_data.sh --source /path/to/kurucz` if you have a "
        "local Kurucz tree."
    )


def _run(cmd: list[str], env: dict[str, str] | None = None) -> str:
    proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
    output = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
        joined = " ".join(cmd)
        raise RuntimeError(f"Command failed ({proc.returncode}): {joined}\n{output}")
    return output


def _trim_fortran_state(path: Path, layers: int) -> bool:
    """Keep only the last `layers` rows if multiple EOS blocks are present."""
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: np.asarray(data[k]) for k in data.files}
    rows = int(arrays["p"].shape[0])
    if rows <= layers:
        return False
    for key, arr in list(arrays.items()):
        if arr.ndim >= 1 and arr.shape[0] == rows:
            arrays[key] = arr[-layers:]
    np.savez(path, **arrays)
    return True


def main() -> int:
    p = argparse.ArgumentParser(description="One-shot Phase-1 EOS/.atm validator")
    p.add_argument("--input-atm", type=Path, required=True, help="Initial .atm file")
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for all validation artifacts",
    )
    p.add_argument("--iterations", type=int, default=1, help="atlas_py iterations")
    p.add_argument(
        "--enable-molecules",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable molecular path in atlas_py (Fortran side is molecules-on "
            "in run_single_iteration)"
        ),
    )
    p.add_argument(
        "--enable-continuum-baseline",
        action="store_true",
        help="Enable atlas_py continuum-only ABROSS baseline during validation",
    )
    p.add_argument(
        "--enable-rt-baseline",
        action="store_true",
        help="Enable atlas_py continuum RT baseline (JOSH+ROSS+RADIAP) during validation",
    )
    p.add_argument(
        "--enable-convec-edens-fd",
        action="store_true",
        help="Enable experimental CONVEC EDENS finite-difference branch in atlas_py",
    )
    p.add_argument(
        "--enable-convec-log",
        action="store_true",
        help="Enable per-depth CONVEC diagnostics for Fortran/Python and compare logs.",
    )
    p.add_argument(
        "--show-worst",
        type=int,
        default=5,
        help="Top-N worst depths in compare_atm output",
    )
    p.add_argument(
        "--atm-frac-threshold",
        type=float,
        default=0.10,
        help="Max allowed per-token fractional error in full .atm comparison",
    )
    p.add_argument(
        "--kurucz-root",
        type=Path,
        default=_default_kurucz_root(),
        help="Path to sibling kurucz/ directory",
    )
    p.add_argument("--atlas12-exe", type=Path, default=None)
    p.add_argument("--molecules-new", type=Path, default=None)
    p.add_argument("--gfpred-bin", type=Path, default=None)
    p.add_argument("--lowobs-bin", type=Path, default=None)
    p.add_argument("--hilines-bin", type=Path, default=None)
    p.add_argument("--diatomics-bin", type=Path, default=None)
    p.add_argument("--tio-bin", type=Path, default=None)
    p.add_argument("--h2o-bin", type=Path, default=None)
    p.add_argument("--nltelinobsat12-bin", type=Path, default=None)
    p.add_argument(
        "--enable-trace",
        action="store_true",
        help="Enable Fortran/Python trace generation and first-divergence report.",
    )
    p.add_argument("--trace-wlo-nm", type=float, default=381.0)
    p.add_argument("--trace-whi-nm", type=float, default=410.0)
    p.add_argument("--trace-jlo", type=int, default=58)
    p.add_argument("--trace-jhi", type=int, default=63)
    p.add_argument("--trace-max-events", type=int, default=50000)
    args = p.parse_args()

    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    layers = load_atm(args.input_atm).layers

    kurucz_root = args.kurucz_root.resolve()
    bin_dir = kurucz_root / "bin_macos"
    lines_dir = kurucz_root / "lines"
    mol_dir = kurucz_root / "molecules"

    atlas12_exe = (args.atlas12_exe or (bin_dir / "atlas12.exe")).resolve()
    molecules_new = (args.molecules_new or (lines_dir / "molecules.new")).resolve()
    gfpred_bin = (args.gfpred_bin or (lines_dir / "gfpred29dec2014.bin")).resolve()
    lowobs_bin = (args.lowobs_bin or (lines_dir / "lowobsat12.bin")).resolve()
    hilines_bin = (args.hilines_bin or (lines_dir / "hilines.bin")).resolve()
    diatomics_bin = (args.diatomics_bin or (lines_dir / "diatomicspacksrt.bin")).resolve()
    tio_bin = (args.tio_bin or (mol_dir / "tio" / "schwenke.bin")).resolve()
    h2o_bin = (args.h2o_bin or (mol_dir / "h2o" / "h2ofastfix.bin")).resolve()
    nltelinobsat12_bin = (
        args.nltelinobsat12_bin or (lines_dir / "nltelinobsat12.bin")
    ).resolve()

    _ensure_gfpred_assembled(gfpred_bin)

    iter_tag = f"iter{max(1, int(args.iterations))}"
    fortran_atm = outdir / f"fortran_{iter_tag}.atm"
    python_atm = outdir / f"python_{iter_tag}.atm"
    fortran_log = outdir / f"fortran_{iter_tag}.log"
    fortran_state = outdir / "fortran_state.npz"
    python_state = outdir / "python_state.npz"
    atm_report = outdir / "compare_atm.txt"
    input_vs_python_report = outdir / "compare_input_vs_python.txt"
    input_vs_fortran_report = outdir / "compare_input_vs_fortran.txt"
    atm_textual_report = outdir / "compare_atm_textual.txt"
    state_report = outdir / "compare_state.txt"
    fortran_trace = outdir / "fortran_trace.csv"
    python_trace = outdir / "python_trace.csv"
    trace_report = outdir / "first_divergence_trace.json"
    trace_text_report = outdir / "first_divergence_trace.txt"
    fortran_convec_log = outdir / "fortran_convec.csv"
    python_convec_log = outdir / "python_convec.csv"
    python_convec_fd_log = outdir / "python_convec_fd.csv"
    python_radiap_log = outdir / "python_radiap.csv"
    python_knu_log = outdir / "python_knu.csv"
    convec_report = outdir / "compare_convec.txt"
    nmolec_edens_log = outdir / "python_nmolec_edens.csv"
    trace_cfg = outdir / "trace_atlas12.cfg"
    if args.enable_trace:
        trace_cfg.write_text(
            f"{args.trace_wlo_nm} {args.trace_whi_nm} "
            f"{args.trace_jlo} {args.trace_jhi} {args.trace_max_events}\n",
            encoding="utf-8",
        )

    fortran_fort12 = outdir / "fortran_iter1_fort12.bin"
    run_fortran_cmd = [
        sys.executable,
        "-m",
        "atlas_py.tools.run_single_iteration",
        "--atlas12-exe",
        str(atlas12_exe),
        "--input-atm",
        str(args.input_atm.resolve()),
        "--output-atm",
        str(fortran_atm),
        "--molecules-new",
        str(molecules_new),
        "--gfpred-bin",
        str(gfpred_bin),
        "--lowobs-bin",
        str(lowobs_bin),
        "--hilines-bin",
        str(hilines_bin),
        "--diatomics-bin",
        str(diatomics_bin),
        "--tio-bin",
        str(tio_bin),
        "--h2o-bin",
        str(h2o_bin),
        "--nltelinobsat12-bin",
        str(nltelinobsat12_bin),
        "--log-path",
        str(fortran_log),
        "--output-lines-bin",
        str(fortran_fort12),
        "--iterations",
        str(max(1, args.iterations)),
    ]
    if args.enable_trace:
        run_fortran_cmd.extend(
            [
                "--trace-config",
                str(trace_cfg),
                "--output-trace-csv",
                str(fortran_trace),
            ]
        )
    if args.enable_convec_log:
        run_fortran_cmd.extend(["--convec-log", str(fortran_convec_log)])
    _run(run_fortran_cmd)

    run_python_cmd = [
        sys.executable,
        "-m",
        "atlas_py.cli",
        str(args.input_atm.resolve()),
        "--output-atm",
        str(python_atm),
        "--iterations",
        str(max(1, args.iterations)),
        "--debug-state",
        str(python_state),
        "--line-selection-bin",
        str(fortran_fort12),
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
        str(nltelinobsat12_bin),
    ]
    if args.enable_molecules:
        run_python_cmd.extend(["--enable-molecules", "--molecules", str(molecules_new)])
    if args.enable_continuum_baseline:
        run_python_cmd.append("--enable-continuum-baseline")
    if args.enable_rt_baseline:
        run_python_cmd.append("--enable-rt-baseline")
    if args.enable_convec_edens_fd:
        run_python_cmd.append("--enable-convec-edens-fd")
    env = None
    if args.enable_trace or args.enable_convec_log:
        env = dict(os.environ)
    if args.enable_trace and env is not None:
        env["ATLAS_TRACE_ENABLE"] = "1"
        env["ATLAS_TRACE_PATH"] = str(python_trace)
        env["ATLAS_TRACE_SOURCE"] = "python"
        env["ATLAS_TRACE_WLO_NM"] = str(args.trace_wlo_nm)
        env["ATLAS_TRACE_WHI_NM"] = str(args.trace_whi_nm)
        env["ATLAS_TRACE_JLO"] = str(args.trace_jlo)
        env["ATLAS_TRACE_JHI"] = str(args.trace_jhi)
        env["ATLAS_TRACE_MAX_EVENTS"] = str(args.trace_max_events)
    if args.enable_convec_log and env is not None:
        env["ATLAS_CONVEC_LOG"] = str(python_convec_log)
        env["ATLAS_CONVEC_FD_LOG"] = str(python_convec_fd_log)
        env["ATLAS_RADIAP_LOG"] = str(python_radiap_log)
        env["ATLAS_KNU_LOG"] = str(python_knu_log)
        env["ATLAS_NMOLEC_EDENS_LOG"] = str(nmolec_edens_log)
        env["ATLAS_NMOLEC_EDENS_MAXJ"] = "4"
    _run(run_python_cmd, env=env)

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

    input_vs_python_cmd = [
        sys.executable,
        "-m",
        "atlas_py.tools.compare_atm",
        str(args.input_atm.resolve()),
        str(python_atm),
        "--show-worst",
        str(max(0, args.show_worst)),
    ]
    input_vs_python_text = _run(input_vs_python_cmd)
    input_vs_python_report.write_text(input_vs_python_text, encoding="utf-8")

    input_vs_fortran_cmd = [
        sys.executable,
        "-m",
        "atlas_py.tools.compare_atm",
        str(args.input_atm.resolve()),
        str(fortran_atm),
        "--show-worst",
        str(max(0, args.show_worst)),
    ]
    input_vs_fortran_text = _run(input_vs_fortran_cmd)
    input_vs_fortran_report.write_text(input_vs_fortran_text, encoding="utf-8")

    compare_atm_textual_cmd = [
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
    try:
        atm_textual_text = _run(compare_atm_textual_cmd)
    except RuntimeError as exc:
        # Still emit report text even when strict textual-equivalence check fails.
        atm_textual_text = (
            "Full .atm textual/token comparison failed threshold.\n"
            "Reason:\n"
            f"{exc}\n"
        )
    atm_textual_report.write_text(atm_textual_text, encoding="utf-8")

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
        state_text = (
            "State comparison unavailable.\n"
            "Reason:\n"
            f"{exc}\n"
        )
    state_report.write_text(state_text, encoding="utf-8")

    if args.enable_trace:
        if fortran_trace.exists() and python_trace.exists():
            trace_cmd = [
                sys.executable,
                "-m",
                "atlas_py.tools.find_first_trace_divergence",
                "--fortran-trace",
                str(fortran_trace),
                "--python-trace",
                str(python_trace),
                "--output-json",
                str(trace_report),
                "--trace-max-events",
                str(args.trace_max_events),
            ]
            trace_text = _run(trace_cmd)
            trace_text_report.write_text(trace_text, encoding="utf-8")
        else:
            msg = (
                "Trace comparison unavailable.\n"
                f"fortran_trace_exists={fortran_trace.exists()}\n"
                f"python_trace_exists={python_trace.exists()}\n"
            )
            trace_text_report.write_text(msg, encoding="utf-8")

    if args.enable_convec_log:
        if fortran_convec_log.exists() and python_convec_log.exists():
            convec_cmd = [
                sys.executable,
                "-m",
                "atlas_py.tools.compare_convec_logs",
                "--fortran-log",
                str(fortran_convec_log),
                "--python-log",
                str(python_convec_log),
                "--frac-threshold",
                "0.01",
                "--show-worst",
                str(max(0, args.show_worst)),
            ]
            try:
                convec_text = _run(convec_cmd)
            except RuntimeError as exc:
                convec_text = (
                    "CONVEC log comparison found divergence.\n"
                    "Reason:\n"
                    f"{exc}\n"
                )
            convec_report.write_text(convec_text, encoding="utf-8")
        else:
            msg = (
                "CONVEC log comparison unavailable.\n"
                f"fortran_convec_exists={fortran_convec_log.exists()}\n"
                f"python_convec_exists={python_convec_log.exists()}\n"
            )
            convec_report.write_text(msg, encoding="utf-8")

    print(f"[ok] Validation artifacts written to {outdir}")
    print(f"[ok] ATM comparison report: {atm_report}")
    print(f"[ok] ATM textual report: {atm_textual_report}")
    print(f"[ok] Input->Python report: {input_vs_python_report}")
    print(f"[ok] Input->Fortran report: {input_vs_fortran_report}")
    print(f"[ok] State comparison report: {state_report}")
    if args.enable_trace:
        print(f"[ok] Trace text report: {trace_text_report}")
        if trace_report.exists():
            print(f"[ok] Trace JSON report: {trace_report}")
    if args.enable_convec_log:
        print(f"[ok] CONVEC report: {convec_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
