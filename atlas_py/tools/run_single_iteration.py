"""Run one Fortran ATLAS12 iteration to produce a reference `.atm`."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="Run atlas12.exe for one iteration")
    p.add_argument("--atlas12-exe", type=Path, required=True)
    p.add_argument("--input-atm", type=Path, required=True)
    p.add_argument("--output-atm", type=Path, required=True)
    p.add_argument("--molecules-new", type=Path, required=True)
    p.add_argument("--gfpred-bin", type=Path, required=True)
    p.add_argument("--lowobs-bin", type=Path, required=True)
    p.add_argument("--hilines-bin", type=Path, required=True)
    p.add_argument("--diatomics-bin", type=Path, required=True)
    p.add_argument("--tio-bin", type=Path, required=True)
    p.add_argument("--h2o-bin", type=Path, required=True)
    p.add_argument("--nltelinobsat12-bin", type=Path, required=True)
    p.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="Optional path to save combined stdout/stderr from both passes",
    )
    p.add_argument(
        "--output-lines-bin",
        type=Path,
        default=None,
        help=(
            "Optional path to save selected-line binary produced by pass 1 "
            "(fort.12), for LINOP1 parity diagnostics."
        ),
    )
    p.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of internal ATLAS12 iterations for pass 2 (default: 1).",
    )
    p.add_argument(
        "--trace-config",
        type=Path,
        default=None,
        help="Optional trace config file to copy as trace_atlas12.cfg for Fortran tracing.",
    )
    p.add_argument(
        "--output-trace-csv",
        type=Path,
        default=None,
        help="Optional output path for atlas12_trace.csv produced by Fortran trace mode.",
    )
    p.add_argument(
        "--convec-log",
        type=Path,
        default=None,
        help="Optional output path for fort.30 CONVEC diagnostics from Fortran.",
    )
    args = p.parse_args()

    deck1 = """READ PUNCH
MOLECULES ON
READ MOLECULES
OPACITY ON LINES
OPACITY ON XLINES
READ LINES
CONVECTION OVER 1.25 0 36
ITERATIONS 1 PRINT 1 PUNCH 0
BEGIN
END
"""

    iterations = max(1, int(args.iterations))
    print_flags = " ".join(["1"] * iterations)
    punch_flags = " ".join(["0"] * (iterations - 1) + ["1"])
    deck2 = f"""READ PUNCH
MOLECULES ON
READ MOLECULES
OPACITY ON LINES
OPACITY ON XLINES
CONVECTION OVER 1.25 0 36
ITERATIONS {iterations}
PRINT {print_flags}
PUNCH {punch_flags}
BEGIN
END
"""

    with tempfile.TemporaryDirectory(prefix="atlas12_single_iter_") as td:
        work = Path(td)
        model = args.input_atm.name
        shutil.copy(args.input_atm, work / model)

        # First pass: read lines and create fort.12
        if args.trace_config is not None:
            shutil.copy(args.trace_config, work / "trace_atlas12.cfg")
        (work / "fort.2").symlink_to(args.molecules_new)
        (work / "fort.3").symlink_to(work / model)
        (work / "fort.11").symlink_to(args.gfpred_bin)
        (work / "fort.111").symlink_to(args.lowobs_bin)
        (work / "fort.21").symlink_to(args.hilines_bin)
        (work / "fort.31").symlink_to(args.diatomics_bin)
        (work / "fort.41").symlink_to(args.tio_bin)
        (work / "fort.51").symlink_to(args.h2o_bin)
        p1 = subprocess.run(
            [str(args.atlas12_exe)],
            input=deck1,
            text=True,
            cwd=work,
            check=True,
            capture_output=True,
        )
        if not (work / "fort.12").exists():
            raise RuntimeError("atlas12 first pass did not produce fort.12")
        shutil.move(work / "fort.12", work / "tmp.bin")
        if args.output_lines_bin is not None:
            args.output_lines_bin.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(work / "tmp.bin", args.output_lines_bin)

        # Second pass: one iteration and punch structure.
        for f in work.glob("fort.*"):
            if f.name != "tmp.bin":
                f.unlink(missing_ok=True)
        (work / "fort.2").symlink_to(args.molecules_new)
        (work / "fort.3").symlink_to(work / model)
        (work / "fort.12").symlink_to(work / "tmp.bin")
        (work / "fort.19").symlink_to(args.nltelinobsat12_bin)
        if args.trace_config is not None:
            shutil.copy(args.trace_config, work / "trace_atlas12.cfg")
        p2 = subprocess.run(
            [str(args.atlas12_exe)],
            input=deck2,
            text=True,
            cwd=work,
            check=True,
            capture_output=True,
        )
        if not (work / "fort.7").exists():
            # Some builds keep unit 7 preconnected to stdout. Try to recover
            # punched deck from stdout before failing.
            stdout_text = p2.stdout
            start = stdout_text.find("TEFF ")
            if start >= 0:
                (work / "fort.7").write_text(stdout_text[start:], encoding="utf-8")
            elif (work / model).exists():
                # Last-resort fallback: if the model deck was rewritten in place.
                src = (work / model).read_text(encoding="utf-8", errors="ignore")
                if "TEFF " in src and "READ DECK" in src:
                    shutil.copy(work / model, work / "fort.7")
        if not (work / "fort.7").exists():
            raise RuntimeError("atlas12 second pass did not produce fort.7")

        args.output_atm.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(work / "fort.7", args.output_atm)
        if args.log_path is not None:
            args.log_path.parent.mkdir(parents=True, exist_ok=True)
            args.log_path.write_text(
                (
                    "=== PASS 1 STDOUT ===\n"
                    + p1.stdout
                    + "\n=== PASS 1 STDERR ===\n"
                    + p1.stderr
                    + "\n=== PASS 2 STDOUT ===\n"
                    + p2.stdout
                    + "\n=== PASS 2 STDERR ===\n"
                    + p2.stderr
                    + "\n"
                ),
                encoding="utf-8",
            )
        if args.output_trace_csv is not None and (work / "atlas12_trace.csv").exists():
            args.output_trace_csv.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(work / "atlas12_trace.csv", args.output_trace_csv)
        if args.convec_log is not None and (work / "fort.30").exists():
            args.convec_log.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(work / "fort.30", args.convec_log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

