"""Run end-to-end SYNTHE parity for one validated ATLAS case.

Pipeline:
1) Run Fortran SYNTHE from case_dir/fortran/fortran_iter1.atm.
2) Run Python SYNTHE from case_dir/python/python_iter1.atm.
3) Compare spectra over a wavelength window.
4) Enforce normalized-flux per-wavelength threshold.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from synthe_py.tools.compare_spectra import compare_spectra, load_spectrum


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_kurucz_root() -> Path:
    repo_root = _default_repo_root()
    # Prefer self-contained data/ tree; fall back to sibling kurucz/ repo.
    data_dir = repo_root / "data"
    if data_dir.is_dir():
        return data_dir
    return repo_root.parent / "kurucz"


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    log_handle=None,
) -> str:
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
    )
    output = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
    if log_handle is not None and output:
        log_handle.write(output)
        if not output.endswith("\n"):
            log_handle.write("\n")
    if proc.returncode != 0:
        joined = " ".join(cmd)
        raise RuntimeError(f"Command failed ({proc.returncode}): {joined}\n{output}")
    return output


def _run_streaming(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    log_handle=None,
) -> str:
    """Run a subprocess and stream merged stdout/stderr to log_handle line-by-line.

    Uses PYTHONUNBUFFERED=1 so child Python processes emit INFO/timing logs immediately.
    """
    merged_env = {**os.environ, **(env or {})}
    merged_env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.Popen(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(cwd) if cwd is not None else None,
        env=merged_env,
        bufsize=1,
    )
    chunks: list[str] = []
    if proc.stdout is not None:
        for line in proc.stdout:
            chunks.append(line)
            if log_handle is not None:
                log_handle.write(line)
                log_handle.flush()
    _ = proc.wait()
    output = "".join(chunks)
    if proc.returncode != 0:
        joined = " ".join(cmd)
        raise RuntimeError(f"Command failed ({proc.returncode}): {joined}\n{output}")
    return output


def _run_fortran_synthe(
    *,
    kurucz_root: Path,
    atm_file: Path,
    output_spec: Path,
    line_list_dir: str,
    log_path: Path,
    save_intermediates_dir: Path | None = None,
) -> None:
    bin_dir = kurucz_root / "bin_macos"
    if not bin_dir.exists():
        bin_dir = kurucz_root / "bin_linux"
    if not bin_dir.exists():
        raise FileNotFoundError(
            f"Neither {kurucz_root / 'bin_macos'} nor {kurucz_root / 'bin_linux'} exists."
        )

    required_bins = [
        "at12tosyn.exe",
        "xnfpelsyn.exe",
        "synthe.exe",
        "spectrv.exe",
        "syntoascanga.exe",
    ]
    for exe in required_bins:
        if not (bin_dir / exe).exists():
            raise FileNotFoundError(f"Missing Fortran executable: {bin_dir / exe}")

    line_dir = kurucz_root / "lines"
    linelist_dir = kurucz_root / "synthe" / line_list_dir
    if not linelist_dir.exists():
        raise FileNotFoundError(f"Missing line-list directory: {linelist_dir}")

    tfort_files = ["tfort.12", "tfort.14", "tfort.19", "tfort.20", "tfort.93"]
    for name in tfort_files:
        if not (linelist_dir / name).exists():
            raise FileNotFoundError(f"Missing line-list file: {linelist_dir / name}")

    output_spec.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_root = output_spec.parent.parent / "logs"
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmpdir = tmp_root / f"tmp_fortran_synthe_{output_spec.stem}"
    if tmpdir.exists():
        shutil.rmtree(tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)

    model_name = atm_file.name
    try:
        with log_path.open("w", encoding="utf-8") as logf:
            logf.write(f"Using BIN_DIR={bin_dir}\n")
            logf.write(f"Using ATM={atm_file}\n")
            logf.write(f"Using line-list dir={linelist_dir}\n")

            _run(
                [str(bin_dir / "at12tosyn.exe"), str(atm_file.resolve()), model_name],
                cwd=tmpdir,
                log_handle=logf,
            )

            (tmpdir / "fort.2").symlink_to(line_dir / "molecules.dat")
            (tmpdir / "fort.17").symlink_to(line_dir / "continua.dat")

            model_text = (tmpdir / model_name).read_text(encoding="utf-8")
            proc = subprocess.run(
                [str(bin_dir / "xnfpelsyn.exe")],
                input=model_text,
                text=True,
                capture_output=True,
                cwd=str(tmpdir),
            )
            xn_output = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
            if xn_output:
                logf.write(xn_output)
                if not xn_output.endswith("\n"):
                    logf.write("\n")
            if proc.returncode != 0:
                raise RuntimeError(f"xnfpelsyn.exe failed ({proc.returncode})")

            mapping = {
                "tfort.12": "fort.12",
                "tfort.14": "fort.14",
                "tfort.19": "fort.19",
                "tfort.20": "fort.20",
                "tfort.93": "fort.93",
            }
            for src, dst in mapping.items():
                (tmpdir / dst).symlink_to(linelist_dir / src)
            (tmpdir / "fort.18").symlink_to(line_dir / "he1tables.dat")

            _run([str(bin_dir / "synthe.exe")], cwd=tmpdir, log_handle=logf)

            (tmpdir / "fort.5").symlink_to(tmpdir / model_name)
            (tmpdir / "fort.25").symlink_to(kurucz_root / "infiles" / "spectrv_std.input")
            _run([str(bin_dir / "spectrv.exe")], cwd=tmpdir, log_handle=logf)

            fort7 = tmpdir / "fort.7"
            if not fort7.exists():
                raise RuntimeError("spectrv.exe did not produce fort.7")
            shutil.move(str(fort7), str(tmpdir / "fort.1"))
            _run([str(bin_dir / "syntoascanga.exe")], cwd=tmpdir, log_handle=logf)

            spec_dat = tmpdir / "specfile.dat"
            if not spec_dat.exists():
                raise RuntimeError("syntoascanga.exe did not produce specfile.dat")
            shutil.move(str(spec_dat), str(output_spec))

            if save_intermediates_dir is not None:
                save_intermediates_dir.mkdir(parents=True, exist_ok=True)
                for fname in ("fort.10", "fort.29", "fort.9"):
                    src = tmpdir / fname
                    if src.exists():
                        shutil.copy2(str(src), str(save_intermediates_dir / fname))
                        logf.write(f"Saved {fname} to {save_intermediates_dir / fname}\n")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _run_python_synthe(
    *,
    repo_root: Path,
    atm_file: Path,
    output_spec: Path,
    npz_path: Path,
    log_path: Path,
    wl_start: float,
    wl_end: float,
    resolution: float,
    n_workers: int | None = None,
) -> None:
    output_spec.parent.mkdir(parents=True, exist_ok=True)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    line_list = repo_root / "lines" / "gfallvac.latest"
    atlas_tables = repo_root / "synthe_py" / "data" / "atlas_tables.npz"
    if not line_list.exists():
        raise FileNotFoundError(f"Missing Python line list: {line_list}")
    if not atlas_tables.exists():
        raise FileNotFoundError(f"Missing atlas tables: {atlas_tables}")

    cpu = os.cpu_count() or 1
    effective_workers = n_workers if n_workers is not None else cpu
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(
            "======================================================================\n"
            + "validate_synthe_e2e: Python SYNTHE log (streaming, line-buffered)\n"
            + f"  n_workers={effective_workers} (default when omitted: all logical CPUs = {cpu})\n"
            + "  Child Python runs with -u and PYTHONUNBUFFERED=1 so Timing: lines appear live.\n"
            + "======================================================================\n\n"
        )
        logf.flush()

        t0 = time.perf_counter()
        logf.write("==== STEP: convert_atm_to_npz START ====\n")
        logf.flush()
        _run_streaming(
            [
                sys.executable,
                "-u",
                str(repo_root / "synthe_py" / "tools" / "convert_atm_to_npz.py"),
                str(atm_file.resolve()),
                str(npz_path.resolve()),
                "--atlas-tables",
                str(atlas_tables.resolve()),
            ],
            cwd=repo_root,
            log_handle=logf,
        )
        logf.write(
            f"==== STEP: convert_atm_to_npz END wall={time.perf_counter() - t0:.3f}s ====\n\n"
        )
        logf.flush()

        t1 = time.perf_counter()
        logf.write("==== STEP: synthe_py.cli START ====\n")
        logf.flush()
        _run_streaming(
            [
                sys.executable,
                "-u",
                "-m",
                "synthe_py.cli",
                str(atm_file.resolve()),
                str(line_list.resolve()),
                "--npz",
                str(npz_path.resolve()),
                "--spec",
                str(output_spec.resolve()),
                "--wl-start",
                str(wl_start),
                "--wl-end",
                str(wl_end),
                "--resolution",
                str(resolution),
                "--n-workers",
                str(effective_workers),
                "--log-level",
                "INFO",
            ],
            cwd=repo_root,
            log_handle=logf,
        )
        logf.write(
            f"==== STEP: synthe_py.cli END wall={time.perf_counter() - t1:.3f}s ====\n"
        )
        logf.flush()


def _resolve_case_atm_paths(case_dir: Path) -> tuple[Path, Path]:
    """Resolve Fortran/Python ATM paths from case artifacts.

    Case directories are not fully uniform; prefer canonical locations first,
    then fall back to discovered artifacts.

    For SYNTHE validation, the Python synthesis uses the same Fortran-generated
    ATM as the Fortran synthesis.  The Python ATLAS12 ATM (python_iter1.atm)
    uses space-separated fields so ``convert_atm_to_npz`` reads the true gas
    pressure (very small at the surface) rather than the FREEFF-misread value
    that Fortran xnfpelsyn sees from the concatenated Fortran ATM.  Using the
    Fortran ATM for both runs ensures both pipelines start from identical
    atmospheric input and that ``_freeff_parse_float`` correctly reproduces
    xnfpelsyn's FREEFF behaviour.
    """
    canonical_fortran = [
        case_dir / "fortran" / "fortran_iter1.atm",
        case_dir / "python" / "fortran_iter1.atm",
    ]
    canonical_python = [
        case_dir / "python" / "python_iter1.atm",
    ]

    fortran_existing = [p for p in canonical_fortran if p.exists()]
    python_existing = [p for p in canonical_python if p.exists()]

    if not fortran_existing:
        discovered = [
            p
            for p in case_dir.rglob("fortran_iter1.atm")
            if "logs" not in p.parts and "inputs" not in p.parts
        ]
        if discovered:
            discovered.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            fortran_existing = [discovered[0]]

    if not python_existing:
        # Fall back to Fortran ATM for Python synthesis: ensures _freeff_parse_float
        # replicates xnfpelsyn FREEFF behaviour for concatenated surface-layer fields.
        if fortran_existing:
            python_existing = [fortran_existing[0]]
        else:
            discovered = [
                p
                for p in case_dir.rglob("python_iter1.atm")
                if "logs" not in p.parts and "inputs" not in p.parts
            ]
            if discovered:
                discovered.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                python_existing = [discovered[0]]

    if not fortran_existing:
        raise FileNotFoundError(
            f"Missing Fortran ATM in case directory: {case_dir / 'fortran' / 'fortran_iter1.atm'}"
        )
    if not python_existing:
        raise FileNotFoundError(
            f"Missing Python ATM in case directory: {case_dir / 'python' / 'python_iter1.atm'}"
        )
    return fortran_existing[0], python_existing[0]


def _compute_norm_flux_errors(
    python_spec: Path,
    fortran_spec: Path,
    wl_start: float,
    wl_end: float,
) -> tuple[float, int, int, list[dict[str, float]]]:
    py_wl, py_flux, py_cont = load_spectrum(python_spec)
    ft_wl, ft_flux, ft_cont = load_spectrum(fortran_spec)
    wl_min = max(float(np.min(py_wl)), float(np.min(ft_wl)), wl_start)
    wl_max = min(float(np.max(py_wl)), float(np.max(ft_wl)), wl_end)

    py_mask = (py_wl >= wl_min) & (py_wl <= wl_max)
    ft_mask = (ft_wl >= wl_min) & (ft_wl <= wl_max)
    py_wl_common = py_wl[py_mask]
    py_flux_common = py_flux[py_mask]
    py_cont_common = py_cont[py_mask]

    ft_wl_common = ft_wl[ft_mask]
    ft_flux_common = ft_flux[ft_mask]
    ft_cont_common = ft_cont[ft_mask]

    ft_flux_interp = np.interp(py_wl_common, ft_wl_common, ft_flux_common)
    ft_cont_interp = np.interp(py_wl_common, ft_wl_common, ft_cont_common)
    py_norm = py_flux_common / np.maximum(py_cont_common, 1e-30)
    ft_norm = ft_flux_interp / np.maximum(ft_cont_interp, 1e-30)
    # Workspace parity rule uses absolute normalized-flux error:
    # |py_norm - ft_norm| < 0.10 for all wavelengths in range.
    abs_err = np.abs(py_norm - ft_norm)

    order = np.argsort(abs_err)[::-1]
    top = []
    for idx in order[:10]:
        top.append(
            {
                "wavelength_nm": float(py_wl_common[idx]),
                "abs_err": float(abs_err[idx]),
                "py_norm": float(py_norm[idx]),
                "ft_norm": float(ft_norm[idx]),
            }
        )
    return float(np.max(abs_err)), int(np.sum(abs_err > 0.10)), int(abs_err.size), top


def main() -> int:
    p = argparse.ArgumentParser(description="End-to-end ATLAS->SYNTHE validator for one case")
    p.add_argument("--case-dir", type=Path, required=True, help="Case dir under tmp_atlas_debug")
    p.add_argument("--kurucz-root", type=Path, default=_default_kurucz_root())
    p.add_argument("--line-list-dir", type=str, default="linelists_full")
    p.add_argument("--wl-start", type=float, default=300.0)
    p.add_argument("--wl-end", type=float, default=1800.0)
    p.add_argument("--resolution", type=float, default=300000.0)
    p.add_argument("--norm-frac-threshold", type=float, default=0.10)
    p.add_argument(
        "--force-rerun-python",
        action="store_true",
        help="Re-run Python SYNTHE even if a cached .spec already exists",
    )
    p.add_argument(
        "--force-rerun-fortran",
        action="store_true",
        help=(
            "Re-run Fortran synthe.exe even if a cached .spec already exists. "
            "Only needed when the input .atm or Fortran line lists change; "
            "Fortran output is otherwise stable ground truth."
        ),
    )
    p.add_argument(
        "--force-rerun",
        action="store_true",
        help="Re-run both Fortran and Python SYNTHE (shorthand for --force-rerun-fortran --force-rerun-python)",
    )
    p.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help=(
            "Parallel workers passed to synthe_py.cli (metal/RT stages). "
            "Default: all logical CPUs (os.cpu_count()) for maximum throughput; "
            "use 1 only for debugging."
        ),
    )
    args = p.parse_args()
    # --force-rerun is a convenience alias for both flags
    if args.force_rerun:
        args.force_rerun_fortran = True
        args.force_rerun_python = True

    repo_root = _default_repo_root()
    case_dir = args.case_dir.resolve()
    fortran_atm, python_atm = _resolve_case_atm_paths(case_dir)

    wl_tag = f"{int(args.wl_start)}_{int(args.wl_end)}"
    fortran_spec = case_dir / "fortran" / f"fortran_synthe_{wl_tag}.spec"
    python_spec = case_dir / "python" / f"python_synthe_{wl_tag}.spec"
    python_npz = case_dir / "python" / "python_iter1_synthe.npz"
    fortran_log = case_dir / "logs" / f"fortran_synthe_{wl_tag}.log"
    python_log = case_dir / "logs" / f"python_synthe_{wl_tag}.log"
    compare_report = case_dir / "logs" / f"compare_synthe_{wl_tag}.txt"
    compare_json = case_dir / "logs" / f"compare_synthe_{wl_tag}.json"

    if args.force_rerun_fortran or not fortran_spec.exists():
        _run_fortran_synthe(
            kurucz_root=args.kurucz_root.resolve(),
            atm_file=fortran_atm,
            output_spec=fortran_spec,
            line_list_dir=args.line_list_dir,
            log_path=fortran_log,
            save_intermediates_dir=case_dir / "fortran",
        )

    if args.force_rerun_python or not python_spec.exists():
        _run_python_synthe(
            repo_root=repo_root,
            atm_file=python_atm,
            output_spec=python_spec,
            npz_path=python_npz,
            log_path=python_log,
            wl_start=args.wl_start,
            wl_end=args.wl_end,
            resolution=args.resolution,
            n_workers=args.n_workers,
        )

    summary = compare_spectra(
        python_file=python_spec,
        fortran_file=fortran_spec,
        wl_range=(args.wl_start, args.wl_end),
        top_n=10,
        quiet=True,
    )
    max_norm_abs, outliers_over_ten_percent, n_points, top = _compute_norm_flux_errors(
        python_spec=python_spec,
        fortran_spec=fortran_spec,
        wl_start=args.wl_start,
        wl_end=args.wl_end,
    )
    threshold = float(args.norm_frac_threshold)
    outliers_over_threshold = outliers_over_ten_percent
    status = "PASS" if max_norm_abs <= threshold else "FAIL"

    report_lines = [
        f"case={case_dir.name}",
        f"python_spec={python_spec}",
        f"fortran_spec={fortran_spec}",
        f"wavelength_window_nm={args.wl_start:.1f}-{args.wl_end:.1f}",
        f"norm_frac_threshold={threshold:.6f}",
        f"status={status}",
        f"n_points={n_points}",
        f"max_norm_abs={max_norm_abs:.6e}",
        f"outliers_over_10pct={outliers_over_ten_percent}",
        f"summary_norm_rms={summary['norm_rms']:.6e}",
        "top_norm_outliers:",
    ]
    for item in top:
        report_lines.append(
            "  wl={wavelength_nm:.6f} abs={abs_err:.6e} py_norm={py_norm:.6e} ft_norm={ft_norm:.6e}".format(
                **item
            )
        )
    compare_report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    compare_payload = {
        "case": case_dir.name,
        "status": status,
        "wavelength_window_nm": [args.wl_start, args.wl_end],
        "norm_frac_threshold": threshold,
        "max_norm_abs": max_norm_abs,
        "outliers_over_10pct": outliers_over_ten_percent,
        "outliers_over_threshold": outliers_over_threshold,
        "n_points": n_points,
        "summary": {k: float(v) for k, v in summary.items()},
        "top_norm_outliers": top,
        "fortran_spec": str(fortran_spec),
        "python_spec": str(python_spec),
        "fortran_log": str(fortran_log),
        "python_log": str(python_log),
    }
    compare_json.write_text(json.dumps(compare_payload, indent=2), encoding="utf-8")

    print(f"[ok] Fortran spectrum: {fortran_spec}")
    print(f"[ok] Python spectrum:  {python_spec}")
    print(f"[ok] Compare report:   {compare_report}")
    print(f"[ok] Compare JSON:     {compare_json}")
    print(
        "[ok] max_norm_abs={:.6e}, threshold={:.6e}, status={}".format(
            max_norm_abs, threshold, status
        )
    )

    if status != "PASS":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
