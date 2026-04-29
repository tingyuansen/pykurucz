"""Generate a grid of Fortran ATLAS12 ground-truth `.atm` files.

This script:
  1) Samples random (Teff, logg, [M/H]) points within a configurable range.
  2) Uses the kurucz-a1 emulator + pyKurucz `.atm` writer to create input
     atmosphere decks for each sampled point.
  3) Runs the Fortran atlas12.exe once for each input deck via
     `atlas_py.tools.run_single_iteration` to obtain `fortran_iter1.atm`.

Python ATLAS is *not* invoked here – this is purely to build a stable library
of Fortran ground-truth `.atm` files that can be reused for later parity tests.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from atlas_py.tools.validate_phase1 import _default_kurucz_root, _ensure_gfpred_assembled


@dataclass
class SampleConfig:
    teff: float
    logg: float
    mh: float
    vturb: float
    case_dir: Path
    input_atm: Path
    fortran_atm: Path
    fortran_log: Path
    fortran_fort12: Path


@dataclass
class SampleResult:
    teff: float
    logg: float
    mh: float
    vturb: float
    case_dir: str
    input_atm: str
    fortran_atm: str
    fortran_log: str
    fortran_fort12: str
    success: bool
    error: str | None


def _repo_root() -> Path:
    # atlas_py/tools/generate_fortran_atm_grid.py -> repo root
    return Path(__file__).resolve().parents[2]


def _build_sample_name(idx: int, teff: float, logg: float, mh: float) -> str:
    teff_i = int(round(teff))
    return f"case_{idx:03d}_t{teff_i:05d}g{logg:+4.2f}_mh{mh:+4.2f}"


def _generate_input_atm(sample: SampleConfig) -> None:
    """Use the kurucz-a1 emulator + pyKurucz writer to create input .atm."""
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "PyTorch is required for the ATLAS12 emulator used to build input .atm "
            "models. Install with `pip install torch`."
        ) from exc

    from emulator import load_emulator
    from pykurucz import derive_emulator_params, write_atm_file

    # For now we keep alpha-enhancement fixed at 0.0; only [M/H] varies.
    eff_mh, eff_am = derive_emulator_params(sample.mh, 0.0, individual=None)

    emulator = load_emulator()
    data_9col = emulator.predict_atmosphere_data(
        sample.teff, sample.logg, eff_mh, eff_am, sample.vturb
    )

    write_atm_file(
        sample.input_atm,
        sample.teff,
        sample.logg,
        data_9col,
        sample.vturb,
        mh=sample.mh,
        am=0.0,
        individual=None,
    )


def _run_fortran_single_iteration(
    sample: SampleConfig,
    atlas12_exe: Path,
    molecules_new: Path,
    gfpred_bin: Path,
    lowobs_bin: Path,
    hilines_bin: Path,
    diatomics_bin: Path,
    tio_bin: Path,
    h2o_bin: Path,
    nltelinobsat12_bin: Path,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "atlas_py.tools.run_single_iteration",
        "--atlas12-exe",
        str(atlas12_exe),
        "--input-atm",
        str(sample.input_atm),
        "--output-atm",
        str(sample.fortran_atm),
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
        str(sample.fortran_log),
        "--output-lines-bin",
        str(sample.fortran_fort12),
    ]

    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "run_single_iteration failed with non-zero exit code "
            f"{proc.returncode} for case {sample.case_dir.name}.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def _sample_parameters(
    n_samples: int,
    seed: int,
    teff_min: float,
    teff_max: float,
    logg_min: float,
    logg_max: float,
    mh_min: float,
    mh_max: float,
    vturb: float,
    out_root: Path,
) -> list[SampleConfig]:
    random.seed(seed)
    samples: list[SampleConfig] = []
    for idx in range(n_samples):
        teff = random.uniform(teff_min, teff_max)
        logg = random.uniform(logg_min, logg_max)
        mh = random.uniform(mh_min, mh_max)
        case_name = _build_sample_name(idx, teff, logg, mh)
        case_dir = out_root / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        samples.append(
            SampleConfig(
                teff=teff,
                logg=logg,
                mh=mh,
                vturb=vturb,
                case_dir=case_dir,
                input_atm=case_dir / "input.atm",
                fortran_atm=case_dir / "fortran_iter1.atm",
                fortran_log=case_dir / "fortran_iter1.log",
                fortran_fort12=case_dir / "fortran_iter1_fort12.bin",
            )
        )
    return samples


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Generate a random grid of Fortran ATLAS12 ground-truth `.atm` files "
            "using the kurucz-a1 emulator for initial structures."
        )
    )
    p.add_argument("--n-samples", type=int, default=50, help="Number of random models")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    p.add_argument("--teff-min", type=float, default=2500.0)
    p.add_argument("--teff-max", type=float, default=50000.0)
    p.add_argument("--logg-min", type=float, default=-1.0)
    p.add_argument("--logg-max", type=float, default=5.5)
    p.add_argument("--mh-min", type=float, default=-4.0)
    p.add_argument("--mh-max", type=float, default=1.5)
    p.add_argument(
        "--vturb",
        type=float,
        default=2.0,
        help="Microturbulent velocity in km/s used for emulator inputs (default: 2.0).",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Root directory for all cases. "
            "Default: tmp_atlas_debug/fortran_atm_grid under repo root."
        ),
    )
    p.add_argument(
        "--kurucz-root",
        type=Path,
        default=None,
        help="Path to sibling kurucz/ directory (defaults to validate_phase1 helper).",
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
    args = p.parse_args()

    repo_root = _repo_root()
    out_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else (repo_root / "tmp_atlas_debug" / "fortran_atm_grid")
    )
    out_root.mkdir(parents=True, exist_ok=True)

    kurucz_root = (
        args.kurucz_root.resolve()
        if args.kurucz_root is not None
        else _default_kurucz_root().resolve()
    )
    bin_dir = kurucz_root / "bin_macos"
    lines_dir = kurucz_root / "lines"
    mol_dir = kurucz_root / "molecules"

    atlas12_exe = (args.atlas12_exe or (bin_dir / "atlas12.exe")).resolve()
    molecules_new = (args.molecules_new or (lines_dir / "molecules.new")).resolve()
    gfpred_bin = (args.gfpred_bin or (lines_dir / "gfpred29dec2014.bin")).resolve()
    lowobs_bin = (args.lowobs_bin or (lines_dir / "lowobsat12.bin")).resolve()
    hilines_bin = (args.hilines_bin or (lines_dir / "hilines.bin")).resolve()
    diatomics_bin = (
        args.diatomics_bin or (lines_dir / "diatomicspacksrt.bin")
    ).resolve()
    tio_bin = (args.tio_bin or (mol_dir / "tio" / "schwenke.bin")).resolve()
    h2o_bin = (args.h2o_bin or (mol_dir / "h2o" / "h2ofastfix.bin")).resolve()
    nltelinobsat12_bin = (
        args.nltelinobsat12_bin or (lines_dir / "nltelinobsat12.bin")
    ).resolve()

    _ensure_gfpred_assembled(gfpred_bin)

    samples = _sample_parameters(
        n_samples=max(0, args.n_samples),
        seed=args.seed,
        teff_min=args.teff_min,
        teff_max=args.teff_max,
        logg_min=args.logg_min,
        logg_max=args.logg_max,
        mh_min=args.mh_min,
        mh_max=args.mh_max,
        vturb=args.vturb,
        out_root=out_root,
    )

    results: list[SampleResult] = []

    for sample in samples:
        error: str | None = None
        success = False
        try:
            _generate_input_atm(sample)
            _run_fortran_single_iteration(
                sample=sample,
                atlas12_exe=atlas12_exe,
                molecules_new=molecules_new,
                gfpred_bin=gfpred_bin,
                lowobs_bin=lowobs_bin,
                hilines_bin=hilines_bin,
                diatomics_bin=diatomics_bin,
                tio_bin=tio_bin,
                h2o_bin=h2o_bin,
                nltelinobsat12_bin=nltelinobsat12_bin,
            )
            success = True
        except Exception as exc:  # pragma: no cover - diagnostics only
            error = str(exc)

        results.append(
            SampleResult(
                teff=sample.teff,
                logg=sample.logg,
                mh=sample.mh,
                vturb=sample.vturb,
                case_dir=str(sample.case_dir),
                input_atm=str(sample.input_atm),
                fortran_atm=str(sample.fortran_atm),
                fortran_log=str(sample.fortran_log),
                fortran_fort12=str(sample.fortran_fort12),
                success=success,
                error=error,
            )
        )

    manifest: dict[str, Any] = {
        "config": {
            "n_samples": len(samples),
            "seed": args.seed,
            "teff_range": [args.teff_min, args.teff_max],
            "logg_range": [args.logg_min, args.logg_max],
            "mh_range": [args.mh_min, args.mh_max],
            "vturb": args.vturb,
            "kurucz_root": str(kurucz_root),
            "atlas12_exe": str(atlas12_exe),
        },
        "samples": [asdict(r) for r in results],
    }

    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    n_ok = sum(1 for r in results if r.success)
    n_fail = len(results) - n_ok
    print(f"[ok] Fortran ground-truth .atm generation root: {out_root}")
    print(f"[ok] Manifest: {manifest_path}")
    print(f"[ok] Successful cases: {n_ok}, failures: {n_fail}")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())