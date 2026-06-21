#!/usr/bin/env python
"""End-to-end Python spectrum synthesis: stellar parameters → synthetic spectrum.

Pipeline (the only supported flow):

    Teff / logg / [M/H] / [α/M]
        └─► kurucz-a1 emulator  ──► warm-start .atm
                                 └─► atlas_py.cli (1 iteration, MOLECULES ON)
                                     └─► iterated .atm
                                         └─► synthe_py.cli
                                             └─► .spec

The emulator plays the same role as READ DECK6 in the Fortran pipeline: it
supplies the starting layer structure so that atlas_py converges quickly
rather than starting from a grey approximation.  atlas_py is always run to
self-consistently iterate the atmospheric structure with the same physics as
Fortran ATLAS12 (MOLECULES ON) — skipping it would silently diverge from
Fortran parity and is therefore not offered as an option.

This module also exposes the building blocks for the same flow as
public helpers so other scripts can reuse them without duplicating subprocess plumbing:

    emulator_warmstart_atm(...)  -> Path    # stellar params → warm-start .atm
    run_atlas_py(...)            -> Path    # warm-start .atm → iterated .atm
    run_synthe_py(...)           -> Path    # iterated .atm → .spec

Requires the self-contained ``data/`` tree (run ``scripts/download_data.py`` once).

Usage:
    python pykurucz.py --teff 5770 --logg 4.44
    python pykurucz.py --teff 4500 --logg 2.0 --mh -1.5 --am 0.3
    python pykurucz.py --teff 5770 --logg 4.44 --abund C:+0.5 --abund Fe:-1.0
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np


# ── Solar abundances and element data (from Kurucz / kurucz.py) ─────────

SOLAR_ABUND = np.array([
    -10.99, -10.66, -9.34, -3.61, -4.21, -3.35,   # 3-8
    -7.48, -4.11, -5.80, -4.44, -5.59, -4.53,     # 9-14
    -6.63, -4.92, -6.54, -5.64, -7.01, -5.70,     # 15-20
    -8.89, -7.09, -8.11, -6.40, -6.61, -4.54,     # 21-26
    -7.05, -5.82, -7.85, -7.48, -9.00, -8.39,     # 27-32
    -9.74, -8.70, -9.50, -8.79, -9.52, -9.17,     # 33-38
    -9.83, -9.46, -10.58, -10.16, -20.00, -10.29,  # 39-44
    -11.13, -10.47, -11.10, -10.33, -11.24, -10.00, # 45-50
    -11.03, -9.86, -10.49, -9.80, -10.96, -9.86,   # 51-56
    -10.94, -10.46, -11.32, -10.62, -20.00, -11.08, # 57-62
    -11.52, -10.97, -11.74, -10.94, -11.56, -11.12, # 63-68
    -11.94, -11.20, -11.94, -11.19, -12.16, -11.19, # 69-74
    -11.78, -10.64, -10.66, -10.42, -11.12, -10.87, # 75-80
    -11.14, -10.29, -11.39, -20.00, -20.00, -20.00, # 81-86
    -20.00, -20.00, -20.00, -12.02, -20.00, -12.58, # 87-92
    -20.00, -20.00, -20.00, -20.00, -20.00, -20.00, -20.00,  # 93-99
], dtype=np.float64)

ALPHA_ELEMENTS = [8, 10, 12, 14, 16, 20, 22]
ALPHA_IDX = np.array([e - 3 for e in ALPHA_ELEMENTS])
HE_ABUNDANCE = 0.078370

ELEM_SYM = {
    1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O',
    9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
    16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti',
    23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu',
    30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr',
    37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc',
    44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
    51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La',
    58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd',
    65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu',
    72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt',
    79: 'Au', 80: 'Hg', 81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At',
    86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U',
    93: 'NP', 94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es',
}

SYM_TO_Z = {sym.lower(): z for z, sym in ELEM_SYM.items()}


# ── Abundance helpers ────────────────────────────────────────────────────

def compute_abundances(mh: float = 0.0, am: float = 0.0,
                       individual: Optional[Dict[int, float]] = None) -> np.ndarray:
    """Compute log abundances for elements 3-99.

    Starts from solar, scales all metals by [M/H], applies additional
    [alpha/M] to alpha elements. Individual offsets (when provided) are
    relative to solar and override the [M/H]/[alpha/M] scaling entirely
    for that element — e.g. {26: -1.0} sets [Fe/H] = -1.0 regardless
    of the --mh value.
    """
    abund = SOLAR_ABUND.copy() + mh
    abund[ALPHA_IDX] += am
    if individual:
        for elem, offset in individual.items():
            if 3 <= elem <= 99:
                abund[elem - 3] = SOLAR_ABUND[elem - 3] + offset
    return abund


def compute_h(mh: float = 0.0, am: float = 0.0,
              individual: Optional[Dict[int, float]] = None) -> float:
    """Compute H abundance: H = 1 - He - sum(10^A_i) for metals."""
    abund = compute_abundances(mh, am, individual)
    z_sum = np.sum(10**abund)
    return 1.0 - HE_ABUNDANCE - z_sum


def derive_emulator_params(
    mh: float,
    am: float,
    individual: Optional[Dict[int, float]],
) -> tuple[float, float]:
    """Derive effective [M/H] and [alpha/M] for the emulator.

    When individual element offsets are provided, the emulator needs
    scalar [M/H] and [alpha/M] to predict the atmospheric structure.
    We derive these as:
      - [M/H] ~ [Fe/H] = offset provided for Fe (if given), else mh
      - [alpha/M] = mean of (alpha_offset_i - eff_mh) for alpha elements
        (O, Ne, Mg, Si, S, Ca, Ti)

    If an element is not individually specified, its abundance follows
    the --mh/--am scaling.
    """
    if not individual:
        return mh, am

    if 26 in individual:
        eff_mh = individual[26]
    else:
        eff_mh = mh

    alpha_offsets = []
    for elem in ALPHA_ELEMENTS:
        if elem in individual:
            alpha_offsets.append(individual[elem] - eff_mh)
        else:
            alpha_offsets.append(mh + am - eff_mh)

    eff_am = float(np.mean(alpha_offsets)) if alpha_offsets else am

    return eff_mh, eff_am


def parse_abund_arg(s: str) -> tuple[int, float]:
    """Parse 'Z:offset' or 'Symbol:offset' into (atomic_number, dex_offset).

    The offset is relative to Asplund solar abundances.
    Examples: 'Fe:+0.3' means [Fe/H]=+0.3, 'C:-0.5' means [C/H]=-0.5
    """
    parts = s.split(':')
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Invalid abundance format '{s}'. Use Z:offset or Symbol:offset "
            f"(e.g. Fe:+0.3 for [Fe/H]=+0.3)")
    key, val_str = parts[0].strip(), parts[1].strip()
    try:
        value = float(val_str)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid abundance value '{val_str}' in '{s}'")

    if key.isdigit():
        z = int(key)
    else:
        z = SYM_TO_Z.get(key.lower())
        if z is None:
            raise argparse.ArgumentTypeError(f"Unknown element symbol '{key}' in '{s}'")

    if not (3 <= z <= 99):
        raise argparse.ArgumentTypeError(f"Element Z={z} out of range (3-99)")
    return z, value


# ── .atm file writer ────────────────────────────────────────────────────

def write_atm_file(path: Path, teff: float, logg: float,
                   data: np.ndarray, vturb: float,
                   mh: float = 0.0, am: float = 0.0,
                   individual: Optional[Dict[int, float]] = None,
                   title: Optional[str] = None) -> None:
    """Write a Kurucz-format .atm file from emulator-predicted atmospheric structure."""
    abund = compute_abundances(mh, am, individual)
    h = compute_h(mh, am, individual)

    lines = []
    lines.append(f'TEFF   {teff:.0f}.  GRAVITY {logg:7.4f} LTE ')
    title_text = title if title is not None else "kurucz-a1 emulator (Li et al. 2025) + pyKurucz"
    lines.append(f'TITLE {title_text:<80}')
    lines.append(' OPACITY IFOP 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 0 0 0')
    lines.append(' CONVECTION ON   1.25 TURBULENCE OFF  0.00  0.00  0.00  0.00')
    lines.append(f'ABUNDANCE SCALE   1.00000 ABUNDANCE CHANGE 1 {h:.5f} 2 {HE_ABUNDANCE:.5f}')

    elem = 3
    while elem <= 99:
        end = min(elem + 5, 99)
        parts = [' ABUNDANCE CHANGE']
        for e in range(elem, end + 1):
            val = abund[e - 3]
            if val > -10:
                parts.append(f' {e:2d}  {val:5.2f}')
            else:
                parts.append(f' {e:2d} {val:6.2f}')
        lines.append(''.join(parts))
        elem = end + 1

    lines.append(' ABUNDANCE TABLE')
    lines.append(f'    1H   {h:.6f}       2He  {HE_ABUNDANCE:.6f}')

    for start in range(3, 100, 5):
        end = min(start + 4, 99)
        parts = []
        for e in range(start, end + 1):
            sym = ELEM_SYM[e]
            val = abund[e - 3]
            if val > -10:
                padding = ' ' * (3 - len(sym))
                parts.append(f'{e:5d}{sym}{padding}{val:6.3f} 0.000')
            else:
                padding = ' ' * (3 - len(sym) - 1)
                parts.append(f'{e:5d}{sym}{padding}{val:7.3f} 0.000')
        lines.append(''.join(parts))

    n_layers = data.shape[0]
    lines.append(f'READ DECK6 {n_layers} RHOX,T,P,XNE,ABROSS,ACCRAD,VTURB')

    for row in data:
        parts = [f' {row[0]:.8E}', f'   {row[1]:.1f}']
        for j in range(2, 6):
            parts.append(f' {row[j]:.3E}')
        # Fortran ATLAS12 writes DECK6 trailing columns as VTURB, FLXCNV, VCONV.
        parts.append(f' {row[6]:.3E}')
        parts.append(f' {row[7]:.3E} {row[8]:.3E}')
        lines.append(''.join(parts))

    lines.append(f'PRADK {0.5:.4E}')
    lines.append('BEGIN                    ITERATION  15 COMPLETED')

    with open(path, 'w') as f:
        f.write('\n'.join(lines))


# ── Shared pipeline helpers ─────────────────────────────────────────────
#
# The three public helpers below are the canonical Python-pipeline building
# blocks.  ``synthesize()`` chains them for the user-facing CLI.

_REPO_ROOT = Path(__file__).resolve().parent


def _default_kurucz_root() -> Path:
    """Self-contained data tree populated by ``scripts/download_data.py``."""
    return _REPO_ROOT / "data"


def _default_atlas_cache_dir() -> Path:
    """Disk cache for Python SELECTLINES (``fort.12``); shared with parity harness."""
    return _REPO_ROOT / "results" / "atlas_fort12_cache"


# Prod optimization defaults.  Applied via setdefault in child processes so an
# setdefault in child processes so an explicit shell override still wins.
_PROD_ENV_DEFAULTS: dict[str, str] = {
    "ATLAS_LINOP1_CACHE_LINE_SCALARS": "1",
    "ATLAS_LINOP1_PREFILTER": "1",
    "ATLAS_LINOP1_TIGHT_PREFILTER": "1",
    "ATLAS_LINOP1_SERIAL": "0",
    "PY_USE_NUMBA_TRANSP": "1",
    "PY_NUMBA_HELIUM": "1",
    "PY_OPT_STEP1_WING_HOIST": "1",
    "PY_HYD_JIT": "1",
}


def _apply_prod_env_defaults(env: dict[str, str]) -> None:
    for key, value in _PROD_ENV_DEFAULTS.items():
        env.setdefault(key, value)


def _run_streaming(
    cmd: list[str],
    *,
    cwd: Optional[Path] = None,
    log_handle=None,
    env_overrides: Optional[dict[str, str]] = None,
) -> None:
    """Stream merged stdout/stderr from *cmd* into *log_handle* line-by-line.

    Runs child Python with ``PYTHONUNBUFFERED=1`` so INFO / ``Timing:`` logs
    appear live instead of being buffered until process exit.
    """
    env = {**os.environ}
    env.setdefault("PYTHONUNBUFFERED", "1")
    _apply_prod_env_defaults(env)
    if env_overrides:
        env.update(env_overrides)
    proc = subprocess.Popen(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        bufsize=1,
    )
    chunks: list[str] = []
    if proc.stdout is not None:
        for line in proc.stdout:
            chunks.append(line)
            if log_handle is not None:
                log_handle.write(line)
                log_handle.flush()
    proc.wait()
    if proc.returncode != 0:
        output = "".join(chunks)
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{output}"
        )


def emulator_warmstart_atm(
    dest: Path,
    *,
    teff: float,
    logg: float,
    mh: float = 0.0,
    am: float = 0.0,
    vturb: float = 2.0,
    abundances: Optional[Dict[int, float]] = None,
) -> Path:
    """Predict a warm-start atmosphere with kurucz-a1 and write it to *dest*.

    The emulator is queried with effective ``[M/H]`` and ``[α/M]`` derived
    from *abundances* (when provided) via :func:`derive_emulator_params`.
    The resulting 9-column layer structure is written as a Kurucz-format
    ``.atm`` file that atlas_py (and the Fortran pipeline) can consume as a
    READ DECK6 starting point.
    """
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for the kurucz-a1 emulator. "
            "Install with: pip install torch"
        ) from exc

    from emulator import load_emulator

    eff_mh, eff_am = derive_emulator_params(mh, am, abundances)
    emulator = load_emulator()
    data_9col = emulator.predict_atmosphere_data(teff, logg, eff_mh, eff_am, vturb)
    dest.parent.mkdir(parents=True, exist_ok=True)

    t_col = data_9col[:, 1]
    rhox_col = data_9col[:, 0]
    bad_t = np.any(t_col <= 0.0) or np.any(~np.isfinite(t_col))
    bad_rhox = np.any(rhox_col <= 0.0) or np.any(~np.isfinite(rhox_col))
    if bad_t or bad_rhox:
        from atlas_py.physics.grey_start import write_grey_atm_file

        write_grey_atm_file(
            dest,
            teff=teff,
            logg=logg,
            mh=mh,
            am=am,
            abundances=abundances,
            vturb_kms=vturb,
            title=(
                f"grey fallback (emulator had non-positive T/RHOX) "
                f"Teff={teff:.0f} logg={logg:.2f}"
            ),
        )
        return dest

    write_atm_file(
        dest, teff, logg, data_9col, vturb,
        mh=mh, am=am, individual=abundances,
    )
    return dest


def run_atlas_py(
    input_atm: Path,
    output_atm: Path,
    *,
    log_path: Path,
    kurucz_root: Optional[Path] = None,
    iterations: int = 1,
    fort12_bin: Optional[Path] = None,
    convergence_epsilon: Optional[float] = None,
    convergence_min_iterations: int = 5,
    convergence_consecutive: int = 1,
    convergence_dlntmax: float = 5.0e-4,
    n_workers: Optional[int] = None,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Run ``atlas_py.cli`` on *input_atm* and write iterated output to *output_atm*.

    ``--enable-molecules`` is always passed to match the Fortran deck, which
    always includes ``MOLECULES ON``.  The molecular opacity path would
    silently diverge otherwise (validated: this was the root cause of the
    90% Na D discrepancy for cool stars before the flag was added).

    Parameters
    ----------
    input_atm:
        Starting atmosphere.  For the standard flow this is the emulator
        warm-start written by :func:`emulator_warmstart_atm`.
    output_atm:
        Destination ``.atm`` file for the iterated atmosphere.
    log_path:
        File receiving combined stdout+stderr from ``atlas_py.cli``.
    kurucz_root:
        Data tree containing ``lines/`` and ``molecules/`` binaries.
        Defaults to ``data/`` inside this repo.
    iterations:
        Number of outer atlas_py iterations (default 1; matches the
        validated single-iteration Fortran parity pipeline).
    fort12_bin:
        Optional Fortran ``fort12`` line-selection binary to replay for
        exact parity with a previous Fortran ATLAS run.  Used only by the
        validation harness — unused in the user-facing flow.
    convergence_epsilon:
        Early-stop enable switch (any positive value enables it). The actual
        stop threshold is ``convergence_dlntmax``.  ``iterations`` is the maximum.
    convergence_dlntmax:
        Deep-layer temperature stop threshold (Fortran checkconv.f90 dlntmax),
        forwarded as ``--checkconv-dlntmax``.  Validated default 5e-4; pass 1e-4
        for Fortran-faithful behavior.
    """
    root = (kurucz_root or _default_kurucz_root()).resolve()
    lines_dir = root / "lines"
    mol_dir = root / "molecules"

    gfpred_bin = lines_dir / "gfpred29dec2014.bin"
    lowobs_bin = lines_dir / "lowobsat12.bin"
    hilines_bin = lines_dir / "hilines.bin"
    diatomics_bin = lines_dir / "diatomicspacksrt.bin"
    tio_bin = mol_dir / "tio" / "schwenke.bin"
    h2o_bin = mol_dir / "h2o" / "h2ofastfix.bin"
    nltelinobsat12_bin = lines_dir / "nltelinobsat12.bin"
    molecules_new = lines_dir / "molecules.new"

    for p in (gfpred_bin, lowobs_bin, hilines_bin, molecules_new):
        if not p.exists():
            raise FileNotFoundError(
                f"Required atlas_py binary not found: {p}\n"
                "Run `python scripts/download_data.py` to populate data/."
            )

    output_atm.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "atlas_py.cli",
        str(input_atm.resolve()),
        "--output-atm", str(output_atm),
        "--iterations", str(iterations),
        "--fort11", str(gfpred_bin),
        "--fort111", str(lowobs_bin),
        "--fort21", str(hilines_bin),
        "--fort31", str(diatomics_bin),
        "--fort41", str(tio_bin),
        "--fort51", str(h2o_bin),
        "--nlteline-bin", str(nltelinobsat12_bin),
        "--enable-molecules",
        "--molecules", str(molecules_new),
    ]
    if convergence_epsilon is not None:
        cmd.extend(
            [
                "--convergence-epsilon", str(convergence_epsilon),
                "--convergence-min-iterations", str(convergence_min_iterations),
                "--convergence-consecutive", str(convergence_consecutive),
                "--checkconv-dlntmax", str(convergence_dlntmax),
            ]
        )
    if fort12_bin is not None and fort12_bin.exists():
        cmd.extend(["--line-selection-bin", str(fort12_bin)])
    if cache_dir is not None:
        cmd.extend(["--cache-dir", str(cache_dir)])
    if n_workers is not None:
        cmd.extend(["--n-workers", str(n_workers)])

    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(
            "=" * 60 + "\n"
            f"  atlas_py.cli  input={input_atm.name}  iter={iterations}\n"
            "=" * 60 + "\n\n"
        )
        logf.flush()
        t0 = time.perf_counter()
        logf.write("==== STEP: atlas_py.cli START ====\n"); logf.flush()
        try:
            _run_streaming(cmd, cwd=_REPO_ROOT, log_handle=logf)
        except RuntimeError as exc:
            raise RuntimeError(
                f"atlas_py.cli failed. See log: {log_path}\n{exc}"
            ) from exc
        logf.write(
            f"==== STEP: atlas_py.cli END wall={time.perf_counter() - t0:.3f}s ====\n"
        )
        logf.flush()
    return output_atm


def run_synthe_py(
    atm: Path,
    *,
    spec: Path,
    npz: Path,
    log_path: Path,
    wl_start: float = 300.0,
    wl_end: float = 1800.0,
    resolution: float = 300_000.0,
    n_workers: Optional[int] = None,
    kurucz_root: Optional[Path] = None,
    use_molecular_lines: bool = True,
    include_tio: bool = True,
    include_h2o: bool = True,
    npz_cache: bool = True,
    npz_cache_dir: Optional[Path] = None,
) -> Path:
    """Convert *atm* to NPZ and run ``synthe_py.cli`` to produce *spec*.

    Two stages are executed and logged into *log_path* with timing markers:
      1. ``synthe_py/tools/convert_atm_to_npz.py`` — populations, molecular
         equilibrium, continuous opacity → ``.npz``.
      2. ``synthe_py.cli`` — line opacity + radiative transfer → ``.spec``.

    Parameters mirror the user-visible knobs of the CLI (wavelength window,
    resolution, molecular-line toggles).  ``kurucz_root`` defaults to the
    self-contained ``data/`` tree; its ``molecules/`` subdirectory is passed
    as ``--molecules-dir`` so that Schwenke TiO and Partridge–Schwenke H₂O
    line lists are picked up consistently with the Fortran pipeline.
    """
    root = (kurucz_root or _default_kurucz_root()).resolve()
    line_list = _REPO_ROOT / "lines" / "gfallvac.latest"
    atlas_tables = _REPO_ROOT / "synthe_py" / "data" / "atlas_tables.npz"
    if not line_list.exists():
        raise FileNotFoundError(f"Missing Python line list: {line_list}")
    if not atlas_tables.exists():
        raise FileNotFoundError(f"Missing atlas tables: {atlas_tables}")

    spec.parent.mkdir(parents=True, exist_ok=True)
    npz.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cpu = os.cpu_count() or 1
    thread_budget = n_workers if n_workers is not None else cpu

    from atlas_py.engine.threading_policy import ThreadingPolicy

    synthe_policy = ThreadingPolicy.for_synthe_grid(
        thread_budget,
        wl_start=wl_start,
        wl_end=wl_end,
        resolution=resolution,
    )
    synthe_env = synthe_policy.env_updates()

    from synthe_py.tools.npz_cache import (
        copy_cached_npz,
        default_npz_cache_dir,
        npz_cache_key,
        resolve_cached_npz,
        store_npz_cache,
    )

    converter_script = _REPO_ROOT / "synthe_py" / "tools" / "convert_atm_to_npz.py"
    molecules_dat = _REPO_ROOT / "lines" / "molecules.dat"
    cache_root = npz_cache_dir or default_npz_cache_dir(_REPO_ROOT)
    cache_key = npz_cache_key(
        atm_path=atm.resolve(),
        atlas_tables=atlas_tables,
        molecules=molecules_dat,
        converter_script=converter_script,
    )
    cached_npz = resolve_cached_npz(cache_root, cache_key)

    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(
            "======================================================================\n"
            "pykurucz.run_synthe_py — streaming log (line-buffered)\n"
            f"  atm:  {atm}\n"
            f"  npz:  {npz}\n"
            f"  spec: {spec}\n"
            f"  wl:   {wl_start:.1f}–{wl_end:.1f} nm  R={resolution:.0f}  "
            f"thread_budget={thread_budget} numba={synthe_policy.numba_threads} "
            f"rt_pool={synthe_policy.rt_pool}\n"
            f"  npz_cache={'on' if npz_cache else 'off'} key={cache_key[:16]}...\n"
            "======================================================================\n\n"
        )
        logf.flush()

        t0 = time.perf_counter()
        if npz_cache and cached_npz.is_file():
            logf.write("==== STEP: convert_atm_to_npz CACHE HIT ====\n")
            logf.flush()
            copy_cached_npz(cached_npz, npz)
            logf.write(
                f"==== STEP: convert_atm_to_npz END wall={time.perf_counter() - t0:.3f}s (cached) ====\n\n"
            )
            logf.flush()
        else:
            logf.write("==== STEP: convert_atm_to_npz START ====\n"); logf.flush()
            _run_streaming(
                [
                    sys.executable, "-u",
                    str(converter_script),
                    str(atm.resolve()),
                    str(npz.resolve()),
                    "--atlas-tables", str(atlas_tables.resolve()),
                ],
                cwd=_REPO_ROOT,
                log_handle=logf,
            )
            logf.write(
                f"==== STEP: convert_atm_to_npz END wall={time.perf_counter() - t0:.3f}s ====\n\n"
            )
            logf.flush()
            if npz_cache:
                store_npz_cache(cache_root, cache_key, npz)

        cmd = [
            sys.executable, "-u", "-m", "synthe_py.cli",
            str(atm.resolve()),
            str(line_list.resolve()),
            "--npz", str(npz.resolve()),
            "--spec", str(spec.resolve()),
            "--wl-start", str(wl_start),
            "--wl-end", str(wl_end),
            "--resolution", str(resolution),
            "--n-workers", str(thread_budget),
            "--log-level", "INFO",
        ]
        mol_dir = root / "molecules"
        if not use_molecular_lines:
            cmd.append("--no-molecular-lines")
        elif mol_dir.is_dir():
            cmd.extend(["--molecules-dir", str(mol_dir)])
        if use_molecular_lines:
            if not include_tio:
                cmd.append("--no-tio")
            if not include_h2o:
                cmd.append("--no-h2o")

        t1 = time.perf_counter()
        logf.write("==== STEP: synthe_py.cli START ====\n"); logf.flush()
        _run_streaming(cmd, cwd=_REPO_ROOT, log_handle=logf, env_overrides=synthe_env)
        logf.write(
            f"==== STEP: synthe_py.cli END wall={time.perf_counter() - t1:.3f}s ====\n"
        )
        logf.flush()

    return spec


# ── End-to-end orchestrator (user-facing) ───────────────────────────────

def synthesize(
    teff: float,
    logg: float,
    mh: float = 0.0,
    am: float = 0.0,
    vturb: float = 2.0,
    wl_start: float = 300.0,
    wl_end: float = 1800.0,
    resolution: float = 300_000.0,
    abundances: Optional[Dict[int, float]] = None,
    output_dir: Optional[str] = None,
    use_molecular_lines: bool = True,
    include_tio: bool = True,
    include_h2o: bool = True,
    atlas_iterations: int = 30,
    atlas_convergence_epsilon: Optional[float] = 1.0e-3,
    atlas_convergence_min_iterations: int = 5,
    atlas_convergence_consecutive: int = 1,
    atlas_checkconv_dlntmax: float = 5.0e-4,
    n_workers: Optional[int] = None,
    atlas_fort12_cache: bool = True,
    atlas_cache_dir: Optional[Path] = None,
    warmstart_atm_override: Optional[str] = None,
    npz_cache: bool = True,
    npz_cache_dir: Optional[Path] = None,
    linop1_serial: Optional[bool] = None,
    convec_fd_parallel: Optional[bool] = None,
    pops_parallel: Optional[bool] = None,
) -> Path:
    """Generate a synthetic spectrum from stellar parameters.

    Runs the canonical Python pipeline:

        emulator → warm-start .atm → atlas_py (MOLECULES ON) →
        iterated .atm → convert_atm_to_npz → synthe_py.cli → .spec

    The emulator provides the starting layer structure (same role as
    READ DECK6 in the Fortran pipeline); atlas_py then self-consistently
    iterates the atmospheric structure with the same physics as Fortran
    ATLAS12 so that the downstream SYNTHE spectrum stays in parity with
    Fortran-generated references.

    Parameters
    ----------
    teff : float
        Effective temperature in Kelvin.
    logg : float
        Surface gravity (log10 cgs).
    mh : float
        Overall metallicity [M/H] (default 0.0, solar).
    am : float
        Alpha enhancement [alpha/M] (default 0.0).
    vturb : float
        Microturbulent velocity in km/s (default 2.0).
    wl_start, wl_end : float
        Wavelength range in nanometres (default 300–1800 nm).
    resolution : float
        Resolving power lambda/delta_lambda (default 300 000).
    abundances : dict, optional
        Individual element offsets {Z: dex_offset_from_solar}.
    output_dir : str, optional
        Directory for output files. Defaults to ``results/``.
    use_molecular_lines : bool
        Pass molecular catalogs to synthe_py.cli (default True).
    include_tio, include_h2o : bool
        Include Schwenke TiO / Partridge–Schwenke H₂O (default True).
    atlas_iterations : int
        Maximum number of atlas_py outer iterations (default 30).
    atlas_convergence_epsilon : float, optional
        Early-stop enable switch (any positive value enables it).  Defaults to
        1e-3; set to None to force all ``atlas_iterations``.  The actual stop
        threshold is ``atlas_checkconv_dlntmax``.
    atlas_checkconv_dlntmax : float
        Deep-layer temperature stop threshold (Fortran checkconv.f90 dlntmax).
        Validated production default 5e-4 (~53% fewer iterations than Fortran's
        1e-4 with worst-case spectrum error max|F/C|=0.0062; see
        results/convergence_criteria/REPORT.md).  Pass 1e-4 for Fortran parity.
    n_workers : int, optional
        Total CPU thread budget for the full pipeline.  Sets Numba
        (LINOP1, metal wings, RT prange), ATLAS frequency-loop pool,
        SYNTHE RT pool, and CONVEC FD pool (``min(4, N)``).  BLAS is
        pinned to 1 thread to avoid oversubscription.  Default: all
        logical CPUs.
    atlas_fort12_cache : bool
        When True (default), reuse a disk cache of Python SELECTLINES output
        under ``results/atlas_fort12_cache/``.
    atlas_cache_dir : Path, optional
        Override directory for the fort.12 cache.  Ignored when
        ``atlas_fort12_cache`` is False.

    Returns
    -------
    Path
        Path to the output ``.spec`` file.
    """
    from atlas_py.engine.threading_policy import (
        ThreadingPolicy,
        estimate_wavelength_grid_size,
        log_threading_banner,
    )

    base_policy = ThreadingPolicy.from_n_workers(
        n_workers,
        linop1_serial=linop1_serial,
        convec_fd_parallel=convec_fd_parallel,
        pops_parallel=pops_parallel,
    )
    n_wl = estimate_wavelength_grid_size(wl_start, wl_end, resolution)
    policy = base_policy.adapt_for_grid(n_wl)
    policy.apply()
    log_threading_banner(policy, stage="synthesize")

    if output_dir is None:
        output_dir = str(_REPO_ROOT / "results")
    output_path = Path(output_dir)

    npz_dir = output_path / "npz"
    spec_dir = output_path / "spec"
    log_dir = output_path / "logs"
    atm_dir = output_path / "atm"
    for d in (npz_dir, spec_dir, log_dir, atm_dir):
        d.mkdir(parents=True, exist_ok=True)

    eff_mh, eff_am = derive_emulator_params(mh, am, abundances)

    warnings = []
    if not (2500 <= teff <= 50000):
        warnings.append(f"Teff={teff:.0f} K outside training range [2500, 50000]")
    if not (-1.0 <= logg <= 5.5):
        warnings.append(f"logg={logg:.2f} outside training range [-1.0, 5.5]")
    if not (-4.0 <= eff_mh <= 1.5):
        warnings.append(f"[M/H]={eff_mh:+.2f} outside training range [-4.0, +1.5]")
    if not (-0.2 <= eff_am <= 0.62):
        warnings.append(f"[alpha/M]={eff_am:+.2f} outside training range [-0.2, +0.6]")
    if warnings:
        print("  WARNING: Parameters outside kurucz-a1 emulator training range:")
        for w in warnings:
            print(f"           {w}")
        print("           Results may be unreliable. Consider using your own .atm file")
        print("           with synthesize_from_atm.py instead.")

    stem = f"t{int(teff):05d}g{logg:.2f}_mh{eff_mh:+.2f}_am{eff_am:+.2f}"
    warmstart_atm = atm_dir / f"{stem}_warmstart.atm"
    atm_path = atm_dir / f"{stem}.atm"
    npz_path = npz_dir / f"{stem}.npz"
    spec_path = spec_dir / f"{stem}_{int(wl_start)}_{int(wl_end)}.spec"
    atlas_log = log_dir / f"{stem}_atlas.log"
    synthe_log = log_dir / f"{stem}_synthe_{int(wl_start)}_{int(wl_end)}.log"

    # ── Stage 1: Emulator warm-start ────────────────────────────────────
    if warmstart_atm_override is not None:
        # Take the donor's READ DECK6 layer block (T, P, RHOX, XNE, ABROSS,
        # ACCRAD, VTURB, FLXCNV, VCONV) and rewrite the header so TEFF /
        # GRAVITY and the ABUNDANCE block reflect the *target* cell —
        # atlas_py reads chemistry from the .atm file only, so a verbatim
        # copy of the donor would give the wrong abundances.
        src_atm = Path(warmstart_atm_override).expanduser().resolve()
        if not src_atm.is_file():
            print(f"ERROR: --warmstart-atm path does not exist: {src_atm}")
            sys.exit(1)
        print(f"[1/3] Using neighbour donor .atm for layer structure, target chemistry: {src_atm}")
        donor_text = src_atm.read_text().splitlines()
        for i_deck, ln in enumerate(donor_text):
            if ln.lstrip().startswith("READ DECK6"):
                break
        else:
            print(f"ERROR: donor .atm has no READ DECK6 line: {src_atm}")
            sys.exit(1)
        layer_lines = []
        for ln in donor_text[i_deck + 1:]:
            stripped = ln.strip()
            if not stripped or stripped.startswith(("PRADK", "BEGIN")):
                break
            layer_lines.append(stripped)
        data = np.array([[float(x) for x in ln.split()] for ln in layer_lines])
        if data.shape[1] < 9:
            pad = np.zeros((data.shape[0], 9 - data.shape[1]))
            data = np.concatenate([data, pad], axis=1)
        write_atm_file(
            warmstart_atm,
            teff=teff, logg=logg, mh=mh, am=am, vturb=vturb,
            individual=abundances,
            data=data,
        )
    else:
        print("[1/3] Predicting warm-start atmosphere with kurucz-a1 emulator...")
        print(f"      Teff={teff:.0f} K, logg={logg:.2f}, vturb={vturb:.1f} km/s")
        if abundances:
            overrides = ", ".join(
                f'[{ELEM_SYM.get(z, f"Z={z}")}/H]={v:+.2f}'
                for z, v in sorted(abundances.items())
            )
            print(f"      Individual abundances: {overrides}")
            if eff_mh != mh or eff_am != am:
                print(f"      Derived emulator params: [M/H]={eff_mh:+.2f}, [alpha/M]={eff_am:+.2f}")
            else:
                print(f"      Emulator params: [M/H]={eff_mh:+.2f}, [alpha/M]={eff_am:+.2f}")
        else:
            print(f"      [M/H]={mh:+.2f}, [alpha/M]={am:+.2f}")

        try:
            emulator_warmstart_atm(
                warmstart_atm,
                teff=teff, logg=logg, mh=mh, am=am, vturb=vturb,
                abundances=abundances,
            )
        except RuntimeError as exc:
            print(f"ERROR in emulator: {exc}")
            sys.exit(1)
    print(f"      Warm-start: {warmstart_atm}")

    cache_dir = None
    if atlas_fort12_cache:
        cache_dir = atlas_cache_dir or _default_atlas_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Stage 2: atlas_py iteration ─────────────────────────────────────
    print(f"[2/3] Running atlas_py ({atlas_iterations} iteration(s), MOLECULES ON)...")
    if cache_dir is not None:
        print(f"      fort.12 cache: {cache_dir}")
    try:
        run_atlas_py(
            input_atm=warmstart_atm,
            output_atm=atm_path,
            log_path=atlas_log,
            iterations=atlas_iterations,
            convergence_epsilon=atlas_convergence_epsilon,
            convergence_min_iterations=atlas_convergence_min_iterations,
            convergence_consecutive=atlas_convergence_consecutive,
            convergence_dlntmax=atlas_checkconv_dlntmax,
            n_workers=policy.atlas_freq_pool,
            cache_dir=cache_dir,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR in atlas_py: {exc}")
        sys.exit(1)
    print(f"      Iterated atmosphere: {atm_path}")
    print(f"      Log: {atlas_log}")

    # ── Stage 3: synthe_py ──────────────────────────────────────────────
    print(
        f"[3/3] Running synthe_py ({wl_start:.0f}–{wl_end:.0f} nm, "
        f"R={resolution:.0f})..."
    )
    try:
        run_synthe_py(
            atm_path,
            spec=spec_path,
            npz=npz_path,
            log_path=synthe_log,
            wl_start=wl_start,
            wl_end=wl_end,
            resolution=resolution,
            n_workers=policy.n_workers,
            use_molecular_lines=use_molecular_lines,
            include_tio=include_tio,
            include_h2o=include_h2o,
            npz_cache=npz_cache,
            npz_cache_dir=npz_cache_dir,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR in synthe_py: {exc}. See log: {synthe_log}")
        sys.exit(1)

    print(f"      NPZ: {npz_path}")
    print(f"      Spectrum: {spec_path}")
    print(f"      Log: {synthe_log}")
    print(f"\nDone! Output: {spec_path}")
    return spec_path


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Python spectrum synthesis: stellar parameters → synthetic spectrum",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline (always):
  kurucz-a1 emulator → warm-start .atm → atlas_py (MOLECULES ON) →
  iterated .atm → convert_atm_to_npz → synthe_py.cli → .spec

Examples:
  # Solar-type star, full wavelength range
  python pykurucz.py --teff 5770 --logg 4.44

  # Metal-poor K giant with alpha enhancement
  python pykurucz.py --teff 4500 --logg 2.0 --mh -1.5 --am 0.3

  # Override individual element abundances relative to solar
  python pykurucz.py --teff 5770 --logg 4.44 --abund Fe:-1.0 --abund C:+0.5
  python pykurucz.py --teff 5770 --logg 4.44 --abund 26:-1.0 --abund 6:+0.5

  # Hot B star, optical only
  python pykurucz.py --teff 15000 --logg 4.0 --wl-start 350 --wl-end 700

  # Low-resolution run
  python pykurucz.py --teff 5770 --logg 4.44 --resolution 50000

  # Fast diagnostic atmosphere pass only (not recommended for science output)
  python pykurucz.py --teff 5770 --logg 4.44 --atlas-iterations 1

Notes:
  --mh scales ALL metals uniformly (like [Fe/H] in a standard grid).
  --am adds an extra offset to alpha elements on top of --mh.
  --abund sets individual element offsets relative to solar (Asplund).
    e.g. --abund Fe:-1.0 means [Fe/H] = -1.0 dex below solar.

  For the emulator's atmospheric structure prediction:
    - [M/H] is proxied by Fe: if --abund Fe is given, [M/H] ~ [Fe/H]_eff
    - [alpha/M] is the mean offset of alpha elements (O,Ne,Mg,Si,S,Ca,Ti)
    - Unspecified elements follow the --mh/--am scaling

  The exact abundances (solar + offsets) are written into the .atm file
  and used by both atlas_py and synthe_py.

  If you already have a .atm file from another source and want to skip both
  the emulator and atlas_py, use synthesize_from_atm.py instead.
""",
    )
    parser.add_argument("--teff", type=float, required=True,
                        help="Effective temperature (K)")
    parser.add_argument("--logg", type=float, required=True,
                        help="Surface gravity (log10 cgs)")
    parser.add_argument("--mh", type=float, default=0.0,
                        help="Overall metallicity [M/H] in dex (default: 0.0). "
                             "Scales all metals uniformly from solar.")
    parser.add_argument("--am", type=float, default=0.0,
                        help="Alpha enhancement [alpha/M] in dex (default: 0.0). "
                             "Extra offset for O, Ne, Mg, Si, S, Ca, Ti.")
    parser.add_argument("--vturb", type=float, default=2.0,
                        help="Microturbulent velocity in km/s (default: 2.0)")
    parser.add_argument("--abund", type=parse_abund_arg, action="append",
                        metavar="ELEM:OFFSET",
                        help="Override element abundance relative to solar (dex). "
                             "e.g. --abund Fe:-1.0 for [Fe/H]=-1.0, "
                             "--abund C:+0.5 for [C/H]=+0.5. Can be repeated.")
    parser.add_argument("--wl-start", type=float, default=300.0,
                        help="Start wavelength in nm (default: 300)")
    parser.add_argument("--wl-end", type=float, default=1800.0,
                        help="End wavelength in nm (default: 1800)")
    parser.add_argument("--resolution", type=float, default=300_000.0,
                        help="Resolving power (default: 300000)")
    parser.add_argument(
        "--atlas-iterations",
        type=int,
        default=30,
        help=(
            "Maximum number of atlas_py atmosphere iterations (default: 30; "
            "early convergence may stop sooner unless disabled)."
        ),
    )
    parser.add_argument(
        "--atlas-convergence-epsilon",
        type=float,
        default=1.0e-3,
        help=(
            "Early-stop threshold for physical atmosphere column changes "
            "(default: 1e-3; use --no-atlas-convergence to force all iterations)."
        ),
    )
    parser.add_argument(
        "--atlas-convergence-min-iterations",
        type=int,
        default=5,
        help="Minimum atlas_py iterations before early stopping can trigger (default: 5).",
    )
    parser.add_argument(
        "--atlas-convergence-consecutive",
        type=int,
        default=1,
        help="Consecutive converged iterations required before early stopping (default: 1).",
    )
    parser.add_argument(
        "--atlas-checkconv-dlntmax",
        type=float,
        default=5.0e-4,
        help=(
            "Deep-layer temperature stop threshold (Fortran checkconv.f90 "
            "dlntmax). Validated production default 5e-4 (~53%% fewer iterations "
            "than Fortran's 1e-4 with negligible spectrum error; see "
            "results/convergence_criteria/REPORT.md). Use --fortran-convergence "
            "for the Fortran-faithful 1e-4."
        ),
    )
    parser.add_argument(
        "--fortran-convergence",
        action="store_true",
        help="Restore the Fortran-faithful checkconv dlntmax=1e-4 threshold "
        "(equivalent to --atlas-checkconv-dlntmax 1e-4).",
    )
    parser.add_argument(
        "--no-atlas-convergence",
        action="store_true",
        help="Disable convergence early stopping and run exactly --atlas-iterations.",
    )
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: results/)")
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help=(
            "Total CPU thread budget for ATLAS + SYNTHE (Numba, freq pool, "
            "RT pool, CONVEC FD). Default: all logical CPUs."
        ),
    )
    parser.add_argument(
        "--linop1-serial",
        action="store_true",
        help="Force serial LINOP1 (overrides auto parallel from --n-workers).",
    )
    parser.add_argument(
        "--convec-fd-parallel",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable parallel CONVEC FD pool (default: on when n-workers>1).",
    )
    parser.add_argument(
        "--pops-parallel",
        action="store_true",
        help="Enable parallel POPS/NELECT (default: off; can hurt e2e).",
    )
    parser.add_argument(
        "--no-npz-cache",
        action="store_true",
        help="Disable convert_atm_to_npz disk cache.",
    )
    parser.add_argument(
        "--atlas-cache-dir",
        type=str,
        default=None,
        help=(
            "Directory for fort.12 SELECTLINES disk cache "
            f"(default: {_default_atlas_cache_dir()})."
        ),
    )
    parser.add_argument(
        "--no-atlas-cache",
        action="store_true",
        help="Disable fort.12 disk cache (rebuild SELECTLINES every run).",
    )
    parser.add_argument(
        "--warmstart-atm", type=str, default=None,
        help="Path to a pre-staged warm-start .atm file. When set, skip "
             "the kurucz-a1 emulator and use this file's layer structure "
             "as ATLAS's initial guess (with the target's chemistry "
             "rewritten on top). Use this when the emulator's prior is "
             "outside its training distribution and ATLAS NaN-explodes "
             "from the from-parameters path.",
    )
    parser.add_argument(
        "--no-molecular-lines",
        action="store_true",
        default=False,
        help="Atomic GFALL lines only (disable molecular line lists).",
    )
    parser.add_argument(
        "--no-tio",
        action="store_true",
        default=False,
        help="Exclude Schwenke TiO lines when molecular data are enabled.",
    )
    parser.add_argument(
        "--no-h2o",
        action="store_true",
        default=False,
        help="Exclude Partridge-Schwenke H2O lines when molecular data are enabled.",
    )
    args = parser.parse_args()
    atlas_iterations = int(args.atlas_iterations)
    if atlas_iterations < 1:
        parser.error("--atlas-iterations must be >= 1")
    if args.atlas_convergence_min_iterations < 1:
        parser.error("--atlas-convergence-min-iterations must be >= 1")
    if args.atlas_convergence_consecutive < 1:
        parser.error("--atlas-convergence-consecutive must be >= 1")
    atlas_convergence_epsilon = (
        None if args.no_atlas_convergence else float(args.atlas_convergence_epsilon)
    )
    if atlas_convergence_epsilon is not None and atlas_convergence_epsilon <= 0.0:
        parser.error("--atlas-convergence-epsilon must be positive")
    atlas_checkconv_dlntmax = (
        1.0e-4 if args.fortran_convergence else float(args.atlas_checkconv_dlntmax)
    )
    if atlas_checkconv_dlntmax <= 0.0:
        parser.error("--atlas-checkconv-dlntmax must be positive")
    if args.n_workers is not None and args.n_workers < 1:
        parser.error("--n-workers must be >= 1")

    individual = None
    if args.abund:
        individual = {z: val for z, val in args.abund}

    synthesize(
        teff=args.teff,
        logg=args.logg,
        mh=args.mh,
        am=args.am,
        vturb=args.vturb,
        wl_start=args.wl_start,
        wl_end=args.wl_end,
        resolution=args.resolution,
        atlas_iterations=atlas_iterations,
        atlas_convergence_epsilon=atlas_convergence_epsilon,
        atlas_convergence_min_iterations=args.atlas_convergence_min_iterations,
        atlas_convergence_consecutive=args.atlas_convergence_consecutive,
        atlas_checkconv_dlntmax=atlas_checkconv_dlntmax,
        abundances=individual,
        output_dir=args.output_dir,
        use_molecular_lines=not args.no_molecular_lines,
        include_tio=not args.no_tio,
        include_h2o=not args.no_h2o,
        n_workers=args.n_workers,
        atlas_fort12_cache=not args.no_atlas_cache,
        atlas_cache_dir=(
            Path(args.atlas_cache_dir).expanduser().resolve()
            if args.atlas_cache_dir is not None
            else None
        ),
        warmstart_atm_override=args.warmstart_atm,
        npz_cache=not args.no_npz_cache,
        linop1_serial=True if args.linop1_serial else None,
        convec_fd_parallel=args.convec_fd_parallel,
        pops_parallel=True if args.pops_parallel else None,
    )


if __name__ == "__main__":
    main()
