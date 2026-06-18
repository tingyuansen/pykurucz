# Branch review — `perf/pipeline-optimizations`

A reviewer's guide to **what this branch changes, why, how it was validated, and
how to stress-test it.** Everything here is **pure Python** (NumPy / SciPy /
Numba) — no Fortran compiler and **no C/native extension** is required or used.
A C prototype of the hottest kernel was built, measured, and then **deleted on
purpose** (see §"What we tried and rejected") to keep the port 100% Python.

If you only read one thing: this branch makes single-thread pykurucz competitive
with single-thread Fortran ATLAS+SYNTHE while keeping flux **within 1%** of the
Fortran reference, and aligns the convergence stop rule with Fortran's
`checkconv.f90`.

---

## 1. TL;DR — what we did

| # | Change | Where | Effect | Parity |
|---|--------|-------|--------|--------|
| 1 | **Numba-JIT `PFSAHA`** (partition/Saha hot path) | `atlas_py` (commit) | removed the single biggest pure-Python hotspot (~40% of a hot-star iter) | bit-identical |
| 2 | **Numba-JIT `NMOLEC`** Newton molecular-equilibrium solver | `atlas_py` (commit) | removed the 2nd Python hotspot | bit-identical |
| 3 | **Cache contiguous `PFTAB`** instead of re-wrapping per call | `atlas_py` (commit) | drops per-call allocation overhead | bit-identical |
| 4 | **`float32` discipline in `LINOP1`/`XLINOP`** line kernels | `atlas_py/physics/line_opacity.py` | LINOP1 ~1.25× faster; **also a Fortran-fidelity fix** (`REAL*4`) | `max\|F/C\|` 0.004 |
| 5 | **Sparse `ASYNTH` + NPZ cache** in SYNTHE | `synthe_py` (commit) | faster synthesis stage, avoids recompute | parity-safe |
| 6 | **Fortran-faithful convergence stop** (`fortran_checkconv`) | `atlas_py/engine/driver.py` | **−14.4% grid wall @1T, −7.4% @12T**; correctness fix | parity-safe |
| 7 | **`float32`/`int32` + hoisted molecular kernel** | `synthe_py/engine/opacity.py` | ~1 GB less RAM, fidelity-true (speed-neutral, kept for correctness) | bit-identical |

Items 1–5 land as commits; items 6–7 plus the CLI/tooling wiring are the working
tree of this branch. Per-case numbers are reproduced inline below (§2, §7); the
long-form investigation lives in `results/` (generated output, not pushed — see
§8).

---

## 2. The headline result (convergence alignment)

Python used to early-stop on a **Python-invented** metric (`physical_max`: the
max normalized change across *all* columns — RHOX, T, P, XNE, ABROSS, VTURB).
That metric is hostage to RHOX/XNE/P oscillation, so converging stars kept
iterating long after the *atmosphere* was settled.

Fortran's real rule (`kurucz/src/checkconv.f90`) is **temperature-only**:

```fortran
REAL, PARAMETER :: dlntmax = 1E-4
dlnt(j) = ABS(temp(j,imax-1) - temp(j,imax)) / temp(j,imax)
IF (MAXVAL(dlnt(40:jmax-5)) .GT. dlntmax) ...   ! deep layers 40..75 (jmax=80)
```

We implemented exactly this as the new default (`fortran_checkconv`). The
per-iteration physics is **unchanged** — only the *stop decision* moves.

Grid bench (10 cases, legacy vs new default, all parity-PASS):

| metric | legacy (`physical`) | `fortran_checkconv` | delta |
|--------|--------------------:|--------------------:|------:|
| grid-sum e2e, 1 thread | 3859 s | 3305 s | **−14.4% (1.17×)** |
| grid-sum e2e, 12 threads | 1637 s | 1516 s | −7.4% (1.08×) |
| mean throughput, 1T | 9.3 /hr | 10.9 /hr | +17% |
| mean throughput, 12T | 22.0 /hr | 23.7 /hr | +8% |

Biggest 1T per-case wins: `t10250` −42.6%, `t05600` −27.9%, `t04000` −21.7%,
`t08250` −16.0%.

---

## 3. Validation / parity gate

Every change is held to the project's gate: on normalized flux F/C over
**300–1800 nm**, `max |F/C_py − F/C_ft| < 0.10` vs the Fortran `synthe_fast`
reference, **0 outliers**. Observed margins are far inside the gate (typically
`max|F/C| ≈ 0.004–0.009`). Kernel-level ports (PFSAHA, NMOLEC, molecular
float32/hoist) are **bit-identical** to the pre-change Python output.

Each change was validated by comparing a full `pykurucz.py` run against the
Fortran reference spectrum over 300–1800 nm before being kept.

---

## 4. How to run

### 4.1 Setup (once)

```bash
pip install -r requirements.txt
pip install torch                  # only needed for Mode B (emulator warm-start)
python scripts/download_data.py    # line lists + molecular data (~5 GB)
```

### 4.2 Synthesize a spectrum

`pykurucz.py` is the single production entry point: stellar parameters in, a
synthetic spectrum out. The atmosphere is recomputed self-consistently with the
requested abundances (no pre-built `.atm` needed).

```bash
python pykurucz.py --teff 5770 --logg 4.44 --wl-start 500 --wl-end 510
```

Output: `results/spec/*.spec` (wavelength, flux, continuum).

---

## 5. How to stress-test (via `pykurucz.py`)

Everything below is driven by the one entry point — no extra harnesses or
external data beyond `download_data.py`. Useful knobs:
`--n-workers` (thread budget), `--no-atlas-convergence` (fixed iterations),
`--mh` / `--am` / `--abund` (chemistry), `--wl-start` / `--wl-end` (window).

```bash
# Hot dwarf — converges fast, exercises the new early-stop
python pykurucz.py --teff 10250 --logg 5.0

# Cool dwarf — molecule-heavy, exercises LINOP1 + molecular kernels
python pykurucz.py --teff 4000 --logg 5.0

# Peculiar abundances (atmosphere responds to the new opacity)
python pykurucz.py --teff 4800 --logg 1.5 \
    --abund Fe:-2.5 --abund C:+1.2 --abund Ba:+1.0 --wl-start 400 --wl-end 700

# Single-thread vs multi-thread on the same case (thread scaling)
python pykurucz.py --teff 8250 --logg 4.0 --n-workers 1
python pykurucz.py --teff 8250 --logg 4.0 --n-workers 12

# Convergence A/B: Fortran-faithful early-stop (default) vs fixed iterations
python pykurucz.py --teff 10250 --logg 5.0                      # checkconv (default)
python pykurucz.py --teff 10250 --logg 5.0 --no-atlas-convergence
```

The convergence threshold is env-tunable without code edits:
`ATLAS_CHECKCONV_DLNTMAX=1e-4` (the Fortran `checkconv.f90` default).

---

## 6. What we tried and rejected (honesty section)

- **C / native LINOP1 kernel** — a faithful C transcription benched at 1.91×
  and passed the parity gate, but was **deleted**: a native extension defeats
  the purpose of a Python port. `atlas_py/native/` no longer exists.
- **Numba `inline="always"` on the wing accumulator** and a row-slice rewrite —
  both **regressed** throughput; reverted.
- **Anderson mixing on temperature** for the one non-converging case (`t03600`)
  — made convergence *worse* at both `beta=1.0` and `beta=0.5`; **removed** from
  the code entirely.
- **Molecular kernel micro-opts** (float32 + hoisting) — bit-identical and
  ~1 GB lighter, but **speed-neutral** at 1T (the kernel is bound by valid-pair
  Voigt-wing scatter writes). Kept for fidelity/memory, documented as neutral.

---

## 7. Known limitations

- **`t03600` (cool giant) does not fully converge in 30 iters** — deep-layer
  `dlnT` bottoms at 1.86e-4 (just above the 1e-4 gate). It remains
  **parity-correct** (`max|F/C| = 0.0045`); it just runs the full iteration cap.
- **Single-thread kernel floor** — `LINOP1`, `XLINOP`, and the molecular wing
  kernel are at a pure-Python (Numba/LLVM) codegen floor. Closing the residual
  gap to clang `-O3` would require a native extension (rejected) or an
  algorithmic far-wing cutoff that risks parity (needs sign-off).
- **Fortran baseline caveat** — the committed `atlas12.exe`/`synthe` are x86_64
  ifort binaries; on Apple Silicon they run under Rosetta 2, so absolute Fortran
  wall-times are emulation-inflated. Compare *trends* and parity, not raw
  seconds, unless you rebuild the Fortran natively.

---

## 8. A note on the measurements

All numbers in this document are reproduced **inline** so the doc is
self-contained. They came from benchmark/validation runs whose raw artifacts
(summaries, plots, per-case logs, the long-form `REPORT.md`) live under
`results/` — which is **generated output and not pushed with the code**. Re-run
the `pykurucz.py` commands in §5 to reproduce the behavior on any machine.
