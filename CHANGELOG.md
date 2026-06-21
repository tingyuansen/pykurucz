# Changelog

Notable updates to pykurucz **since the initial published release**. The README
and public docs continue to track the original published version; this file is
the running record of what has changed since.

## 2026-06-21 — Pipeline performance optimizations (student-contributed)

Adopted the student-contributed performance pass onto `main` (code only; the
git-LFS data files and `data/` download layout are unchanged). The optimizations
were validated end-to-end before merging.

### What changed (code)

- **Numba-JIT hot paths** — `PFSAHA` (partition/Saha) and `NMOLEC` (molecular
  Newton solver) ported to Numba kernels; cached contiguous `PFTAB`.
- **Parallelization** — frequency-loop and CONVEC finite-difference workers
  (`atlas_py/engine/{threading_policy,convec_fd_worker}.py`,
  `physics/pops_parallel.py`); numerically equivalent to the serial kernels.
- **Fortran-faithful convergence stop** — early-stop now uses Fortran's real
  `checkconv.f90` rule (deep-layer `max|ΔT/T|` over layers 40..jmax−5) instead of
  the old all-column "physical_max" metric. Default `--atlas-checkconv-dlntmax
  5e-4`; pass `--fortran-convergence` to restore strict `1e-4`. This is the main
  driver of the iteration-count reduction.
- **SYNTHE** — sparse `ASYNTH` wing kernel, NPZ atmosphere caching, Numba RT
  kernel (`synthe_py/engine/_rt_numba.py`), `float32` leaf-kernel discipline to
  match Fortran `REAL*4` line precision.
- New doc: `docs/physical_accuracy_deviations.md` (Fortran-parity accuracy notes).

### Validation (this repo's data, full emulator → ATLAS → SYNTHE)

Four stars across the HR diagram, 480–520 nm at R=300000, compared against the
pre-optimization `main`. Fair timing: warm cache, sequential, all 10 cores each.

| star | speedup | max\|ΔF/C\| vs old main (gate < 0.10) |
|------|--------:|--------------------------------------:|
| hot dwarf 10250/5.0 | 7.1× | 0.0026 |
| Sun 5770/4.44       | 9.0× | 0.0012 |
| cool dwarf 4500/4.5 | 13.1× | 0.0071 |
| giant 4800/2.0      | 11.9× | 0.0069 |
| **overall**         | **~10×** (25 min → 2.5 min) | — |

- **SYNTHE is bit-faithful**: molecules-off → identical (0.000); same-atmosphere
  cool dwarf → 9.5e-5. The cool-star residual (~0.7%) is the convergence/atmosphere
  term (the T-only stop leaves P/RHOX further from settled), not a synthesis error.
- **Memory** unchanged: ~280 MB private footprint per run.

### Known follow-ups (minor; empirically negligible for optical spectra)

Flagged during review for later cleanup — none affect the validated results above:

- `atlas_py/physics/kapp_continuum.py` — Si II level-25 photoionization
  cross-section double-counted in the "Group-2b" block (affects λ ≲ 152 nm only).
- `atlas_py/physics/selectlines.py` — hardcoded CENRATIO constant `0.014999`
  vs exact `0.0149725` (over-includes marginal lines; never drops any).
- `atlas_py/physics/grey_start.py` — emulator-fallback passes an unsupported
  `title=` kwarg (`TypeError` on the rare non-physical-emulator path).
- `atlas_py/engine/driver.py` — removed degenerate/NaN-atmosphere guards and an
  unguarded `max()` over a possibly-empty metric set (robustness regression that
  only manifests on a diverging atmosphere).
- Opt-in only, off by default: `fort12` cache key omits abundances/logg/vturb;
  `npz_cache` omits `continua.dat`; `ATLAS_POPS_PARALLEL=1` has data races.
