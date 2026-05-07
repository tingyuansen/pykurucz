<div align="center">

# pykurucz

**Pure Python ATLAS12 & SYNTHE — Synthetic Stellar Spectra**

By **Elliot M. Kim** (Cornell) &nbsp;·&nbsp; **[Yuan-Sen Ting](https://ysting.space/)** (OSU)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-pykurucz.vercel.app-1F6FEB.svg)](https://pykurucz.vercel.app)

📖 **Full documentation:** **[pykurucz.vercel.app](https://pykurucz.vercel.app)** — installation, user guide, architecture, physics reference, and worked examples.

</div>

> *Dedicated to the memory of **Robert L. Kurucz** (1944–2025), whose ATLAS and SYNTHE codes laid the foundations for modern stellar spectroscopy.*

pykurucz is a faithful, performance-tuned reimplementation of Kurucz's ATLAS12 (stellar atmosphere modeling) and SYNTHE (spectrum synthesis) in pure Python — no Fortran compiler required. Two abundance workflows are first-class: bulk scaling via `--mh` / `--am` for the standard scaled-solar / α-enhanced cases, and per-element overrides via `--abund Fe:-1.0 --abund C:+0.4` for peculiar patterns (CEMP, Ap, etc.). Either way the **atmosphere is rebuilt with the matching opacity** so line blanketing reshapes the temperature structure self-consistently — not just the spectrum on top of a generic atmosphere. Numba-JIT'd hot loops keep wall-time competitive with Fortran, and the result is validated end-to-end with sub-0.1% flux differences.

## Quick start

```bash
git clone https://github.com/tingyuansen/pykurucz.git
cd pykurucz
pip install -r requirements.txt

# Download line lists and molecular data (~5 GB, once)
python scripts/download_data.py

# Synthesize a solar spectrum end-to-end
pip install torch
python pykurucz.py --teff 5770 --logg 4.44 --wl-start 500 --wl-end 510
```

Output: `results/spec/*.spec` (wavelength, flux, continuum).

## Two modes

| | Mode A | Mode B |
|---|---|---|
| **Entry point** | `python synthesize_from_atm.py` | `python pykurucz.py` |
| **Input** | Your `.atm` file | Stellar parameters + any per-element abundances |
| **Atmosphere** | Pre-computed (ATLAS12, MARCS, PHOENIX, …) | Emulator warm-start → Python ATLAS12 iteration with the requested abundances |
| **Best for** | Re-using an externally computed atmosphere | Exploring abundance-peculiar patterns where the atmosphere itself must respond |
| **Needs PyTorch** | No | Yes |

```bash
# Mode A — from an existing atmosphere (atmosphere abundances fixed by the .atm file)
python synthe_py/tools/convert_atm_to_npz.py model.atm model.npz
python -m synthe_py.cli model.atm lines/gfallvac.latest --npz model.npz --spec output.spec

# Mode B — from stellar parameters; bulk metallicity + α
python pykurucz.py --teff 4500 --logg 2.0 --mh -1.5 --am 0.3

# Mode B — with arbitrary per-element abundances (CEMP-s example)
python pykurucz.py --teff 4800 --logg 1.5 \
    --abund Fe:-2.5 --abund C:+1.2 --abund Ba:+1.0 \
    --wl-start 400 --wl-end 700
```

## Features

- **Bulk and per-element abundance control** — `--mh` / `--am` for
  standard scaled-solar / α-enhanced runs, or `--abund` (repeatable)
  to override individual elements for peculiar patterns. Either way
  the atmosphere is recomputed self-consistently with the new opacity.
- **Pure Python** — NumPy, SciPy, Numba. No Fortran toolchain.
- **Fortran-validated** — sub-0.1% flux differences vs. original
  ATLAS12 + SYNTHE on the validation grid.
- **Neural warm-start** — PyTorch emulator drops `atlas_py` into the
  convergence basin so iteration takes ~10–15 steps; cold starts can
  stall.
- **Molecular lines** — TiO (Schwenke), H₂O (Partridge–Schwenke), CN,
  CO, C₂, CH, OH, MgH, FeH, and ~50 molecular species in total. On by
  default when `data/molecules/` is populated.

## Known limitation: extreme cool-RSG α-perturbed cells

In a narrow corner — cool ($T_{\rm eff} \lesssim 4500$ K), low-gravity ($\log g = 0$) atmospheres with non-zero `--am` and small `--abund C:` perturbations — the kurucz-a1 emulator's prior is far enough from the true converged solution that ATLAS iterates into all-NaN before recovering. The defensive guards in this branch (Fixes 12 / 13 in `PYKURUCZ_FIXES.md`) catch the failure cleanly with a `RuntimeError` rather than silently writing a degenerate `.atm`. The escape hatch when this happens is a **neighbour warmstart** — pre-stage a converged neighbour cell's `.atm` as the initial guess for ATLAS, with the target's chemistry rewritten on top. See [`docs/user-guide/neighbour-warmstart.md`](docs/user-guide/neighbour-warmstart.md) for the recipe.

## Documentation

Full documentation with installation guides, architecture deep-dives, physics reference, Python reference, and worked examples:

**→ [pykurucz.vercel.app](https://pykurucz.vercel.app)** *(or build locally: `pip install -r requirements-docs.txt && mkdocs serve`)*

## Package layout

```
pykurucz.py              # End-to-end: stellar params → spectrum
synthesize_from_atm.py   # Mode A: synthesis from existing .atm
atlas_py/                # Python ATLAS12 atmosphere engine
synthe_py/               # Python SYNTHE spectrum synthesis engine
emulator/                # Neural-network warm-start (PyTorch)
scripts/download_data.py # Fetch large binaries from GitHub releases
```

## Requirements

- Python 3.10+
- Core: NumPy, SciPy, Numba, Matplotlib
- End-to-end: + PyTorch
- Data: run `python scripts/download_data.py` once (~5 GB)

## Citation

```bibtex
@article{kim2026pykurucz,
  title   = {pyKurucz: A Pure Python Reimplementation of Kurucz SYNTHE
             for Stellar Spectrum Synthesis},
  author  = {Kim, Elliot M. and Ting, Yuan-Sen},
  journal = {Journal of Open Source Software},
  year    = {2026},
  note    = {in review}
}
```

If you use the kurucz-a1 emulator, also cite:

```bibtex
@article{li2025kurucza1,
  title   = {Differentiable Stellar Atmospheres with Physics-Informed
             Neural Networks},
  author  = {Li, Jiadong and Jian, Mingjie and Ting, Yuan-Sen
             and Green, Gregory M.},
  journal = {arXiv e-prints},
  year    = {2025},
  eprint  = {2507.06357}
}
```

## License

MIT License. See [LICENSE](LICENSE).
