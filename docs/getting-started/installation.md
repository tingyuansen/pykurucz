# Installation

pykurucz runs on Python 3.10 or newer and requires only standard scientific Python packages for its core functionality. The neural-network atmosphere emulator adds a single optional dependency on PyTorch.

## System Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| **Python** | 3.10 | 3.12 |
| **Operating System** | Linux, macOS, Windows | Linux or macOS |
| **RAM** | 8 GB | 16 GB for full-range synthesis |
| **Disk** | 6 GB free | 10 GB free |

## Core Dependencies

The following packages are required for **all** modes of operation:

| Package | Minimum Version | Purpose |
|---------|-----------------|---------|
| NumPy | 1.24 | Array computation, linear algebra, interpolation |
| SciPy | 1.10 | Special functions, optimization, sparse arrays |
| Numba | 0.58 | JIT compilation of hot loops (opacity accumulation, RT) |
| Matplotlib | 3.7 | Plotting utilities |

!!! note "Numba is optional but strongly recommended"
    Without Numba, the line-opacity accumulation and radiative-transfer loops fall back to pure NumPy. The code is still correct, but synthesis can be 5–10× slower.

## Installing from Source

Clone the repository and install the Python dependencies:

```bash
git clone https://github.com/tingyuansen/pykurucz.git
cd pykurucz
pip install -r requirements.txt
```

The `requirements.txt` contains only the core scientific stack:

```text
numpy>=1.24
scipy>=1.10
numba>=0.58
matplotlib>=3.7
```

## Optional: PyTorch for the Emulator

If you plan to use **Stellar Parameters** (end-to-end synthesis from \(T_{\rm eff}\), \(\log g\), [M/H], [α/M]), install PyTorch:

```bash
pip install torch
```

!!! tip "CPU-only PyTorch is sufficient"
    The emulator inference is fast even on CPU. A GPU is not required for any pykurucz workflow.

## Downloading Data Files

pykurucz requires large binary data files (atomic line lists, molecular
catalogs, and physics tables) that are distributed via GitHub release assets
rather than committed to the repository. Run the downloader once after
installation:

```bash
# Full data (~5.2 GB extracted) — needed for Stellar Parameters mode
python scripts/download_data.py

# Synthesis-only (~1.3 GB) — sufficient if you only use Existing Atmosphere
python scripts/download_data.py --synthe-only
```

See **[Downloading Data](downloading-data.md)** for the full description of
what gets fetched, where it lands on disk, how to pin a specific release
(`--tag`), and how to assemble the parts manually for offline installs.

!!! warning "Do not skip the data download"
    Without the data files, `synthe_py` cannot load line lists and `atlas_py`
    cannot perform line selection. The CLI will raise a clear `FileNotFoundError`
    pointing back to this step.

## Verifying the Installation

A single command tests the full chain end-to-end:

```bash
python pykurucz.py --teff 5770 --logg 4.44 --wl-start 500 --wl-end 501 --atlas-iterations 1
```

If this writes a `.spec` file under `results/spec/`, your install is fully
functional (the run uses the emulator + 1 ATLAS iteration + synthesis, so
every code path is touched). For a step-by-step walk-through of the same
example, see [Your First Spectrum](first-spectrum.md).

## Development Installation

If you plan to modify pykurucz or run the validation suite, install in editable mode and include test dependencies:

```bash
pip install -e ".[dev]"
```

or, if no `setup.py` / `pyproject.toml` extras are defined:

```bash
pip install -e .
pip install pytest black ruff
```

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `ModuleNotFoundError: No module named 'torch'` | PyTorch not installed | `pip install torch` |
| `FileNotFoundError: Required atlas_py binary not found` | Data files missing | `python scripts/download_data.py` |
| `ImportError: cannot import name '...' from numba` | Numba version too old | `pip install --upgrade numba` |
| Slow synthesis on first run | Numba compiling JIT kernels | Expected; subsequent runs are fast |
