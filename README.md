<div align="center">

```
                                           ....                                           
                              ......:::::::=*#%*-:.                                       
                        ..:=+##**+=+#%@@@%@%%%%%@%*.                                      
                       .=+#@@@@@%%@@@@%%@@@@@%%%%%%%#*-                                   
                     .-*#%%%%%%%%%%%%%@@%%%%@@@@%%%%%%%*--:                               
                   .=#@@@%%%%%@@@@@@@@@@@@@@@@@@@%%%%%%@@@#.                              
                  -%@@%%%@%%@@@@@@@@@@@@@@@@@@@@@@@@@@@%%%%*=:                            
                 -%@%%%%%%%@@@%%%%%%%%%%%@@@@@%@@%%%%%%%%%#%@*.                           
               :+%%%%%%%%%%%%%@@@@@@@@@@@@@%%@@%%%##%%%%%%%%%#%+                          
              =#%%@%%%@%%%@@@@%##****##%%%%@@%%%#####%%%%@@@%#%@*                         
             =%#%%%%@%*=+**+=-:..::-----=++**++++++++*#%%%%%%%%%*:                        
            :%%@@@@@%*-...      ..::--:::::----===++++++#%@@%%%%**-                       
            #%%%%%%%#=-:..  ..    ...::--::---==++**###*+#%%%@@%%#%-                      
            +##%#%%%*=:..  .--::::::::-:--:---+**********++**#%@%##+                      
          .+*##%%%%%*-::.:::==----=====::-===+*+++*****###=%@%%%@%%#=.                    
         ::+#%%%%%@%#=-:-:::---=+*++=-::.:=*+*####%@###*##-*%@@@%@@##+-                   
           :#%%%%%%%%#+=:::.:*++%%*++-:..:-+##%%#******+*+=**%%%%%%%##%=                  
           .=#%##%%%%*:::.. ....:---=:....:=+**###******+=+==*%%#%%%%:.                   
             #**#%%%%*:.. .     .:::......::--**+******++**+=+%#+##*%                     
             -*+#**+#*-..   .....::....  ..-=-=##***+**####*==+++*%#+                     
              :++---++-:..    .   . .:.:-::=#%#+%@%#*++=*##*=-=*##@%*                     
                -..--:=:....... .  .=:.+#--+*%@%%%@%***==##*+-=+#*#%+                     
                  .-: .::........ .=: .:--=+=+*#*#%%%##*=*#+=-=#*#+-                      
                  ..::..:--.  ....-=:-==+*********#%%%%*+*#=--==--.                       
                  ..  .::-=:......-=+**###*##*#%%%%%%%#****+-:.                           
                     ..:::-:::::.:=+##*++===+*##%%%%%%#***+*=:                            
                          ----:-:-=*%+::::-=+++*##%%%%###***=.                            
                     .:-++::::-:-=+*%=--=++*++**#%%%@%%###*=*%#*=-:.                      
               .-=+*#%%@@@#+=---==**#**+==++++**#%%%%##**++:.#@@@@%%#+=-.                 
         .:=+*###%%%%%@%%%@#.:-==-=+++**+=++***##%##*#*+==#- =%%%%@@%%%%%#*+-.            
   .:-=+*########%%%%%%%%%*.  .:----=+***##**#######***++%*::*%%%%%@@%%%%%%%%%#*+-.       
=+*#########%%%%%%%%%%%%##*.   ..:::-==+***#**#%#**++#%%%#-:-%%%%@@@@%@@@@%%%%%%@@%#*+=:. 
#######%%%%%%%%%%%%%%%%##*%=    .:::::---==***+*#*++#%%%#+--#@%%@@@@@@@@%%@@@@%%%%%%%%%%#*
##%%%%%%%%%%%%%%%%%%%%####%.     .::::::---=+**###*%%%%##*-*@%@@@@@@@@@@@@@@%@@@@@@%%%%%%%
%%%%%%%%%%%%%%%%%%#%%####%#        .:-------+####%%%%###*=*@%@@@@@@@@@@@@@@@@@@@@@@@@@@@@%
%%%%%%%%%%%%%%%%%##########.         .:--=--#%%%%%##****+*@%%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
%%%%%%%%%%%%%%%%#########*=              :=+##*=+++===++*%%%%%@@@%@@@@@@@@@@@@@@@@@@@@@@@@
%%%%%%%%%%%%%%%###########: .           :*%%#%#=::::--+*#%%#*%@%%%@@@@@@@@@@@@@@@@@@@@@@@@
%%%%%%%%%%%%%%############. -         :*@@%@@%@@%*-:-+*####+*@%%%%@@@@@@@@@@@@@@@@@@@@@@%%
%%%%%%%%%%%%%############%- ::        #@@%%%%%%%@@%*+*###*+-#%%%%%@@@@%%@@@@@@@@@@@@@@%%%%
%%%%%%%%%@@%%#############*  :        .%#%%%%%@@%%@%+###*=-+%%%%%%@@@@%%%@@@@@@@@@@@@%%%%%
%%%%%#%%%@@%##############%+ :.      .+--%%%@@@@@%%#+*+=---%%%%%%%@@@@@%%%@@@@@@@@@@%%%%%%
```

</div>

<p align="center"><em>
This project is dedicated to the memory of <strong>Robert L. Kurucz</strong> (1944–2025), whose nearly six decades of work at the Center for Astrophysics | Harvard & Smithsonian laid the foundations for modern stellar spectroscopy. Bob's ATLAS and SYNTHE codes, his atomic and molecular line lists, and his freely shared data have been used by thousands of astronomers worldwide, accumulating tens of thousands of citations. He received the AAS Van Biesbroeck Prize in 1992 for "long-term extraordinary or unselfish service to astronomy." Bob spent seven days a week in his office for decades, driven by a simple goal he never stopped pursuing: "I wanted to determine stellar effective temperatures, gravities, and abundances to study solar and stellar evolution. I still want to." This Python reimplementation of his SYNTHE code is our tribute to his extraordinary legacy of open science.
</em></p>

<p align="center"><sub>Portrait rendered from <a href="https://www.cfa.harvard.edu/people/robert-kurucz">CfA biography photo</a> — credit: Center for Astrophysics | Harvard & Smithsonian</sub></p>

---

# pyKurucz — Pure Python Stellar Spectrum Synthesis

**A ground-truth reimplementation of Kurucz's ATLAS12 and SYNTHE in pure Python** — atmosphere iteration (`atlas_py`) and spectrum synthesis (`synthe_py`) validated against Fortran end-to-end spectra; molecular lines use the same Fortran-grounded path and are **on by default** when `data/molecules/` is populated (TiO/H₂O binaries are included if the files exist). Use `--no-molecular-lines` for atomic-only runs. No Fortran compiler needed. Just Python, NumPy, SciPy, and Numba.

**Authors:** Elliot M. Kim (Cornell) and Yuan-Sen Ting (The Ohio State), building on the original Fortran codes by Robert L. Kurucz (CfA/Harvard & Smithsonian).

## Quick start

```bash
git clone https://github.com/emk255/pykurucz.git
cd pykurucz
pip install -r requirements.txt
mkdir -p results
```

**Download the data directory** (line-list binaries and molecule tables — only needed once):

```bash
pip install -r requirements.txt
python scripts/download_data.py                 # full data (~5.2 GB)
# or, if you don't need atlas_py atmosphere iteration:
python scripts/download_data.py --synthe-only   # ~1.3 GB, enough for synthesis
```

This pulls the data from a public Google Drive folder via `gdown` (no login, no OAuth, no tokens). Two files are fetched:

- `pykurucz-data-synthe-v1.0.tar.gz` (1.3 GB compressed, ~3.3 GB extracted) — atomic line lists and all molecular catalogs (TiO Schwenke, H2O Partridge-Schwenke, VO, CN, C2, CO, H2, ...) used by `synthe_py`.
- `gfpred29dec2014.bin` (3.9 GB) — Kurucz 29-Dec-2014 predicted atomic line list, used by `atlas_py` for full atmosphere iteration (optional for `synthe_py`).

SHA256 is verified automatically. See [data/README.md](data/README.md) for the full file layout.

**Synthesize from an existing atmosphere file** (no PyTorch needed; requires `python scripts/download_data.py` for line list and molecule data):

```bash
# Replace paths with your ATLAS12-format .atm
python synthe_py/tools/convert_atm_to_npz.py path/to/model.atm results/model.npz
python -m synthe_py.cli path/to/model.atm lines/gfallvac.latest \
    --npz results/model.npz --spec results/output.spec --wl-start 500 --wl-end 510
```

**Or go end-to-end from stellar parameters** (requires `pip install torch`):

```bash
pip install torch
python pykurucz.py --teff 5770 --logg 4.44 --wl-start 500 --wl-end 510

# Fast diagnostic atmosphere pass only, not recommended for science output
python pykurucz.py --teff 5770 --logg 4.44 --atlas-iterations 1
```

Both produce a `.spec` file with wavelength, flux, and continuum columns. By default, `pykurucz.py` allows up to 30 `atlas_py` atmosphere iterations so the model is not left at the warm-start/single-step stage, but it can stop earlier once the physical atmosphere columns converge (`--atlas-convergence-epsilon`, default `1e-3`). Use `--no-atlas-convergence` to force all requested iterations. Use a narrow wavelength range (like 500-510 nm above) for a quicker first run; the full 300-1800 nm range is slower but covers the complete optical+NIR.


## Why this exists

Bob Kurucz's codes — ATLAS, SYNTHE, DFSYNTHE, WIDTH, BALMER — together with his atomic and molecular line lists, form one of the most consequential software ecosystems in astrophysics. They have been used to analyze spectra from nearly every major telescope and spectroscopic survey for decades, accumulating tens of thousands of citations. But the original Fortran codebase, developed continuously since the 1960s, is increasingly difficult to compile, install, modify, and integrate with modern workflows.

pyKurucz is a line-by-line numerical reimplementation — not a wrapper around Fortran — ensuring that this extraordinary body of work remains accessible, extensible, and usable for the next generation of astronomers. The core is faithful SYNTHE-class spectrum synthesis from a model atmosphere, including **molecular line opacity** loaded automatically from `data/molecules/` after running `python scripts/download_data.py` (with Schwenke TiO and Partridge–Schwenke H₂O). Override with `--molecules-dir` or switch off with `--no-molecular-lines`.


## Two modes of operation

Spectrum synthesis requires a **model atmosphere** as input — a description of how temperature, pressure, and density vary with depth in the star. `synthe_py` is agnostic about where that atmosphere comes from. This repository offers two entry points:

| | Mode A (`synthesize_from_atm.py`) | Mode B (`pykurucz.py`) |
|---|---|---|
| **Input** | Your own `.atm` file | Stellar parameters (Teff, logg, [M/H], [α/M]) |
| **Atmosphere source** | Pre-computed (ATLAS12, MARCS, PHOENIX, etc.) | Emulator warm-start → `atlas_py` self-consistent iteration |
| **Dependencies** | Core only | Core + PyTorch + `data/` (populated via `python scripts/download_data.py`) |
| **Atmosphere physics** | Exact (whatever generated the `.atm`) | Full Python ATLAS12 (Fortran-parity) |
| **Best for** | Full control, outside emulator range, or reusing an external atmosphere | End-to-end synthesis straight from stellar parameters |

### Mode A examples

```bash
# Step 1: Preprocess the atmosphere (populations, molecular equilibrium, opacities)
python synthe_py/tools/convert_atm_to_npz.py your_model.atm results/your_model.npz

# Step 2: Run synthesis (molecular lines default ON if data/molecules/ is populated;
# TiO/H2O load when schwenke.bin / h2ofastfix.bin are found — use --no-tio / --no-h2o to skip)
python -m synthe_py.cli your_model.atm lines/gfallvac.latest \
    --npz results/your_model.npz --spec results/your_model.spec \
    --wl-start 300 --wl-end 1800

# Atomic lines only (faster setup when you have no molecule data)
python -m synthe_py.cli your_model.atm lines/gfallvac.latest \
    --npz results/your_model.npz --spec results/your_model_atomic.spec \
    --no-molecular-lines --wl-start 300 --wl-end 1800

# Custom molecule trees (repeat --molecules-dir for multiple directories)
python -m synthe_py.cli your_model.atm lines/gfallvac.latest \
    --npz results/your_model.npz --spec results/your_model_custommol.spec \
    --molecules-dir /path/to/extra/molecules --wl-start 300 --wl-end 1800
```

### Mode B examples

`pykurucz.py` runs a single canonical pipeline:

```
kurucz-a1 emulator ──► warm-start .atm ──► atlas_py (MOLECULES ON)
                                     └──► iterated .atm ──► synthe_py ──► .spec
```

The emulator plays the same role as `READ DECK6` in the Fortran pipeline: it supplies the starting layer structure so that `atlas_py` converges quickly rather than starting from a grey approximation. `atlas_py` then self-consistently iterates the atmospheric structure with the same physics as Fortran ATLAS12 (always `MOLECULES ON`, matching the Fortran deck), so the downstream SYNTHE spectrum stays in parity with Fortran references. The CLI defaults to a maximum of 30 atmosphere iterations, matching the full Fortran-style validation/convergence path, and stops early when the physical columns (`RHOX`, `T`, `P`, `XNE`, `ABROSS`, `VTURB`) change by less than `1e-3` after at least five iterations. `ACCRAD` is logged as a diagnostic but is not part of the stop rule. Use `--no-atlas-convergence` to force the full iteration count, or `--atlas-iterations 1` only for fast diagnostics.

Requires `data/` to be populated once via `python scripts/download_data.py` (see Quick start above).

```bash
# Solar-type star, full wavelength range
python pykurucz.py --teff 5770 --logg 4.44

# Fast diagnostic atmosphere pass only (not recommended for science output)
python pykurucz.py --teff 5770 --logg 4.44 --atlas-iterations 1

# Force all 30 atmosphere iterations, disabling convergence early stopping
python pykurucz.py --teff 5770 --logg 4.44 --no-atlas-convergence

# Metal-poor K giant with alpha enhancement
python pykurucz.py --teff 4500 --logg 2.0 --mh -1.5 --am 0.3

# Override individual element abundances (relative to solar)
python pykurucz.py --teff 5770 --logg 4.44 --abund C:+0.5 --abund Fe:-1.0

# Optical only, lower resolution (faster)
python pykurucz.py --teff 5770 --logg 4.44 --wl-start 400 --wl-end 700 --resolution 50000

# Cool star (molecular lines default on when data/molecules/ is populated)
python pykurucz.py --teff 4000 --logg 4.5

# Turn off molecular lines or specific binaries
python pykurucz.py --teff 5770 --logg 4.44 --no-molecular-lines
python pykurucz.py --teff 4000 --logg 4.5 --no-tio --no-h2o
```

The emulator is trained on 104,269 ATLAS12 models ($T_{\text{eff}}$ 2,500–50,000 K, $\log g$ −1.0–5.5, [M/H] −4.0–+1.5, [α/M] −0.2–+0.6). Outside this range the emulator warm-start becomes unreliable and the code warns you — use Mode A instead with an externally computed atmosphere.

**Output layout** (under `results/` or `--output-dir`):
```
atm/<stem>_warmstart.atm   emulator prediction (READ DECK6 for atlas_py)
atm/<stem>.atm             atlas_py iterated atmosphere
npz/<stem>.npz             preprocessed populations / opacity tables
spec/<stem>_<wl0>_<wl1>.spec   final spectrum
logs/<stem>_atlas.log          atlas_py.cli log
logs/<stem>_synthe_<wl0>_<wl1>.log   convert_atm_to_npz + synthe_py.cli log
```

`<stem>` is `t{Teff:05d}g{logg:.2f}_mh{eff_mh:+.2f}_am{eff_am:+.2f}`.

If you need to skip both the emulator and `atlas_py` (e.g. you already have a `.atm` from another source), use Mode A (`synthesize_from_atm.py`) directly.


## What `synthe_py` does under the hood

The heart of this repository is **`synthe_py/`** — a pure Python reimplementation of Kurucz's SYNTHE spectral synthesis code. Given a model atmosphere and an atomic line list, it computes the emergent stellar spectrum wavelength by wavelength:

1. **Continuum opacity** — H⁻ bound-free/free-free (the dominant source in Sun-like stars), H I bound-free (Karsas & Latter cross-sections), He I/II, metal photoionization, Rayleigh scattering (H, He, H₂), Thomson scattering — interpolated from pre-tabulated arrays following the original KAPP subroutine logic. For cool atmospheres, the COOLOP path also adds CH, OH, and H₂ collisional opacity when molecular populations from equilibrium are available.
2. **Line opacity — atomic** — every transition in the Kurucz GFALL catalog (~1.3 million lines) near the current wavelength contributes a **Voigt profile** (thermal Doppler + van der Waals + Stark + radiative broadening). Hydrogen Balmer/Lyman lines get dedicated Stark-broadened profiles (HPROF4); helium lines use tabulated BCS/Griem/Dimitrijević profiles.
3. **Line opacity — molecular** — Kurucz ASCII molecular catalogs (e.g. CH, OH, CO, CN, C₂, MgH, …) from `data/molecules/` (populated by `python scripts/download_data.py`) or explicit `--molecules-dir`, plus Schwenke TiO and Partridge–Schwenke H₂O when enabled (default on; binaries skipped quietly if missing). Use `--no-molecular-lines`, `--no-tio`, or `--no-h2o` to disable. Molecular lines use the same opacity accumulation and radiative-transfer loop as atoms, with populations tied to the **NELION** dispatch and molecular equilibrium (NMOLEC-class solver) from preprocessing.
4. **Radiative transfer** — the JOSH solver integrates the transfer equation on a fixed log-τ grid with parabolic optical depth quadrature and Lambda iteration for scattering, yielding both line+continuum $F_\lambda$ and continuum-only $F_{\rm cont}$.

Plus supporting physics: Saha–Boltzmann populations with detailed partition functions, molecular equilibrium for ~300 species, and Doppler widths at every atmospheric layer.

### Validated against the Fortran original

Both codes were run with identical inputs: same `.atm` files, same line list, same wavelength range (300–1800 nm), same resolution (R=300,000). The end-to-end `atlas_py` + `synthe_py` validation uses normalized flux (`F/F_cont`) as the scientific comparison target; matched Fortran-reference checks are counted only when the Fortran ATLAS30 → SYNTHE pipeline itself produces a valid reference spectrum. For those valid matched references, recent checks passed with no wavelength exceeding the `0.10` outlier threshold.

Very cool, edge-of-grid giants can be difficult for the original Fortran validation pipeline as well; if Fortran ATLAS30 or Fortran SYNTHE cannot produce a clean reference spectrum for a case, that case is treated as an invalid reference sample rather than a Python pass/fail. Use such edge cases cautiously until a trusted external atmosphere/reference is available.

### Runtime

Runtime depends strongly on stellar type, wavelength range, line-list density, worker count, and whether atmosphere convergence triggers early. In matched 300–1800 nm, R=300,000 validation on the same machine, a 4000 K dwarf with a valid Fortran30 reference took about 3230 s for the full Fortran+Python comparison run: the Python branch (`atlas_py` + `synthe_py`) took about 1095 s, while the Fortran branch took about 2136 s, so Python was roughly 2x faster for that case. Solar-like runs benefit more from convergence early stopping; for one 5770 K validation case, Python ATLAS stopped at 18/30 iterations and took about 233 s for the atmosphere stage.

| Model | $T_{\text{eff}}$ | $\log g$ | Type | Median Δ | 95th pctl | 99th pctl |
|-------|-------|-------|------|----------|-----------|-----------|
| `t02500g-1.0` | 2500 K | −1.0 | Cool giant | 0.0006% | 0.023% | 0.065% |
| `t04000g5.00` | 4000 K | 5.0 | K dwarf | 0.00004% | 0.002% | 0.006% |
| `t08250g4.00` | 8250 K | 4.0 | A star | 0.0008% | 0.023% | 0.056% |
| `t10250g5.00` | 10250 K | 5.0 | Late B | 0.0008% | 0.020% | 0.047% |
| `t44000g4.50` | 44000 K | 4.5 | O star | 0.002% | 0.109% | 0.224% |


## How the pipeline works

```
  MODE A (own .atm)                       MODE B (stellar parameters)
 ──────────────────                ───────────────────────────────────
                                    Teff, logg, [M/H], [α/M]
  Your .atm file                              │
  (from ATLAS12,                              v
   MARCS, PHOENIX,                   [kurucz-a1 emulator]
   or any source)                    (warm-start atmosphere)
        │                                     │
        │                              warm-start .atm
        │                                     │
        │                                     v
        │                              [atlas_py.cli]
        │                          (Python ATLAS12 iteration,
        │                              MOLECULES ON)
        │                                     │
        │                              iterated .atm
        │                           (+ optional abundance
        │                             overrides via --abund)
        │                                     │
        └────────────────┬────────────────────┘
                         │
                         v
              [convert_atm_to_npz.py]
              (populations, molecular equil.,
               continuous opacity coefficients)
                         │
                         v
                  [synthe_py.cli]
              (line-by-line radiative transfer;
               molecular catalogs + TiO/H2O by default when data exist)
                         │
                         v
                     .spec file
              (wavelength, flux, continuum)
```

**Stage 1 — Atmosphere**: Mode A reads your `.atm` file directly. Mode B runs the kurucz-a1 emulator for a warm-start and then iterates it with `atlas_py` (full Python ATLAS12, `MOLECULES ON`). The default is `--atlas-iterations 30` as the maximum, with convergence early stopping enabled at `--atlas-convergence-epsilon 1e-3` after at least five iterations. Pass `--no-atlas-convergence` when you intentionally want exactly the full iteration count, or `--atlas-iterations 1` only when you want a fast diagnostic atmosphere pass.

**Stage 2 — Preprocessing** (`convert_atm_to_npz.py`): Reads the `.atm` file and computes Saha–Boltzmann populations, molecular equilibrium (~300 species), continuous opacity coefficients (H⁻, H I, He, metals, scattering), and Doppler widths at each atmospheric layer.

**Stage 3 — Synthesis** (`synthe_py.cli`): The core loop. For each wavelength point: evaluate continuum opacity, search atomic and (by default) molecular line lists when catalogs are available, compute profiles, solve the transfer equation. Outputs wavelength, $F_\lambda$, and $F_{\text{cont}}$.


## Individual element abundances (Mode B)

When using `pykurucz.py` (Mode B) and specifying individual element abundances via `--abund`, values are **dex offsets relative to Asplund solar** (e.g., `Fe:-1.0` means [Fe/H] = −1.0). The script automatically derives the best-matching 4-parameter model for the emulator warm-start:

- **[M/H]**: proxied by the Fe offset (if given): $[\text{M/H}] \approx [\text{Fe/H}]$
- **[α/M]**: computed as the average offset of alpha elements (O, Ne, Mg, Si, S, Ca, Ti) relative to the derived [M/H]

The emulator predicts the closest atmospheric structure for those derived parameters. Your exact abundances (solar + offsets) are written into the `.atm` file's abundance table, so SYNTHE uses the precise values you requested for all line opacity calculations.

```bash
# Example: metal-poor star with enhanced Mg and Ca
python pykurucz.py --teff 5000 --logg 3.0 --abund Fe:-1.0 --abund Mg:+0.4 --abund Ca:+0.3
# -> derives [M/H]=-1.0, [alpha/M]=+1.13 for the emulator
# -> writes exact Fe, Mg, Ca abundances into .atm for SYNTHE
```

For fully self-consistent treatment where the atmospheric structure itself responds to non-standard abundances, use **Mode A** with an atmosphere computed by a full stellar-atmosphere code (not generated here).


## Output

Each `.spec` file has three whitespace-delimited columns: `wavelength(nm)  F_lambda  F_continuum`. The **normalized spectrum** is $F_\lambda / F_{\text{cont}}$. Wavelengths are in vacuum nm; fluxes in erg cm⁻² s⁻¹ nm⁻¹.


## Input data

Small physics tables and code are in the repository. Large binary data (line lists, molecule tables) are distributed via a public Google Drive folder — run `python scripts/download_data.py` once to populate `data/`.

### Line list (`data/lines/` — from `python scripts/download_data.py`)

All files below live under `data/lines/` after running `python scripts/download_data.py`:

| File | Description |
|------|-------------|
| `gfallvac.latest` | Kurucz GFALL atomic line list — ~1.3M transitions with wavelength, $\log gf$, excitation energies, damping constants |
| `gfpred29dec2014.bin` | Kurucz GFALL predicted-line binary used by `atlas_py` for line selection (~3.9 GB) |
| `continua.dat` | Bound-free absorption edge wavelengths and cross-sections for computing continuous opacity |
| `molecules.dat` | Dissociation energies and equilibrium constants for ~300 molecular species |
| `he1tables.dat` | Tabulated helium line broadening profiles |

### Molecule tables (`data/molecules/` — from `python scripts/download_data.py`)

TiO (Schwenke) and H₂O (Partridge–Schwenke) binary line lists (~2.8 GB total) are downloaded and loaded automatically when present.

### Physics tables (`synthe_py/data/` — in-repo)

Pre-extracted from the original Fortran binary data arrays:

| File | Contents |
|------|----------|
| `atlas_tables.npz` | Rosseland mean opacity interpolation tables |
| `fortran_data.npz` | Physical constants, element properties, ionization stages |
| `pfsaha_ion_pots.npz` | Ionization potentials for all elements and ionization stages |
| `pfsaha_levels.npz` | Energy levels and statistical weights for partition function calculations |
| `pfiron_data.npz` | Detailed iron-group partition functions |
| `kapp_tables.npz` | Continuous opacity coefficient tables (KAPP subroutine data) |
| `he1_tables.npz` | Helium profile interpolation tables |
| `helium_aux.npz` | Auxiliary helium broadening data |

### ATLAS12 emulator (`emulator/`)

| File | Description |
|------|-------------|
| `a_one_weights.pt` | Trained MLP weights (512-dim stellar encoder + 512-dim tau encoder + predictor) |
| `norm_params.pt` | Input/output normalization parameters (min-max scaling with log transforms) |

## Reproducing the validation

```bash
# Example: loop over your own .atm files (same stem for model.spec vs reference.spec)
for atm in path/to/atmospheres/*.atm; do
    python synthesize_from_atm.py "$atm"
done

for atm in path/to/atmospheres/*.atm; do
    stem=$(basename "$atm" .atm)
    python synthe_py/tools/compare_spectra.py \
        "results/spec/${stem}_300_1800.spec" \
        "path/to/fortran_specs/${stem}.spec" \
        --range 300 1800 --top 5
done
```


## Package structure

```
pykurucz/
├── pykurucz.py                     # End-to-end: stellar params -> emulator -> atlas_py -> synthe_py -> spectrum
├── synthesize_from_atm.py          # Synthesis from an existing .atm file
├── requirements.txt                # Core dependencies (Python 3.10+)
├── LICENSE                         # MIT License
│
├── atlas_py/                       # Python ATLAS12 atmosphere engine
│   ├── cli.py                      # Command-line interface
│   ├── atmosphere.py               # Atmospheric structure and iteration
│   ├── physics/                    # Opacity sources, equation of state, convection
│   └── tools/                      # Utilities (validator, comparator)
│
├── synthe_py/                      # Python SYNTHE spectrum synthesis engine
│   ├── cli.py                      # Command-line interface
│   ├── config.py                   # Configuration dataclasses
│   ├── data/                       # Pre-extracted physics tables (.npz) — in-repo
│   ├── engine/                     # Core synthesis loop + radiative transfer
│   ├── io/                         # Atmosphere, line list, and spectrum I/O
│   ├── physics/                    # Opacity, broadening, populations, profiles
│   └── tools/                      # Utilities (converter, comparator, plotter)
│
├── emulator/                       # ATLAS12 neural network warm-start (needs PyTorch)
│   ├── model.py                    # PyTorch MLP architecture
│   ├── emulator.py                 # Prediction interface
│   └── normalization.py            # Input/output normalization
│
└── data/                           # Large runtime binaries — populated via `python scripts/download_data.py`
    ├── lines/                      # ATLAS line-list binaries (gfpred29dec2014.bin, hilines.bin, …)
    ├── molecules/                  # Molecular line lists (TiO, H₂O binaries)
    └── README.md                   # Describes the layout and how to obtain the files
```


## CLI reference

### `pykurucz.py` — end-to-end from stellar parameters (requires `pip install torch`)

```bash
python pykurucz.py --teff <Teff> --logg <logg> [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--teff` | (required) | Effective temperature (K) |
| `--logg` | (required) | Surface gravity (log10 cgs) |
| `--mh` | 0.0 | Overall metallicity [M/H] — scales all metals uniformly |
| `--am` | 0.0 | Alpha enhancement [α/M] — extra offset for O, Ne, Mg, Si, S, Ca, Ti |
| `--vturb` | 2.0 | Microturbulent velocity (km/s) |
| `--abund` | — | Element offset relative to solar: `--abund Fe:-1.0` for [Fe/H]=−1.0 (repeatable) |
| `--wl-start` | 300 | Start wavelength (nm) |
| `--wl-end` | 1800 | End wavelength (nm) |
| `--resolution` | 300000 | Resolving power $\lambda / \Delta\lambda$ |
| `--atlas-iterations` | 30 | Maximum number of `atlas_py` atmosphere iterations |
| `--atlas-convergence-epsilon` | 1e-3 | Early-stop threshold on physical atmosphere column changes |
| `--atlas-convergence-min-iterations` | 5 | Minimum iterations before convergence early stopping can trigger |
| `--atlas-convergence-consecutive` | 1 | Consecutive converged iterations required before stopping |
| `--no-atlas-convergence` | off | Disable convergence early stopping and run exactly `--atlas-iterations` |
| `--output-dir` | results/ | Output directory |
| `--no-molecular-lines` | off | Disable all molecular line opacity (GFALL only) |
| `--no-tio` | off | Exclude Schwenke TiO (when molecular lines are enabled) |
| `--no-h2o` | off | Exclude Partridge–Schwenke H₂O (when molecular lines are enabled) |

### `synthe_py.cli` — synthesis from `.atm` file (no PyTorch needed)

```bash
python synthe_py/tools/convert_atm_to_npz.py <atm_file> <output.npz>
python -m synthe_py.cli <atm_file> lines/gfallvac.latest --npz <output.npz> --spec <output.spec> [options]
```

**Molecular line behavior (default on):** If you omit `--molecules-dir`, the code looks for `data/molecules/` inside the pykurucz repo (populated by `python scripts/download_data.py`). TiO and H₂O are included by default when the Schwenke / Partridge–Schwenke binaries are present. Use `--no-molecular-lines` for atomic-only synthesis; `--no-tio` / `--no-h2o` to drop specific species. **`--molecules-dir DIR`** (repeatable) overrides the search paths; **`--tio-bin`** / **`--h2o-bin`** set explicit binary paths.


## Dependencies

**Core** (all modes): NumPy, SciPy, Numba, Matplotlib — Python 3.10+

```bash
pip install -r requirements.txt
```

**End-to-end pipeline** (emulator + atlas_py + synthe_py): PyTorch, plus the `data/` tree populated once from Google Drive.

```bash
pip install torch
pip install -r requirements.txt
python scripts/download_data.py
```


## Current limitations and roadmap

pyKurucz reimplements both the ATLAS12 atmosphere stage (`atlas_py`) and the SYNTHE synthesis stage (`synthe_py`) in pure Python — atomic lines throughout, **molecular lines by default** when `data/molecules/` is populated via `python scripts/download_data.py` (TiO/H₂O binaries used if present). Use `--no-molecular-lines` for GFALL-only runs. Remaining gaps are mostly about breadth of physics (NLTE, geometry) and workflow (external atmosphere codes for self-consistency).

### What works today

- Full atomic line synthesis from any `.atm` file
- **Molecular line synthesis** — Schwenke TiO and Partridge–Schwenke H₂O from `data/molecules/` (via `python scripts/download_data.py`), plus Kurucz ASCII molecular catalogs; opt out with `--no-molecular-lines` / `--no-tio` / `--no-h2o`
- End-to-end synthesis from stellar parameters: **emulator warm-start → `atlas_py` (Python ATLAS12, `MOLECULES ON`) → `synthe_py`**, for Fortran-faithful self-consistent atmospheres
- Full Python ATLAS12 atmosphere iteration (`atlas_py`): opacity, convection, equation of state, and temperature correction fully reimplemented
- All continuous opacity sources (H⁻, H I, He I/II, metals, Rayleigh, Thomson), including cool-star COOLOP molecular continuum (CH, OH, H₂) when populations are present
- Voigt line profiles with van der Waals, Stark, and radiative broadening
- Dedicated hydrogen Stark-broadened (Balmer/Lyman) and helium tabulated profiles
- Saha–Boltzmann populations, molecular equilibrium (~300 species)
- Individual element abundance overrides with automatic emulator parameter derivation

### What comes next

| Limitation | Impact | Next step |
|---|---|---|
| **No self-consistent atmosphere iteration for arbitrary abundance patterns** | The emulator warm-start is trained on a 4-parameter grid; highly non-solar abundance patterns may need extra `atlas_py` iterations to converge. | Increase `--atlas-iterations` for such cases. |
| **LTE only** | NLTE effects matter for specific lines (Li I, Na D, O I triplet) in metal-poor/hot stars. | Ingest departure coefficients from external NLTE codes as correction factors. |
| **1D plane-parallel geometry** | Breaks down for evolved giants with extended atmospheres. | Ingest 3D model atmospheres (Stagger, CO⁵BOLD) as stratifications for post-processing. |

## Relation to tingyuansen/kurucz

[tingyuansen/kurucz](https://github.com/tingyuansen/kurucz) provides the original Fortran ATLAS12 + SYNTHE pipeline with pre-compiled binaries, plus the kurucz-a1 emulator. **This repository** (`pykurucz`) is a self-contained Python reimplementation of both ATLAS12 (`atlas_py`) and SYNTHE (`synthe_py`), including molecular lines via `data/molecules/` populated by `python scripts/download_data.py`. No local clone of the `kurucz` repo is needed at runtime. Use the `kurucz` repo if you need Fortran tools or want to run the original Fortran pipeline for ground-truth comparison.


## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use pyKurucz in your research, please cite the JOSS paper:

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

If you use the kurucz-a1 atmosphere emulator (the warm-start stage of Mode B), please also cite:

```bibtex
@article{li2025kurucza1,
  title   = {Differentiable Stellar Atmospheres with Physics-Informed
             Neural Networks},
  author  = {Li, Jiadong and Jian, Mingjie and Ting, Yuan-Sen
             and Green, Gregory M.},
  journal = {arXiv e-prints},
  year    = {2025},
  eprint  = {2507.06357},
  url     = {https://arxiv.org/abs/2507.06357}
}
```
