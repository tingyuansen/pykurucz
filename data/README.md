# `data/` — runtime binaries and line-list assets

Large binary files (line lists, molecule tables) are hosted on a public Google
Drive folder. They are **not** stored in git. No authentication is required —
the files are shared as "Anyone with the link – Viewer".

## Get the data (all users)

```bash
pip install -r requirements.txt
python scripts/download_data.py                 # full data (~5.2 GB)
# or, if you don't need atlas_py atmosphere iteration:
python scripts/download_data.py --synthe-only   # ~1.3 GB, enough for synthesis
```

Two files are fetched via `gdown`:

| File | Size | Used by |
|---|---|---|
| `pykurucz-data-synthe-v1.0.tar.gz` | 1.3 GB (→ 3.3 GB extracted) | `synthe_py` |
| `gfpred29dec2014.bin` | 3.9 GB | `atlas_py` (optional) |

SHA256 is verified automatically after each download. The synthe tarball is
extracted into `data/lines/` and `data/molecules/` and then deleted.

### Resulting layout

```
data/
├── lines/
│   ├── gfpred29dec2014.bin          # Kurucz GFPRED predicted lines (fort.11) ~3.9 GB
│   ├── hilines.bin                  # High-excitation lines (fort.21)
│   ├── diatomicspacksrt.bin         # Diatomic molecular lines
│   ├── lowobsat12.bin               # Observed low lines (fort.111)
│   ├── nltelinobsat12.bin           # NLTE lines
│   ├── continua.dat                 # Bound-free absorption edges
│   ├── molecules.dat                # Dissociation energies / equilibrium constants
│   ├── molecules.new                # Extended molecule list for ATLAS12
│   └── he1tables.dat                # Helium broadening tables
└── molecules/
    ├── tio/
    │   ├── schwenke.bin             # Schwenke TiO line list
    │   └── eschwenke.bin
    ├── h2o/
    │   └── h2ofastfix.bin           # Partridge-Schwenke H2O line list
    └── [*.dat / *.asc molecule catalogs]
```

## Google Drive quota

Google Drive enforces a per-file download cap of roughly 190 requests per
24 hours per public file. For a typical academic release this is more than
enough, but if you hit a `Quota exceeded` error the downloader will print a
clear message. Options:

1. Wait 24 hours and retry.
2. If you have a local Kurucz data tree, populate `data/` without going
   through Google Drive (see below).