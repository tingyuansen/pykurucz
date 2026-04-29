#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# scripts/setup_data.sh — DEVELOPER / LAB MACHINES ONLY
#
# Populate pykurucz/data/ by copying files from an existing Kurucz data tree.
# Requires local access to a kurucz directory (e.g. tingyuansen/kurucz clone).
#
# External / public users: download the release tarball instead.
# See data/README.md for instructions.
#
# Usage:
#   bash scripts/setup_data.sh --source /path/to/kurucz
#   bash scripts/setup_data.sh                     # default: ../kurucz (sibling dir)
#   bash scripts/setup_data.sh --source /path/to/kurucz --no-synthe
#   bash scripts/setup_data.sh --dry-run
#
# Flags:
#   --source DIR    Path to the kurucz repository or data directory.
#                   Default: ../kurucz (sibling directory — lab convention only).
#   --no-synthe     Skip the ~13 GB synthe/linelists_full/ tfort.* files.
#                   Use when you only need the Python pipeline (not Fortran SYNTHE).
#   --dry-run       Print what would be copied without actually copying.
#
# After running, pykurucz.py and all tools use data/ by default.
# Large files are git-ignored (see .gitignore); only small text files are tracked.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DEST="$REPO_ROOT/data"

SOURCE="$(dirname "$REPO_ROOT")/kurucz"  # default: ../kurucz
SKIP_SYNTHE=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)   SOURCE="$2"; shift 2 ;;
        --no-synthe) SKIP_SYNTHE=1; shift ;;
        --dry-run)  DRY_RUN=1; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

SOURCE="$(cd "$SOURCE" && pwd)"  # resolve to absolute path

echo "Source : $SOURCE"
echo "Dest   : $DEST"
[[ $DRY_RUN -eq 1 ]] && echo "(dry run — no files will be copied)"
echo

CP() {
    local src="$1" dst="$2"
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  [dry] cp $src -> $dst"
    else
        cp -v "$src" "$dst"
    fi
}

CP_GLOB() {
    # Copy files matching a glob; silently skip if none match.
    local pattern="$1" dst="$2"
    local count=0
    for f in $pattern; do
        [[ -e "$f" ]] || continue
        CP "$f" "$dst/"
        count=$((count+1))
    done
    if [[ $count -eq 0 ]]; then
        echo "  (no files matched: $pattern)"
    fi
}

# ---------------------------------------------------------------------------
# 1. ATLAS line/molecule binaries (required for atlas_py / Mode C)
# ---------------------------------------------------------------------------
echo "=== lines/ binaries ==="
mkdir -p "$DEST/lines"
CP_GLOB "$SOURCE/lines/gfpred29dec2014.bin"        "$DEST/lines"
CP_GLOB "$SOURCE/lines/gfpred29dec2014.bin.part*"  "$DEST/lines"
CP_GLOB "$SOURCE/lines/lowobsat12.bin"             "$DEST/lines"
CP_GLOB "$SOURCE/lines/hilines.bin"                "$DEST/lines"
CP_GLOB "$SOURCE/lines/diatomicspacksrt.bin"       "$DEST/lines"
CP_GLOB "$SOURCE/lines/nltelinobsat12.bin"         "$DEST/lines"

echo
echo "=== lines/ text files (tracked in git) ==="
CP_GLOB "$SOURCE/lines/molecules.new"   "$DEST/lines"
CP_GLOB "$SOURCE/lines/molecules.dat"   "$DEST/lines"
CP_GLOB "$SOURCE/lines/continua.dat"    "$DEST/lines"
CP_GLOB "$SOURCE/lines/he1tables.dat"   "$DEST/lines"

echo
echo "=== molecules/ (TiO + H2O binaries) ==="
mkdir -p "$DEST/molecules/tio" "$DEST/molecules/h2o"
CP_GLOB "$SOURCE/molecules/tio/*.bin"   "$DEST/molecules/tio"
CP_GLOB "$SOURCE/molecules/h2o/*.bin"   "$DEST/molecules/h2o"

echo
echo "=== molecules/ ASCII catalogs ==="
CP_GLOB "$SOURCE/molecules/*.dat"       "$DEST/molecules"
CP_GLOB "$SOURCE/molecules/*.asc"       "$DEST/molecules"

echo
echo "=== molecules/ subdirectory catalogs (vo/) ==="
if [[ -d "$SOURCE/molecules/vo" ]]; then
    mkdir -p "$DEST/molecules/vo"
    CP_GLOB "$SOURCE/molecules/vo/*"    "$DEST/molecules/vo"
fi

# ---------------------------------------------------------------------------
# 2. Fortran executables (required for run_e2e_pipeline.py Fortran branch)
# ---------------------------------------------------------------------------
echo
echo "=== bin_macos/ executables ==="
mkdir -p "$DEST/bin_macos"
CP_GLOB "$SOURCE/bin_macos/*.exe"  "$DEST/bin_macos"

echo
echo "=== bin_linux/ executables ==="
mkdir -p "$DEST/bin_linux"
CP_GLOB "$SOURCE/bin_linux/*.exe"  "$DEST/bin_linux"

# ---------------------------------------------------------------------------
# 3. Fortran SYNTHE line lists (~13 GB; skip with --no-synthe)
# ---------------------------------------------------------------------------
if [[ $SKIP_SYNTHE -eq 1 ]]; then
    echo
    echo "=== synthe/linelists_full/ — SKIPPED (--no-synthe) ==="
else
    echo
    echo "=== synthe/linelists_full/ (~13 GB) ==="
    mkdir -p "$DEST/synthe/linelists_full"
    CP_GLOB "$SOURCE/synthe/linelists_full/tfort.*"  "$DEST/synthe/linelists_full"
fi

# ---------------------------------------------------------------------------
# 4. Fortran control input + source
# ---------------------------------------------------------------------------
echo
echo "=== infiles/ ==="
mkdir -p "$DEST/infiles"
CP_GLOB "$SOURCE/infiles/spectrv_std.input"  "$DEST/infiles"

echo
echo "=== src/ (atlas12.for) ==="
mkdir -p "$DEST/src"
CP_GLOB "$SOURCE/src/atlas12.for"  "$DEST/src"

# ---------------------------------------------------------------------------
echo
if [[ $DRY_RUN -eq 1 ]]; then
    echo "Dry run complete. Re-run without --dry-run to perform the copy."
else
    echo "setup_data.sh complete."
    echo
    echo "Quick smoke-tests:"
    echo "  python pykurucz.py --teff 5770 --logg 4.44 --wl-start 500 --wl-end 510"
fi
