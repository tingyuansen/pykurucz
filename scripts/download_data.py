"""Download pykurucz runtime data from Google Drive.

Usage:
    python scripts/download_data.py                 # full data (~5.2 GB)
    python scripts/download_data.py --synthe-only   # ~1.3 GB, enough for synthesis

Downloads:
  * pykurucz-data-synthe-v1.0.tar.gz  (1.3 GB compressed, 3.3 GB extracted)
      Atomic line lists + molecular catalogs. Required by synthe_py for
      spectrum synthesis from a .atm file.
  * gfpred29dec2014.bin               (3.9 GB)
      Kurucz 29-Dec-2014 predicted atomic line list. Required by atlas_py
      for full atmosphere iteration. Not needed for synthe_py alone.

No authentication required. Data is hosted on a public Google Drive folder
and served through gdown (handles the >100 MB confirmation cookie and
supports resume).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
from pathlib import Path


# Google Drive file IDs (public: "Anyone with the link - Viewer")
SYNTHE_ID = "15-G3VTrF4niqpkkrGz5ZVfj-nQ9bxfym"
GFPRED_ID = "1sbFJC4kX-X-58iDsloDsJY8t81mzIrug"

SYNTHE_FILENAME = "pykurucz-data-synthe-v1.0.tar.gz"
GFPRED_FILENAME = "gfpred29dec2014.bin"

SYNTHE_SHA256 = "62537e444ff5130cc593a424e2cf7537b2db79575187e95411664605ca3f02d2"
GFPRED_SHA256 = "2e9afd8ad1765d79768469216ac8f5751dcefb9b9ab49b23ea6642e289035593"

SYNTHE_BYTES = 1_408_060_085
GFPRED_BYTES = 4_195_274_400

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def _ensure_gdown():
    try:
        import gdown  # noqa: F401
    except ImportError:
        print("gdown is not installed. Run: pip install gdown", file=sys.stderr)
        sys.exit(1)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(file_id: str, dest: Path, expected_sha: str, expected_bytes: int) -> None:
    """Download one Google Drive file to *dest* with SHA256 verification.

    If *dest* already exists with the right SHA256, skip the download.
    """
    import gdown

    if dest.exists() and dest.stat().st_size == expected_bytes:
        print(f"  {dest.name}: already present, verifying SHA256 ...")
        if _sha256(dest) == expected_sha:
            print(f"  {dest.name}: SHA256 OK, skipping download.")
            return
        print(f"  {dest.name}: SHA256 mismatch, re-downloading.")
        dest.unlink()

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"

    try:
        out = gdown.download(url, str(dest), quiet=False, resume=True)
    except Exception as exc:  # noqa: BLE001
        if "quota" in str(exc).lower():
            _print_quota_error(dest.name)
        raise
    if out is None:
        _print_quota_error(dest.name)
        raise RuntimeError(f"gdown failed to download {dest.name}")

    print(f"  {dest.name}: verifying SHA256 ...")
    actual = _sha256(dest)
    if actual != expected_sha:
        raise RuntimeError(
            f"SHA256 mismatch for {dest}\n"
            f"  expected {expected_sha}\n"
            f"  got      {actual}\n"
            f"Delete {dest} and re-run the downloader."
        )
    print(f"  {dest.name}: SHA256 OK.")


def _print_quota_error(filename: str) -> None:
    print("", file=sys.stderr)
    print("=" * 72, file=sys.stderr)
    print(f"ERROR: Google Drive quota exceeded for {filename}.", file=sys.stderr)
    print("", file=sys.stderr)
    print("This file has hit Google's per-file download cap (~190 requests/day).", file=sys.stderr)
    print("Please try again in 24 hours, or populate data/ from a local Kurucz", file=sys.stderr)
    print("tree with:", file=sys.stderr)
    print("    bash scripts/setup_data.sh --source /path/to/kurucz", file=sys.stderr)
    print("=" * 72, file=sys.stderr)
    print("", file=sys.stderr)


def _extract_synthe(tarball: Path) -> None:
    """Extract the synthe tarball into the repo root (populates data/lines and data/molecules)."""
    print(f"  Extracting {tarball.name} into {REPO_ROOT} ...")
    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(REPO_ROOT)
    print(f"  Extract complete. Removing tarball ...")
    tarball.unlink()


def _synthe_extracted() -> bool:
    """Cheap check: do the main synthe files already exist?"""
    markers = [
        DATA_DIR / "lines" / "hilines.bin",
        DATA_DIR / "lines" / "lowobsat12.bin",
        DATA_DIR / "molecules" / "tio" / "schwenke.bin",
        DATA_DIR / "molecules" / "h2o" / "h2ofastfix.bin",
    ]
    return all(p.exists() for p in markers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--synthe-only",
        action="store_true",
        help="Download only the synthe bundle (skip the 3.9 GB gfpred file). "
             "Enough for synthe_py synthesis from an existing .atm file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files appear to be present.",
    )
    args = parser.parse_args()

    _ensure_gdown()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "lines").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "molecules").mkdir(parents=True, exist_ok=True)

    print("Downloading pykurucz runtime data from Google Drive ...")
    print(f"  Destination: {DATA_DIR}")
    print()

    # synthe bundle
    if args.force or not _synthe_extracted():
        synthe_tar = DATA_DIR / SYNTHE_FILENAME
        print(f"[1/2] {SYNTHE_FILENAME}  (1.3 GB compressed)")
        _download(SYNTHE_ID, synthe_tar, SYNTHE_SHA256, SYNTHE_BYTES)
        _extract_synthe(synthe_tar)
    else:
        print(f"[1/2] {SYNTHE_FILENAME}: already extracted, skipping.")
    print()

    # gfpred (optional)
    if args.synthe_only:
        print("[2/2] gfpred29dec2014.bin: skipped (--synthe-only).")
        print("      atlas_py atmosphere iteration will not be available.")
    else:
        gfpred_path = DATA_DIR / "lines" / GFPRED_FILENAME
        print(f"[2/2] {GFPRED_FILENAME}  (3.9 GB)")
        _download(GFPRED_ID, gfpred_path, GFPRED_SHA256, GFPRED_BYTES)
    print()

    print("Done. Data is ready:")
    print("  data/lines/                              -- atomic line lists (for synthe_py and atlas_py)")
    print("  data/molecules/                          -- TiO, H2O, and molecular catalogs (for synthe_py)")
    if not args.synthe_only:
        print("  data/lines/gfpred29dec2014.bin           -- predicted atomic line list (for atlas_py)")


if __name__ == "__main__":
    main()
