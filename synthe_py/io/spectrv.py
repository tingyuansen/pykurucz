"""Helpers for parsing spectrv input cards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class SpectrvParams:
    """Subset of the spectrv control parameters used in the LTE run."""

    rhoxj: float
    ph1: float
    pc1: float
    psi1: float
    prddop: float
    prdpow: float


def load(path: Path) -> SpectrvParams:
    """Parse the standard spectrv input deck (e.g. ``spectrv_std.input``)."""

    text = path.read_text(encoding="ascii").splitlines()
    if not text:
        raise ValueError(f"spectrv input {path} is empty")
    first = text[0].strip()
    if not first:
        raise ValueError(f"spectrv input {path} missing parameter line")
    parts = first.replace(",", " ").split()
    if len(parts) < 8:
        raise ValueError(
            f"spectrv input {path} expected eight floats on first line, found {parts}"
        )
    values = tuple(float(item) for item in parts[:8])
    rhoxj, r1, r101, ph1, pc1, psi1, prddop, prdpow = values
    return SpectrvParams(
        rhoxj=rhoxj,
        ph1=ph1,
        pc1=pc1,
        psi1=psi1,
        prddop=prddop,
        prdpow=prdpow,
    )

