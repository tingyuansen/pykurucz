"""Parser for ATLAS12 control-deck style commands (READIN semantics)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class AtlasDeck:
    """Normalized control deck fields used by atlas_py."""

    molecules_on: bool = True
    read_molecules: bool = True
    read_punch: bool = True
    opacity_lines: bool = True
    opacity_xlines: bool = True
    convection_over: float = 1.25
    iterations: int = 1
    print_level: int = 1
    punch_level: int = 1
    scale_model_fields: List[float] | None = None
    vturb_cms: float = 2.0e5


def parse_readin_deck(path: Path) -> AtlasDeck:
    """Parse a simple ATLAS12 stdin deck used by atlas12.exe.

    This parser supports the subset needed for single-iteration validation:
    MOLECULES, READ MOLECULES, READ PUNCH, OPACITY ON LINES/XLINES,
    CONVECTION OVER, ITERATIONS, PRINT, PUNCH, SCALE MODEL, VTURB.
    """

    deck = AtlasDeck()
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip().upper()
        if not line:
            continue
        parts = line.split()
        if parts[:2] == ["MOLECULES", "OFF"]:
            deck.molecules_on = False
        elif parts[:2] == ["MOLECULES", "ON"]:
            deck.molecules_on = True
        elif parts[:2] == ["READ", "MOLECULES"]:
            deck.read_molecules = True
        elif parts[:2] == ["READ", "PUNCH"]:
            deck.read_punch = True
        elif parts[:3] == ["OPACITY", "ON", "LINES"]:
            deck.opacity_lines = True
        elif parts[:3] == ["OPACITY", "ON", "XLINES"]:
            deck.opacity_xlines = True
        elif parts and parts[0] == "ITERATIONS" and len(parts) >= 2:
            deck.iterations = int(float(parts[1]))
        elif parts and parts[0] == "PRINT" and len(parts) >= 2:
            deck.print_level = int(float(parts[1]))
        elif parts and parts[0] == "PUNCH" and len(parts) >= 2:
            deck.punch_level = int(float(parts[1]))
        elif parts[:2] == ["CONVECTION", "OVER"] and len(parts) >= 3:
            deck.convection_over = float(parts[2])
        elif parts[:2] == ["SCALE", "MODEL"] and len(parts) >= 7:
            deck.scale_model_fields = [float(x) for x in parts[2:]]
        elif parts and parts[0] == "VTURB" and len(parts) >= 2:
            deck.vturb_cms = float(parts[1])
    return deck

