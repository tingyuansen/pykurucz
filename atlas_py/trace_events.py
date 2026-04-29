"""Shared trace event schema for Fortran/Python divergence checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import csv
import re


TRACE_VERSION = "TRACEV1"
TRACE_HEADER = [
    "version",
    "source",
    "event",
    "iter",
    "line",
    "depth",
    "nu",
    "type_code",
    "wlvac_nm",
    "center",
    "adamp",
    "cv",
    "tabcont",
    "branch",
    "reason",
]


@dataclass(frozen=True)
class TraceEvent:
    version: str
    source: str
    event: str
    iter: int
    line: int
    depth: int
    nu: int
    type_code: int
    wlvac_nm: float
    center: float
    adamp: float
    cv: float
    tabcont: float
    branch: str
    reason: str

    def key(self) -> tuple[int, int, int, int, str]:
        """Deterministic alignment key for first-divergence matching."""
        return (self.iter, self.line, self.depth, self.nu, self.event)


def write_trace_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(TRACE_HEADER)


def append_trace_event(path: Path, ev: TraceEvent) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                ev.version,
                ev.source,
                ev.event,
                ev.iter,
                ev.line,
                ev.depth,
                ev.nu,
                ev.type_code,
                f"{ev.wlvac_nm:.9e}",
                f"{ev.center:.9e}",
                f"{ev.adamp:.9e}",
                f"{ev.cv:.9e}",
                f"{ev.tabcont:.9e}",
                ev.branch,
                ev.reason,
            ]
        )


def read_trace_events(path: Path) -> list[TraceEvent]:
    def _f(s: str) -> float:
        t = s.strip()
        if not t:
            return 0.0
        try:
            return float(t)
        except ValueError:
            # Fortran fixed format can emit exponent without explicit 'E',
            # e.g. "5.615354239-315" -> "5.615354239E-315".
            m = re.match(r"^([+-]?\d*\.?\d+)([+-]\d+)$", t)
            if m:
                return float(f"{m.group(1)}E{m.group(2)}")
            raise

    out: list[TraceEvent] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(
                TraceEvent(
                    version=row["version"].strip(),
                    source=row["source"].strip(),
                    event=row["event"].strip(),
                    iter=int(row["iter"]),
                    line=int(row["line"]),
                    depth=int(row["depth"]),
                    nu=int(row["nu"]),
                    type_code=int(row["type_code"]),
                    wlvac_nm=_f(row["wlvac_nm"]),
                    center=_f(row["center"]),
                    adamp=_f(row["adamp"]),
                    cv=_f(row["cv"]),
                    tabcont=_f(row["tabcont"]),
                    branch=row["branch"].strip(),
                    reason=row["reason"].strip(),
                )
            )
    return out


def write_trace_events(path: Path, events: Iterable[TraceEvent]) -> None:
    write_trace_header(path)
    for ev in events:
        append_trace_event(path, ev)
