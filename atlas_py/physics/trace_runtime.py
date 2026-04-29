"""Runtime-controlled trace emitter for physics kernels."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from ..trace_events import TRACE_VERSION, TraceEvent, append_trace_event, write_trace_header


@dataclass
class TraceConfig:
    enabled: bool
    path: Path
    source: str
    wlo_nm: float
    whi_nm: float
    jlo_1b: int
    jhi_1b: int
    max_events: int


_TRACE_CONFIG: TraceConfig | None = None
_TRACE_EVENT_COUNT = 0
_TRACE_HEADER_WRITTEN = False


def _load_config() -> TraceConfig:
    enabled = os.environ.get("ATLAS_TRACE_ENABLE", "0") == "1"
    path = Path(os.environ.get("ATLAS_TRACE_PATH", "atlas_py_trace.csv"))
    source = os.environ.get("ATLAS_TRACE_SOURCE", "python")
    wlo_nm = float(os.environ.get("ATLAS_TRACE_WLO_NM", "381.0"))
    whi_nm = float(os.environ.get("ATLAS_TRACE_WHI_NM", "410.0"))
    jlo_1b = int(os.environ.get("ATLAS_TRACE_JLO", "58"))
    jhi_1b = int(os.environ.get("ATLAS_TRACE_JHI", "63"))
    max_events = int(os.environ.get("ATLAS_TRACE_MAX_EVENTS", "50000"))
    return TraceConfig(
        enabled=enabled,
        path=path,
        source=source,
        wlo_nm=wlo_nm,
        whi_nm=whi_nm,
        jlo_1b=jlo_1b,
        jhi_1b=jhi_1b,
        max_events=max_events,
    )


def trace_config() -> TraceConfig:
    global _TRACE_CONFIG
    if _TRACE_CONFIG is None:
        _TRACE_CONFIG = _load_config()
    return _TRACE_CONFIG


def trace_enabled() -> bool:
    return trace_config().enabled


def trace_in_focus(*, wlvac_nm: float, j0: int) -> bool:
    cfg = trace_config()
    if not cfg.enabled:
        return False
    j1 = j0 + 1  # Python 0-based -> Fortran 1-based
    return (cfg.wlo_nm <= wlvac_nm <= cfg.whi_nm) and (cfg.jlo_1b <= j1 <= cfg.jhi_1b)


def trace_emit(
    *,
    event: str,
    iter_num: int,
    line_num_1b: int,
    depth_1b: int,
    nu_1b: int,
    type_code: int,
    wlvac_nm: float,
    center: float,
    adamp: float,
    cv: float,
    tabcont: float,
    branch: str,
    reason: str,
) -> None:
    global _TRACE_EVENT_COUNT, _TRACE_HEADER_WRITTEN
    cfg = trace_config()
    if not cfg.enabled:
        return
    if _TRACE_EVENT_COUNT >= cfg.max_events:
        return
    if not _TRACE_HEADER_WRITTEN:
        write_trace_header(cfg.path)
        _TRACE_HEADER_WRITTEN = True
    append_trace_event(
        cfg.path,
        TraceEvent(
            version=TRACE_VERSION,
            source=cfg.source,
            event=event,
            iter=iter_num,
            line=line_num_1b,
            depth=depth_1b,
            nu=nu_1b,
            type_code=type_code,
            wlvac_nm=wlvac_nm,
            center=center,
            adamp=adamp,
            cv=cv,
            tabcont=tabcont,
            branch=branch,
            reason=reason,
        ),
    )
    _TRACE_EVENT_COUNT += 1
