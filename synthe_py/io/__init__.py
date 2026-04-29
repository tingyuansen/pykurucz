"""I/O helpers for the Python SYNTHE pipeline."""

from typing import Any

__all__ = ["atmosphere", "export", "persist", "lines"]


def __getattr__(name: str) -> Any:  # pragma: no cover - simple lazy loader
    if name in {"atmosphere", "export", "persist", "lines"}:
        module = __import__(f"synthe_py.io.{name}", fromlist=[name])
        return module
    raise AttributeError(f"module 'synthe_py.io' has no attribute {name!r}")
