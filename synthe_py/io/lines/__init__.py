"""Line catalogue helpers."""

from typing import Any

__all__ = [
    "atomic",
    "molecular",
    "fort9",
    "fort19",
    "compiler",
    "parsed_cache",
    "tfort_write",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - thin lazy loader
    if name in __all__:
        module = __import__(f"synthe_py.io.lines.{name}", fromlist=[name])
        return module
    raise AttributeError(f"module 'synthe_py.io.lines' has no attribute {name!r}")
