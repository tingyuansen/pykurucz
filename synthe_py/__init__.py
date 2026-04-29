"""Python reimplementation of the SYNTHE spectrum synthesis pipeline."""

from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Entry-point wrapper exposing ``synthe_py.main`` lazily."""

    from .cli import main as _main

    return _main(argv)


__all__ = ["main"]
