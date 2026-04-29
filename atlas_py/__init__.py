"""Python reimplementation scaffold for Kurucz ATLAS12."""

from .config import AtlasConfig
from .engine.driver import run_atlas

__all__ = ["AtlasConfig", "run_atlas"]

