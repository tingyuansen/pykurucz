from __future__ import annotations

from typing import Dict

import numpy as np


def initialize_departure_tables(n_layers: int) -> Dict[str, np.ndarray]:
    """
    Create in-memory departure coefficient tables matching Fortran COMMON blocks.

    All arrays are initialized to 1.0, matching the DATA statements in atlas7v.
    """

    def ones(columns: int) -> np.ndarray:
        return np.ones((n_layers, columns), dtype=np.float64)

    tables: Dict[str, np.ndarray] = {
        "bhyd": ones(8),
        "bhe1": ones(29),
        "bhe2": ones(6),
        "bc1": ones(14),
        "bc2": ones(6),
        "bmg1": ones(11),
        "bmg2": ones(6),
        "bal1": ones(9),
        "bsi1": ones(11),
        "bsi2": ones(10),
        "bca1": ones(8),
        "bca2": ones(5),
        "bo1": ones(13),
        "bna1": ones(8),
        "bb1": ones(7),
        "bk1": ones(8),
        "bmin": np.ones((n_layers,), dtype=np.float64),
    }

    return tables

