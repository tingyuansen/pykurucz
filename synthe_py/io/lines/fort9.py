"""Helpers for fort.9 line opacity archives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class Fort9Data:
    """Container for the fort.9 (ALINEC + metadata) payload."""

    wavelength: np.ndarray
    alinec: np.ndarray  # shape: (n_lines, depth)
    lindat4: np.ndarray  # (n_lines, 28) float32 records
    lindat8: np.ndarray  # (n_lines, 14) float64 records
    n_layers: int
    length: int
    wlbeg: float
    wlend: float
    resolution: float
    linout: int
    turbv: float
    ifvac: int
    stride: int
    continuum_edges: bytes

    @property
    def n_lines(self) -> int:
        return self.wavelength.size

    @property
    def lindat4_int(self) -> np.ndarray:
        """Raw view of LINDAT4 as signed 32-bit integers."""

        return self.lindat4.view(np.int32).reshape(self.lindat4.shape)

    def get_field_float(self, index: int) -> np.ndarray:
        """Return the column from LINDAT4 interpreted as float32."""

        return self.lindat4[:, index]

    def get_field_int(self, index: int) -> np.ndarray:
        """Return the column from LINDAT4 interpreted as int32."""

        return self.lindat4_int[:, index]


@dataclass(frozen=True)
class Fort9Metadata:
    """Decoded scalar metadata aligned with the fort.9 wavelength grid."""

    nelion: np.ndarray
    nblo: np.ndarray
    nbup: np.ndarray
    code: np.ndarray
    ncon: np.ndarray
    nelionx: np.ndarray
    excitation_lower: np.ndarray
    gf: np.ndarray
    gamma_rad: np.ndarray
    gamma_stark: np.ndarray
    gamma_vdw: np.ndarray
    gf_log: np.ndarray
    extra1: np.ndarray
    extra2: np.ndarray
    extra3: np.ndarray


_CODEX = np.array(
    [
        1.0,
        2.0,
        2.01,
        6.0,
        6.01,
        12.0,
        12.01,
        13.0,
        13.01,
        14.0,
        14.01,
        20.0,
        20.01,
        8.0,
        11.0,
        5.0,
        19.0,
    ],
    dtype=np.float64,
)


def decode_metadata(data: Fort9Data) -> Fort9Metadata:
    """Decode typed metadata arrays from a :class:`Fort9Data` instance."""

    nelion = data.get_field_int(0)
    gamma_rad = data.get_field_float(1)
    gamma_stark = data.get_field_float(2)
    gamma_vdw = data.get_field_float(3)
    nblo = data.get_field_int(5)
    nbup = data.get_field_int(6)
    iso1 = data.get_field_int(7)
    iso2 = data.get_field_int(9)
    code = data.get_field_float(14)
    e_lower = data.get_field_float(15)
    gf = data.get_field_float(16)
    gf_log = data.get_field_float(11)
    extra1 = data.get_field_float(25)
    extra2 = data.get_field_float(26)
    extra3 = data.get_field_float(27)

    # Continuum index (NCON) only defined when ISO1 == 0 in the legacy tapes.
    ncon = np.where(iso1 == 0, iso2, 0)

    # Map CODE onto the discrete CODEX table (1..17). Values not in CODEX map to 0.
    diff = np.abs(code[:, None] - _CODEX[None, :])
    nelionx = diff.argmin(axis=1) + 1
    mask_unmatched = np.min(diff, axis=1) > 1e-6
    nelionx[mask_unmatched] = 0

    return Fort9Metadata(
        nelion=nelion,
        nblo=nblo,
        nbup=nbup,
        code=code,
        ncon=ncon,
        nelionx=nelionx,
        excitation_lower=e_lower,
        gf=gf,
        gamma_rad=gamma_rad,
        gamma_stark=gamma_stark,
        gamma_vdw=gamma_vdw,
        gf_log=gf_log,
        extra1=extra1,
        extra2=extra2,
        extra3=extra3,
    )


def load(path: Path) -> Fort9Data:
    """Load a fort.9 archive produced by :mod:`convert_fort9`."""

    with np.load(path, allow_pickle=False) as data:
        wavelength = np.asarray(data["wavelength"], dtype=np.float64)
        alinec = np.asarray(data["alinec"], dtype=np.float64)
        lindat4 = np.asarray(data["lindat4"], dtype=np.float32)
        lindat8 = np.asarray(data["lindat8"], dtype=np.float64)
        wlbeg = float(data["wlbeg"])
        wlend = float(data["wlend"])
        resolution = float(data["resolu"])
        length = int(data["length"])
        n_layers = int(data["nrhox"])
        linout = int(data["linout"])
        turbv = float(data["turbv"])
        ifvac = int(data["ifvac"])
        stride = int(data["stride"])
        continuum_edges = bytes(data["continuum_edges"].tobytes())

    if alinec.shape[0] != wavelength.size:
        raise ValueError("Mismatch between fort.9 wavelength and ALINEC shape")
    if lindat4.shape[0] != wavelength.size or lindat8.shape[0] != wavelength.size:
        raise ValueError("Mismatch between fort.9 metadata and wavelength grid")

    return Fort9Data(
        wavelength=wavelength,
        alinec=alinec,
        lindat4=lindat4,
        lindat8=lindat8,
        n_layers=n_layers,
        length=length,
        wlbeg=wlbeg,
        wlend=wlend,
        resolution=resolution,
        linout=linout,
        turbv=turbv,
        ifvac=ifvac,
        stride=stride,
        continuum_edges=continuum_edges,
    )

