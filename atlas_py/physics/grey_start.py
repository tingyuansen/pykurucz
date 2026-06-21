"""Grey atmosphere cold-start generator.

Port of Fortran ATLAS12 ``CALCULATE`` deck card (atlas12.for label 700).

The grey starting model is the classic Eddington T-tau relation with
Kurucz's polynomial fit, fed through ``TTAUP`` (P-tau-kappa hydrostatic
coupling) using a stub ``ROSSTAB`` (returns 1.0 since the table is empty
on cold start).  This produces RHOX, T, P, ABROSS for an 80-layer model
that ATLAS12 then iterates from.

Fortran reference (kurucz/src/atlas12.for lines 1944-1966)::

      700 NRHOX=FREEFF(CARD)
          TAU1LG=FREEFF(CARD)
          STEPLG=FREEFF(CARD)
          DO 701 J=1,NRHOX
          TAUROS(J)=EXP10(TAU1LG+DFLOAT(J-1)*STEPLG)
      701 T(J)=TEFF*(.75*(.710+TAUROS(J)-.1331*EXP(-3.4488*TAUROS(J))))**.25
      702 DO 703 J=1,NRHOX
          XNE(J)=0.
          PRADK(J)=2.521D-15*DMAX1(T(J)**4,TEFF**4/2.)
          VTURB(J)=0.
      703 PTURB(J)=0.
          PRADK0=PRADK(1)
          PCON=0.
          PTURB0=0.
          PZERO=PRADK0
          DO 704 J=1,NRHOX
      704 PRAD(J)=PRADK(J)-PRADK0
          CALL TTAUP(T,TAUROS,ABROSS,PTOTAL,P,PRAD,PTURB,VTURB,GRAV,NRHOX)
          DO 705 J=1,NRHOX
          RHOX(J)=PTOTAL(J)/GRAV
      705 PTOTAL(J)=PTOTAL(J)+PZERO
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tcorr import TcorrState, _ttaup, init_tcorr


@dataclass
class GreyAtmosphere:
    """Output of the grey CALCULATE pass.

    Columns ordered to match the emulator/`write_atm_file` 9-column layout:
    RHOX, T, P, XNE, ABROSS, ACCRAD, VTURB, FLXCNV, VCONV.
    """

    tauros: np.ndarray
    rhox: np.ndarray
    temperature: np.ndarray
    p_gas: np.ndarray
    xne: np.ndarray
    abross: np.ndarray
    accrad: np.ndarray
    vturb: np.ndarray
    flxcnv: np.ndarray
    vconv: np.ndarray
    pradk: np.ndarray

    def as_deck6(self) -> np.ndarray:
        """Return 9-column ``data`` array compatible with ``write_atm_file``."""
        return np.column_stack(
            [
                self.rhox,
                self.temperature,
                self.p_gas,
                self.xne,
                self.abross,
                self.accrad,
                self.vturb,
                self.flxcnv,
                self.vconv,
            ]
        )


def grey_t_tau(teff: float, tauros: np.ndarray) -> np.ndarray:
    """Eddington-Kurucz grey T(tau).

    Fortran: ``T(J)=TEFF*(.75*(.710+TAU-.1331*EXP(-3.4488*TAU)))**.25``.
    """
    tau = np.asarray(tauros, dtype=np.float64)
    bracket = 0.75 * (0.710 + tau - 0.1331 * np.exp(-3.4488 * tau))
    return float(teff) * np.power(np.maximum(bracket, 1e-300), 0.25)


def generate_grey_atm(
    *,
    teff: float,
    logg: float,
    nrhox: int = 80,
    tau1lg: float = -6.875,
    steplg: float = 0.125,
) -> GreyAtmosphere:
    """Build a grey starting atmosphere for ATLAS12.

    Parameters mirror the Fortran ``CALCULATE`` card defaults
    (atlas12.for lines 1729-1731): ``TAU1LG=-6.875``, ``STEPLG=0.125``,
    ``NRHOX=80``.

    Notes
    -----
    The TTAUP pass is run with an *empty* ROSSTAB table (the same way
    Fortran starts: NROSS=0).  ``rosstab_eval`` returns 1.0 in that
    regime, so ABROSS is effectively unity except for the seed
    ``ABSTD[0]=0.1`` from the Fortran TTAUP boundary clause.  This is the
    intentional crude-opacity bootstrap; the proper ABROSS is built up
    by ROSSTAB over subsequent ATLAS iterations.
    """
    if nrhox <= 0:
        raise ValueError("nrhox must be positive")
    gravity_cgs = 10.0 ** float(logg)

    j = np.arange(nrhox, dtype=np.float64)
    tauros = np.power(10.0, tau1lg + j * steplg)
    temperature = grey_t_tau(teff, tauros)

    pradk = 2.521e-15 * np.maximum(
        np.power(temperature, 4.0),
        (float(teff) ** 4) / 2.0,
    )
    pradk0 = float(pradk[0])
    prad = pradk - pradk0
    pturb = np.zeros(nrhox, dtype=np.float64)
    vturb = np.zeros(nrhox, dtype=np.float64)
    xne = np.zeros(nrhox, dtype=np.float64)

    st: TcorrState = init_tcorr(nrhox)
    abross, ptotal, p_gas = _ttaup(
        st=st,
        t=temperature,
        tau=tauros,
        prad=prad,
        pturb=pturb,
        gravity_cgs=gravity_cgs,
    )

    rhox = ptotal / gravity_cgs

    return GreyAtmosphere(
        tauros=tauros,
        rhox=rhox,
        temperature=temperature,
        p_gas=p_gas,
        xne=xne,
        abross=abross,
        accrad=np.zeros(nrhox, dtype=np.float64),
        vturb=vturb,
        flxcnv=np.zeros(nrhox, dtype=np.float64),
        vconv=np.zeros(nrhox, dtype=np.float64),
        pradk=pradk,
    )


def write_grey_atm_file(
    dest,
    *,
    teff: float,
    logg: float,
    mh: float = 0.0,
    am: float = 0.0,
    abundances: dict[int, float] | None = None,
    nrhox: int = 80,
    tau1lg: float = -6.875,
    steplg: float = 0.125,
    vturb_kms: float = 2.0,
    title: str | None = None,
):
    """Generate a grey atmosphere and write it as a Kurucz ``.atm`` file.

    The ``vturb_kms`` argument is recorded in the header only; the layer
    structure VTURB column stays at 0 (matching Fortran CALCULATE).
    """
    from pathlib import Path
    from pykurucz import write_atm_file

    grey = generate_grey_atm(
        teff=teff,
        logg=logg,
        nrhox=nrhox,
        tau1lg=tau1lg,
        steplg=steplg,
    )
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_atm_file(
        dest,
        teff,
        logg,
        grey.as_deck6(),
        vturb_kms,
        mh=mh,
        am=am,
        individual=abundances,
        title=(
            title
            if title is not None
            else f"grey CALCULATE start (atlas12.for label 700) NRHOX={nrhox}"
        ),
    )
    return dest
