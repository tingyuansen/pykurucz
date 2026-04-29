"""ATLAS12 neural network emulator (kurucz-a1).

Predicts stellar atmospheric structure from 4 parameters
(Teff, logg, [Fe/H], [alpha/Fe]) using a pre-trained MLP.

Requires PyTorch (optional dependency).

Reference:
    Li, Jian, Ting & Green (2025), "Differentiable Stellar Atmospheres
    with Physics-Informed Neural Networks", arXiv:2507.06357
"""

__version__ = "0.1.0"

from .emulator import AtmosphereEmulator, load_emulator

__all__ = ["AtmosphereEmulator", "load_emulator"]
