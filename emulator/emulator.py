#!/usr/bin/env python
"""
Atmosphere emulator for Kurucz stellar atmospheres.

This module provides a neural network-based emulator that predicts atmospheric 
structure from 4D stellar parameters (Teff, logg, [Fe/H], [α/Fe]).
"""

import torch
import numpy as np
from pathlib import Path

from .model import AtmosphereNetMLPtau
from .normalization import NormalizationHelper, load_norm_params


# Default paths to model files (relative to this module)
EMULATOR_DIR = Path(__file__).parent.resolve()
DEFAULT_WEIGHTS = EMULATOR_DIR / 'a_one_weights.pt'
DEFAULT_NORM_PARAMS = EMULATOR_DIR / 'norm_params.pt'


class AtmosphereEmulator:
    """
    Emulator for Kurucz stellar atmosphere models.
    
    This class provides an interface to predict atmospheric structure
    based on stellar parameters and optical depth using a pre-trained
    neural network model.
    
    Input: 4D stellar parameters (Teff, logg, [Fe/H], [α/Fe])
    Output: 80 depth points × 6 quantities (RHOX, T, P, XNE, ABROSS, ACCRAD)
    """
    
    def __init__(self, model, normalizer, default_tau_grid=None, device='cpu'):
        """
        Initialize the emulator with a pre-trained model and normalization helper.
        
        Parameters:
            model (torch.nn.Module): Pre-trained neural network model
            normalizer (NormalizationHelper): Normalization helper object
            default_tau_grid (torch.Tensor, optional): Default optical depth grid
            device (str): Device to run the model on ('cpu' or 'cuda')
        """
        self.model = model
        self.normalizer = normalizer
        self.default_tau_grid = default_tau_grid
        self.device = device
        self.model.to(device)
        self.model.eval()  # Set model to evaluation mode
        
        # Create default tau grid if none provided
        if self.default_tau_grid is None:
            self.default_tau_grid = torch.logspace(-6, 2, 80)  # 80 points from 1e-6 to 100
        
    def predict(self, teff, logg, feh, afe, tau_grid=None):
        """
        Predict atmospheric structure for given stellar parameters.
        
        Parameters:
            teff (float): Effective temperature in K
            logg (float): Surface gravity (log10 cgs)
            feh (float): Metallicity [Fe/H]
            afe (float): Alpha enhancement [α/Fe]
            tau_grid (array-like, optional): Optical depth grid (80 points)
                                            If None, uses the default grid
        
        Returns:
            dict: Dictionary containing atmospheric parameters for each depth point
                 Keys: 'RHOX', 'T', 'P', 'XNE', 'ABROSS', 'ACCRAD', 'TAU'
                 Each is a numpy array of shape (80,)
        """
        # Create stellar params tensor
        stellar_params = torch.tensor([[teff, logg, feh, afe]], dtype=torch.float32)
        stellar_params = stellar_params.to(self.device)
        
        # Use default tau grid if none provided
        if tau_grid is None:
            tau_grid = self.default_tau_grid.unsqueeze(0)
        else:
            if not isinstance(tau_grid, torch.Tensor):
                tau_grid = torch.tensor(tau_grid, dtype=torch.float32)
            if tau_grid.dim() == 1:
                tau_grid = tau_grid.unsqueeze(0)
                
        tau_grid = tau_grid.to(self.device)
        
        # Store original tau grid for output
        original_tau_grid = tau_grid.clone()
        
        # Normalize all parameters
        teff_norm = self.normalizer.normalize('teff', stellar_params[:, 0:1])
        logg_norm = self.normalizer.normalize('gravity', stellar_params[:, 1:2])
        feh_norm = self.normalizer.normalize('feh', stellar_params[:, 2:3])
        afe_norm = self.normalizer.normalize('afe', stellar_params[:, 3:4])
        tau_norm = self.normalizer.normalize('TAU', tau_grid)
        
        # Combine normalized parameters
        params_normalized = torch.cat([
            teff_norm,
            logg_norm,
            feh_norm,
            afe_norm,
            tau_norm
        ], dim=1)
        
        # Get predictions
        with torch.no_grad():
            predictions = self.model(params_normalized)
        
        # Reshape predictions (batch_size=1, depth_points=80, 6 features)
        predictions = predictions.view(1, 80, 6)
        
        # Denormalize predictions
        output = {
            'RHOX': self.normalizer.denormalize('RHOX', predictions[:, :, 0]).cpu().numpy()[0],
            'T': self.normalizer.denormalize('T', predictions[:, :, 1]).cpu().numpy()[0],
            'P': self.normalizer.denormalize('P', predictions[:, :, 2]).cpu().numpy()[0],
            'XNE': self.normalizer.denormalize('XNE', predictions[:, :, 3]).cpu().numpy()[0],
            'ABROSS': self.normalizer.denormalize('ABROSS', predictions[:, :, 4]).cpu().numpy()[0],
            'ACCRAD': self.normalizer.denormalize('ACCRAD', predictions[:, :, 5]).cpu().numpy()[0],
            'TAU': original_tau_grid.cpu().numpy()[0],
        }
        
        return output
    
    def predict_atmosphere_data(self, teff, logg, feh, afe, vturb=2.0, tau_grid=None):
        """
        Predict the 80x9 atmosphere data array for use in .atm files.
        
        Parameters:
            teff (float): Effective temperature in K
            logg (float): Surface gravity (log10 cgs)
            feh (float): Metallicity [Fe/H]
            afe (float): Alpha enhancement [α/Fe]
            vturb (float): Microturbulent velocity in km/s (default: 2.0)
            tau_grid (array-like, optional): TAU grid from closest atmosphere.
                     IMPORTANT: Must use TAU from a real atmosphere for accurate
                     predictions. The MLP was trained with TAU calculated from
                     each atmosphere, not a fixed grid.
        
        Returns:
            numpy.ndarray: 80x9 atmosphere data array with columns:
                          RHOX, T, P, XNE, ABROSS, ACCRAD, VTURB, 0, 0
        """
        # Get predictions from model using provided TAU grid
        pred = self.predict(teff, logg, feh, afe, tau_grid=tau_grid)
        
        # Build 80x9 array
        data = np.zeros((80, 9))
        data[:, 0] = pred['RHOX']      # Column 1: RHOX (mass column density)
        data[:, 1] = pred['T']          # Column 2: Temperature
        data[:, 2] = pred['P']          # Column 3: Gas pressure
        data[:, 3] = pred['XNE']        # Column 4: Electron number density
        data[:, 4] = pred['ABROSS']     # Column 5: Rosseland mean opacity
        data[:, 5] = pred['ACCRAD']     # Column 6: Radiative acceleration
        data[:, 6] = vturb * 1e5        # Column 7: Vturb (km/s -> cm/s)
        data[:, 7] = 0.0                # Column 8: Convective velocity (set to 0)
        data[:, 8] = 0.0                # Column 9: Convective flux ratio (set to 0)
        
        return data


def load_emulator(weights_path=None, norm_params_path=None, device='cpu'):
    """
    Load the pre-trained atmosphere emulator.
    
    Parameters:
        weights_path (str or Path, optional): Path to model weights file.
                                              If None, uses bundled weights.
        norm_params_path (str or Path, optional): Path to normalization parameters.
                                                  If None, uses bundled params.
        device (str): Device to load the model on ('cpu' or 'cuda')
    
    Returns:
        AtmosphereEmulator: Initialized emulator object
    """
    if weights_path is None:
        weights_path = DEFAULT_WEIGHTS
    if norm_params_path is None:
        norm_params_path = DEFAULT_NORM_PARAMS
        
    weights_path = Path(weights_path)
    norm_params_path = Path(norm_params_path)
    
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    if not norm_params_path.exists():
        raise FileNotFoundError(f"Normalization params not found: {norm_params_path}")
    
    # Load normalization parameters
    norm_params = load_norm_params(str(norm_params_path))
    normalizer = NormalizationHelper(norm_params)
    
    # Load model
    model = AtmosphereNetMLPtau(
        stellar_embed_dim=512, tau_embed_dim=512
    ).to(device)
    
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return AtmosphereEmulator(model, normalizer, device=device)

