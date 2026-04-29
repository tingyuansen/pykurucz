#!/usr/bin/env python
"""
Normalization utilities for the atmosphere emulator.
"""

import torch


def load_norm_params(norm_params_path):
    """
    Load normalization parameters from a standalone file.
    
    Parameters:
        norm_params_path (str): Path to the normalization parameters file
        
    Returns:
        dict: Normalization parameters
    """
    return torch.load(norm_params_path, map_location='cpu', weights_only=True)


class NormalizationHelper:
    """
    Helper class for normalizing and denormalizing data using saved parameters.
    """
    
    def __init__(self, norm_params):
        """
        Initialize with normalization parameters.
        
        Parameters:
            norm_params (dict): Normalization parameters
        """
        self.norm_params = norm_params
        
    def normalize(self, param_name, data):
        """
        Normalize data to [-1, 1] range with optional log transform.

        Parameters:
            param_name (str): Name of the parameter to normalize
            data (torch.Tensor): Data to normalize

        Returns:
            torch.Tensor: Normalized data in [-1, 1] range
        """
        params = self.norm_params[param_name]
        
        # Apply log transform if needed
        if params['log_scale']:
            transformed_data = torch.log10(data + 1e-30)
        else:
            transformed_data = data
        
        # Apply min-max scaling to [-1, 1] range
        normalized = 2.0 * (transformed_data - params['min']) / (params['max'] - params['min']) - 1.0
        
        return normalized

    def denormalize(self, param_name, normalized_data):
        """
        Denormalize data from [-1, 1] range back to original scale.

        Parameters:
            param_name (str): Name of the parameter to denormalize
            normalized_data (torch.Tensor): Normalized data to convert back

        Returns:
            torch.Tensor: Denormalized data with gradients preserved
        """
        params = self.norm_params[param_name]
        
        # Reverse min-max scaling from [-1, 1] range
        transformed_data = (normalized_data + 1.0) / 2.0 * (params['max'] - params['min']) + params['min']
        
        # Reverse log transform if needed
        if params['log_scale']:
            denormalized_data = torch.pow(10.0, transformed_data) - 1e-30
        else:
            denormalized_data = transformed_data
        
        return denormalized_data

