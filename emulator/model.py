#!/usr/bin/env python
"""
Neural network model for Kurucz stellar atmosphere prediction.

This model takes 4D stellar parameters (Teff, logg, [Fe/H], [α/Fe]) plus 
optical depth tau values and predicts 6D atmospheric structure 
(RHOX, T, P, XNE, ABROSS, ACCRAD).
"""

import torch
import torch.nn as nn


class StellarParamEncoder(nn.Module):
    """Encoder for global stellar parameters (Teff, logg, [M/H], [alpha/Fe])"""
    def __init__(self, input_dim=4, embed_dim=128):
        super(StellarParamEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        
    def forward(self, x):
        # x shape: (batch_size, 4)
        return self.encoder(x)  # output: (batch_size, embed_dim)


class TauPositionEncoder(nn.Module):
    """Position encoder for tau values at each depth point"""
    def __init__(self, embed_dim=64, depth_points=80):
        super(TauPositionEncoder, self).__init__()
        self.depth_points = depth_points
        self.encoder = nn.Sequential(
            nn.Linear(1, embed_dim//2),
            nn.GELU(),
            nn.Linear(embed_dim//2, embed_dim),
            nn.GELU()
        )
        
    def forward(self, tau):
        # tau shape: (batch_size, depth_points)
        batch_size = tau.size(0)
        # Reshape to process each tau value independently
        tau_reshaped = tau.reshape(batch_size * self.depth_points, 1)
        # Encode each tau value
        encoded = self.encoder(tau_reshaped)
        # Reshape back to separate batch and depth dimensions
        return encoded.reshape(batch_size, self.depth_points, -1)


class AtmosphereNetMLPtau(nn.Module):
    """
    MLP-based atmosphere model with separate encoders for stellar params and tau.
    
    Input: 4 stellar parameters + 80 tau values
    Output: 80 depth points × 6 atmospheric quantities
    """
    def __init__(self, output_size=6, depth_points=80, 
                 stellar_embed_dim=128, tau_embed_dim=64):
        super(AtmosphereNetMLPtau, self).__init__()
        self.depth_points = depth_points
        self.output_size = output_size
        
        # Encoders
        self.stellar_encoder = StellarParamEncoder(input_dim=4, embed_dim=stellar_embed_dim)
        self.tau_encoder = TauPositionEncoder(embed_dim=tau_embed_dim, depth_points=depth_points)
        
        # Combined embedding dimension
        combined_dim = stellar_embed_dim + tau_embed_dim
        
        # Atmospheric parameter predictor network
        self.predictor = nn.Sequential(
            nn.Linear(combined_dim, combined_dim*2),
            nn.GELU(),
            nn.Dropout(0.01),
            nn.Linear(combined_dim*2, combined_dim*2),
            nn.GELU(),
            nn.Dropout(0.01),
            nn.Linear(combined_dim*2, output_size)
        )
        
    def forward(self, x):
        # Extract stellar parameters and tau values
        # x shape: (batch_size, 84)
        batch_size = x.size(0)
        stellar_params = x[:, :4]  # (batch_size, 4)
        tau_values = x[:, 4:].reshape(batch_size, self.depth_points)  # (batch_size, depth_points)
        
        # Encode stellar parameters
        stellar_embedding = self.stellar_encoder(stellar_params)  # (batch_size, stellar_embed_dim)
        
        # Encode tau values
        tau_embedding = self.tau_encoder(tau_values)  # (batch_size, depth_points, tau_embed_dim)
        
        # Expand stellar embedding to match depth points dimension
        stellar_embedding = stellar_embedding.unsqueeze(1).expand(-1, self.depth_points, -1)
        
        # Combine embeddings
        combined = torch.cat([stellar_embedding, tau_embedding], dim=2)
        
        # Generate atmospheric parameters for each depth point
        outputs = self.predictor(combined)  # (batch_size, depth_points, output_size)
        
        return outputs

