"""
Geometric Matching Module (GMM)
Learns spatial transformation to align garment with target person pose
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .networks import FeatureExtraction, CorrelationLayer, TPSGridGen, ResidualBlock


class GeometricMatchingModule(nn.Module):
    """
    GMM: Aligns garment image with person pose using Thin Plate Spline transformation
    
    Architecture:
    1. Feature extraction from person representation and garment
    2. Correlation layer to find correspondences
    3. Regression network to predict TPS parameters
    4. Grid generation and warping
    """
    def __init__(self, input_nc=22, feature_dim=512, num_points=5):
        super(GeometricMatchingModule, self).__init__()
        
        self.num_points = num_points
        self.feature_dim = feature_dim
        
        # Feature extraction networks
        # Person representation has many channels (pose + mask + agnostic)
        self.feature_extraction_person = FeatureExtraction(input_nc=input_nc, ngf=64, n_layers=4)
        # Garment is just RGB (3 channels)
        self.feature_extraction_garment = FeatureExtraction(input_nc=3, ngf=64, n_layers=4)
        
        # Correlation layer
        self.correlation = CorrelationLayer(in_channels=feature_dim, max_displacement=4)
        
        # Regression network to predict control points
        # Input: correlation features
        # Output: TPS control point offsets
        self.regression = nn.Sequential(
            nn.Conv2d(self.correlation.out_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_points * num_points * 2)  # Grid of control points
        )
        
        # Initialize TPS grid
        self.grid_size = num_points
        gridY = torch.linspace(-1, 1, steps=self.grid_size)
        gridX = torch.linspace(-1, 1, steps=self.grid_size)
        gridY, gridX = torch.meshgrid(gridY, gridX, indexing='ij')
        target_control_points = torch.stack([gridX.flatten(), gridY.flatten()], dim=1)
        self.register_buffer('target_control_points', target_control_points)
        
    def forward(self, person_repr, garment, garment_mask=None):
        """
        Forward pass of GMM
        
        Args:
            person_repr: Person representation (B, C, H, W) 
                        Concatenated (pose + body_mask + agnostic)
            garment: Garment image (B, 3, H, W)
            garment_mask: Garment mask (B, 1, H, W), optional
        Returns:
            warped_garment: Spatially transformed garment (B, 3, H, W)
            theta: TPS transformation parameters (B, K*K, 2)
        """
        batch_size, _, H, W = garment.size()
        
        # Extract features
        feat_person = self.feature_extraction_person(person_repr)
        feat_garment = self.feature_extraction_garment(garment)
        
        # Compute correlation
        correlation = self.correlation(feat_person, feat_garment)
        
        # Predict control point offsets
        theta = self.regression(correlation)
        theta = theta.view(batch_size, self.grid_size * self.grid_size, 2)
        
        # Add offsets to target control points
        source_control_points = self.target_control_points.unsqueeze(0).expand(batch_size, -1, -1)
        source_control_points = source_control_points + theta * 0.1  # Scale offsets
        
        # Generate TPS grid
        tps_grid = TPSGridGen(H, W, self.target_control_points)
        sampling_grid = tps_grid(source_control_points)
        
        # Warp garment
        warped_garment = F.grid_sample(
            garment, 
            sampling_grid, 
            mode='bilinear', 
            padding_mode='border',
            align_corners=True
        )
        
        # Also warp mask if provided
        if garment_mask is not None:
            warped_mask = F.grid_sample(
                garment_mask,
                sampling_grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            )
            return warped_garment, warped_mask, theta
        
        return warped_garment, theta


class GMMWithRefinement(nn.Module):
    """
    GMM with additional refinement network for better detail preservation
    """
    def __init__(self, input_nc=3, feature_dim=512, num_points=5):
        super(GMMWithRefinement, self).__init__()
        
        self.gmm = GeometricMatchingModule(input_nc, feature_dim, num_points)
        
        # Refinement network (residual blocks)
        self.refinement = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, padding=1),  # Input: warped + original
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            ResidualBlock(64),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, person_repr, garment, garment_mask=None):
        """
        GMM with refinement
        """
        # Coarse warping
        if garment_mask is not None:
            warped_garment, warped_mask, theta = self.gmm(person_repr, garment, garment_mask)
        else:
            warped_garment, theta = self.gmm(person_repr, garment)
            warped_mask = None
        
        # Refinement
        refinement_input = torch.cat([warped_garment, garment], dim=1)
        residual = self.refinement(refinement_input)
        refined_garment = warped_garment + residual * 0.1  # Small residual
        refined_garment = torch.clamp(refined_garment, 0, 1)
        
        if warped_mask is not None:
            return refined_garment, warped_mask, theta
        return refined_garment, theta

