"""
Loss functions for Vesaki-VTON training
"""

from .losses import (
    L1Loss,
    PerceptualLoss,
    StyleLoss,
    TotalVariationLoss,
    CompositeLoss
)

__all__ = [
    'L1Loss',
    'PerceptualLoss',
    'StyleLoss',
    'TotalVariationLoss',
    'CompositeLoss'
]

