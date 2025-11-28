"""
Loss functions for virtual try-on training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vgg import VGG19, gram_matrix


class L1Loss(nn.Module):
    """
    Simple L1 reconstruction loss
    """
    def __init__(self):
        super(L1Loss, self).__init__()
        self.criterion = nn.L1Loss()
        
    def forward(self, pred, target, mask=None):
        """
        Args:
            pred: Predicted image (B, 3, H, W)
            target: Target image (B, 3, H, W)
            mask: Optional mask (B, 1, H, W)
        """
        if mask is not None:
            # Masked L1 loss (focus on specific regions)
            loss = torch.abs(pred - target) * mask
            return loss.sum() / (mask.sum() + 1e-6)
        else:
            return self.criterion(pred, target)


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19 features
    """
    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(Perceptual Loss, self).__init__()
        self.vgg = VGG19(requires_grad=False)
        self.criterion = nn.L1Loss()
        self.weights = weights
        
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def forward(self, pred, target):
        """
        Compute perceptual loss
        
        Args:
            pred: Predicted image (B, 3, H, W) in range [0, 1]
            target: Target image (B, 3, H, W) in range [0, 1]
        Returns:
            loss: Weighted sum of feature differences
        """
        # Normalize
        pred_norm = (pred - self.mean) / self.std
        target_norm = (target - self.mean) / self.std
        
        # Extract features
        pred_features = self.vgg(pred_norm)
        target_features = self.vgg(target_norm)
        
        # Compute loss at each layer
        loss = 0
        for i, (pred_feat, target_feat) in enumerate(zip(pred_features, target_features)):
            loss += self.weights[i] * self.criterion(pred_feat, target_feat.detach())
        
        return loss


class StyleLoss(nn.Module):
    """
    Style loss using Gram matrices
    """
    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(StyleLoss, self).__init__()
        self.vgg = VGG19(requires_grad=False)
        self.criterion = nn.L1Loss()
        self.weights = weights
        
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def forward(self, pred, target):
        """
        Compute style loss using Gram matrices
        
        Args:
            pred: Predicted image (B, 3, H, W) in range [0, 1]
            target: Target image (B, 3, H, W) in range [0, 1]
        Returns:
            loss: Weighted sum of Gram matrix differences
        """
        # Normalize
        pred_norm = (pred - self.mean) / self.std
        target_norm = (target - self.mean) / self.std
        
        # Extract features
        pred_features = self.vgg(pred_norm)
        target_features = self.vgg(target_norm)
        
        # Compute Gram matrices and loss
        loss = 0
        for i, (pred_feat, target_feat) in enumerate(zip(pred_features, target_features)):
            pred_gram = gram_matrix(pred_feat)
            target_gram = gram_matrix(target_feat.detach())
            loss += self.weights[i] * self.criterion(pred_gram, target_gram)
        
        return loss


class TotalVariationLoss(nn.Module):
    """
    Total Variation loss for smoothness regularization
    """
    def __init__(self):
        super(TotalVariationLoss, self).__init__()
        
    def forward(self, x):
        """
        Args:
            x: Image tensor (B, C, H, W)
        Returns:
            loss: TV loss value
        """
        batch_size, channels, height, width = x.size()
        
        # Horizontal differences
        h_tv = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).sum()
        
        # Vertical differences
        w_tv = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).sum()
        
        return (h_tv + w_tv) / (batch_size * channels * height * width)


class CompositeLoss(nn.Module):
    """
    Composite loss combining multiple loss functions
    """
    def __init__(self, config):
        super(CompositeLoss, self).__init__()
        
        self.l1_loss = L1Loss()
        self.perceptual_loss = PerceptualLoss()
        self.style_loss = StyleLoss()
        self.tv_loss = TotalVariationLoss()
        
        # Loss weights from config
        self.l1_weight = config.get('l1_weight', 1.0)
        self.perceptual_weight = config.get('perceptual_weight', 1.0)
        self.style_weight = config.get('style_weight', 100.0)
        self.tv_weight = config.get('tv_weight', 0.0001)
        
    def forward(self, pred, target, mask=None):
        """
        Compute weighted sum of all losses
        
        Args:
            pred: Predicted image (B, 3, H, W)
            target: Target image (B, 3, H, W)
            mask: Optional mask (B, 1, H, W)
        Returns:
            total_loss: Weighted sum
            loss_dict: Dictionary of individual losses
        """
        # Compute individual losses
        l1 = self.l1_loss(pred, target, mask)
        perceptual = self.perceptual_loss(pred, target)
        style = self.style_loss(pred, target)
        tv = self.tv_loss(pred)
        
        # Weighted sum
        total_loss = (
            self.l1_weight * l1 +
            self.perceptual_weight * perceptual +
            self.style_weight * style +
            self.tv_weight * tv
        )
        
        # Return individual losses for logging
        loss_dict = {
            'l1': l1.item(),
            'perceptual': perceptual.item(),
            'style': style.item(),
            'tv': tv.item(),
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict


class GMMSmoothnessLoss(nn.Module):
    """
    Smoothness loss for GMM transformation
    Encourages smooth TPS transformations
    """
    def __init__(self):
        super(GMMSmoothnessLoss, self).__init__()
        
    def forward(self, theta):
        """
        Args:
            theta: TPS control point offsets (B, K*K, 2)
        Returns:
            loss: Smoothness penalty
        """
        # Compute second-order differences for smoothness
        batch_size, num_points, coords = theta.size()
        grid_size = int(num_points ** 0.5)
        
        # Reshape to grid
        theta_grid = theta.view(batch_size, grid_size, grid_size, coords)
        
        # Horizontal smoothness
        h_diff = theta_grid[:, :, 1:, :] - theta_grid[:, :, :-1, :]
        h_smoothness = torch.abs(h_diff[:, :, 1:, :] - h_diff[:, :, :-1, :]).mean()
        
        # Vertical smoothness
        v_diff = theta_grid[:, 1:, :, :] - theta_grid[:, :-1, :, :]
        v_smoothness = torch.abs(v_diff[:, 1:, :, :] - v_diff[:, :-1, :, :]).mean()
        
        return h_smoothness + v_smoothness

