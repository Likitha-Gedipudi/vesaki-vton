"""
VGG19 network for perceptual and style loss computation
"""

import torch
import torch.nn as nn
import torchvision.models as models


class VGG19(nn.Module):
    """
    VGG19 feature extractor for perceptual loss
    Uses pre-trained ImageNet weights
    """
    def __init__(self, requires_grad=False):
        super(VGG19, self).__init__()
        
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        
        # Split VGG19 into different feature levels
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        
        for x in range(2):  # relu1_2
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):  # relu2_2
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):  # relu3_2
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):  # relu4_2
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):  # relu5_2
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
                
    def forward(self, x):
        """
        Forward pass through VGG19
        Returns features from multiple layers
        """
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]


class VGGLoss(nn.Module):
    """
    Perceptual loss using VGG19 features
    """
    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19(requires_grad=False)
        self.criterion = nn.L1Loss()
        self.weights = weights
        
        # Normalization for ImageNet
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def forward(self, x, y):
        """
        Compute perceptual loss between x and y
        
        Args:
            x: Generated image (B, 3, H, W) in range [0, 1]
            y: Target image (B, 3, H, W) in range [0, 1]
        Returns:
            loss: Perceptual loss value
        """
        # Normalize to ImageNet mean and std
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        
        x_vgg = self.vgg(x)
        y_vgg = self.vgg(y)
        
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
            
        return loss


def gram_matrix(features):
    """
    Compute Gram matrix for style loss
    
    Args:
        features: Feature map (B, C, H, W)
    Returns:
        gram: Gram matrix (B, C, C)
    """
    B, C, H, W = features.size()
    features = features.view(B, C, H * W)
    gram = torch.bmm(features, features.transpose(1, 2))
    gram = gram / (C * H * W)
    return gram

