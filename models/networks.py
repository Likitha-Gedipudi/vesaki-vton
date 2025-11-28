"""
Shared network components and building blocks for Vesaki-VTON
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtraction(nn.Module):
    """
    Feature extraction network using modified ResNet-style architecture
    Extracts multi-scale features from input images
    """
    def __init__(self, input_nc=3, ngf=64, n_layers=5):
        super(FeatureExtraction, self).__init__()
        
        # Initial convolution
        layers = [
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling layers
        for i in range(n_layers):
            mult = 2 ** i
            layers += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(ngf * mult * 2),
                nn.ReLU(inplace=True)
            ]
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)


class FeaturePyramid(nn.Module):
    """
    Feature Pyramid Network for multi-scale feature extraction
    """
    def __init__(self, input_nc=3, ngf=64):
        super(FeaturePyramid, self).__init__()
        
        # Encoder (downsampling)
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True)
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True)
        )
        
        self.enc4 = nn.Sequential(
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """
        Returns multi-scale features
        """
        e1 = self.enc1(x)  # 1/1 scale
        e2 = self.enc2(e1)  # 1/2 scale
        e3 = self.enc3(e2)  # 1/4 scale
        e4 = self.enc4(e3)  # 1/8 scale
        
        return [e1, e2, e3, e4]


class ResidualBlock(nn.Module):
    """
    Residual block with batch normalization
    """
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(ResidualBlock, self).__init__()
        
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(f'padding {padding_type} is not implemented')
        
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p),
            norm_layer(dim),
            nn.ReLU(True)
        ]
        
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p),
            norm_layer(dim)
        ]
        
        self.conv_block = nn.Sequential(*conv_block)
        
    def forward(self, x):
        return x + self.conv_block(x)


class CorrelationLayer(nn.Module):
    """
    Correlation layer for geometric matching
    Computes correlation between two feature maps
    """
    def __init__(self, in_channels, kernel_size=1, max_displacement=20):
        super(CorrelationLayer, self).__init__()
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = 1
        self.stride2 = 1
        self.pad_size = max_displacement
        
        # Output channels: (2 * max_displacement + 1) ** 2
        self.out_channels = (2 * max_displacement + 1) ** 2
        
    def forward(self, x1, x2):
        """
        Args:
            x1: Feature map 1 (B, C, H, W)
            x2: Feature map 2 (B, C, H, W)
        Returns:
            correlation: Correlation map (B, D*D, H, W) where D = 2*max_displacement+1
        """
        batch_size, channels, height, width = x1.size()
        
        # Pad x2
        x2 = F.pad(x2, [self.pad_size] * 4)
        
        # Compute correlation
        correlation = []
        for i in range(-self.max_displacement, self.max_displacement + 1):
            for j in range(-self.max_displacement, self.max_displacement + 1):
                # Shifted version of x2
                x2_shift = x2[:, :, 
                              self.pad_size + i:self.pad_size + i + height,
                              self.pad_size + j:self.pad_size + j + width]
                
                # Element-wise multiplication and sum over channel dimension
                corr = (x1 * x2_shift).mean(dim=1, keepdim=True)
                correlation.append(corr)
        
        # Stack along channel dimension
        correlation = torch.cat(correlation, dim=1)
        
        return correlation


class TPSGridGen(nn.Module):
    """
    Thin Plate Spline (TPS) grid generator
    Generates sampling grid for spatial transformer based on TPS transformation
    """
    def __init__(self, target_height, target_width, target_control_points):
        super(TPSGridGen, self).__init__()
        
        self.num_points = target_control_points.size(0)
        self.target_height = target_height
        self.target_width = target_width
        
        # Create padded kernel matrix
        forward_kernel = torch.zeros(self.num_points + 3, self.num_points + 3)
        target_control_partial_repr = self.compute_partial_repr(target_control_points, target_control_points)
        forward_kernel[:self.num_points, :self.num_points].copy_(target_control_partial_repr)
        forward_kernel[:self.num_points, -3].fill_(1)
        forward_kernel[-3, :self.num_points].fill_(1)
        forward_kernel[:self.num_points, -2:].copy_(target_control_points)
        forward_kernel[-2:, :self.num_points].copy_(target_control_points.transpose(0, 1))
        
        # Compute inverse matrix
        inverse_kernel = torch.inverse(forward_kernel)
        
        # Create target coordinate matrix
        HW = target_height * target_width
        target_coordinate = list(itertools.product(range(target_height), range(target_width)))
        target_coordinate = torch.Tensor(target_coordinate)  # HW x 2
        Y, X = target_coordinate.split(1, dim=1)
        Y = Y * 2 / (target_height - 1) - 1
        X = X * 2 / (target_width - 1) - 1
        target_coordinate = torch.cat([X, Y], dim=1)  # Convert to (x, y)
        target_coordinate_partial_repr = self.compute_partial_repr(target_coordinate, target_control_points)
        target_coordinate_repr = torch.cat([
            target_coordinate_partial_repr, torch.ones(HW, 1), target_coordinate
        ], dim=1)
        
        # Register buffers
        self.register_buffer('inverse_kernel', inverse_kernel)
        self.register_buffer('padding_matrix', torch.zeros(3, 2))
        self.register_buffer('target_coordinate_repr', target_coordinate_repr)
        
    def compute_partial_repr(self, input_points, control_points):
        """
        Compute radial basis function for TPS
        """
        N = input_points.size(0)
        M = control_points.size(0)
        pairwise_diff = input_points.view(N, 1, 2) - control_points.view(1, M, 2)
        pairwise_diff_square = pairwise_diff * pairwise_diff
        pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :, 1]
        repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist + 1e-6)
        return repr_matrix
        
    def forward(self, source_control_points):
        """
        Args:
            source_control_points: (B, K, 2)
        Returns:
            grid: (B, H, W, 2)
        """
        batch_size = source_control_points.size(0)
        
        Y = torch.cat([source_control_points, self.padding_matrix.expand(batch_size, 3, 2)], dim=1)
        mapping_matrix = torch.matmul(self.inverse_kernel, Y)
        source_coordinate = torch.matmul(self.target_coordinate_repr, mapping_matrix)
        
        grid = source_coordinate.view(batch_size, self.target_height, self.target_width, 2)
        return grid


class UNetDown(nn.Module):
    """
    U-Net downsampling block
    """
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """
    U-Net upsampling block with skip connection
    """
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), dim=1)
        return x


class SelfAttention(nn.Module):
    """
    Self-attention module for detail preservation
    """
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        """
        Args:
            x: Input feature map (B, C, H, W)
        Returns:
            out: Self-attention output (B, C, H, W)
        """
        batch_size, C, H, W = x.size()
        
        # Query, Key, Value
        proj_query = self.query_conv(x).view(batch_size, -1, H * W).permute(0, 2, 1)  # B x HW x C'
        proj_key = self.key_conv(x).view(batch_size, -1, H * W)  # B x C' x HW
        proj_value = self.value_conv(x).view(batch_size, -1, H * W)  # B x C x HW
        
        # Attention map
        attention = torch.bmm(proj_query, proj_key)  # B x HW x HW
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x HW
        out = out.view(batch_size, C, H, W)
        
        # Residual connection
        out = self.gamma * out + x
        
        return out


import itertools

