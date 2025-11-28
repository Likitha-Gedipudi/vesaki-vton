"""
Try-On Module (TOM)
Synthesizes final try-on result with attention and composition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .networks import UNetDown, UNetUp, SelfAttention, ResidualBlock


class TryOnModule(nn.Module):
    """
    TOM: Generates final try-on result from warped garment and person representation
    
    Architecture:
    - U-Net encoder-decoder with skip connections
    - Self-attention for detail preservation
    - Composition mask prediction for blending
    """
    def __init__(self, input_nc=9, output_nc=4):  # 4 = RGB + composition mask
        super(TryOnModule, self).__init__()
        
        # U-Net Encoder (downsampling)
        self.down1 = UNetDown(input_nc, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512)
        self.down6 = UNetDown(512, 512)
        self.down7 = UNetDown(512, 512, normalize=False)
        
        # Self-attention at bottleneck
        self.attention = SelfAttention(512)
        
        # U-Net Decoder (upsampling with skip connections)
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)  # 1024 = 512 + 512 from skip
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 256)
        self.up5 = UNetUp(512, 128)
        self.up6 = UNetUp(256, 64)
        
        # Final output layer
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, output_nc, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        """
        Forward pass of TOM
        
        Args:
            x: Input tensor (B, C, H, W)
               Concatenated (agnostic_person + warped_garment + person_representation)
        Returns:
            output: Generated image + composition mask (B, 4, H, W)
        """
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        
        # Attention at bottleneck
        d7 = self.attention(d7)
        
        # Decoder with skip connections
        u1 = self.up1(d7, d6)
        u2 = self.up2(u1, d5)
        u3 = self.up3(u2, d4)
        u4 = self.up4(u3, d3)
        u5 = self.up5(u4, d2)
        u6 = self.up6(u5, d1)
        
        # Final output
        output = self.final(u6)
        
        return output


class TryOnModuleAdvanced(nn.Module):
    """
    Advanced TOM with multi-scale processing and better composition
    """
    def __init__(self, input_nc=9):
        super(TryOnModuleAdvanced, self).__init__()
        
        # Multi-scale encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(64)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(128)
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(256)
        )
        
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(512)
        )
        
        # Bottleneck with attention
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            SelfAttention(512),
            ResidualBlock(512),
            ResidualBlock(512)
        )
        
        # Decoder
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(512)
        )
        
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1),  # 1024 = 512 + 512
            nn.ReLU(inplace=True),
            ResidualBlock(256)
        )
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),  # 512 = 256 + 256
            nn.ReLU(inplace=True),
            ResidualBlock(128)
        )
        
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),  # 256 = 128 + 128
            nn.ReLU(inplace=True),
            ResidualBlock(64)
        )
        
        # Output heads
        self.rgb_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 128 = 64 + 64
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        self.mask_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # Composition mask in [0, 1]
        )
        
    def forward(self, agnostic, warped_garment, person_repr):
        """
        Forward pass with separate inputs
        
        Args:
            agnostic: Agnostic person (B, 3, H, W)
            warped_garment: Warped garment from GMM (B, 3, H, W)
            person_repr: Person representation (B, 3, H, W)
        Returns:
            rendered: Final try-on result (B, 3, H, W)
            composition_mask: Blending mask (B, 1, H, W)
        """
        # Concatenate inputs
        x = torch.cat([agnostic, warped_garment, person_repr], dim=1)
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # Bottleneck
        bottleneck = self.bottleneck(e4)
        
        # Decoder with skip connections
        d4 = self.dec4(bottleneck)
        d4 = torch.cat([d4, e4], dim=1)
        
        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        
        d1 = self.dec1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        
        # Output
        rendered_image = self.rgb_head(d1)
        composition_mask = self.mask_head(d1)
        
        # Compose final result
        # result = warped_garment * composition_mask + agnostic * (1 - composition_mask)
        # But we output the rendered image directly from the network
        
        return rendered_image, composition_mask


class CompleteTryOnModel(nn.Module):
    """
    Complete model: GMM + TOM pipeline
    """
    def __init__(self, gmm_model, tom_model):
        super(CompleteTryOnModel, self).__init__()
        self.gmm = gmm_model
        self.tom = tom_model
        
    def forward(self, person_repr, garment, agnostic, garment_mask=None):
        """
        End-to-end forward pass
        
        Args:
            person_repr: Person representation for GMM (B, C, H, W)
            garment: Garment image (B, 3, H, W)
            agnostic: Agnostic person for TOM (B, 3, H, W)
            garment_mask: Optional garment mask (B, 1, H, W)
        Returns:
            final_output: Try-on result (B, 3, H, W)
            composition_mask: Blending mask (B, 1, H, W)
            warped_garment: Intermediate warped garment (B, 3, H, W)
        """
        # Stage 1: Geometric matching
        if garment_mask is not None:
            warped_garment, warped_mask, theta = self.gmm(person_repr, garment, garment_mask)
        else:
            warped_garment, theta = self.gmm(person_repr, garment)
        
        # Stage 2: Try-on synthesis
        if isinstance(self.tom, TryOnModuleAdvanced):
            rendered_image, composition_mask = self.tom(agnostic, warped_garment, person_repr)
        else:
            # Original TOM expects concatenated input
            tom_input = torch.cat([agnostic, warped_garment, person_repr], dim=1)
            output = self.tom(tom_input)
            rendered_image = output[:, :3]
            composition_mask = output[:, 3:4]
        
        return rendered_image, composition_mask, warped_garment

