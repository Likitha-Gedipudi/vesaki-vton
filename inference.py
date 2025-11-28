#!/usr/bin/env python3
"""
Vesaki-VTON Inference Script

Run inference on single images or batches
"""

import os
import argparse
import torch
import yaml
from PIL import Image
import torchvision.transforms as transforms

from models import GeometricMatchingModule, TryOnModuleAdvanced
from utils import load_checkpoint, tensor_to_image


class VITONInference:
    """
    Inference wrapper for Vesaki-VTON
    """
    def __init__(self, gmm_checkpoint, tom_checkpoint, config_path='configs/train_config.yaml', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize models
        self.gmm = GeometricMatchingModule(
            input_nc=config['model']['gmm']['input_nc'],
            feature_dim=config['model']['gmm']['feature_dim'],
            num_points=config['model']['gmm']['num_points']
        ).to(self.device)
        
        self.tom = TryOnModuleAdvanced(
            input_nc=config['model']['tom']['input_nc']
        ).to(self.device)
        
        # Load checkpoints
        load_checkpoint(gmm_checkpoint, self.gmm, device=self.device)
        load_checkpoint(tom_checkpoint, self.tom, device=self.device)
        
        self.gmm.eval()
        self.tom.eval()
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((1024, 768)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        print(f"Models loaded on {self.device}")
    
    @torch.no_grad()
    def try_on(self, person_path, garment_path, person_repr_path=None, agnostic_path=None):
        """
        Perform virtual try-on
        
        Args:
            person_path: Path to person image
            garment_path: Path to garment image
            person_repr_path: Path to person representation (optional, computed if not provided)
            agnostic_path: Path to agnostic person (optional, computed if not provided)
        Returns:
            result: Try-on result as PIL Image
        """
        # Load and preprocess images
        person_img = Image.open(person_path).convert('RGB')
        garment_img = Image.open(garment_path).convert('RGB')
        
        person = self.transform(person_img).unsqueeze(0).to(self.device)
        garment = self.transform(garment_img).unsqueeze(0).to(self.device)
        
        # TODO: In production, you'd need to generate person_repr and agnostic
        # For now, use placeholders
        if person_repr_path:
            person_repr_img = Image.open(person_repr_path).convert('RGB')
            person_repr = self.transform(person_repr_img).unsqueeze(0).to(self.device)
        else:
            # Placeholder: In practice, generate from pose + mask + agnostic
            person_repr = torch.cat([
                torch.zeros(1, 19, 1024, 768),  # 18 pose + 1 mask
                person  # 3 channels
            ], dim=1).to(self.device)
        
        if agnostic_path:
            agnostic_img = Image.open(agnostic_path).convert('RGB')
            agnostic = self.transform(agnostic_img).unsqueeze(0).to(self.device)
        else:
            # Placeholder: Use original person
            agnostic = person
        
        # Stage 1: GMM - Warp garment
        warped_garment, theta = self.gmm(person_repr, garment)
        
        # Stage 2: TOM - Generate result
        rendered, composition_mask = self.tom(agnostic, warped_garment, person_repr[:, -3:, :, :])
        
        # Convert to image
        result_tensor = rendered.squeeze(0).cpu()
        result_img = tensor_to_image(result_tensor)
        
        return result_img


def main():
    parser = argparse.ArgumentParser(description='Vesaki-VTON Inference')
    parser.add_argument('--person', type=str, required=True, help='Path to person image')
    parser.add_argument('--garment', type=str, required=True, help='Path to garment image')
    parser.add_argument('--output', type=str, required=True, help='Output path')
    parser.add_argument('--gmm_checkpoint', type=str, default='checkpoints/gmm_final.pth')
    parser.add_argument('--tom_checkpoint', type=str, default='checkpoints/tom_final.pth')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()
    
    print("Vesaki-VTON Inference")
    print("=" * 70)
    
    # Initialize inference
    inferencer = VITONInference(
        gmm_checkpoint=args.gmm_checkpoint,
        tom_checkpoint=args.tom_checkpoint,
        config_path=args.config,
        device=args.device
    )
    
    # Run inference
    print(f"Processing: {args.person} + {args.garment}")
    result = inferencer.try_on(args.person, args.garment)
    
    # Save result
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    result.save(args.output)
    print(f"Result saved to: {args.output}")


if __name__ == '__main__':
    main()

