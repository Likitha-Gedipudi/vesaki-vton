#!/usr/bin/env python3
"""
Vesaki-VTON Training Script

Train GMM and TOM models for virtual try-on
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import GeometricMatchingModule, TryOnModuleAdvanced
from data import get_dataloader
from losses import L1Loss, PerceptualLoss, StyleLoss, GMMSmoothnessLoss
from utils import save_checkpoint, load_checkpoint, compute_ssim, visualize_batch


def train_gmm(config):
    """Train Geometric Matching Module"""
    
    print("=" * 70)
    print("Training GMM (Geometric Matching Module)")
    print("=" * 70)
    
    # Setup device
    device = torch.device(f"cuda:{config['hardware']['gpu_ids'][0]}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    gmm = GeometricMatchingModule(
        input_nc=config['model']['gmm']['input_nc'],
        feature_dim=config['model']['gmm']['feature_dim'],
        num_points=config['model']['gmm']['num_points']
    ).to(device)
    
    # Create dataloaders
    train_loader = get_dataloader(
        config['training']['data_dir'],
        mode='train',
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        shuffle=True
    )
    
    # Loss functions
    l1_loss = L1Loss()
    smoothness_loss = GMMSmoothnessLoss()
    
    # Optimizer
    optimizer = optim.Adam(
        gmm.parameters(),
        lr=config['training']['learning_rate_gmm'],
        betas=(config['training']['beta1'], config['training']['beta2']),
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['training']['lr_decay_epochs'],
        gamma=config['training']['lr_decay_gamma']
    )
    
    # Tensorboard
    writer = SummaryWriter(os.path.join(config['training']['log_dir'], 'gmm'))
    
    # Training loop
    num_epochs = config['training']['num_epochs_gmm']
    global_step = 0
    
    for epoch in range(num_epochs):
        gmm.train()
        epoch_losses = {'l1': 0, 'smoothness': 0, 'total': 0}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            person_repr = batch['person_repr'].to(device)
            garment = batch['garment'].to(device)
            target = batch['person'].to(device)  # Target is person wearing garment
            
            # Forward pass
            warped_garment, theta = gmm(person_repr, garment)
            
            # Compute losses
            l1 = l1_loss(warped_garment, target)
            smoothness = smoothness_loss(theta)
            
            total_loss = (
                config['loss']['gmm_l1_weight'] * l1 +
                config['loss']['gmm_smoothness_weight'] * smoothness
            )
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Logging
            epoch_losses['l1'] += l1.item()
            epoch_losses['smoothness'] += smoothness.item()
            epoch_losses['total'] += total_loss.item()
            
            pbar.set_postfix({'loss': total_loss.item()})
            
            if batch_idx % config['training']['print_freq'] == 0:
                writer.add_scalar('GMM/l1_loss', l1.item(), global_step)
                writer.add_scalar('GMM/smoothness_loss', smoothness.item(), global_step)
                writer.add_scalar('GMM/total_loss', total_loss.item(), global_step)
            
            global_step += 1
        
        # Average epoch losses
        for key in epoch_losses:
            epoch_losses[key] /= len(train_loader)
        
        print(f"Epoch {epoch+1} - L1: {epoch_losses['l1']:.4f}, "
              f"Smoothness: {epoch_losses['smoothness']:.4f}, "
              f"Total: {epoch_losses['total']:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config['training']['save_epoch_freq'] == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': gmm.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_losses['total']
            }, config['training']['checkpoint_dir'], f'gmm_epoch_{epoch+1}.pth')
        
        scheduler.step()
    
    # Save final model
    save_checkpoint({
        'epoch': num_epochs,
        'model_state_dict': gmm.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, config['training']['checkpoint_dir'], 'gmm_final.pth')
    
    writer.close()
    print("GMM training complete!")
    

def train_tom(config):
    """Train Try-On Module"""
    
    print("=" * 70)
    print("Training TOM (Try-On Module)")
    print("=" * 70)
    
    device = torch.device(f"cuda:{config['hardware']['gpu_ids'][0]}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load trained GMM
    gmm = GeometricMatchingModule(
        input_nc=config['model']['gmm']['input_nc'],
        feature_dim=config['model']['gmm']['feature_dim'],
        num_points=config['model']['gmm']['num_points']
    ).to(device)
    
    gmm_checkpoint = os.path.join(config['training']['checkpoint_dir'], 'gmm_final.pth')
    load_checkpoint(gmm_checkpoint, gmm, device=device)
    gmm.eval()  # Freeze GMM
    
    # Create TOM
    tom = TryOnModuleAdvanced(input_nc=config['model']['tom']['input_nc']).to(device)
    
    # Dataloaders
    train_loader = get_dataloader(
        config['training']['data_dir'],
        mode='train',
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers']
    )
    
    # Loss functions
    l1_loss = L1Loss()
    perceptual_loss = PerceptualLoss().to(device)
    style_loss = StyleLoss().to(device)
    
    # Optimizer
    optimizer = optim.Adam(
        tom.parameters(),
        lr=config['training']['learning_rate_tom'],
        betas=(config['training']['beta1'], config['training']['beta2'])
    )
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    writer = SummaryWriter(os.path.join(config['training']['log_dir'], 'tom'))
    
    # Training loop
    num_epochs = config['training']['num_epochs_tom']
    global_step = 0
    
    for epoch in range(num_epochs):
        tom.train()
        epoch_losses = {'l1': 0, 'perceptual': 0, 'style': 0, 'total': 0}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            person_repr = batch['person_repr'].to(device)
            garment = batch['garment'].to(device)
            agnostic = batch['agnostic'].to(device)
            target = batch['person'].to(device)
            
            # Get warped garment from GMM (no gradients)
            with torch.no_grad():
                warped_garment, _ = gmm(person_repr, garment)
            
            # Forward through TOM
            rendered, composition_mask = tom(agnostic, warped_garment, person_repr)
            
            # Compute losses
            l1 = l1_loss(rendered, target)
            perceptual = perceptual_loss(rendered, target)
            style = style_loss(rendered, target)
            
            total_loss = (
                config['loss']['tom_l1_weight'] * l1 +
                config['loss']['tom_perceptual_weight'] * perceptual +
                config['loss']['tom_style_weight'] * style
            )
            
            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Logging
            epoch_losses['l1'] += l1.item()
            epoch_losses['perceptual'] += perceptual.item()
            epoch_losses['style'] += style.item()
            epoch_losses['total'] += total_loss.item()
            
            pbar.set_postfix({'loss': total_loss.item()})
            
            if batch_idx % config['training']['print_freq'] == 0:
                writer.add_scalar('TOM/l1_loss', l1.item(), global_step)
                writer.add_scalar('TOM/perceptual_loss', perceptual.item(), global_step)
                writer.add_scalar('TOM/style_loss', style.item(), global_step)
                writer.add_scalar('TOM/total_loss', total_loss.item(), global_step)
            
            global_step += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= len(train_loader)
        
        print(f"Epoch {epoch+1} - L1: {epoch_losses['l1']:.4f}, "
              f"Perceptual: {epoch_losses['perceptual']:.4f}, "
              f"Style: {epoch_losses['style']:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config['training']['save_epoch_freq'] == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': tom.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, config['training']['checkpoint_dir'], f'tom_epoch_{epoch+1}.pth')
        
        scheduler.step()
    
    # Save final
    save_checkpoint({
        'epoch': num_epochs,
        'model_state_dict': tom.state_dict()
    }, config['training']['checkpoint_dir'], 'tom_final.pth')
    
    writer.close()
    print("TOM training complete!")


def main():
    parser = argparse.ArgumentParser(description='Train Vesaki-VTON')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml', help='Path to config file')
    parser.add_argument('--stage', type=str, choices=['gmm', 'tom', 'both'], default='both', help='Training stage')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Train based on stage
    if args.stage in ['gmm', 'both']:
        train_gmm(config)
    
    if args.stage in ['tom', 'both']:
        train_tom(config)
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Checkpoints saved in: {config['training']['checkpoint_dir']}")
    print(f"Logs saved in: {config['training']['log_dir']}")


if __name__ == '__main__':
    main()

