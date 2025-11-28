"""
Checkpoint saving and loading utilities
"""

import os
import torch


def save_checkpoint(state, checkpoint_dir, filename='checkpoint.pth'):
    """
    Save model checkpoint
    
    Args:
        state: Dictionary containing model state, optimizer state, etc.
        checkpoint_dir: Directory to save checkpoint
        filename: Checkpoint filename
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath, model, optimizer=None, device='cuda'):
    """
    Load model checkpoint
    
    Args:
        filepath: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state
        device: Device to load checkpoint to
    Returns:
        epoch: Epoch number from checkpoint
        best_metric: Best metric value from checkpoint
    """
    if not os.path.exists(filepath):
        print(f"Checkpoint not found: {filepath}")
        return 0, 0.0
    
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    best_metric = checkpoint.get('best_metric', 0.0)
    
    print(f"Checkpoint loaded from {filepath} (epoch {epoch})")
    
    return epoch, best_metric

