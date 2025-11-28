"""
Visualization utilities
"""

import os
import torch
import torchvision.utils as vutils
import numpy as np
from PIL import Image


def tensor_to_image(tensor):
    """
    Convert tensor to PIL Image
    
    Args:
        tensor: Image tensor (C, H, W) in range [-1, 1] or [0, 1]
    Returns:
        image: PIL Image
    """
    # Denormalize if in range [-1, 1]
    if tensor.min() < 0:
        tensor = (tensor + 1) / 2
    
    # Clamp to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy
    np_image = tensor.cpu().numpy()
    np_image = np.transpose(np_image, (1, 2, 0))
    np_image = (np_image * 255).astype(np.uint8)
    
    return Image.fromarray(np_image)


def save_images(images_dict, save_dir, epoch, step):
    """
    Save a grid of images
    
    Args:
        images_dict: Dictionary of image tensors
        save_dir: Directory to save images
        epoch: Current epoch
        step: Current step
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create grid for each image type
    for name, images in images_dict.items():
        if images is not None and len(images) > 0:
            # Denormalize if needed
            if images.min() < 0:
                images = (images + 1) / 2
            
            # Create grid
            grid = vutils.make_grid(images, nrow=4, normalize=False)
            
            # Save
            filename = f"{name}_epoch{epoch:03d}_step{step:06d}.png"
            filepath = os.path.join(save_dir, filename)
            vutils.save_image(grid, filepath)
    
    print(f"Images saved to {save_dir}")


def visualize_batch(batch, outputs, save_path):
    """
    Visualize a batch of inputs and outputs
    
    Args:
        batch: Input batch dictionary
        outputs: Model outputs
        save_path: Path to save visualization
    """
    # Extract data
    person = batch['person'][:4]  # Take first 4 samples
    garment = batch['garment'][:4]
    agnostic = batch['agnostic'][:4]
    
    # Extract outputs
    if isinstance(outputs, dict):
        rendered = outputs.get('rendered', outputs.get('output'))[:4]
        warped = outputs.get('warped_garment')[:4] if 'warped_garment' in outputs else None
    else:
        rendered = outputs[:4]
        warped = None
    
    # Create comparison grid
    images_to_show = [person, garment, agnostic, rendered]
    if warped is not None:
        images_to_show.insert(3, warped)
    
    # Denormalize
    images_to_show = [(img + 1) / 2 for img in images_to_show]
    
    # Concatenate horizontally
    comparison = torch.cat(images_to_show, dim=3)  # Concatenate along width
    
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    vutils.save_image(comparison, save_path, nrow=4)

