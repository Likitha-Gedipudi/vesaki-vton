"""
Data augmentation and transformations
"""

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random


class RandomHorizontalFlip:
    """
    Random horizontal flip with consistent flipping across all inputs
    """
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, *images):
        if random.random() < self.p:
            return [TF.hflip(img) for img in images]
        return list(images)


class Compose:
    """
    Compose multiple transforms and apply to all inputs
    """
    def __init__(self, transforms_list):
        self.transforms = transforms_list
        
    def __call__(self, *images):
        for t in self.transforms:
            images = t(*images)
        return images


def get_transforms(mode='train'):
    """
    Get standard transforms for training/testing
    
    Args:
        mode: 'train' or 'test'
    Returns:
        transform: Composition of transforms
    """
    if mode == 'train':
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    else:
        # Test transforms (no augmentation)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    return transform

