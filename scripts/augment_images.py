#!/usr/bin/env python3
"""
Data Augmentation Pipeline for Person Images
Generates 5-8x variations per image with extensive transformations
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("Warning: albumentations not installed, using basic augmentations")

def create_augmentation_pipeline():
    """Create augmentation pipeline using albumentations"""
    if not ALBUMENTATIONS_AVAILABLE:
        return None
    
    # Define augmentation transforms
    transforms = {
        'horizontal_flip': A.HorizontalFlip(p=1.0),
        
        'brightness_up': A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0, p=1.0)
        ]),
        
        'brightness_down': A.Compose([
            A.RandomBrightnessContrast(brightness_limit=-0.2, contrast_limit=0, p=1.0)
        ]),
        
        'contrast_up': A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.15, p=1.0)
        ]),
        
        'color_jitter': A.Compose([
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=1.0)
        ]),
        
        'slight_rotate': A.Compose([
            A.Rotate(limit=5, p=1.0, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255))
        ]),
        
        'crop_variation': A.Compose([
            A.RandomResizedCrop(height=1024, width=768, scale=(0.95, 1.0), ratio=(0.75, 0.75), p=1.0)
        ])
    }
    
    return transforms

def augment_with_basic(image):
    """Basic augmentation without albumentations"""
    augmented_images = []
    
    # 1. Horizontal flip
    flipped = cv2.flip(image, 1)
    augmented_images.append(('flip', flipped))
    
    # 2. Brightness up
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.2, 0, 255)
    bright = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    augmented_images.append(('bright', bright))
    
    # 3. Brightness down
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.8, 0, 255)
    dark = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    augmented_images.append(('dark', dark))
    
    # 4. Contrast
    alpha = 1.15  # Contrast control
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    augmented_images.append(('contrast', adjusted))
    
    return augmented_images

def augment_with_albumentations(image, transforms):
    """Augment image using albumentations"""
    augmented_images = []
    
    for name, transform in transforms.items():
        try:
            augmented = transform(image=image)['image']
            augmented_images.append((name, augmented))
        except Exception as e:
            print(f"    Warning: {name} augmentation failed: {e}")
    
    return augmented_images

def augment_single_image(image_path, output_dir, use_albumentations=True):
    """Augment a single image and save variations"""
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        return []
    
    # Get base name
    base_name = Path(image_path).stem
    
    # Generate augmentations
    if use_albumentations and ALBUMENTATIONS_AVAILABLE:
        transforms = create_augmentation_pipeline()
        augmented_images = augment_with_albumentations(image, transforms)
    else:
        augmented_images = augment_with_basic(image)
    
    # Save augmented images
    saved_files = []
    for i, (aug_type, aug_img) in enumerate(augmented_images, 1):
        output_filename = f"{base_name}_aug{i}_{aug_type}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        
        cv2.imwrite(output_path, aug_img)
        saved_files.append({
            'original': str(image_path),
            'augmented': output_path,
            'augmentation_type': aug_type,
            'augmentation_id': i
        })
    
    return saved_files

def augment_dataset(data_dir, output_dir=None):
    """Augment all person images in dataset"""
    person_dir = os.path.join(data_dir, 'person')
    
    if output_dir is None:
        output_dir = person_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all person images
    image_files = list(Path(person_dir).glob('*.jpg')) + list(Path(person_dir).glob('*.png'))
    
    # Filter out already augmented images
    original_images = [f for f in image_files if '_aug' not in f.stem]
    
    if not original_images:
        print("No original images found for augmentation!")
        return []
    
    print(f"Augmenting {len(original_images)} person images...")
    print()
    
    # Check if albumentations is available
    use_albumentations = ALBUMENTATIONS_AVAILABLE
    if use_albumentations:
        print("Using albumentations for professional augmentation")
    else:
        print("Using basic OpenCV augmentation")
    print()
    
    # Augment each image
    all_augmentations = []
    
    for image_path in tqdm(original_images, desc="Augmenting images"):
        augmented = augment_single_image(image_path, output_dir, use_albumentations)
        all_augmentations.extend(augmented)
    
    return all_augmentations

def save_augmentation_map(augmentation_data, output_file):
    """Save augmentation mapping to JSON"""
    with open(output_file, 'w') as f:
        json.dump(augmentation_data, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Augment person images')
    parser.add_argument('--data_dir', type=str, default='dataset/train',
                        help='Path to dataset directory (default: dataset/train)')
    parser.add_argument('--output_map', type=str, default='augmentation_map.json',
                        help='Output JSON file for augmentation mapping')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  DATA AUGMENTATION PIPELINE")
    print("=" * 70)
    print()
    
    # Check if person directory exists
    person_dir = os.path.join(args.data_dir, 'person')
    if not os.path.exists(person_dir):
        print(f"Error: Person directory not found: {person_dir}")
        print("Please run scraper first.")
        return
    
    # Count original images
    original_count = len([f for f in os.listdir(person_dir) 
                         if f.endswith(('.jpg', '.png')) and '_aug' not in f])
    
    if original_count == 0:
        print("No original images found!")
        return
    
    print(f"Found {original_count} original person images")
    print()
    
    # Perform augmentation
    augmentation_data = augment_dataset(args.data_dir)
    
    # Count final images
    final_count = len([f for f in os.listdir(person_dir) 
                      if f.endswith(('.jpg', '.png'))])
    augmented_count = final_count - original_count
    
    print()
    print(f"✓ Generated {augmented_count} augmented images")
    print(f"  Original: {original_count} → Total: {final_count}")
    print(f"  Expansion: {final_count/original_count:.1f}x")
    print()
    
    # Save augmentation mapping
    save_augmentation_map(augmentation_data, args.output_map)
    print(f"✓ Saved augmentation map to: {args.output_map}")
    print()
    
    print("=" * 70)
    print("  AUGMENTATION COMPLETE")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Resize images: python3 preprocess.py --data_dir dataset/train --resize")
    print("  2. Generate annotations: python3 main_advanced.py")
    print()

if __name__ == "__main__":
    main()

