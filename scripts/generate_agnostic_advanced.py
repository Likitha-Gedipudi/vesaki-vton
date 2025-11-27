#!/usr/bin/env python3
"""
Advanced Agnostic Mask Generator
Uses SCHP segmentation for precise clothing region masking
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

def generate_agnostic_from_schp(person_image_path, parse_mask_path, output_dir):
    """Generate agnostic mask using SCHP parsing"""
    # Read person image
    person_img = cv2.imread(str(person_image_path))
    if person_img is None:
        return False
    
    # Read segmentation mask
    seg_mask = cv2.imread(str(parse_mask_path), cv2.IMREAD_GRAYSCALE)
    
    if seg_mask is None:
        return generate_simple_agnostic(person_image_path, output_dir)
    
    h, w = person_img.shape[:2]
    
    # Resize mask if needed
    if seg_mask.shape[:2] != (h, w):
        seg_mask = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Define clothing labels to mask out
    # ATR labels: 4=Upper-clothes, 5=Skirt, 6=Pants, 7=Dress, 8=Belt
    clothing_labels = [4, 5, 6, 7, 8]
    
    # Create clothing mask
    clothing_mask = np.zeros((h, w), dtype=np.uint8)
    for label in clothing_labels:
        clothing_mask[seg_mask == label] = 255
    
    # Expand mask slightly
    kernel = np.ones((15, 15), np.uint8)
    clothing_mask = cv2.dilate(clothing_mask, kernel, iterations=1)
    
    # Apply Gaussian blur for smooth edges
    clothing_mask_blur = cv2.GaussianBlur(clothing_mask.astype(float), (31, 31), 10)
    clothing_mask_blur = clothing_mask_blur / 255.0
    
    # Create agnostic image
    agnostic_img = person_img.copy()
    gray_color = np.array([128, 128, 128])
    
    # Blend
    for i in range(3):
        agnostic_img[:, :, i] = (
            person_img[:, :, i] * (1 - clothing_mask_blur) + 
            gray_color[i] * clothing_mask_blur
        ).astype(np.uint8)
    
    # Save agnostic image
    output_path = os.path.join(output_dir, Path(person_image_path).stem + '.jpg')
    cv2.imwrite(output_path, agnostic_img)
    
    # Save mask
    mask_path = os.path.join(output_dir, Path(person_image_path).stem + '_mask.png')
    cv2.imwrite(mask_path, clothing_mask)
    
    return True

def generate_simple_agnostic(person_image_path, output_dir):
    """Fallback: Simple region-based agnostic mask"""
    person_img = cv2.imread(str(person_image_path))
    if person_img is None:
        return False
    
    h, w = person_img.shape[:2]
    
    # Create elliptical mask for torso region
    mask = np.zeros((h, w), dtype=np.uint8)
    
    center_x = w // 2
    center_y = (int(h * 0.2) + int(h * 0.7)) // 2
    axes_x = (int(w * 0.75) - int(w * 0.25)) // 2
    axes_y = (int(h * 0.7) - int(h * 0.2)) // 2
    
    cv2.ellipse(mask, (center_x, center_y), (axes_x, axes_y), 0, 0, 360, 255, -1)
    
    # Blur
    mask_blur = cv2.GaussianBlur(mask.astype(float), (31, 31), 0) / 255.0
    
    # Create agnostic
    agnostic_img = person_img.copy()
    gray_color = np.array([128, 128, 128])
    
    for i in range(3):
        agnostic_img[:, :, i] = (
            person_img[:, :, i] * (1 - mask_blur) + 
            gray_color[i] * mask_blur
        ).astype(np.uint8)
    
    # Save
    output_path = os.path.join(output_dir, Path(person_image_path).stem + '.jpg')
    cv2.imwrite(output_path, agnostic_img)
    
    mask_path = os.path.join(output_dir, Path(person_image_path).stem + '_mask.png')
    cv2.imwrite(mask_path, mask)
    
    return True

def process_dataset(data_dir):
    """Process all person images"""
    person_dir = os.path.join(data_dir, 'person')
    parse_dir = os.path.join(data_dir, 'person-parse')
    agnostic_dir = os.path.join(data_dir, 'agnostic-mask')
    
    os.makedirs(agnostic_dir, exist_ok=True)
    
    image_files = list(Path(person_dir).glob('*.jpg')) + list(Path(person_dir).glob('*.png'))
    
    if not image_files:
        print("No person images found!")
        return
    
    print(f"Processing {len(image_files)} person images...")
    print()
    
    has_parsing = os.path.exists(parse_dir) and len(list(Path(parse_dir).glob('*.png'))) > 0
    
    if has_parsing:
        print("✓ Using SCHP segmentation masks for precise agnostic generation")
    else:
        print("⚠ No segmentation masks found, using simple method")
        print("  Run generate_parse_schp.py first for better results")
    print()
    
    success_count = 0
    
    for image_path in tqdm(image_files, desc="Generating agnostic masks"):
        parse_mask_path = os.path.join(parse_dir, image_path.stem + '.png')
        
        if generate_agnostic_from_schp(str(image_path), parse_mask_path, agnostic_dir):
            success_count += 1
    
    print()
    print(f"✓ Successfully processed: {success_count}/{len(image_files)}")
    print(f"✓ Output saved to: {agnostic_dir}")

def main():
    parser = argparse.ArgumentParser(description='Generate advanced agnostic masks')
    parser.add_argument('--data_dir', type=str, default='dataset/train',
                        help='Path to dataset directory')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  ADVANCED AGNOSTIC MASK GENERATOR")
    print("=" * 70)
    print()
    
    process_dataset(args.data_dir)
    
    print()
    print("=" * 70)
    print("  AGNOSTIC MASK GENERATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()

