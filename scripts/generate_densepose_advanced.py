#!/usr/bin/env python3
"""
Advanced Body Segmentation
Uses MediaPipe Selfie Segmentation for high-quality body masks
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Error: MediaPipe not installed!")

def generate_advanced_segmentation(image_path, output_dir):
    """Generate body segmentation using MediaPipe"""
    if not MEDIAPIPE_AVAILABLE:
        return generate_simple_mask(image_path, output_dir)
    
    # Initialize MediaPipe Selfie Segmentation
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        return False
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_seg:
        results = selfie_seg.process(image_rgb)
    
    # Get mask
    if results.segmentation_mask is None:
        return generate_simple_mask(image_path, output_dir)
    
    # Convert to binary mask
    mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255
    
    # Clean up mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Save mask
    output_path = os.path.join(output_dir, Path(image_path).stem + '_mask.png')
    cv2.imwrite(output_path, mask)
    
    # Generate visualization
    vis_path = os.path.join(output_dir, Path(image_path).stem + '.jpg')
    
    # Create colored overlay
    overlay = image.copy()
    overlay[mask == 255] = [0, 255, 0]  # Green for person
    blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    cv2.imwrite(vis_path, blended)
    
    return True

def generate_simple_mask(image_path, output_dir):
    """Fallback: Simple background subtraction"""
    image = cv2.imread(str(image_path))
    if image is None:
        return False
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    _, mask = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find largest contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros_like(mask)
        cv2.drawContours(clean_mask, [largest_contour], -1, 255, -1)
        
        output_path = os.path.join(output_dir, Path(image_path).stem + '_mask.png')
        cv2.imwrite(output_path, clean_mask)
        
        # Visualization
        vis_path = os.path.join(output_dir, Path(image_path).stem + '.jpg')
        overlay = image.copy()
        overlay[clean_mask == 255] = [0, 255, 0]
        blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        cv2.imwrite(vis_path, blended)
        
        return True
    
    return False

def process_dataset(data_dir):
    """Process all person images"""
    person_dir = os.path.join(data_dir, 'person')
    densepose_dir = os.path.join(data_dir, 'densepose')
    
    os.makedirs(densepose_dir, exist_ok=True)
    
    image_files = list(Path(person_dir).glob('*.jpg')) + list(Path(person_dir).glob('*.png'))
    
    if not image_files:
        print("No person images found!")
        return
    
    print(f"Processing {len(image_files)} person images...")
    print()
    
    if MEDIAPIPE_AVAILABLE:
        print("Using MediaPipe Selfie Segmentation (high quality)")
    else:
        print("Using simple background subtraction")
    print()
    
    success_count = 0
    
    for image_path in tqdm(image_files, desc="Generating body masks"):
        if generate_advanced_segmentation(str(image_path), densepose_dir):
            success_count += 1
    
    print()
    print(f"✓ Successfully processed: {success_count}/{len(image_files)}")
    print(f"✓ Output saved to: {densepose_dir}")

def main():
    parser = argparse.ArgumentParser(description='Generate advanced body segmentation')
    parser.add_argument('--data_dir', type=str, default='dataset/train',
                        help='Path to dataset directory')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  ADVANCED BODY SEGMENTATION")
    print("=" * 70)
    print()
    
    process_dataset(args.data_dir)
    
    print()
    print("=" * 70)
    print("  SEGMENTATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()

