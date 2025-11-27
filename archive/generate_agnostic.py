"""
Agnostic Mask Generator for Virtual Try-On Dataset
Creates masked person images where the try-on area is grayed out

The agnostic mask tells the model which parts of the person to preserve
and which parts to replace with the new garment.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

def generate_agnostic_mask(person_image_path, parse_mask_path, output_dir):
    """
    Generate agnostic mask by masking out clothing regions
    
    Args:
        person_image_path: Path to person image
        parse_mask_path: Path to segmentation mask
        output_dir: Output directory for agnostic masks
    """
    # Read person image
    person_img = cv2.imread(str(person_image_path))
    if person_img is None:
        return False
    
    # Read segmentation mask
    seg_mask = cv2.imread(str(parse_mask_path), cv2.IMREAD_GRAYSCALE)
    if seg_mask is None:
        # If no segmentation available, create a simple center mask
        return generate_simple_agnostic(person_image_path, output_dir)
    
    h, w = person_img.shape[:2]
    
    # Define labels to mask out (clothing regions)
    # 5=Upper-clothes, 6=Dress, 7=Coat, 10=Jumpsuits, 12=Skirt
    clothing_labels = [5, 6, 7, 10, 12]
    
    # Create clothing mask
    clothing_mask = np.zeros((h, w), dtype=np.uint8)
    for label in clothing_labels:
        clothing_mask[seg_mask == label] = 255
    
    # Expand mask slightly to cover more area
    kernel = np.ones((15, 15), np.uint8)
    clothing_mask = cv2.dilate(clothing_mask, kernel, iterations=1)
    
    # Create agnostic image (gray out clothing regions)
    agnostic_img = person_img.copy()
    gray_color = np.array([128, 128, 128])
    
    # Apply gray color to clothing regions
    agnostic_img[clothing_mask > 0] = gray_color
    
    # Optional: Add edge blending for smoother transition
    # Create smooth transition at mask edges
    blur_kernel_size = 15
    clothing_mask_blur = cv2.GaussianBlur(clothing_mask.astype(float), (blur_kernel_size, blur_kernel_size), 0)
    clothing_mask_blur = clothing_mask_blur / 255.0
    
    # Blend original and grayed image
    for i in range(3):
        agnostic_img[:, :, i] = (
            person_img[:, :, i] * (1 - clothing_mask_blur) + 
            gray_color[i] * clothing_mask_blur
        ).astype(np.uint8)
    
    # Save agnostic image
    output_path = os.path.join(output_dir, Path(person_image_path).stem + '.jpg')
    cv2.imwrite(output_path, agnostic_img)
    
    # Also save the mask itself
    mask_output_path = os.path.join(output_dir, Path(person_image_path).stem + '_mask.png')
    cv2.imwrite(mask_output_path, clothing_mask)
    
    return True

def generate_simple_agnostic(person_image_path, output_dir):
    """
    Generate simple agnostic mask without segmentation
    Masks out central region where clothing typically is
    """
    # Read person image
    person_img = cv2.imread(str(person_image_path))
    if person_img is None:
        return False
    
    h, w = person_img.shape[:2]
    
    # Create mask for center torso region
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Define approximate clothing region (center of image)
    y_start = int(h * 0.2)   # Start below head
    y_end = int(h * 0.7)     # End at upper legs
    x_start = int(w * 0.25)  # Left side
    x_end = int(w * 0.75)    # Right side
    
    # Create elliptical mask for more natural shape
    center_x = w // 2
    center_y = (y_start + y_end) // 2
    axes_x = (x_end - x_start) // 2
    axes_y = (y_end - y_start) // 2
    
    cv2.ellipse(mask, (center_x, center_y), (axes_x, axes_y), 0, 0, 360, 255, -1)
    
    # Apply Gaussian blur for smooth edges
    mask_blur = cv2.GaussianBlur(mask.astype(float), (31, 31), 0) / 255.0
    
    # Create agnostic image
    agnostic_img = person_img.copy()
    gray_color = np.array([128, 128, 128])
    
    # Blend
    for i in range(3):
        agnostic_img[:, :, i] = (
            person_img[:, :, i] * (1 - mask_blur) + 
            gray_color[i] * mask_blur
        ).astype(np.uint8)
    
    # Save
    output_path = os.path.join(output_dir, Path(person_image_path).stem + '.jpg')
    cv2.imwrite(output_path, agnostic_img)
    
    return True

def process_dataset(data_dir):
    """Process all person images"""
    person_dir = os.path.join(data_dir, 'person')
    parse_dir = os.path.join(data_dir, 'person-parse')
    agnostic_dir = os.path.join(data_dir, 'agnostic-mask')
    
    # Create output directory
    os.makedirs(agnostic_dir, exist_ok=True)
    
    # Get all person images
    image_files = list(Path(person_dir).glob('*.jpg')) + list(Path(person_dir).glob('*.png'))
    
    if not image_files:
        print("❌ No person images found!")
        return
    
    print(f"📸 Processing {len(image_files)} person images...")
    print()
    
    # Check if segmentation masks exist
    has_segmentation = os.path.exists(parse_dir) and len(list(Path(parse_dir).glob('*.png'))) > 0
    
    if has_segmentation:
        print("✓ Using segmentation masks for precise agnostic generation")
    else:
        print("ℹ️  No segmentation masks found, using simple method")
    
    print()
    
    success_count = 0
    failed_images = []
    
    for image_path in tqdm(image_files, desc="Generating agnostic masks"):
        # Find corresponding segmentation mask
        parse_mask_path = os.path.join(parse_dir, image_path.stem + '.png')
        
        if generate_agnostic_mask(str(image_path), parse_mask_path, agnostic_dir):
            success_count += 1
        else:
            failed_images.append(image_path.name)
    
    print()
    print(f"✅ Successfully processed: {success_count}/{len(image_files)}")
    
    if failed_images:
        print(f"⚠️  Failed to process {len(failed_images)} images")
    
    print(f"📁 Output saved to: {agnostic_dir}")

def main():
    parser = argparse.ArgumentParser(description='Generate agnostic masks for person images')
    parser.add_argument('--data_dir', type=str, default='dataset/train',
                        help='Path to dataset directory (default: dataset/train)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  🎭 AGNOSTIC MASK GENERATOR")
    print("=" * 70)
    print()
    print("This script creates masked person images where clothing regions")
    print("are grayed out to guide the virtual try-on model.")
    print()
    
    process_dataset(args.data_dir)
    
    print()
    print("=" * 70)
    print("  ✅ AGNOSTIC MASK GENERATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()

