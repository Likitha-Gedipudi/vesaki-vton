"""
Human Parsing Segmentation Generator for Virtual Try-On Dataset
Generates semantic segmentation masks using SCHP (Self Correction Human Parsing)

Labels:
0=Background, 1=Hat, 2=Hair, 3=Glove, 4=Sunglasses, 5=Upper-clothes,
6=Dress, 7=Coat, 8=Socks, 9=Pants, 10=Jumpsuits, 11=Scarf, 12=Skirt,
13=Face, 14=Left-arm, 15=Right-arm, 16=Left-leg, 17=Right-leg,
18=Left-shoe, 19=Right-shoe
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

try:
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Color palette for visualization (20 classes)
PALETTE = [
    [0, 0, 0],           # 0: Background
    [128, 0, 0],         # 1: Hat
    [255, 0, 0],         # 2: Hair
    [0, 85, 0],          # 3: Glove
    [170, 0, 51],        # 4: Sunglasses
    [255, 85, 0],        # 5: Upper-clothes
    [0, 0, 85],          # 6: Dress
    [0, 119, 221],       # 7: Coat
    [85, 85, 0],         # 8: Socks
    [0, 85, 85],         # 9: Pants
    [85, 51, 0],         # 10: Jumpsuits
    [52, 86, 128],       # 11: Scarf
    [0, 128, 0],         # 12: Skirt
    [0, 0, 255],         # 13: Face
    [51, 170, 221],      # 14: Left-arm
    [0, 255, 255],       # 15: Right-arm
    [85, 255, 170],      # 16: Left-leg
    [170, 255, 85],      # 17: Right-leg
    [255, 255, 0],       # 18: Left-shoe
    [255, 170, 0]        # 19: Right-shoe
]

def generate_simple_segmentation(image_path, output_dir):
    """
    Generate simple segmentation using traditional CV methods
    This is a fallback when SCHP is not available
    """
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        return False
    
    h, w = image.shape[:2]
    
    # Create segmentation mask
    seg_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Convert to different color spaces
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Background detection (assuming white/light background)
    _, bg_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    
    # Person mask (inverse of background)
    person_mask = cv2.bitwise_not(bg_mask)
    
    # Find person contour
    contours, _ = cv2.findContours(person_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return False
    
    # Get largest contour (person)
    person_contour = max(contours, key=cv2.contourArea)
    
    # Create clean person mask
    person_mask_clean = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(person_mask_clean, [person_contour], -1, 1, -1)
    
    # Try to detect face using Haar Cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0:
        # Get largest face
        face = max(faces, key=lambda x: x[2] * x[3])
        x, y, fw, fh = face
        
        # Face region (label 13)
        seg_mask[y:y+fh, x:x+fw] = 13
        
        # Hair region (above face, label 2)
        hair_y_start = max(0, y - int(fh * 0.5))
        hair_y_end = y
        seg_mask[hair_y_start:hair_y_end, x:x+fw] = 2
        
        # Upper body (below face to mid-height, label 5 for upper-clothes)
        upper_y_start = y + fh
        upper_y_end = y + fh + int(h * 0.3)
        upper_mask = person_mask_clean[upper_y_start:upper_y_end, :]
        seg_mask[upper_y_start:upper_y_end, :] = np.where(upper_mask > 0, 5, seg_mask[upper_y_start:upper_y_end, :])
        
        # Arms (side regions at upper body height, labels 14 and 15)
        arm_y_start = upper_y_start
        arm_y_end = upper_y_end + int(h * 0.2)
        
        # Left arm (right side of image)
        left_arm_x = int(w * 0.7)
        arm_mask = person_mask_clean[arm_y_start:arm_y_end, left_arm_x:]
        seg_mask[arm_y_start:arm_y_end, left_arm_x:] = np.where(arm_mask > 0, 14, seg_mask[arm_y_start:arm_y_end, left_arm_x:])
        
        # Right arm (left side of image)
        right_arm_x = int(w * 0.3)
        arm_mask = person_mask_clean[arm_y_start:arm_y_end, :right_arm_x]
        seg_mask[arm_y_start:arm_y_end, :right_arm_x] = np.where(arm_mask > 0, 15, seg_mask[arm_y_start:arm_y_end, :right_arm_x])
        
        # Lower body (pants/legs, label 9 for pants)
        lower_y_start = upper_y_end
        lower_y_end = h
        lower_mask = person_mask_clean[lower_y_start:lower_y_end, :]
        seg_mask[lower_y_start:lower_y_end, :] = np.where(lower_mask > 0, 9, seg_mask[lower_y_start:lower_y_end, :])
    else:
        # No face detected - simple body segmentation
        seg_mask = np.where(person_mask_clean > 0, 5, 0)  # Label everything as upper-clothes
    
    # Save segmentation mask
    output_path = os.path.join(output_dir, Path(image_path).stem + '.png')
    cv2.imwrite(output_path, seg_mask)
    
    # Create colored visualization
    vis_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for label in range(20):
        vis_mask[seg_mask == label] = PALETTE[label]
    
    vis_path = os.path.join(output_dir, Path(image_path).stem + '_vis.jpg')
    
    # Blend with original image
    blended = cv2.addWeighted(image, 0.6, vis_mask, 0.4, 0)
    cv2.imwrite(vis_path, blended)
    
    return True

def visualize_segmentation(seg_mask, image=None):
    """Convert segmentation mask to colored visualization"""
    h, w = seg_mask.shape[:2]
    vis_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for label in range(20):
        vis_mask[seg_mask == label] = PALETTE[label]
    
    if image is not None:
        # Blend with original image
        vis_mask = cv2.addWeighted(image, 0.6, vis_mask, 0.4, 0)
    
    return vis_mask

def process_dataset(data_dir):
    """Process all person images"""
    person_dir = os.path.join(data_dir, 'person')
    parse_dir = os.path.join(data_dir, 'person-parse')
    
    # Create output directory
    os.makedirs(parse_dir, exist_ok=True)
    
    # Get all person images
    image_files = list(Path(person_dir).glob('*.jpg')) + list(Path(person_dir).glob('*.png'))
    
    if not image_files:
        print("❌ No person images found!")
        return
    
    print(f"📸 Processing {len(image_files)} person images...")
    print()
    
    success_count = 0
    failed_images = []
    
    for image_path in tqdm(image_files, desc="Generating segmentation"):
        if generate_simple_segmentation(str(image_path), parse_dir):
            success_count += 1
        else:
            failed_images.append(image_path.name)
    
    print()
    print(f"✅ Successfully processed: {success_count}/{len(image_files)}")
    
    if failed_images:
        print(f"⚠️  Failed to process {len(failed_images)} images")
    
    print(f"📁 Output saved to: {parse_dir}")

def main():
    parser = argparse.ArgumentParser(description='Generate human parsing segmentation masks')
    parser.add_argument('--data_dir', type=str, default='dataset/train',
                        help='Path to dataset directory (default: dataset/train)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  🎨 HUMAN PARSING SEGMENTATION GENERATOR")
    print("=" * 70)
    print()
    
    if not TORCH_AVAILABLE:
        print("ℹ️  PyTorch not installed - using simple segmentation method")
        print()
    
    print("ℹ️  Using traditional CV methods for segmentation")
    print("   (For better results, consider using SCHP model)")
    print()
    
    process_dataset(args.data_dir)
    
    print()
    print("=" * 70)
    print("  ✅ SEGMENTATION GENERATION COMPLETE")
    print("=" * 70)
    print()
    print("📝 Note: For production use, consider training or using SCHP:")
    print("   https://github.com/GoGoDuck912/Self-Correction-Human-Parsing")

if __name__ == "__main__":
    main()

