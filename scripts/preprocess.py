"""
Preprocessing Script for Virtual Try-On Dataset
Handles image resizing and prepares images for annotation
"""

import os
import cv2
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

# Target dimensions for VITON-HD
TARGET_WIDTH = 768
TARGET_HEIGHT = 1024

def resize_and_pad(image_path, output_path, target_width=TARGET_WIDTH, target_height=TARGET_HEIGHT):
    """
    Resize image to target dimensions while maintaining aspect ratio
    Pads with white background if needed
    """
    try:
        # Open image
        img = Image.open(image_path)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Calculate aspect ratios
        img_aspect = img.width / img.height
        target_aspect = target_width / target_height
        
        # Resize while maintaining aspect ratio
        if img_aspect > target_aspect:
            # Image is wider - fit to width
            new_width = target_width
            new_height = int(target_width / img_aspect)
        else:
            # Image is taller - fit to height
            new_height = target_height
            new_width = int(target_height * img_aspect)
        
        img_resized = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Create white canvas
        canvas = Image.new('RGB', (target_width, target_height), (255, 255, 255))
        
        # Paste resized image in center
        offset_x = (target_width - new_width) // 2
        offset_y = (target_height - new_height) // 2
        canvas.paste(img_resized, (offset_x, offset_y))
        
        # Save
        canvas.save(output_path, quality=95)
        return True
        
    except Exception as e:
        print(f"  Error processing {image_path}: {e}")
        return False

def process_directory(input_dir, output_dir=None):
    """Process all images in a directory"""
    if output_dir is None:
        output_dir = input_dir
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(f'*{ext}'))
        image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"  No images found in {input_dir}")
        return 0
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process images
    success_count = 0
    print(f"  Processing {len(image_files)} images...")
    
    for img_path in tqdm(image_files, desc="  Resizing"):
        output_path = os.path.join(output_dir, img_path.name)
        if resize_and_pad(str(img_path), output_path):
            success_count += 1
    
    return success_count

def verify_dataset_structure(data_dir):
    """Check if dataset has proper structure"""
    required_dirs = ['person', 'garment']
    optional_dirs = ['person-parse', 'densepose', 'openpose', 'agnostic-mask']
    
    missing = []
    for dir_name in required_dirs:
        dir_path = os.path.join(data_dir, dir_name)
        if not os.path.exists(dir_path):
            missing.append(dir_name)
    
    if missing:
        print(f"❌ Missing required directories: {', '.join(missing)}")
        return False
    
    return True

def count_images(directory):
    """Count images in a directory"""
    if not os.path.exists(directory):
        return 0
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    count = 0
    
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            count += 1
    
    return count

def check_image_dimensions(directory):
    """Check if images are properly sized"""
    image_files = list(Path(directory).glob('*.jpg')) + list(Path(directory).glob('*.png'))
    
    if not image_files:
        return True
    
    # Check first few images
    sample_size = min(5, len(image_files))
    correctly_sized = 0
    
    for img_path in image_files[:sample_size]:
        try:
            img = Image.open(img_path)
            if img.width == TARGET_WIDTH and img.height == TARGET_HEIGHT:
                correctly_sized += 1
        except:
            pass
    
    return correctly_sized == sample_size

def main():
    parser = argparse.ArgumentParser(description='Preprocess images for Virtual Try-On dataset')
    parser.add_argument('--data_dir', type=str, default='dataset/train',
                        help='Path to dataset directory (default: dataset/train)')
    parser.add_argument('--resize', action='store_true',
                        help='Resize images to target dimensions (768x1024)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify dataset structure and image sizes')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  🔧 VIRTUAL TRY-ON DATASET PREPROCESSOR")
    print("=" * 70)
    print()
    
    # Verify dataset structure
    if not verify_dataset_structure(args.data_dir):
        print("\n💡 Expected directory structure:")
        print("  dataset/train/")
        print("    ├── person/")
        print("    └── garment/")
        return
    
    print(f"📁 Dataset directory: {args.data_dir}")
    print()
    
    # Count existing images
    person_count = count_images(os.path.join(args.data_dir, 'person'))
    garment_count = count_images(os.path.join(args.data_dir, 'garment'))
    
    print(f"📊 Current dataset:")
    print(f"   Person images:  {person_count}")
    print(f"   Garment images: {garment_count}")
    print()
    
    if person_count == 0 and garment_count == 0:
        print("⚠️  No images found! Run the scraper first:")
        print("   python scraper_improved.py")
        return
    
    # Verification mode
    if args.verify:
        print("🔍 Verifying dataset...")
        
        person_dir = os.path.join(args.data_dir, 'person')
        garment_dir = os.path.join(args.data_dir, 'garment')
        
        person_ok = check_image_dimensions(person_dir)
        garment_ok = check_image_dimensions(garment_dir)
        
        print(f"   Person images:  {'✅ Correctly sized' if person_ok else '❌ Need resizing'}")
        print(f"   Garment images: {'✅ Correctly sized' if garment_ok else '❌ Need resizing'}")
        
        if not person_ok or not garment_ok:
            print("\n💡 Run with --resize to fix dimensions:")
            print("   python preprocess.py --data_dir dataset/train --resize")
        
        return
    
    # Resize mode
    if args.resize:
        print(f"🔄 Resizing images to {TARGET_WIDTH}x{TARGET_HEIGHT}...")
        print()
        
        # Process person images
        print("📸 Processing person images...")
        person_dir = os.path.join(args.data_dir, 'person')
        person_success = process_directory(person_dir)
        print(f"   ✅ Processed {person_success} person images")
        print()
        
        # Process garment images
        print("👔 Processing garment images...")
        garment_dir = os.path.join(args.data_dir, 'garment')
        garment_success = process_directory(garment_dir)
        print(f"   ✅ Processed {garment_success} garment images")
        print()
        
        print("=" * 70)
        print("  ✅ PREPROCESSING COMPLETE")
        print("=" * 70)
        print()
        print("📝 Next steps:")
        print("   1. Generate OpenPose annotations:  python generate_openpose.py")
        print("   2. Generate DensePose maps:        python generate_densepose.py")
        print("   3. Generate segmentation masks:    python generate_parse.py")
        print("   4. Generate agnostic masks:        python generate_agnostic.py")
        print()
        print("   Or run all at once:")
        print("   python build_dataset.py --all")
        
    else:
        print("💡 Usage:")
        print("   • Resize images:        python preprocess.py --data_dir dataset/train --resize")
        print("   • Verify dataset:       python preprocess.py --data_dir dataset/train --verify")
        print()

if __name__ == "__main__":
    main()

