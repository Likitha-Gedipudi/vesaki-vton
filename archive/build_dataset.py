"""
Virtual Try-On Dataset Builder - Main Pipeline
Orchestrates the entire dataset building process
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

BANNER = """
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║         🎯 VIRTUAL TRY-ON DATASET BUILDER                         ║
║                                                                    ║
║         Build complete datasets for VITON-HD, CP-VTON+            ║
║         and other virtual try-on models                           ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
"""

def print_step(step_num, total_steps, title):
    """Print formatted step header"""
    print()
    print("=" * 70)
    print(f"  [{step_num}/{total_steps}] {title}")
    print("=" * 70)
    print()

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🚀 {description}...")
    print(f"   Command: {' '.join(command)}")
    print()
    
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=False,
            text=True
        )
        print()
        print(f"✅ {description} - Complete!")
        return True
    except subprocess.CalledProcessError as e:
        print()
        print(f"❌ {description} - Failed!")
        print(f"   Error: {e}")
        return False
    except FileNotFoundError:
        print()
        print(f"❌ {description} - Script not found!")
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    print_step(0, 6, "CHECKING DEPENDENCIES")
    
    required_packages = {
        'opencv-python': 'cv2',
        'pillow': 'PIL',
        'numpy': 'numpy',
        'requests': 'requests',
        'aiohttp': 'aiohttp',
        'selenium': 'selenium',
        'undetected-chromedriver': 'undetected_chromedriver',
        'tqdm': 'tqdm'
    }
    
    optional_packages = {
        'torch': 'torch',
        'mediapipe': 'mediapipe'
    }
    
    missing_required = []
    missing_optional = []
    
    print("Checking required packages...")
    for package, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - MISSING")
            missing_required.append(package)
    
    print()
    print("Checking optional packages (for better results)...")
    for package, import_name in optional_packages.items():
        try:
            __import__(import_name)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ⚠ {package} - Not installed (optional)")
            missing_optional.append(package)
    
    print()
    
    if missing_required:
        print("❌ Missing required packages!")
        print()
        print("Install with:")
        print(f"   pip install {' '.join(missing_required)}")
        print()
        print("Or install all requirements:")
        print("   pip install -r requirements.txt")
        return False
    
    if missing_optional:
        print("ℹ️  Optional packages not installed:")
        for pkg in missing_optional:
            print(f"   - {pkg}")
        print()
        print("Install for better results:")
        print("   pip install -r requirements_ml.txt")
        print()
    
    print("✅ All required dependencies are installed!")
    return True

def check_files():
    """Check if required files exist"""
    required_files = [
        'product_urls.txt',
        'haarcascade_frontalface_default.xml',
        'scraper_improved.py',
        'preprocess.py',
        'generate_openpose.py',
        'generate_densepose.py',
        'generate_parse.py',
        'generate_agnostic.py'
    ]
    
    missing = []
    for file in required_files:
        if not os.path.exists(file):
            missing.append(file)
    
    if missing:
        print("❌ Missing required files:")
        for file in missing:
            print(f"   - {file}")
        return False
    
    return True

def count_images(directory):
    """Count images in a directory"""
    if not os.path.exists(directory):
        return 0
    
    image_extensions = {'.jpg', '.jpeg', '.png'}
    count = 0
    
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            count += 1
    
    return count

def main():
    parser = argparse.ArgumentParser(description='Build Virtual Try-On Dataset')
    parser.add_argument('--data_dir', type=str, default='dataset/train',
                        help='Path to dataset directory (default: dataset/train)')
    parser.add_argument('--skip_scrape', action='store_true',
                        help='Skip scraping (use existing images)')
    parser.add_argument('--scrape_only', action='store_true',
                        help='Only run scraping step')
    parser.add_argument('--all', action='store_true',
                        help='Run complete pipeline')
    
    args = parser.parse_args()
    
    print(BANNER)
    
    # Check dependencies
    if not check_dependencies():
        print("Please install missing dependencies first.")
        return
    
    time.sleep(1)
    
    # Check files
    if not check_files():
        print("Please ensure all required files are present.")
        return
    
    print()
    input("Press Enter to start building the dataset...")
    
    # Step 1: Scrape images
    if not args.skip_scrape:
        print_step(1, 6, "SCRAPING PRODUCT IMAGES")
        
        if not run_command(
            [sys.executable, 'scraper_improved.py'],
            "Scraping product images from Google Shopping"
        ):
            print("⚠️  Scraping failed or was interrupted")
            
            person_count = count_images(os.path.join(args.data_dir, 'person'))
            garment_count = count_images(os.path.join(args.data_dir, 'garment'))
            
            if person_count == 0 or garment_count == 0:
                print("Cannot continue without images. Exiting.")
                return
            
            print(f"Found {person_count} person and {garment_count} garment images")
            print("Continuing with existing images...")
    else:
        print("⏭️  Skipping scraping step")
    
    if args.scrape_only:
        print()
        print("✅ Scraping complete! Run without --scrape_only to continue.")
        return
    
    # Check if we have images
    person_count = count_images(os.path.join(args.data_dir, 'person'))
    garment_count = count_images(os.path.join(args.data_dir, 'garment'))
    
    if person_count == 0:
        print("❌ No person images found! Run scraper first.")
        return
    
    print()
    print(f"📊 Dataset: {person_count} person images, {garment_count} garment images")
    print()
    
    # Step 2: Preprocess (resize)
    print_step(2, 6, "PREPROCESSING IMAGES")
    
    if not run_command(
        [sys.executable, 'preprocess.py', '--data_dir', args.data_dir, '--resize'],
        "Resizing images to 768x1024"
    ):
        print("Failed to preprocess images. Exiting.")
        return
    
    # Step 3: Generate OpenPose keypoints
    print_step(3, 6, "GENERATING OPENPOSE KEYPOINTS")
    
    if not run_command(
        [sys.executable, 'generate_openpose.py', '--data_dir', args.data_dir],
        "Generating pose keypoints"
    ):
        print("⚠️  OpenPose generation failed")
        print("Continuing to next step...")
    
    # Step 4: Generate DensePose
    print_step(4, 6, "GENERATING DENSEPOSE MAPS")
    
    if not run_command(
        [sys.executable, 'generate_densepose.py', '--data_dir', args.data_dir, '--simple'],
        "Generating body masks"
    ):
        print("⚠️  DensePose generation failed")
        print("Continuing to next step...")
    
    # Step 5: Generate segmentation masks
    print_step(5, 6, "GENERATING SEGMENTATION MASKS")
    
    if not run_command(
        [sys.executable, 'generate_parse.py', '--data_dir', args.data_dir],
        "Generating human parsing segmentation"
    ):
        print("⚠️  Segmentation generation failed")
        print("Continuing to next step...")
    
    # Step 6: Generate agnostic masks
    print_step(6, 6, "GENERATING AGNOSTIC MASKS")
    
    if not run_command(
        [sys.executable, 'generate_agnostic.py', '--data_dir', args.data_dir],
        "Generating agnostic masks"
    ):
        print("⚠️  Agnostic mask generation failed")
    
    # Summary
    print()
    print("=" * 70)
    print("  🎉 DATASET BUILDING COMPLETE!")
    print("=" * 70)
    print()
    
    # Count files in each directory
    person_dir = os.path.join(args.data_dir, 'person')
    garment_dir = os.path.join(args.data_dir, 'garment')
    openpose_dir = os.path.join(args.data_dir, 'openpose')
    densepose_dir = os.path.join(args.data_dir, 'densepose')
    parse_dir = os.path.join(args.data_dir, 'person-parse')
    agnostic_dir = os.path.join(args.data_dir, 'agnostic-mask')
    
    print("📊 Dataset Statistics:")
    print(f"   Person images:       {count_images(person_dir)}")
    print(f"   Garment images:      {count_images(garment_dir)}")
    print(f"   OpenPose files:      {count_images(openpose_dir)}")
    print(f"   DensePose files:     {count_images(densepose_dir)}")
    print(f"   Segmentation masks:  {count_images(parse_dir)}")
    print(f"   Agnostic masks:      {count_images(agnostic_dir)}")
    print()
    
    print(f"📁 Dataset location: {os.path.abspath(args.data_dir)}")
    print()
    
    print("🎯 Your dataset is now ready for virtual try-on training!")
    print()
    print("Compatible with:")
    print("  • VITON-HD")
    print("  • CP-VTON+")
    print("  • HR-VITON")
    print("  • Other virtual try-on models")
    print()
    
    print("📝 Next steps:")
    print("  1. Review the generated annotations for quality")
    print("  2. Create pairs.txt matching person and garment images")
    print("  3. Train your virtual try-on model")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Build interrupted by user")
        sys.exit(1)

