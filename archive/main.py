#!/usr/bin/env python3
"""
Virtual Try-On Dataset Builder - Main Entry Point
Run this script to build your complete dataset with clear progress tracking
"""

import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_banner():
    """Print welcome banner"""
    banner = f"""
{Colors.CYAN}{Colors.BOLD}
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║         🎯 VIRTUAL TRY-ON DATASET BUILDER                         ║
║                                                                    ║
║         Automatically build complete datasets for                  ║
║         VITON-HD, CP-VTON+, HR-VITON and similar models           ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
{Colors.ENDC}
"""
    print(banner)

def print_step_header(step_num, total_steps, title, emoji="🔹"):
    """Print formatted step header"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print()
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{emoji} STEP {step_num}/{total_steps}: {title}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.ENDC}")
    print(f"{Colors.YELLOW}⏰ Time: {timestamp}{Colors.ENDC}")
    print()

def print_success(message):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {message}{Colors.ENDC}")

def print_error(message):
    """Print error message"""
    print(f"{Colors.RED}❌ {message}{Colors.ENDC}")

def print_warning(message):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.ENDC}")

def print_info(message):
    """Print info message"""
    print(f"{Colors.CYAN}ℹ️  {message}{Colors.ENDC}")

def print_progress(message):
    """Print progress message"""
    print(f"{Colors.BLUE}▶️  {message}{Colors.ENDC}")

def check_dependencies():
    """Check if required packages are installed"""
    print_step_header(0, 6, "CHECKING DEPENDENCIES", "🔍")
    
    required_packages = {
        'cv2': 'opencv-python',
        'PIL': 'pillow',
        'numpy': 'numpy',
        'requests': 'requests',
        'aiohttp': 'aiohttp',
        'selenium': 'selenium',
        'undetected_chromedriver': 'undetected-chromedriver',
        'tqdm': 'tqdm'
    }
    
    optional_packages = {
        'torch': 'torch',
        'mediapipe': 'mediapipe'
    }
    
    missing_required = []
    missing_optional = []
    
    print_info("Checking required packages...")
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"  {Colors.GREEN}✓{Colors.ENDC} {package_name}")
        except ImportError:
            print(f"  {Colors.RED}✗{Colors.ENDC} {package_name} - MISSING")
            missing_required.append(package_name)
    
    print()
    print_info("Checking optional packages (for better results)...")
    for import_name, package_name in optional_packages.items():
        try:
            __import__(import_name)
            print(f"  {Colors.GREEN}✓{Colors.ENDC} {package_name}")
        except ImportError:
            print(f"  {Colors.YELLOW}⚠{Colors.ENDC} {package_name} - Not installed (optional)")
            missing_optional.append(package_name)
    
    print()
    
    if missing_required:
        print_error("Missing required packages!")
        print()
        print(f"Install with: {Colors.BOLD}pip install {' '.join(missing_required)}{Colors.ENDC}")
        print()
        print(f"Or install all: {Colors.BOLD}pip install -r requirements.txt{Colors.ENDC}")
        return False
    
    if missing_optional:
        print_warning("Some optional packages not installed")
        print(f"For better results: {Colors.BOLD}pip install -r requirements_ml.txt{Colors.ENDC}")
        print()
    
    print_success("All required dependencies installed!")
    return True

def check_files():
    """Check if all required scripts exist"""
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
        print_error("Missing required files:")
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

def run_script(script_name, args=None, description="", allow_failure=False):
    """Run a Python script with progress tracking"""
    command = [sys.executable, script_name]
    if args:
        command.extend(args)
    
    print_progress(f"Running: {Colors.BOLD}{' '.join(command)}{Colors.ENDC}")
    print()
    
    start_time = time.time()
    
    try:
        # Run with real-time output
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
        elapsed = time.time() - start_time
        
        print()
        if process.returncode == 0:
            print_success(f"{description} completed in {elapsed:.1f}s")
            return True
        else:
            if allow_failure:
                print_warning(f"{description} had issues but continuing...")
                return True
            else:
                print_error(f"{description} failed with code {process.returncode}")
                return False
            
    except KeyboardInterrupt:
        print()
        print_warning("Interrupted by user")
        return False
    except Exception as e:
        print_error(f"Error running {script_name}: {e}")
        return False

def show_summary(data_dir):
    """Show final dataset summary"""
    print()
    print(f"{Colors.BOLD}{Colors.GREEN}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.GREEN}  🎉 DATASET BUILD COMPLETE!{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.GREEN}{'='*70}{Colors.ENDC}")
    print()
    
    # Count files
    person_dir = os.path.join(data_dir, 'person')
    garment_dir = os.path.join(data_dir, 'garment')
    openpose_dir = os.path.join(data_dir, 'openpose')
    densepose_dir = os.path.join(data_dir, 'densepose')
    parse_dir = os.path.join(data_dir, 'person-parse')
    agnostic_dir = os.path.join(data_dir, 'agnostic-mask')
    
    person_count = count_images(person_dir)
    garment_count = count_images(garment_dir)
    openpose_count = count_images(openpose_dir)
    densepose_count = count_images(densepose_dir)
    parse_count = count_images(parse_dir)
    agnostic_count = count_images(agnostic_dir)
    
    print(f"{Colors.BOLD}📊 Dataset Statistics:{Colors.ENDC}")
    print(f"   {Colors.CYAN}Person images:{Colors.ENDC}       {person_count}")
    print(f"   {Colors.CYAN}Garment images:{Colors.ENDC}      {garment_count}")
    print(f"   {Colors.CYAN}OpenPose files:{Colors.ENDC}      {openpose_count}")
    print(f"   {Colors.CYAN}DensePose files:{Colors.ENDC}     {densepose_count}")
    print(f"   {Colors.CYAN}Segmentation masks:{Colors.ENDC}  {parse_count}")
    print(f"   {Colors.CYAN}Agnostic masks:{Colors.ENDC}      {agnostic_count}")
    print()
    
    print(f"{Colors.BOLD}📁 Dataset Location:{Colors.ENDC}")
    print(f"   {os.path.abspath(data_dir)}")
    print()
    
    print(f"{Colors.BOLD}🎯 Ready for Training!{Colors.ENDC}")
    print()
    print(f"Your dataset is compatible with:")
    print(f"  {Colors.GREEN}•{Colors.ENDC} VITON-HD")
    print(f"  {Colors.GREEN}•{Colors.ENDC} CP-VTON+")
    print(f"  {Colors.GREEN}•{Colors.ENDC} HR-VITON")
    print(f"  {Colors.GREEN}•{Colors.ENDC} Other virtual try-on models")
    print()
    
    print(f"{Colors.BOLD}📝 Next Steps:{Colors.ENDC}")
    print(f"  1. Review generated annotations for quality")
    print(f"  2. Create pairs.txt to match person-garment images")
    print(f"  3. Train your virtual try-on model")
    print()
    
    print(f"{Colors.YELLOW}💡 Tip:{Colors.ENDC} Run {Colors.BOLD}python verify_dataset.py --inspect{Colors.ENDC} to inspect the dataset")
    print()

def main():
    """Main execution flow"""
    start_time = time.time()
    
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        print()
        print_error("Please install missing dependencies first")
        print(f"Run: {Colors.BOLD}pip install -r requirements.txt{Colors.ENDC}")
        return 1
    
    time.sleep(1)
    
    # Check required files
    if not check_files():
        print_error("Missing required files")
        return 1
    
    print()
    print_info("All checks passed! Starting dataset build...")
    print()
    
    # Ask for confirmation
    try:
        response = input(f"{Colors.BOLD}Press Enter to start building the dataset (or Ctrl+C to cancel)...{Colors.ENDC}")
    except KeyboardInterrupt:
        print()
        print_warning("Build cancelled by user")
        return 0
    
    data_dir = 'dataset/train'
    
    # STEP 1: Scrape images
    print_step_header(1, 6, "SCRAPING PRODUCT IMAGES", "🛍️")
    
    print_info("Downloading images from Google Shopping...")
    print_info("This may take 10-20 minutes depending on network speed")
    print()
    
    if not run_script('scraper_improved.py', description="Image scraping", allow_failure=True):
        # Check if we have any images despite failure
        person_count = count_images(os.path.join(data_dir, 'person'))
        garment_count = count_images(os.path.join(data_dir, 'garment'))
        
        if person_count == 0 or garment_count == 0:
            print_error("No images found. Cannot continue.")
            return 1
        
        print_warning(f"Found {person_count} person and {garment_count} garment images")
        print_info("Continuing with existing images...")
    
    # Check image counts
    person_count = count_images(os.path.join(data_dir, 'person'))
    garment_count = count_images(os.path.join(data_dir, 'garment'))
    
    print()
    print_success(f"Images collected: {person_count} person, {garment_count} garment")
    time.sleep(2)
    
    # STEP 2: Preprocess
    print_step_header(2, 6, "PREPROCESSING IMAGES", "🔧")
    
    print_info("Resizing images to 768×1024...")
    print()
    
    if not run_script('preprocess.py', 
                     args=['--data_dir', data_dir, '--resize'],
                     description="Image preprocessing"):
        print_error("Preprocessing failed")
        return 1
    
    time.sleep(1)
    
    # STEP 3: OpenPose
    print_step_header(3, 6, "GENERATING POSE KEYPOINTS", "🦴")
    
    print_info("Detecting human poses and generating 18-point skeletons...")
    print()
    
    run_script('generate_openpose.py',
              args=['--data_dir', data_dir],
              description="OpenPose generation",
              allow_failure=True)
    
    time.sleep(1)
    
    # STEP 4: DensePose
    print_step_header(4, 6, "GENERATING BODY MASKS", "🎯")
    
    print_info("Creating body surface maps...")
    print()
    
    run_script('generate_densepose.py',
              args=['--data_dir', data_dir, '--simple'],
              description="DensePose generation",
              allow_failure=True)
    
    time.sleep(1)
    
    # STEP 5: Segmentation
    print_step_header(5, 6, "GENERATING SEGMENTATION MASKS", "🎨")
    
    print_info("Segmenting body parts (face, hair, clothes, arms, legs, etc.)...")
    print()
    
    run_script('generate_parse.py',
              args=['--data_dir', data_dir],
              description="Segmentation generation",
              allow_failure=True)
    
    time.sleep(1)
    
    # STEP 6: Agnostic masks
    print_step_header(6, 6, "GENERATING AGNOSTIC MASKS", "🎭")
    
    print_info("Creating masked person images for try-on guidance...")
    print()
    
    run_script('generate_agnostic.py',
              args=['--data_dir', data_dir],
              description="Agnostic mask generation",
              allow_failure=True)
    
    # Show summary
    total_time = time.time() - start_time
    show_summary(data_dir)
    
    print(f"{Colors.BOLD}⏱️  Total build time: {total_time/60:.1f} minutes{Colors.ENDC}")
    print()
    print(f"{Colors.GREEN}{Colors.BOLD}🎊 Success! Your dataset is ready for training! 🎊{Colors.ENDC}")
    print()
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        print()
        print_warning("Build interrupted by user")
        print()
        sys.exit(1)
    except Exception as e:
        print()
        print_error(f"Unexpected error: {e}")
        print()
        sys.exit(1)

