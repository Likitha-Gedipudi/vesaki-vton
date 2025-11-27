#!/usr/bin/env python3
"""
Vesaki-VTON - Dataset Builder Pipeline
Orchestrates the complete dataset building process with state-of-the-art models
"""

import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path

# ANSI color codes
class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

def print_banner():
    """Print welcome banner"""
    banner = f"""
{Colors.CYAN}{Colors.BOLD}
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║      Vesaki-VTON Dataset Builder Pipeline                        ║
║                                                                    ║
║      State-of-the-art models for maximum accuracy                 ║
║      • SCHP Human Parsing                                         ║
║      • MediaPipe Holistic Pose                                    ║
║      • Extensive Data Augmentation (5-8x)                         ║
║      • Quality Filtering                                          ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
{Colors.ENDC}
"""
    print(banner)

def print_step(step_num, total_steps, title, emoji="🔹"):
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

def check_dependencies():
    """Check if required packages are installed"""
    print_step(0, 11, "CHECKING DEPENDENCIES", "🔍")
    
    required_packages = {
        'cv2': 'opencv-python',
        'PIL': 'pillow',
        'numpy': 'numpy',
        'torch': 'torch',
        'mediapipe': 'mediapipe',
        'albumentations': 'albumentations',
        'gdown': 'gdown'
    }
    
    missing = []
    
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"  {Colors.GREEN}✓{Colors.ENDC} {package_name}")
        except ImportError:
            print(f"  {Colors.RED}✗{Colors.ENDC} {package_name} - MISSING")
            missing.append(package_name)
    
    print()
    
    if missing:
        print_error("Missing required packages!")
        print()
        print("Install with:")
        print(f"   {Colors.BOLD}pip3 install -r requirements_advanced.txt{Colors.ENDC}")
        return False
    
    print_success("All dependencies installed!")
    return True

def check_models():
    """Check if model weights are downloaded"""
    print()
    print_info("Checking model weights...")
    
    model_dir = "models"
    required_models = [
        "exp-schp-201908261155-atr.pth",
        "exp-schp-201908301523-lip.pth"
    ]
    
    if not os.path.exists(model_dir):
        print_warning("Models directory not found")
        print(f"Run: {Colors.BOLD}python3 download_models.py{Colors.ENDC}")
        return False
    
    missing_models = []
    for model_file in required_models:
        model_path = os.path.join(model_dir, model_file)
        if os.path.exists(model_path):
            print(f"  {Colors.GREEN}✓{Colors.ENDC} {model_file}")
        else:
            print(f"  {Colors.RED}✗{Colors.ENDC} {model_file} - MISSING")
            missing_models.append(model_file)
    
    if missing_models:
        print()
        print_warning("Some models are missing")
        print(f"Run: {Colors.BOLD}python3 download_models.py{Colors.ENDC}")
        return False
    
    print()
    print_success("All models available!")
    return True

def run_script(script_name, args=None, description="", allow_failure=False):
    """Run a Python script"""
    # Scripts are now in scripts/ directory
    script_path = os.path.join('scripts', script_name)
    command = [sys.executable, script_path]
    if args:
        command.extend(args)
    
    print_info(f"Running: {Colors.BOLD}{' '.join(command)}{Colors.ENDC}")
    print()
    
    start_time = time.time()
    
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
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
                print_error(f"{description} failed")
                return False
            
    except Exception as e:
        print_error(f"Error: {e}")
        return False

def count_images(directory):
    """Count images in directory"""
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory) if f.endswith(('.jpg', '.png'))])

def main():
    """Main execution flow"""
    start_time = time.time()
    
    print_banner()
    
    # Step 0: Check dependencies
    if not check_dependencies():
        return 1
    
    if not check_models():
        response = input(f"\n{Colors.BOLD}Continue without SCHP models? (y/n): {Colors.ENDC}")
        if response.lower() != 'y':
            print_info("Download models and run again")
            return 1
    
    time.sleep(1)
    
    print()
    print(f"{Colors.BOLD}This pipeline will:{Colors.ENDC}")
    print("  1. Scrape 2000+ images from Google Shopping")
    print("  2. Apply quality filtering")
    print("  3. Generate 5-8x augmented variations")
    print("  4. Resize to 768×1024")
    print("  5. Generate SCHP human parsing")
    print("  6. Generate MediaPipe Holistic pose")
    print("  7. Generate advanced body segmentation")
    print("  8. Generate agnostic masks")
    print("  9. Apply quality post-filtering")
    print("  10. Validate dataset")
    print("  11. Export metadata")
    print()
    print(f"{Colors.YELLOW}Estimated time: 2-4 hours{Colors.ENDC}")
    print()
    
    try:
        input(f"{Colors.BOLD}Press Enter to start...{Colors.ENDC}")
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
        return 0
    
    data_dir = 'dataset/train'
    
    # STEP 1: Scrape images
    print_step(1, 11, "SCRAPING PRODUCT IMAGES", "🛍️")
    if not run_script('scraper_improved.py', description="Image scraping", allow_failure=True):
        person_count = count_images(os.path.join(data_dir, 'person'))
        if person_count == 0:
            print_error("No images collected")
            return 1
    
    # STEP 2: Pre-filter quality
    print_step(2, 11, "PRE-FILTERING LOW QUALITY IMAGES", "🔍")
    run_script('filter_quality.py', 
               args=['--data_dir', data_dir],
               description="Quality pre-filtering",
               allow_failure=True)
    
    # STEP 3: Data augmentation
    print_step(3, 11, "DATA AUGMENTATION", "🔄")
    if not run_script('augment_images.py',
                     args=['--data_dir', data_dir],
                     description="Image augmentation"):
        print_warning("Augmentation failed, continuing with original images")
    
    # STEP 4: Resize images
    print_step(4, 11, "RESIZING IMAGES", "🔧")
    if not run_script('preprocess.py',
                     args=['--data_dir', data_dir, '--resize'],
                     description="Image resizing"):
        print_error("Resizing failed")
        return 1
    
    # STEP 5: Generate SCHP parsing
    print_step(5, 11, "GENERATING HUMAN PARSING (SCHP)", "🎨")
    run_script('generate_parse_schp.py',
              args=['--data_dir', data_dir],
              description="SCHP parsing",
              allow_failure=True)
    
    # STEP 6: Generate holistic pose
    print_step(6, 11, "GENERATING POSE KEYPOINTS (HOLISTIC)", "🦴")
    run_script('generate_openpose_holistic.py',
              args=['--data_dir', data_dir],
              description="Holistic pose",
              allow_failure=True)
    
    # STEP 7: Generate advanced body segmentation
    print_step(7, 11, "GENERATING BODY SEGMENTATION", "🎯")
    run_script('generate_densepose_advanced.py',
              args=['--data_dir', data_dir],
              description="Body segmentation",
              allow_failure=True)
    
    # STEP 8: Generate agnostic masks
    print_step(8, 11, "GENERATING AGNOSTIC MASKS", "🎭")
    run_script('generate_agnostic_advanced.py',
              args=['--data_dir', data_dir],
              description="Agnostic masks",
              allow_failure=True)
    
    # STEP 9: Post-filter quality
    print_step(9, 11, "POST-FILTERING QUALITY", "✨")
    run_script('filter_quality.py',
              args=['--data_dir', data_dir],
              description="Quality post-filtering",
              allow_failure=True)
    
    # STEP 10: Validate dataset
    print_step(10, 11, "VALIDATING DATASET", "✅")
    run_script('validate_advanced.py',
              args=['--data_dir', data_dir],
              description="Dataset validation",
              allow_failure=True)
    
    # STEP 11: Export metadata
    print_step(11, 11, "EXPORTING METADATA", "📋")
    run_script('export_metadata.py',
              args=['--data_dir', data_dir],
              description="Metadata export",
              allow_failure=True)
    
    # Final summary
    total_time = time.time() - start_time
    
    print()
    print(f"{Colors.BOLD}{Colors.GREEN}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.GREEN}  🎉 ADVANCED PIPELINE COMPLETE!{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.GREEN}{'='*70}{Colors.ENDC}")
    print()
    
    person_count = count_images(os.path.join(data_dir, 'person'))
    garment_count = count_images(os.path.join(data_dir, 'garment'))
    
    print(f"{Colors.BOLD}📊 Final Statistics:{Colors.ENDC}")
    print(f"   Person images:  {person_count}")
    print(f"   Garment images: {garment_count}")
    print()
    print(f"{Colors.BOLD}⏱️  Total time: {total_time/60:.1f} minutes{Colors.ENDC}")
    print()
    print(f"{Colors.BOLD}📁 Dataset location:{Colors.ENDC}")
    print(f"   {os.path.abspath(data_dir)}")
    print()
    print(f"{Colors.BOLD}📝 Generated files:{Colors.ENDC}")
    print("   • pairs.txt - Person-garment pairings")
    print("   • train_list.txt - Training image list")
    print("   • quality_scores.json - Quality metrics")
    print("   • dataset_info.json - Dataset information")
    print()
    print(f"{Colors.GREEN}🚀 Your advanced dataset is ready for training!{Colors.ENDC}")
    print()
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        print()
        print_warning("Build interrupted by user")
        sys.exit(1)

