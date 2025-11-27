#!/usr/bin/env python3
"""
Download Pre-trained Models for Advanced Pipeline
Downloads SCHP and other model weights automatically
"""

import os
import sys
from pathlib import Path

try:
    import gdown
    import torch
except ImportError:
    print("Error: Required packages not installed!")
    print("Please run: pip3 install -r requirements_advanced.txt")
    sys.exit(1)

MODEL_DIR = "models"
MODELS = {
    "schp_atr": {
        "name": "SCHP ATR Model",
        "url": "https://drive.google.com/uc?id=1k4dllHpu0bdx38J7H28rVVLpU-kOHmnH",
        "filename": "exp-schp-201908261155-atr.pth",
        "description": "Self-Correction Human Parsing (ATR dataset)"
    },
    "schp_lip": {
        "name": "SCHP LIP Model",  
        "url": "https://drive.google.com/uc?id=1E5YwNKW2VOEayK9mWCS3Kpsxf-3z04ZE",
        "filename": "exp-schp-201908301523-lip.pth",
        "description": "Self-Correction Human Parsing (LIP dataset)"
    }
}

def create_model_directory():
    """Create models directory if it doesn't exist"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Created/verified models directory: {MODEL_DIR}")

def download_model(model_key):
    """Download a single model"""
    model_info = MODELS[model_key]
    output_path = os.path.join(MODEL_DIR, model_info["filename"])
    
    print()
    print(f"Downloading: {model_info['name']}")
    print(f"Description: {model_info['description']}")
    
    if os.path.exists(output_path):
        print(f"Model already exists: {output_path}")
        return True
    
    try:
        print(f"Downloading to: {output_path}")
        gdown.download(model_info["url"], output_path, quiet=False)
        
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"Download complete! Size: {file_size:.1f} MB")
            return True
        else:
            print(f"Download failed for {model_info['name']}")
            return False
            
    except Exception as e:
        print(f"Error downloading {model_info['name']}: {e}")
        return False

def verify_models():
    """Verify all models are downloaded"""
    print()
    print("Verifying models...")
    all_present = True
    
    for model_key, model_info in MODELS.items():
        output_path = os.path.join(MODEL_DIR, model_info["filename"])
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"  ✓ {model_info['name']}: {file_size:.1f} MB")
        else:
            print(f"  ✗ {model_info['name']}: Missing")
            all_present = False
    
    return all_present

def main():
    print("=" * 70)
    print("  MODEL DOWNLOADER")
    print("=" * 70)
    print()
    print("This script downloads pre-trained models for:")
    print("  - SCHP (Self-Correction Human Parsing)")
    print("  - ATR and LIP datasets")
    print()
    print("Total download size: ~500 MB")
    print()
    
    # Create directory
    create_model_directory()
    
    # Check if models already exist
    if verify_models():
        print()
        print("All models are already downloaded!")
        print()
        response = input("Re-download models? (y/n): ")
        if response.lower() != 'y':
            print("Skipping download.")
            return
    
    print()
    print("Starting downloads...")
    
    # Download each model
    success_count = 0
    for model_key in MODELS.keys():
        if download_model(model_key):
            success_count += 1
    
    print()
    print("=" * 70)
    
    if success_count == len(MODELS):
        print("  ✓ ALL MODELS DOWNLOADED SUCCESSFULLY")
        print("=" * 70)
        print()
        print("Models ready for use!")
        print()
        print("Next steps:")
        print("  1. Clean dataset: python3 cleanup_dataset.py")
        print("  2. Run pipeline: python3 main_advanced.py")
    else:
        print("  ⚠ SOME DOWNLOADS FAILED")
        print("=" * 70)
        print()
        print(f"Downloaded: {success_count}/{len(MODELS)} models")
        print()
        print("You can:")
        print("  1. Re-run this script")
        print("  2. Manually download models from:")
        print("     https://github.com/GoGoDuck912/Self-Correction-Human-Parsing")
        print()
    
    print()

if __name__ == "__main__":
    main()

