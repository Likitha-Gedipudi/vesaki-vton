"""
Dataset Verification and Inspection Tool
Checks dataset completeness and provides statistics
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse

def count_files(directory, extension='*'):
    """Count files in directory"""
    if not os.path.exists(directory):
        return 0
    
    if extension == '*':
        extensions = {'.jpg', '.jpeg', '.png', '.json', '.npy'}
        return sum(1 for f in os.listdir(directory) 
                  if any(f.lower().endswith(ext) for ext in extensions))
    else:
        return len(list(Path(directory).glob(f'*.{extension}')))

def check_image_dimensions(directory, expected_size=(768, 1024)):
    """Check if images have correct dimensions"""
    image_files = list(Path(directory).glob('*.jpg')) + list(Path(directory).glob('*.png'))
    
    if not image_files:
        return None, None
    
    correct = 0
    incorrect = []
    
    for img_path in image_files[:20]:  # Sample first 20
        try:
            img = cv2.imread(str(img_path))
            if img is not None:
                h, w = img.shape[:2]
                if w == expected_size[0] and h == expected_size[1]:
                    correct += 1
                else:
                    incorrect.append((img_path.name, w, h))
        except:
            pass
    
    total_checked = correct + len(incorrect)
    return correct, total_checked, incorrect

def verify_dataset(data_dir):
    """Verify dataset structure and completeness"""
    
    print("=" * 70)
    print("  🔍 DATASET VERIFICATION")
    print("=" * 70)
    print()
    
    print(f"📁 Dataset directory: {os.path.abspath(data_dir)}")
    print()
    
    # Define expected structure
    components = {
        'person': 'Person images',
        'garment': 'Garment images',
        'person-parse': 'Segmentation masks',
        'densepose': 'DensePose maps',
        'openpose': 'Pose keypoints',
        'agnostic-mask': 'Agnostic masks'
    }
    
    stats = {}
    issues = []
    
    # Check each component
    print("📊 Component Status:")
    print()
    
    for dir_name, description in components.items():
        dir_path = os.path.join(data_dir, dir_name)
        
        if not os.path.exists(dir_path):
            print(f"  ❌ {description:25} - Directory missing")
            issues.append(f"Missing directory: {dir_name}")
            stats[dir_name] = 0
            continue
        
        # Count files
        if dir_name == 'openpose':
            # Count JSON files
            file_count = count_files(dir_path, 'json')
        else:
            file_count = count_files(dir_path)
        
        stats[dir_name] = file_count
        
        if file_count == 0:
            print(f"  ⚠️  {description:25} - No files ({file_count})")
            issues.append(f"No files in {dir_name}")
        else:
            print(f"  ✅ {description:25} - {file_count} files")
    
    print()
    
    # Check image dimensions
    print("🖼️  Image Dimensions:")
    print()
    
    for dir_name in ['person', 'garment']:
        dir_path = os.path.join(data_dir, dir_name)
        if os.path.exists(dir_path):
            result = check_image_dimensions(dir_path)
            if result[0] is not None:
                correct, total, incorrect = result
                if correct == total:
                    print(f"  ✅ {dir_name:10} - All images correctly sized (768×1024)")
                else:
                    print(f"  ⚠️  {dir_name:10} - {correct}/{total} images correctly sized")
                    if incorrect:
                        print(f"      Incorrect sizes found:")
                        for name, w, h in incorrect[:5]:
                            print(f"        - {name}: {w}×{h}")
    
    print()
    
    # Dataset completeness
    print("📈 Dataset Completeness:")
    print()
    
    person_count = stats.get('person', 0)
    garment_count = stats.get('garment', 0)
    
    if person_count == 0 and garment_count == 0:
        print("  ❌ Dataset is empty!")
        print("     Run: python scraper_improved.py")
        return
    
    # Calculate completeness
    required_components = ['person', 'garment']
    optional_components = ['person-parse', 'densepose', 'openpose', 'agnostic-mask']
    
    required_complete = all(stats.get(comp, 0) > 0 for comp in required_components)
    optional_complete = [comp for comp in optional_components if stats.get(comp, 0) > 0]
    
    if required_complete:
        print(f"  ✅ Core dataset complete")
        print(f"     - {person_count} person images")
        print(f"     - {garment_count} garment images")
    else:
        print(f"  ❌ Core dataset incomplete")
    
    print()
    
    if optional_complete:
        print(f"  ✅ Annotations: {len(optional_complete)}/4 types generated")
        for comp in optional_complete:
            print(f"     - {components[comp]}: {stats[comp]} files")
    else:
        print(f"  ⚠️  No annotations generated yet")
        print(f"     Run: python build_dataset.py --all")
    
    print()
    
    # Recommendations
    print("💡 Recommendations:")
    print()
    
    if person_count == 0:
        print("  1. Run scraper to collect images:")
        print("     python scraper_improved.py")
    elif person_count < 50:
        print(f"  1. Consider collecting more images (currently: {person_count})")
        print("     Add more URLs to product_urls.txt and rerun scraper")
    
    if person_count > 0 and stats.get('openpose', 0) == 0:
        print("  2. Generate annotations:")
        print("     python build_dataset.py --skip_scrape")
    
    if any(issues):
        print()
        print("⚠️  Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    
    print()
    print("=" * 70)
    
    # Check for scraping stats
    if os.path.exists('scraping_stats.json'):
        try:
            with open('scraping_stats.json', 'r') as f:
                scrape_stats = json.load(f)
            
            print()
            print("📊 Scraping Statistics:")
            print(f"   Total images found:  {scrape_stats.get('total_images_found', 0)}")
            print(f"   Person images saved: {scrape_stats.get('person_images', 0)}")
            print(f"   Garment images saved: {scrape_stats.get('garment_images', 0)}")
            print(f"   Failed downloads:    {scrape_stats.get('failed_downloads', 0)}")
        except:
            pass

def inspect_sample(data_dir, show_images=False):
    """Show sample images from dataset"""
    print()
    print("=" * 70)
    print("  🔎 SAMPLE INSPECTION")
    print("=" * 70)
    print()
    
    person_dir = os.path.join(data_dir, 'person')
    garment_dir = os.path.join(data_dir, 'garment')
    
    # List sample files
    person_files = list(Path(person_dir).glob('*.jpg'))[:5]
    garment_files = list(Path(garment_dir).glob('*.jpg'))[:5]
    
    if person_files:
        print("👤 Sample person images:")
        for f in person_files:
            img = cv2.imread(str(f))
            if img is not None:
                h, w = img.shape[:2]
                print(f"   {f.name} ({w}×{h})")
    
    print()
    
    if garment_files:
        print("👔 Sample garment images:")
        for f in garment_files:
            img = cv2.imread(str(f))
            if img is not None:
                h, w = img.shape[:2]
                print(f"   {f.name} ({w}×{h})")
    
    print()

def main():
    parser = argparse.ArgumentParser(description='Verify and inspect dataset')
    parser.add_argument('--data_dir', type=str, default='dataset/train',
                        help='Path to dataset directory (default: dataset/train)')
    parser.add_argument('--inspect', action='store_true',
                        help='Show sample files from dataset')
    
    args = parser.parse_args()
    
    verify_dataset(args.data_dir)
    
    if args.inspect:
        inspect_sample(args.data_dir)
    
    print()

if __name__ == "__main__":
    main()

