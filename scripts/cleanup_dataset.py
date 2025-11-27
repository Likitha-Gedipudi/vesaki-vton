#!/usr/bin/env python3
"""
Dataset Cleanup Script
Removes all existing images and annotations while preserving directory structure
"""

import os
import shutil
from pathlib import Path

def cleanup_directory(directory):
    """Remove all files in directory but keep the directory"""
    if os.path.exists(directory):
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
                
def count_files(directory):
    """Count files in directory"""
    if not os.path.exists(directory):
        return 0
    return sum(1 for item in os.listdir(directory) if os.path.isfile(os.path.join(directory, item)))

def main():
    print("=" * 70)
    print("  DATASET CLEANUP")
    print("=" * 70)
    print()
    
    # Directories to clean
    datasets = ['dataset/train', 'dataset/test']
    subdirs = ['person', 'garment', 'person-parse', 'densepose', 'openpose', 'agnostic-mask']
    
    # Count files before cleanup
    total_files = 0
    print("Current dataset:")
    for dataset in datasets:
        for subdir in subdirs:
            dir_path = os.path.join(dataset, subdir)
            count = count_files(dir_path)
            total_files += count
            if count > 0:
                print(f"  {dir_path}: {count} files")
    
    print()
    print(f"Total files to remove: {total_files}")
    print()
    
    if total_files == 0:
        print("Dataset is already clean!")
        return
    
    # Confirm
    response = input("Are you sure you want to delete all dataset files? (yes/no): ")
    if response.lower() != 'yes':
        print("Cleanup cancelled.")
        return
    
    print()
    print("Cleaning dataset...")
    
    # Clean each directory
    for dataset in datasets:
        for subdir in subdirs:
            dir_path = os.path.join(dataset, subdir)
            cleanup_directory(dir_path)
            print(f"  Cleaned {dir_path}")
    
    # Remove temp directories
    temp_dirs = ['temp_downloads']
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"  Removed {temp_dir}")
    
    # Clear scraping stats
    stats_file = 'scraping_stats.json'
    if os.path.exists(stats_file):
        os.remove(stats_file)
        print(f"  Removed {stats_file}")
    
    print()
    print("=" * 70)
    print("  CLEANUP COMPLETE")
    print("=" * 70)
    print()
    print("Dataset is ready for fresh data collection!")
    print()
    print("Next steps:")
    print("  1. Install advanced requirements: pip3 install -r requirements_advanced.txt")
    print("  2. Download models: python3 download_models.py")
    print("  3. Run pipeline: python3 main_advanced.py")
    print()

if __name__ == "__main__":
    main()

