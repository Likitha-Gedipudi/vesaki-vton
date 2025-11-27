#!/usr/bin/env python3
"""
Quality Filtering for Dataset
Automatically removes low-quality images based on multiple criteria
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

def check_sharpness(image_path, threshold=100):
    """Check image sharpness using Laplacian variance"""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False, 0
    
    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
    return laplacian_var > threshold, laplacian_var

def check_brightness(image_path, min_brightness=50, max_brightness=200):
    """Check if image has proper brightness"""
    img = cv2.imread(str(image_path))
    if img is None:
        return False, 0
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    
    is_good = min_brightness < mean_brightness < max_brightness
    return is_good, mean_brightness

def check_face_confidence(image_path, min_confidence=0.3):
    """Check face detection confidence"""
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    img = cv2.imread(str(image_path))
    if img is None:
        return False, 0
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        return False, 0
    
    # Calculate confidence based on face size relative to image
    largest_face = max(faces, key=lambda x: x[2] * x[3])
    _, _, fw, fh = largest_face
    
    face_area = fw * fh
    image_area = img.shape[0] * img.shape[1]
    face_ratio = face_area / image_area
    
    confidence = min(face_ratio * 10, 1.0)  # Normalize to 0-1
    
    return confidence > min_confidence, confidence

def check_resolution(image_path, min_size=300):
    """Check if image has adequate resolution"""
    img = cv2.imread(str(image_path))
    if img is None:
        return False, (0, 0)
    
    h, w = img.shape[:2]
    is_good = min(h, w) >= min_size
    
    return is_good, (w, h)

def assess_image_quality(image_path):
    """Comprehensive quality assessment"""
    scores = {
        'sharpness': {'pass': False, 'score': 0},
        'brightness': {'pass': False, 'score': 0},
        'face_confidence': {'pass': False, 'score': 0},
        'resolution': {'pass': False, 'score': (0, 0)}
    }
    
    # Check sharpness
    pass_sharp, sharp_score = check_sharpness(image_path)
    scores['sharpness'] = {'pass': bool(pass_sharp), 'score': float(sharp_score)}
    
    # Check brightness
    pass_bright, bright_score = check_brightness(image_path)
    scores['brightness'] = {'pass': bool(pass_bright), 'score': float(bright_score)}
    
    # Check face
    pass_face, face_score = check_face_confidence(image_path)
    scores['face_confidence'] = {'pass': bool(pass_face), 'score': float(face_score)}
    
    # Check resolution
    pass_res, res_size = check_resolution(image_path)
    scores['resolution'] = {'pass': bool(pass_res), 'score': tuple(map(int, res_size))}
    
    # Overall pass: at least 3 out of 4 criteria
    passed_count = sum(1 for metric in scores.values() if metric['pass'])
    overall_pass = bool(passed_count >= 3)
    
    return overall_pass, scores

def filter_dataset(data_dir, dry_run=False):
    """Filter dataset and remove low-quality images"""
    person_dir = os.path.join(data_dir, 'person')
    
    if not os.path.exists(person_dir):
        print(f"Person directory not found: {person_dir}")
        return
    
    # Get all person images
    image_files = list(Path(person_dir).glob('*.jpg')) + list(Path(person_dir).glob('*.png'))
    
    if not image_files:
        print("No images found!")
        return
    
    print(f"Assessing quality of {len(image_files)} images...")
    print()
    
    quality_results = {}
    passed_images = []
    failed_images = []
    
    for image_path in tqdm(image_files, desc="Quality assessment"):
        passed, scores = assess_image_quality(str(image_path))
        
        quality_results[str(image_path)] = {
            'passed': passed,
            'scores': scores
        }
        
        if passed:
            passed_images.append(image_path)
        else:
            failed_images.append(image_path)
    
    print()
    print(f"Quality assessment complete:")
    print(f"  Passed: {len(passed_images)}/{len(image_files)}")
    print(f"  Failed: {len(failed_images)}/{len(image_files)}")
    print()
    
    # Save quality report
    report_file = 'quality_assessment.json'
    with open(report_file, 'w') as f:
        json.dump(quality_results, f, indent=2)
    print(f"✓ Quality report saved to: {report_file}")
    print()
    
    # Remove failed images
    if failed_images and not dry_run:
        print(f"Removing {len(failed_images)} low-quality images...")
        
        removed_count = 0
        for image_path in failed_images:
            try:
                # Remove image
                os.remove(image_path)
                
                # Remove associated annotations
                base_name = image_path.stem
                
                # Remove from other directories
                for subdir in ['openpose', 'densepose', 'person-parse', 'agnostic-mask']:
                    dir_path = os.path.join(data_dir, subdir)
                    if os.path.exists(dir_path):
                        for ext in ['.json', '.png', '.jpg', '_keypoints.json', '_mask.png', '_vis.jpg']:
                            file_path = os.path.join(dir_path, base_name + ext)
                            if os.path.exists(file_path):
                                os.remove(file_path)
                
                removed_count += 1
            except Exception as e:
                print(f"  Error removing {image_path.name}: {e}")
        
        print(f"✓ Removed {removed_count} images and their annotations")
    
    elif failed_images and dry_run:
        print("Dry run mode - no files removed")
        print()
        print("Failed images:")
        for img in failed_images[:10]:
            print(f"  - {img.name}")
        if len(failed_images) > 10:
            print(f"  ... and {len(failed_images) - 10} more")
    
    return len(passed_images), len(failed_images)

def main():
    parser = argparse.ArgumentParser(description='Filter low-quality images from dataset')
    parser.add_argument('--data_dir', type=str, default='dataset/train',
                        help='Path to dataset directory')
    parser.add_argument('--dry_run', action='store_true',
                        help='Show what would be removed without actually removing')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  DATASET QUALITY FILTER")
    print("=" * 70)
    print()
    
    print("Quality criteria:")
    print("  • Sharpness (Laplacian variance > 100)")
    print("  • Brightness (mean 50-200)")
    print("  • Face detection confidence (> 0.3)")
    print("  • Resolution (min dimension > 300px)")
    print()
    print("Images must pass at least 3/4 criteria")
    print()
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be removed")
        print()
    
    passed, failed = filter_dataset(args.data_dir, args.dry_run)
    
    print()
    print("=" * 70)
    print("  QUALITY FILTERING COMPLETE")
    print("=" * 70)
    print()
    
    if passed:
        print(f"✓ Dataset quality improved: {passed} high-quality images retained")

if __name__ == "__main__":
    main()

