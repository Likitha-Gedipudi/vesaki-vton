#!/usr/bin/env python3
"""
Advanced Dataset Validator
Validates completeness and quality of generated dataset
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
import argparse

def count_files(directory, extensions=['.jpg', '.png', '.json']):
    """Count files with specific extensions"""
    if not os.path.exists(directory):
        return 0
    
    count = 0
    for file in os.listdir(directory):
        if any(file.endswith(ext) for ext in extensions):
            count += 1
    return count

def validate_image_dimensions(directory, expected_size=(768, 1024)):
    """Check if images have correct dimensions"""
    if not os.path.exists(directory):
        return True, []
    
    incorrect_images = []
    
    for image_file in Path(directory).glob('*.jpg'):
        img = cv2.imread(str(image_file))
        if img is not None:
            h, w = img.shape[:2]
            if w != expected_size[0] or h != expected_size[1]:
                incorrect_images.append((image_file.name, w, h))
    
    return len(incorrect_images) == 0, incorrect_images

def validate_annotations(data_dir):
    """Check if all person images have corresponding annotations"""
    person_dir = os.path.join(data_dir, 'person')
    
    if not os.path.exists(person_dir):
        return False, "Person directory not found"
    
    person_images = list(Path(person_dir).glob('*.jpg')) + list(Path(person_dir).glob('*.png'))
    
    if not person_images:
        return False, "No person images found"
    
    missing_annotations = {
        'openpose': [],
        'densepose': [],
        'person-parse': [],
        'agnostic-mask': []
    }
    
    for image_path in person_images:
        base_name = image_path.stem
        
        # Check OpenPose JSON
        openpose_file = os.path.join(data_dir, 'openpose', f'{base_name}_keypoints.json')
        if not os.path.exists(openpose_file):
            missing_annotations['openpose'].append(base_name)
        
        # Check DensePose mask
        densepose_file = os.path.join(data_dir, 'densepose', f'{base_name}_mask.png')
        if not os.path.exists(densepose_file):
            missing_annotations['densepose'].append(base_name)
        
        # Check segmentation
        parse_file = os.path.join(data_dir, 'person-parse', f'{base_name}.png')
        if not os.path.exists(parse_file):
            missing_annotations['person-parse'].append(base_name)
        
        # Check agnostic mask
        agnostic_file = os.path.join(data_dir, 'agnostic-mask', f'{base_name}.jpg')
        if not os.path.exists(agnostic_file):
            missing_annotations['agnostic-mask'].append(base_name)
    
    all_complete = all(len(v) == 0 for v in missing_annotations.values())
    
    return all_complete, missing_annotations

def validate_augmentation_mapping(augmentation_map_file):
    """Validate augmentation mapping file"""
    if not os.path.exists(augmentation_map_file):
        return True, "No augmentation mapping (not required)"
    
    try:
        with open(augmentation_map_file, 'r') as f:
            aug_data = json.load(f)
        
        if not isinstance(aug_data, list):
            return False, "Invalid augmentation map format"
        
        return True, f"{len(aug_data)} augmented images tracked"
    
    except Exception as e:
        return False, f"Error reading augmentation map: {e}"

def generate_report(data_dir):
    """Generate comprehensive validation report"""
    print("=" * 70)
    print("  DATASET VALIDATION REPORT")
    print("=" * 70)
    print()
    
    # Component statistics
    print("📊 Dataset Components:")
    print()
    
    person_count = count_files(os.path.join(data_dir, 'person'), ['.jpg', '.png'])
    garment_count = count_files(os.path.join(data_dir, 'garment'), ['.jpg', '.png'])
    openpose_count = count_files(os.path.join(data_dir, 'openpose'), ['.json'])
    densepose_count = count_files(os.path.join(data_dir, 'densepose'), ['.png'])
    parse_count = count_files(os.path.join(data_dir, 'person-parse'), ['.png'])
    agnostic_count = count_files(os.path.join(data_dir, 'agnostic-mask'), ['.jpg', '.png'])
    
    print(f"  Person images:        {person_count}")
    print(f"  Garment images:       {garment_count}")
    print(f"  OpenPose keypoints:   {openpose_count}")
    print(f"  DensePose masks:      {densepose_count}")
    print(f"  Segmentation masks:   {parse_count}")
    print(f"  Agnostic masks:       {agnostic_count}")
    print()
    
    # Check dimensions
    print("🖼️  Image Dimensions:")
    print()
    
    person_dims_ok, person_incorrect = validate_image_dimensions(
        os.path.join(data_dir, 'person'), (768, 1024)
    )
    
    if person_dims_ok:
        print(f"  ✓ All person images: 768×1024")
    else:
        print(f"  ⚠ {len(person_incorrect)} person images have incorrect dimensions")
        for name, w, h in person_incorrect[:3]:
            print(f"     - {name}: {w}×{h}")
    
    garment_dims_ok, garment_incorrect = validate_image_dimensions(
        os.path.join(data_dir, 'garment')
    )
    
    print()
    
    # Check annotation completeness
    print("📋 Annotation Completeness:")
    print()
    
    annotations_ok, missing = validate_annotations(data_dir)
    
    if annotations_ok:
        print(f"  ✓ All person images have complete annotations")
    else:
        if not isinstance(missing, str):
            for ann_type, missing_list in missing.items():
                if missing_list:
                    print(f"  ⚠ {ann_type}: {len(missing_list)} missing")
        else:
            print(f"  ⚠ {missing}")
    
    print()
    
    # Check augmentation
    print("🔄 Augmentation:")
    print()
    
    aug_ok, aug_msg = validate_augmentation_mapping('augmentation_map.json')
    if aug_ok:
        print(f"  ✓ {aug_msg}")
    else:
        print(f"  ⚠ {aug_msg}")
    
    print()
    
    # Overall status
    print("=" * 70)
    
    if person_count >= 100 and annotations_ok:
        print("  ✅ DATASET VALIDATION PASSED")
        print("=" * 70)
        print()
        print("Your dataset is ready for virtual try-on training!")
    elif person_count < 100:
        print("  ⚠ DATASET VALIDATION: NEEDS MORE IMAGES")
        print("=" * 70)
        print()
        print(f"Current: {person_count} person images")
        print(f"Required: 100+ person images")
        print()
        print("Run scraper again to collect more images")
    else:
        print("  ⚠ DATASET VALIDATION: INCOMPLETE ANNOTATIONS")
        print("=" * 70)
        print()
        print("Some images are missing annotations")
        print("Re-run annotation generation scripts")
    
    print()
    
    return person_count >= 100 and annotations_ok

def main():
    parser = argparse.ArgumentParser(description='Validate dataset completeness and quality')
    parser.add_argument('--data_dir', type=str, default='dataset/train',
                        help='Path to dataset directory')
    
    args = parser.parse_args()
    
    validation_passed = generate_report(args.data_dir)
    
    return 0 if validation_passed else 1

if __name__ == "__main__":
    exit(main())

