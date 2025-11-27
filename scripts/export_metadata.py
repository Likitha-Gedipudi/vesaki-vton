#!/usr/bin/env python3
"""
Export Metadata for Training
Creates pairs.txt, train_list.txt, and quality scores
"""

import os
import json
import random
from pathlib import Path
import argparse

def create_pairs_txt(data_dir, output_file='pairs.txt'):
    """Create person-garment pairings"""
    person_dir = os.path.join(data_dir, 'person')
    garment_dir = os.path.join(data_dir, 'garment')
    
    person_images = sorted([f.name for f in Path(person_dir).glob('*.jpg')])
    garment_images = sorted([f.name for f in Path(garment_dir).glob('*.jpg')])
    
    if not person_images or not garment_images:
        print("Error: No person or garment images found")
        return False
    
    # Create pairings
    # Strategy: Each person with random garments
    pairs = []
    
    for person_img in person_images:
        # Pair with 3-5 random garments
        num_pairs = min(random.randint(3, 5), len(garment_images))
        selected_garments = random.sample(garment_images, num_pairs)
        
        for garment_img in selected_garments:
            pairs.append(f"{person_img} {garment_img}")
    
    # Save to file
    with open(output_file, 'w') as f:
        for pair in pairs:
            f.write(pair + '\n')
    
    return True, len(pairs)

def create_train_list(data_dir, output_file='train_list.txt'):
    """Create list of all training images"""
    person_dir = os.path.join(data_dir, 'person')
    
    person_images = sorted([f.name for f in Path(person_dir).glob('*.jpg')])
    
    with open(output_file, 'w') as f:
        for img in person_images:
            f.write(img + '\n')
    
    return len(person_images)

def export_quality_scores(quality_file='quality_assessment.json', output_file='quality_scores.json'):
    """Export simplified quality scores"""
    if not os.path.exists(quality_file):
        print(f"Warning: Quality assessment file not found: {quality_file}")
        return False
    
    try:
        # Try to read and fix malformed JSON
        with open(quality_file, 'r') as f:
            content = f.read()
        
        # Handle empty or malformed JSON
        if not content.strip():
            print(f"Warning: Quality assessment file is empty")
            return False
        
        # Try to parse JSON
        try:
            quality_data = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse quality assessment JSON: {e}")
            # Try to salvage by reading line by line
            quality_data = {}
        
        if not quality_data:
            print(f"Warning: No quality data available")
            return False
        
        # Simplify scores
        simplified_scores = {}
        
        for image_path, data in quality_data.items():
            if not isinstance(data, dict):
                continue
                
            image_name = Path(image_path).name
            
            scores = data.get('scores', {})
            
            simplified_scores[image_name] = {
                'passed': data.get('passed', False),
                'sharpness': float(scores.get('sharpness', {}).get('score', 0)) if isinstance(scores.get('sharpness'), dict) else 0,
                'brightness': float(scores.get('brightness', {}).get('score', 0)) if isinstance(scores.get('brightness'), dict) else 0,
                'face_confidence': float(scores.get('face_confidence', {}).get('score', 0)) if isinstance(scores.get('face_confidence'), dict) else 0
            }
        
        with open(output_file, 'w') as f:
            json.dump(simplified_scores, f, indent=2)
        
        return True
    
    except Exception as e:
        print(f"Error exporting quality scores: {e}")
        # Create empty quality scores file as fallback
        with open(output_file, 'w') as f:
            json.dump({}, f, indent=2)
        return False

def create_dataset_info(data_dir, output_file='dataset_info.json'):
    """Create comprehensive dataset information file"""
    person_dir = os.path.join(data_dir, 'person')
    garment_dir = os.path.join(data_dir, 'garment')
    
    person_count = len(list(Path(person_dir).glob('*.jpg')))
    garment_count = len(list(Path(garment_dir).glob('*.jpg')))
    
    # Check augmentation
    aug_count = len([f for f in Path(person_dir).glob('*.jpg') if '_aug' in f.name])
    original_count = person_count - aug_count
    
    dataset_info = {
        'version': '1.0',
        'format': 'VITON-HD Compatible',
        'statistics': {
            'total_person_images': person_count,
            'original_person_images': original_count,
            'augmented_person_images': aug_count,
            'garment_images': garment_count,
            'augmentation_ratio': f"{person_count/max(original_count, 1):.1f}x" if original_count > 0 else "N/A"
        },
        'image_dimensions': {
            'width': 768,
            'height': 1024
        },
        'components': {
            'person': 'Person images (models)',
            'garment': 'Garment images (products)',
            'openpose': 'Pose keypoints (OpenPose 25-point format)',
            'densepose': 'Body segmentation masks',
            'person-parse': 'Human parsing segmentation (SCHP)',
            'agnostic-mask': 'Masked person images for try-on'
        },
        'compatible_models': [
            'VITON-HD',
            'CP-VTON+',
            'HR-VITON',
            'PFAFN'
        ]
    }
    
    with open(output_file, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Export dataset metadata')
    parser.add_argument('--data_dir', type=str, default='dataset/train',
                        help='Path to dataset directory')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  METADATA EXPORT")
    print("=" * 70)
    print()
    
    # Create pairs.txt
    print("Creating pairs.txt...")
    success, num_pairs = create_pairs_txt(args.data_dir)
    if success:
        print(f"  ✓ Created {num_pairs} person-garment pairs")
    else:
        print("  ✗ Failed to create pairs")
    print()
    
    # Create train_list.txt
    print("Creating train_list.txt...")
    num_train = create_train_list(args.data_dir)
    print(f"  ✓ Listed {num_train} training images")
    print()
    
    # Export quality scores
    print("Exporting quality scores...")
    if export_quality_scores():
        print(f"  ✓ Quality scores exported")
    else:
        print(f"  ⚠ Quality scores not available")
    print()
    
    # Create dataset info
    print("Creating dataset_info.json...")
    if create_dataset_info(args.data_dir):
        print(f"  ✓ Dataset information exported")
    print()
    
    print("=" * 70)
    print("  METADATA EXPORT COMPLETE")
    print("=" * 70)
    print()
    print("Generated files:")
    print("  • pairs.txt - Person-garment pairings for training")
    print("  • train_list.txt - List of all training images")
    print("  • quality_scores.json - Quality metrics per image")
    print("  • dataset_info.json - Dataset statistics and info")
    print()

if __name__ == "__main__":
    main()

