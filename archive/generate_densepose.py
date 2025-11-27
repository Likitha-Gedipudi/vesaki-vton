"""
DensePose UV Map Generator for Virtual Try-On Dataset
Generates dense surface coordinates for body warping

Note: DensePose requires Detectron2 which is complex to install.
This script provides installation instructions and a basic implementation.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

try:
    import torch
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from densepose import add_densepose_config
    from densepose.vis.densepose_results import DensePoseResultsFineSegmentationVisualizer as Visualizer
    from densepose.vis.extractor import DensePoseResultExtractor
    DENSEPOSE_AVAILABLE = True
except ImportError:
    DENSEPOSE_AVAILABLE = False

def setup_densepose():
    """Setup DensePose predictor"""
    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("DensePose/densepose_rcnn_R_50_FPN_s1x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
    predictor = DefaultPredictor(cfg)
    return predictor

def generate_densepose(image_path, predictor, output_dir):
    """Generate DensePose IUV map"""
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        return False
    
    # Run DensePose
    with torch.no_grad():
        outputs = predictor(image)
    
    # Check if person detected
    if not outputs["instances"].has("pred_densepose"):
        return False
    
    # Extract DensePose results
    results = outputs["instances"]
    
    # Save IUV map
    iuv_array = DensePoseResultExtractor()(results)[0]
    
    # Save as numpy file
    output_path = os.path.join(output_dir, Path(image_path).stem + '_iuv.npy')
    np.save(output_path, iuv_array)
    
    # Save visualization
    visualizer = Visualizer(alpha=0.7)
    visualization = visualizer.visualize(image, results)
    
    vis_path = os.path.join(output_dir, Path(image_path).stem + '.jpg')
    cv2.imwrite(vis_path, visualization)
    
    return True

def generate_simple_mask(image_path, output_dir):
    """
    Generate a simple body mask using background subtraction
    This is a fallback when DensePose is not available
    """
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        return False
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Threshold to create binary mask
    # Assuming white or light background
    _, mask = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological operations to clean up mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find largest contour (should be the person)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create clean mask
        clean_mask = np.zeros_like(mask)
        cv2.drawContours(clean_mask, [largest_contour], -1, 255, -1)
        
        # Save mask
        output_path = os.path.join(output_dir, Path(image_path).stem + '_mask.png')
        cv2.imwrite(output_path, clean_mask)
        
        return True
    
    return False

def process_dataset(data_dir, use_simple=False):
    """Process all person images"""
    person_dir = os.path.join(data_dir, 'person')
    densepose_dir = os.path.join(data_dir, 'densepose')
    
    # Create output directory
    os.makedirs(densepose_dir, exist_ok=True)
    
    # Get all person images
    image_files = list(Path(person_dir).glob('*.jpg')) + list(Path(person_dir).glob('*.png'))
    
    if not image_files:
        print("❌ No person images found!")
        return
    
    print(f"📸 Processing {len(image_files)} person images...")
    print()
    
    if use_simple or not DENSEPOSE_AVAILABLE:
        print("ℹ️  Using simple mask generation (DensePose not available)")
        print()
        
        success_count = 0
        for image_path in tqdm(image_files, desc="Generating masks"):
            if generate_simple_mask(str(image_path), densepose_dir):
                success_count += 1
        
        print()
        print(f"✅ Successfully processed: {success_count}/{len(image_files)}")
        print(f"📁 Output saved to: {densepose_dir}")
        
    else:
        print("🚀 Using DensePose (this may take a while)...")
        print()
        
        predictor = setup_densepose()
        
        success_count = 0
        failed_images = []
        
        for image_path in tqdm(image_files, desc="Generating DensePose"):
            if generate_densepose(str(image_path), predictor, densepose_dir):
                success_count += 1
            else:
                failed_images.append(image_path.name)
        
        print()
        print(f"✅ Successfully processed: {success_count}/{len(image_files)}")
        
        if failed_images:
            print(f"⚠️  Failed to detect person in {len(failed_images)} images")
        
        print(f"📁 Output saved to: {densepose_dir}")

def main():
    parser = argparse.ArgumentParser(description='Generate DensePose maps for person images')
    parser.add_argument('--data_dir', type=str, default='dataset/train',
                        help='Path to dataset directory (default: dataset/train)')
    parser.add_argument('--simple', action='store_true',
                        help='Use simple mask generation instead of DensePose')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  🎯 DENSEPOSE GENERATOR")
    print("=" * 70)
    print()
    
    if not DENSEPOSE_AVAILABLE and not args.simple:
        print("⚠️  DensePose/Detectron2 not installed!")
        print()
        print("DensePose installation is complex. You have two options:")
        print()
        print("Option 1: Use simple mask generation (recommended for quick start)")
        print("   python generate_densepose.py --simple")
        print()
        print("Option 2: Install DensePose (advanced)")
        print("   See: https://github.com/facebookresearch/detectron2")
        print("   See: https://github.com/facebookresearch/DensePose")
        print()
        print("For now, we'll use simple mask generation...")
        print()
        args.simple = True
    
    process_dataset(args.data_dir, use_simple=args.simple)
    
    print()
    print("=" * 70)
    print("  ✅ DENSEPOSE GENERATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()

