"""
OpenPose Keypoint Detection for Virtual Try-On Dataset
Generates 18-point skeleton keypoints for person images

This script uses MediaPipe as an alternative to OpenPose (easier to install)
MediaPipe provides 33 pose landmarks which we map to OpenPose's 18 keypoints
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️  MediaPipe not installed. Install with: pip install mediapipe")

# OpenPose 18-point format mapping from MediaPipe 33-point
# MediaPipe -> OpenPose mapping
MEDIAPIPE_TO_OPENPOSE = {
    0: 0,   # Nose
    2: 14,  # Right eye
    5: 15,  # Left eye
    7: 16,  # Right ear
    8: 17,  # Left ear
    11: 5,  # Right shoulder
    12: 2,  # Left shoulder
    13: 6,  # Right elbow
    14: 3,  # Left elbow
    15: 7,  # Right wrist
    16: 4,  # Left wrist
    23: 8,  # Right hip
    24: 11, # Left hip
    25: 9,  # Right knee
    26: 12, # Left knee
    27: 10, # Right ankle
    28: 13, # Left ankle
}

def generate_openpose_json(image_path, output_dir):
    """Generate OpenPose-style JSON keypoints"""
    if not MEDIAPIPE_AVAILABLE:
        return False
    
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        return False
    
    h, w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.5
    ) as pose:
        results = pose.process(image_rgb)
    
    # Check if pose detected
    if not results.pose_landmarks:
        return False
    
    # Convert to OpenPose format (18 keypoints)
    openpose_keypoints = [[0, 0, 0] for _ in range(18)]
    
    for mp_idx, op_idx in MEDIAPIPE_TO_OPENPOSE.items():
        landmark = results.pose_landmarks.landmark[mp_idx]
        openpose_keypoints[op_idx] = [
            landmark.x * w,  # x coordinate
            landmark.y * h,  # y coordinate
            landmark.visibility  # confidence
        ]
    
    # Calculate neck (midpoint between shoulders)
    if openpose_keypoints[2][2] > 0 and openpose_keypoints[5][2] > 0:
        openpose_keypoints[1] = [
            (openpose_keypoints[2][0] + openpose_keypoints[5][0]) / 2,
            (openpose_keypoints[2][1] + openpose_keypoints[5][1]) / 2,
            (openpose_keypoints[2][2] + openpose_keypoints[5][2]) / 2
        ]
    
    # Create JSON structure
    pose_data = {
        "version": 1.3,
        "people": [{
            "person_id": [-1],
            "pose_keypoints_2d": [coord for kp in openpose_keypoints for coord in kp],
            "face_keypoints_2d": [],
            "hand_left_keypoints_2d": [],
            "hand_right_keypoints_2d": [],
            "pose_keypoints_3d": [],
            "face_keypoints_3d": [],
            "hand_left_keypoints_3d": [],
            "hand_right_keypoints_3d": []
        }]
    }
    
    # Save JSON
    output_path = os.path.join(output_dir, Path(image_path).stem + '_keypoints.json')
    with open(output_path, 'w') as f:
        json.dump(pose_data, f, indent=2)
    
    return True

def generate_openpose_visualization(image_path, json_path, output_dir):
    """Generate visualization of pose keypoints"""
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        return False
    
    # Read keypoints
    try:
        with open(json_path, 'r') as f:
            pose_data = json.load(f)
    except:
        return False
    
    if not pose_data['people']:
        return False
    
    keypoints = pose_data['people'][0]['pose_keypoints_2d']
    keypoints = np.array(keypoints).reshape(-1, 3)
    
    # OpenPose skeleton connections
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Left arm
        (1, 5), (5, 6), (6, 7),          # Right arm
        (1, 8), (8, 9), (9, 10),         # Left leg
        (1, 11), (11, 12), (12, 13),     # Right leg
        (0, 14), (14, 16),               # Left face
        (0, 15), (15, 17)                # Right face
    ]
    
    # Draw skeleton
    for connection in connections:
        pt1_idx, pt2_idx = connection
        if keypoints[pt1_idx][2] > 0.3 and keypoints[pt2_idx][2] > 0.3:
            pt1 = tuple(keypoints[pt1_idx][:2].astype(int))
            pt2 = tuple(keypoints[pt2_idx][:2].astype(int))
            cv2.line(image, pt1, pt2, (0, 255, 0), 2)
    
    # Draw keypoints
    for i, kp in enumerate(keypoints):
        if kp[2] > 0.3:  # confidence threshold
            center = tuple(kp[:2].astype(int))
            cv2.circle(image, center, 4, (0, 0, 255), -1)
    
    # Save visualization
    output_path = os.path.join(output_dir, Path(image_path).stem + '.jpg')
    cv2.imwrite(output_path, image)
    
    return True

def process_dataset(data_dir):
    """Process all person images in dataset"""
    person_dir = os.path.join(data_dir, 'person')
    openpose_dir = os.path.join(data_dir, 'openpose')
    
    # Create output directory
    os.makedirs(openpose_dir, exist_ok=True)
    
    # Get all person images
    image_files = list(Path(person_dir).glob('*.jpg')) + list(Path(person_dir).glob('*.png'))
    
    if not image_files:
        print("❌ No person images found!")
        return
    
    print(f"📸 Processing {len(image_files)} person images...")
    print()
    
    success_count = 0
    failed_images = []
    
    for image_path in tqdm(image_files, desc="Generating OpenPose"):
        if generate_openpose_json(str(image_path), openpose_dir):
            success_count += 1
            
            # Also generate visualization
            json_path = os.path.join(openpose_dir, image_path.stem + '_keypoints.json')
            generate_openpose_visualization(str(image_path), json_path, openpose_dir)
        else:
            failed_images.append(image_path.name)
    
    print()
    print(f"✅ Successfully processed: {success_count}/{len(image_files)}")
    
    if failed_images:
        print(f"⚠️  Failed to detect pose in {len(failed_images)} images:")
        for name in failed_images[:10]:  # Show first 10
            print(f"   - {name}")
        if len(failed_images) > 10:
            print(f"   ... and {len(failed_images) - 10} more")
    
    print()
    print(f"📁 Output saved to: {openpose_dir}")

def main():
    parser = argparse.ArgumentParser(description='Generate OpenPose keypoints for person images')
    parser.add_argument('--data_dir', type=str, default='dataset/train',
                        help='Path to dataset directory (default: dataset/train)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  🦴 OPENPOSE KEYPOINT GENERATOR")
    print("=" * 70)
    print()
    
    if not MEDIAPIPE_AVAILABLE:
        print("❌ MediaPipe not installed!")
        print()
        print("Install with:")
        print("   pip install mediapipe")
        print()
        print("Or install all ML requirements:")
        print("   pip install -r requirements_ml.txt")
        return
    
    process_dataset(args.data_dir)
    
    print()
    print("=" * 70)
    print("  ✅ OPENPOSE GENERATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()

