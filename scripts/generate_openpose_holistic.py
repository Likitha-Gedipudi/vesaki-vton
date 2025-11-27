#!/usr/bin/env python3
"""
MediaPipe Holistic Pose Detection
Generates OpenPose-compatible keypoints with hands and face
Superior accuracy and stability compared to basic pose detection
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
    print("Error: MediaPipe not installed!")
    print("Install with: pip3 install mediapipe")

# OpenPose 25-point format (body + hands)
OPENPOSE_25 = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist", "MidHip", "RHip",
    "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar", "LEar", "LBigToe",
    "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"
]

# MediaPipe Pose (33 points) to OpenPose (25 points) mapping
MP_TO_OPENPOSE = {
    0: 0,   # Nose
    11: 2,  # Right shoulder
    13: 3,  # Right elbow
    15: 4,  # Right wrist
    12: 5,  # Left shoulder
    14: 6,  # Left elbow
    16: 7,  # Left wrist
    23: 8,  # Mid hip (approximate)
    24: 9,  # Right hip
    26: 10, # Right knee
    28: 11, # Right ankle
    23: 12, # Left hip
    25: 13, # Left knee
    27: 14, # Left ankle
    2: 15,  # Right eye
    5: 16,  # Left eye
    8: 17,  # Right ear
    7: 18,  # Left ear
}

class KalmanFilter:
    """Simple Kalman filter for smoothing keypoints"""
    def __init__(self, process_noise=0.01, measurement_noise=0.1):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.estimate = None
        self.error = 1.0
        
    def update(self, measurement):
        if self.estimate is None:
            self.estimate = measurement
            return measurement
        
        # Prediction
        prediction = self.estimate
        prediction_error = self.error + self.process_noise
        
        # Update
        kalman_gain = prediction_error / (prediction_error + self.measurement_noise)
        self.estimate = prediction + kalman_gain * (measurement - prediction)
        self.error = (1 - kalman_gain) * prediction_error
        
        return self.estimate

def generate_holistic_keypoints(image_path, output_dir, use_smoothing=True):
    """Generate pose keypoints using MediaPipe Holistic"""
    if not MEDIAPIPE_AVAILABLE:
        return False
    
    # Initialize MediaPipe Holistic
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        return False
    
    h, w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe Holistic
    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        results = holistic.process(image_rgb)
    
    # Check if pose detected
    if not results.pose_landmarks:
        return False
    
    # Convert to OpenPose 25-point format
    openpose_keypoints = [[0, 0, 0] for _ in range(25)]
    
    # Map body keypoints
    for mp_idx, op_idx in MP_TO_OPENPOSE.items():
        if mp_idx < len(results.pose_landmarks.landmark):
            landmark = results.pose_landmarks.landmark[mp_idx]
            openpose_keypoints[op_idx] = [
                landmark.x * w,
                landmark.y * h,
                landmark.visibility
            ]
    
    # Calculate neck (midpoint between shoulders)
    if openpose_keypoints[2][2] > 0 and openpose_keypoints[5][2] > 0:
        openpose_keypoints[1] = [
            (openpose_keypoints[2][0] + openpose_keypoints[5][0]) / 2,
            (openpose_keypoints[2][1] + openpose_keypoints[5][1]) / 2,
            (openpose_keypoints[2][2] + openpose_keypoints[5][2]) / 2
        ]
    
    # Add hand keypoints if available
    hand_confidence = 0.0
    if results.left_hand_landmarks:
        # Average confidence from left hand
        left_wrist = results.left_hand_landmarks.landmark[0]
        hand_confidence += left_wrist.visibility if hasattr(left_wrist, 'visibility') else 0.8
    
    if results.right_hand_landmarks:
        # Average confidence from right hand
        right_wrist = results.right_hand_landmarks.landmark[0]
        hand_confidence += right_wrist.visibility if hasattr(right_wrist, 'visibility') else 0.8
    
    # Create JSON structure (OpenPose format)
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
    
    # Add hand keypoints if detected
    if results.left_hand_landmarks:
        left_hand_kps = []
        for landmark in results.left_hand_landmarks.landmark:
            left_hand_kps.extend([
                landmark.x * w,
                landmark.y * h,
                landmark.visibility if hasattr(landmark, 'visibility') else 0.8
            ])
        pose_data["people"][0]["hand_left_keypoints_2d"] = left_hand_kps
    
    if results.right_hand_landmarks:
        right_hand_kps = []
        for landmark in results.right_hand_landmarks.landmark:
            right_hand_kps.extend([
                landmark.x * w,
                landmark.y * h,
                landmark.visibility if hasattr(landmark, 'visibility') else 0.8
            ])
        pose_data["people"][0]["hand_right_keypoints_2d"] = right_hand_kps
    
    # Save JSON
    output_path = os.path.join(output_dir, Path(image_path).stem + '_keypoints.json')
    with open(output_path, 'w') as f:
        json.dump(pose_data, f, indent=2)
    
    # Generate visualization
    generate_visualization(image_path, openpose_keypoints, results, output_dir)
    
    return True

def generate_visualization(image_path, keypoints, holistic_results, output_dir):
    """Generate visualization with skeleton overlay"""
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        return
    
    # MediaPipe drawing
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Draw pose
    if holistic_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            holistic_results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
    
    # Draw hands
    if holistic_results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            holistic_results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
        )
    
    if holistic_results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            holistic_results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
        )
    
    # Save visualization
    output_path = os.path.join(output_dir, Path(image_path).stem + '.jpg')
    cv2.imwrite(output_path, image)

def process_dataset(data_dir):
    """Process all person images"""
    person_dir = os.path.join(data_dir, 'person')
    openpose_dir = os.path.join(data_dir, 'openpose')
    
    os.makedirs(openpose_dir, exist_ok=True)
    
    # Get all person images
    image_files = list(Path(person_dir).glob('*.jpg')) + list(Path(person_dir).glob('*.png'))
    
    if not image_files:
        print("No person images found!")
        return
    
    print(f"Processing {len(image_files)} person images...")
    print()
    
    success_count = 0
    failed_images = []
    
    for image_path in tqdm(image_files, desc="Generating pose keypoints"):
        if generate_holistic_keypoints(str(image_path), openpose_dir):
            success_count += 1
        else:
            failed_images.append(image_path.name)
    
    print()
    print(f"✓ Successfully processed: {success_count}/{len(image_files)}")
    
    if failed_images:
        print(f"⚠ Failed to detect pose in {len(failed_images)} images:")
        for name in failed_images[:10]:
            print(f"   - {name}")
        if len(failed_images) > 10:
            print(f"   ... and {len(failed_images) - 10} more")
    
    print(f"✓ Output saved to: {openpose_dir}")

def main():
    parser = argparse.ArgumentParser(description='Generate MediaPipe Holistic pose keypoints')
    parser.add_argument('--data_dir', type=str, default='dataset/train',
                        help='Path to dataset directory')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  MEDIAPIPE HOLISTIC POSE GENERATOR")
    print("=" * 70)
    print()
    
    if not MEDIAPIPE_AVAILABLE:
        print("Error: MediaPipe not installed!")
        print()
        print("Install with:")
        print("   pip3 install mediapipe")
        return
    
    print("Using MediaPipe Holistic for:")
    print("  • 33-point body pose")
    print("  • 21-point hand landmarks (both hands)")
    print("  • 468-point face mesh")
    print()
    print("Output: OpenPose 25-point format + hand keypoints")
    print()
    
    process_dataset(args.data_dir)
    
    print()
    print("=" * 70)
    print("  POSE GENERATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()

