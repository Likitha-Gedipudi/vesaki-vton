"""
PyTorch Dataset for Vesaki-VTON
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class VITONDataset(Dataset):
    """
    VITON-style dataset for virtual try-on
    
    Loads:
    - Person images
    - Garment images
    - Person parsing (SCHP)
    - Pose keypoints (OpenPose)
    - Body masks (DensePose)
    - Agnostic masks
    """
    def __init__(self, data_dir, mode='train', transform=None):
        """
        Args:
            data_dir: Root dataset directory
            mode: 'train' or 'test'
            transform: Image transformations
        """
        self.data_dir = os.path.join(data_dir, mode)
        self.mode = mode
        self.transform = transform
        
        # Load pairs file
        pairs_file = os.path.join(self.data_dir, 'pairs.txt')
        if os.path.exists(pairs_file):
            with open(pairs_file, 'r') as f:
                self.pairs = [line.strip().split() for line in f.readlines()]
        else:
            # If no pairs file, create all combinations
            person_files = sorted(os.listdir(os.path.join(self.data_dir, 'person')))
            garment_files = sorted(os.listdir(os.path.join(self.data_dir, 'garment')))
            self.pairs = [[p, g] for p in person_files for g in garment_files[:5]]  # 5 garments per person
        
        # Default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        
        # Simple to tensor (for masks)
        self.to_tensor = transforms.ToTensor()
        
    def __len__(self):
        return len(self.pairs)
    
    def load_image(self, path):
        """Load image and convert to RGB"""
        img = Image.open(path).convert('RGB')
        return img
    
    def load_mask(self, path):
        """Load mask (grayscale)"""
        if path.endswith('.png'):
            mask = Image.open(path).convert('L')
        else:
            mask = Image.open(path).convert('RGB')  # Parse visualization
        return mask
    
    def load_pose(self, path):
        """Load pose keypoints from JSON"""
        with open(path, 'r') as f:
            pose_data = json.load(f)
        
        if 'people' in pose_data and len(pose_data['people']) > 0:
            keypoints = pose_data['people'][0]['pose_keypoints_2d']
        else:
            # Return zeros if no pose detected
            keypoints = [0] * 75  # 25 points * 3 (x, y, confidence)
        
        # Convert to numpy array and reshape
        keypoints = np.array(keypoints).reshape(-1, 3)
        return keypoints
    
    def draw_pose(self, keypoints, height=1024, width=768):
        """
        Draw pose keypoints on a blank canvas
        Returns tensor representation of pose
        """
        pose_map = torch.zeros(18, height, width)
        
        # Draw each keypoint as a Gaussian blob
        for i, (x, y, confidence) in enumerate(keypoints[:18]):
            if confidence > 0.1:
                x = int(x * width) if x < 2 else int(x)
                y = int(y * height) if y < 2 else int(y)
                
                if 0 <= x < width and 0 <= y < height:
                    # Simple point representation
                    y1 = max(0, y - 2)
                    y2 = min(height, y + 3)
                    x1 = max(0, x - 2)
                    x2 = min(width, x + 3)
                    pose_map[i, y1:y2, x1:x2] = 1.0
        
        return pose_map
    
    def __getitem__(self, idx):
        """
        Get a training sample
        
        Returns dict with:
        - person: Person image (3, H, W)
        - garment: Garment image (3, H, W)
        - parse: Person parsing (1, H, W)
        - pose: Pose representation (18, H, W)
        - body_mask: Body segmentation (1, H, W)
        - agnostic: Agnostic person (3, H, W)
        - person_name: Person filename
        - garment_name: Garment filename
        """
        person_name, garment_name = self.pairs[idx]
        
        # Construct file paths
        person_path = os.path.join(self.data_dir, 'person', person_name)
        garment_path = os.path.join(self.data_dir, 'garment', garment_name)
        
        # Parse paths (need to handle _person suffix)
        person_base = person_name.replace('.jpg', '')
        parse_path = os.path.join(self.data_dir, 'person-parse', f'{person_base}_person.png')
        pose_json_path = os.path.join(self.data_dir, 'openpose', f'{person_base}_person_keypoints.json')
        body_mask_path = os.path.join(self.data_dir, 'densepose', f'{person_base}_person.png')
        agnostic_path = os.path.join(self.data_dir, 'agnostic-mask', f'{person_base}_person.jpg')
        
        # Load images
        person_img = self.load_image(person_path)
        garment_img = self.load_image(garment_path)
        
        # Load annotations
        parse_img = self.load_mask(parse_path)
        body_mask = self.load_mask(body_mask_path)
        agnostic_img = self.load_image(agnostic_path)
        
        # Load pose
        if os.path.exists(pose_json_path):
            keypoints = self.load_pose(pose_json_path)
            pose_map = self.draw_pose(keypoints, height=1024, width=768)
        else:
            pose_map = torch.zeros(18, 1024, 768)
        
        # Apply transforms
        person = self.transform(person_img)
        garment = self.transform(garment_img)
        agnostic = self.transform(agnostic_img)
        
        parse = self.to_tensor(parse_img)
        body_mask = self.to_tensor(body_mask)
        
        # Create person representation (concatenate pose + body_mask + agnostic)
        # For GMM input: (18 + 1 + 3 = 22 channels)
        person_repr = torch.cat([pose_map, body_mask, agnostic], dim=0)
        
        return {
            'person': person,
            'garment': garment,
            'parse': parse,
            'pose': pose_map,
            'body_mask': body_mask,
            'agnostic': agnostic,
            'person_repr': person_repr,
            'person_name': person_name,
            'garment_name': garment_name
        }


def get_dataloader(data_dir, mode='train', batch_size=4, num_workers=4, shuffle=True):
    """
    Create dataloader
    
    Args:
        data_dir: Root dataset directory
        mode: 'train' or 'test'
        batch_size: Batch size
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle
    Returns:
        dataloader: PyTorch DataLoader
    """
    dataset = VITONDataset(data_dir, mode=mode)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop incomplete batches
    )
    
    return dataloader

