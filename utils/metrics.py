"""
Evaluation metrics for virtual try-on
"""

import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def compute_ssim(pred, target):
    """
    Compute SSIM between predicted and target images
    
    Args:
        pred: Predicted images (B, 3, H, W) in range [0, 1]
        target: Target images (B, 3, H, W) in range [0, 1]
    Returns:
        ssim_score: Average SSIM across batch
    """
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    batch_size = pred_np.shape[0]
    ssim_scores = []
    
    for i in range(batch_size):
        # Transpose from (C, H, W) to (H, W, C)
        pred_img = np.transpose(pred_np[i], (1, 2, 0))
        target_img = np.transpose(target_np[i], (1, 2, 0))
        
        score = ssim(pred_img, target_img, multichannel=True, data_range=1.0)
        ssim_scores.append(score)
    
    return np.mean(ssim_scores)


def compute_psnr(pred, target):
    """
    Compute PSNR between predicted and target images
    
    Args:
        pred: Predicted images (B, 3, H, W) in range [0, 1]
        target: Target images (B, 3, H, W) in range [0, 1]
    Returns:
        psnr_score: Average PSNR across batch
    """
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    batch_size = pred_np.shape[0]
    psnr_scores = []
    
    for i in range(batch_size):
        pred_img = np.transpose(pred_np[i], (1, 2, 0))
        target_img = np.transpose(target_np[i], (1, 2, 0))
        
        score = psnr(pred_img, target_img, data_range=1.0)
        psnr_scores.append(score)
    
    return np.mean(psnr_scores)


def compute_fid(real_features, fake_features):
    """
    Compute Fréchet Inception Distance (simplified)
    
    Args:
        real_features: Features from real images (N, D)
        fake_features: Features from generated images (N, D)
    Returns:
        fid_score: FID value
    """
    # Compute mean and covariance
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)
    
    # Compute FID
    diff = mu_real - mu_fake
    
    # Product might be almost singular
    covmean = np.sqrt(sigma_real.dot(sigma_fake))
    
    # Numerical error might give complex numbers
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
    
    return fid


def compute_lpips(pred, target, lpips_model=None):
    """
    Compute LPIPS (Learned Perceptual Image Patch Similarity)
    Requires lpips package: pip install lpips
    
    Args:
        pred: Predicted images (B, 3, H, W) in range [-1, 1]
        target: Target images (B, 3, H, W) in range [-1, 1]
        lpips_model: LPIPS model instance
    Returns:
        lpips_score: Average LPIPS distance
    """
    if lpips_model is None:
        try:
            import lpips
            lpips_model = lpips.LPIPS(net='alex').to(pred.device)
        except ImportError:
            print("LPIPS package not installed. Run: pip install lpips")
            return 0.0
    
    with torch.no_grad():
        distance = lpips_model(pred, target)
    
    return distance.mean().item()

