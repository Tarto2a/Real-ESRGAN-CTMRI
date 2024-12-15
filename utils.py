import os
import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import save_image
import cv2

# Function to save the model checkpoint
def save_checkpoint(model, epoch, optimizer, models_dir):
    """
    Saves the model checkpoint.
    
    Args:
    - model: The model to save.
    - epoch: The current epoch.
    - optimizer: The optimizer being used.
    - models_dir: Directory to save the model.
    """
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    checkpoint_path = os.path.join(models_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


# Function to load the model from a checkpoint
def load_model(model, checkpoint_path):
    """
    Loads the model state from a checkpoint.

    Args:
    - model: The model to load.
    - checkpoint_path: The path to the checkpoint file.
    
    Returns:
    - model: The model with loaded weights.
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {checkpoint_path}")
    return model


# Function to compute PSNR (Peak Signal-to-Noise Ratio)
def compute_psnr(sr, hr, max_pixel=1.0):
    """
    Compute PSNR between the super-resolved and ground-truth images.

    Args:
    - sr: The super-resolved image.
    - hr: The ground truth high-resolution image.
    - max_pixel: The maximum pixel value in the image (default: 1.0).

    Returns:
    - psnr: The computed PSNR value.
    """
    mse = np.mean((sr - hr) ** 2)
    if mse == 0:
        return 100  # Perfect match
    return 20 * np.log10(max_pixel / np.sqrt(mse))


# Function to save the generated images (optional, for visualization)
def save_generated_images(epoch, sr, hr, logs_dir):
    """
    Saves a comparison of the super-resolved and ground-truth images.
    
    Args:
    - epoch: The current epoch number.
    - sr: The super-resolved image.
    - hr: The high-resolution image.
    - logs_dir: Directory to save the images.
    """
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # Convert tensors to images and save
    sr_image = sr.cpu().detach().numpy().transpose(1, 2, 0)  # Convert to HWC
    hr_image = hr.cpu().detach().numpy().transpose(1, 2, 0)  # Convert to HWC
    sr_image = np.clip(sr_image * 255, 0, 255).astype(np.uint8)
    hr_image = np.clip(hr_image * 255, 0, 255).astype(np.uint8)
    
    save_path = os.path.join(logs_dir, f"epoch_{epoch}_generated.png")
    comparison_image = np.concatenate([sr_image, hr_image], axis=1)  # Side by side
    cv2.imwrite(save_path, comparison_image)
    print(f"Comparison image saved to {save_path}")
