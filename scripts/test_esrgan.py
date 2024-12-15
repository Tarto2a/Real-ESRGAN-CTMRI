import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from models import RealESRGAN
from utils import load_model, compute_psnr, save_generated_images
from prepare_data import CTMRIDataset

# Define the function to test the model
def test_model(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    psnr_total = 0.0
    num_images = 0

    with torch.no_grad():
        for lr, hr in test_loader:
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)  # Super-resolved output from the model

            # Convert to numpy arrays for PSNR calculation
            sr = sr.cpu().numpy().transpose(0, 2, 3, 1)  # Convert to HWC format
            hr = hr.cpu().numpy().transpose(0, 2, 3, 1)  # Convert to HWC format

            # Compute PSNR for this batch
            for i in range(lr.size(0)):
                psnr = compute_psnr(sr[i], hr[i])
                psnr_total += psnr
                num_images += 1

            # Optionally save the generated images for visual comparison
            save_generated_images(epoch=0, sr=sr[0], hr=hr[0], logs_dir='test_results')

    average_psnr = psnr_total / num_images
    print(f"Average PSNR on test dataset: {average_psnr:.4f}")

# Main testing function
if __name__ == "__main__":
    # Set the device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model = RealESRGAN().to(device)

    # Load the trained weights
    checkpoint_path = "path_to_your_checkpoint.pth"  # Replace with your model checkpoint path
    model = load_model(model, checkpoint_path)

    # Prepare the test data loader
    test_image_dir = "path_to_test_images"  # Replace with your test images directory
    test_loader = DataLoader(CTMRIDataset(test_image_dir, hr_size=256, lr_size=64, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])), batch_size=16, shuffle=False)

    # Test the model
    test_model(model, test_loader, device)
