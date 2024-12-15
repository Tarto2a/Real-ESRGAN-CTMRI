import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from models.models import RealESRGAN
from datasets.datasets import CustomDataset
from utils import compute_psnr, save_checkpoint, load_model
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np

# Load configuration from train_config.py
from configs.train_config import DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG, LOSS_CONFIG, OPTIMIZER_CONFIG, SCHEDULER_CONFIG, OUTPUT_CONFIG

# Device configuration
device = torch.device(TRAINING_CONFIG["device"])

def train():
    # Dataset preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_dataset = CustomDataset(
        lr_dir=DATA_CONFIG["train_lr_path"],
        hr_dir=DATA_CONFIG["train_hr_path"],
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=DATA_CONFIG["batch_size"], shuffle=True, num_workers=DATA_CONFIG["num_workers"])

    val_dataset = CustomDataset(
        lr_dir=DATA_CONFIG["val_lr_path"],
        hr_dir=DATA_CONFIG["val_hr_path"],
        transform=transform
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Initialize Real-ESRGAN model
    model = RealESRGAN(scale_factor=MODEL_CONFIG["scale_factor"]).to(device)

    # Load pre-trained model if exists
    if os.path.exists(MODEL_CONFIG["pretrained_model_path"]):
        print(f"Loading pre-trained model from {MODEL_CONFIG['pretrained_model_path']}")
        model = load_model(model, MODEL_CONFIG["pretrained_model_path"])

    # Loss function (Pixel loss, Perceptual loss, Adversarial loss)
    criterion_pixel = nn.L1Loss().to(device)  # Using L1 loss for simplicity

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=TRAINING_CONFIG["learning_rate"], betas=OPTIMIZER_CONFIG["betas"], weight_decay=OPTIMIZER_CONFIG["weight_decay"])
    scheduler = StepLR(optimizer, step_size=SCHEDULER_CONFIG["step_size"], gamma=SCHEDULER_CONFIG["gamma"])

    # Training loop
    for epoch in range(TRAINING_CONFIG["num_epochs"]):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TRAINING_CONFIG['num_epochs']}")

        for i, (lr, hr) in enumerate(pbar):
            lr, hr = lr.to(device), hr.to(device)

            # Forward pass
            sr = model(lr)

            # Pixel loss (L1 loss)
            pixel_loss = criterion_pixel(sr, hr)

            # Backprop and optimize
            optimizer.zero_grad()
            pixel_loss.backward()
            optimizer.step()

            running_loss += pixel_loss.item()

            # Update the progress bar
            pbar.set_postfix({"loss": running_loss / (i + 1)})

        # Adjust learning rate
        scheduler.step()

        # Save model and log loss
        if (epoch + 1) % TRAINING_CONFIG["save_every_n_epochs"] == 0:
            save_checkpoint(model, epoch, optimizer, OUTPUT_CONFIG["models_dir"])

        # Validation (on a subset of data or full dataset)
        if (epoch + 1) % TRAINING_CONFIG["validate_every_n_epochs"] == 0:
            model.eval()
            psnr_values = []
            with torch.no_grad():
                for lr, hr in val_loader:
                    lr, hr = lr.to(device), hr.to(device)
                    sr = model(lr)
                    psnr_value = compute_psnr(sr, hr)
                    psnr_values.append(psnr_value)
            avg_psnr = np.mean(psnr_values)
            print(f"Epoch {epoch+1}: Average PSNR on validation set: {avg_psnr:.2f} dB")

        # Save logs (Optional)
        if (epoch + 1) % TRAINING_CONFIG["log_every_n_steps"] == 0:
            print(f"Epoch [{epoch+1}/{TRAINING_CONFIG['num_epochs']}] Loss: {running_loss / len(train_loader)}")

    # Final model save
    save_checkpoint(model, TRAINING_CONFIG["num_epochs"], optimizer, OUTPUT_CONFIG["models_dir"])

# Ensure that the script only runs the training loop when executed directly
if __name__ == "__main__":
    train()
