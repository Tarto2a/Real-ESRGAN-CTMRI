import os
import torch

# General settings
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_NAME = "Real-ESRGAN-CTMRI"
DATASET_PATH = os.path.join(BASE_DIR, "../datasets")
EXPERIMENTS_PATH = os.path.join(BASE_DIR, "../experiments")
PRETRAINED_MODEL = os.path.join(BASE_DIR, "../pretrained/Real-ESRGANx4plus.pth")

# Dataset configuration
DATA_CONFIG = {
    "train_lr_path": os.path.join(DATASET_PATH, "train/lr"),
    "train_hr_path": os.path.join(DATASET_PATH, "train/hr"),
    "val_lr_path": os.path.join(DATASET_PATH, "val/lr"),
    "val_hr_path": os.path.join(DATASET_PATH, "val/hr"),
    "batch_size": 8,                 # Adjust based on GPU memory
    "num_workers": 4,               # Number of data loader workers
    "image_size": 128,              # Cropped image size for training
}

# Model configuration
MODEL_CONFIG = {
    "scale_factor": 4,              # Upscaling factor (e.g., 4x)
    "pretrained_model_path": PRETRAINED_MODEL,
}

# Training configuration
TRAINING_CONFIG = {
    "learning_rate": 1e-4,          # Learning rate for the optimizer
    "num_epochs": 100,              # Total number of training epochs
    "save_every_n_epochs": 5,       # Save the model every N epochs
    "log_every_n_steps": 10,        # Log training progress every N steps
    "use_gpu": True,                # Set to False to use CPU
    "device": "cuda" if MODEL_CONFIG["pretrained_model_path"] and torch.cuda.is_available() else "cpu",
}

# Loss configuration
LOSS_CONFIG = {
    "pixel_loss_weight": 1.0,       # Weight for pixel-wise loss (e.g., L1 or L2)
    "perceptual_loss_weight": 0.1,  # Weight for perceptual loss
    "adversarial_loss_weight": 0.01 # Weight for adversarial loss (GAN component)
}

# Optimizer configuration
OPTIMIZER_CONFIG = {
    "type": "Adam",                 # Optimizer type (e.g., Adam or SGD)
    "betas": (0.9, 0.999),          # Adam betas
    "weight_decay": 1e-4,           # Weight decay
}

# Scheduler configuration (optional)
SCHEDULER_CONFIG = {
    "step_size": 10,    # Interval (in epochs) for adjusting the learning rate
    "gamma": 0.1,       # Factor by which the learning rate is reduced
}


# Output paths
OUTPUT_CONFIG = {
    "logs_dir": os.path.join(EXPERIMENTS_PATH, "logs"),
    "models_dir": os.path.join(EXPERIMENTS_PATH, "models"),
}

# Debugging/Validation
DEBUG_CONFIG = {
    "validate_every_n_epochs": 1,   # Perform validation every N epochs
    "visualize_results": True,      # Save visual comparison of results during validation
}

# Print configurations (for debugging purposes)
if __name__ == "__main__":
    print("Training Configuration:")
    print("DATA_CONFIG:", DATA_CONFIG)
    print("MODEL_CONFIG:", MODEL_CONFIG)
    print("TRAINING_CONFIG:", TRAINING_CONFIG)
    print("LOSS_CONFIG:", LOSS_CONFIG)
    print("OPTIMIZER_CONFIG:", OPTIMIZER_CONFIG)
    print("SCHEDULER_CONFIG:", SCHEDULER_CONFIG)
    print("OUTPUT_CONFIG:", OUTPUT_CONFIG)
    print("DEBUG_CONFIG:", DEBUG_CONFIG)
