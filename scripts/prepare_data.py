import os
import cv2
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class CTMRIDataset(Dataset):
    def __init__(self, image_dir, hr_size=256, lr_size=64, transform=None):
        """
        Args:
            image_dir (str): Path to the directory containing CT and MRI images.
            hr_size (int): Size of the high-resolution images (for training).
            lr_size (int): Size of the low-resolution images (to simulate downscaling).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.hr_size = hr_size
        self.lr_size = lr_size
        self.transform = transform
        
        # Get the list of image file paths
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the image
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # High-resolution image (original)
        hr_img = cv2.resize(img, (self.hr_size, self.hr_size))

        # Low-resolution image (downscaled version)
        lr_img = cv2.resize(img, (self.lr_size, self.lr_size))

        # Apply transform if provided
        if self.transform:
            hr_img = self.transform(hr_img)
            lr_img = self.transform(lr_img)
        
        return lr_img, hr_img

def prepare_data(image_dir, batch_size=16, hr_size=256, lr_size=64):
    """
    Prepares and returns a DataLoader for the CT and MRI dataset.
    
    Args:
    - image_dir (str): Directory containing the images.
    - batch_size (int): Number of samples per batch.
    - hr_size (int): High-resolution size.
    - lr_size (int): Low-resolution size.

    Returns:
    - train_loader (DataLoader): DataLoader for training.
    """
    
    # Define data transformations (normalization, tensor conversion)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Create the dataset
    dataset = CTMRIDataset(image_dir, hr_size, lr_size, transform)
    
    # Create the DataLoader
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return train_loader

if __name__ == "__main__":
    image_dir = "path_to_your_image_folder"  # Replace with the path to your image folder
    train_loader = prepare_data(image_dir)

    # Print some sample data to verify
    for lr, hr in tqdm(train_loader):
        print(lr.shape, hr.shape)
        break  # Only printing one batch for testing
