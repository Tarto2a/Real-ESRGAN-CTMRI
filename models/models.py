import torch
import torch.nn as nn
import torch.nn.functional as F

# Residual Block for ESRGAN
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return x + residual  # Skip connection

# Generator network for ESRGAN
class RealESRGAN(nn.Module):
    def __init__(self, scale_factor=4, num_residual_blocks=16, num_channels=64):
        super(RealESRGAN, self).__init__()
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=9, padding=4)
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_channels, num_channels) for _ in range(num_residual_blocks)]
        )
        
        # Upsampling layers (to scale the image)
        self.upsample1 = nn.Conv2d(num_channels, num_channels * scale_factor**2, kernel_size=3, padding=1)
        self.upsample2 = nn.PixelShuffle(scale_factor)
        
        # Final output convolution to get the 3 channels of the image
        self.conv2 = nn.Conv2d(num_channels, 3, kernel_size=9, padding=4)

    def forward(self, x):
        # Initial convolution
        x = F.relu(self.conv1(x))
        
        # Apply residual blocks
        x = self.residual_blocks(x)
        
        # Upsample the image
        x = self.upsample1(x)
        x = self.upsample2(x)
        
        # Final convolution
        x = self.conv2(x)
        
        return x

