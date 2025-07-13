import torch
import torch.nn as nn
import torch.nn.functional as F


class VolumeEncoder(nn.Module):
    def __init__(self, output_channels):
        super(VolumeEncoder, self).__init__()
        self.conv3d_1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3d_2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv3d_3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)

        self.bn3d_1 = nn.BatchNorm3d(32)
        self.bn3d_2 = nn.BatchNorm3d(64)
        self.bn3d_3 = nn.BatchNorm3d(128)

        self.conv2d_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2d_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv2d_3 = nn.Conv2d(in_channels=256, out_channels=output_channels, kernel_size=3, stride=2, padding=1)

        self.bn2d_1 = nn.BatchNorm2d(128)
        self.bn2d_2 = nn.BatchNorm2d(256)
        self.bn2d_3 = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        # Apply 3D convolutional layers with ReLU and batch normalization
        x = F.relu(self.bn3d_1(self.conv3d_1(x)))
        x = F.relu(self.bn3d_2(self.conv3d_2(x)))
        x = F.relu(self.bn3d_3(self.conv3d_3(x)))

        # Convert to bird's-eye view (sum over height dimension)
        x = torch.mean(x, dim=2)

        # Apply 2D convolutional layers
        x = F.relu(self.bn2d_1(self.conv2d_1(x)))
        x = F.relu(self.bn2d_2(self.conv2d_2(x)))
        x = F.relu(self.bn2d_3(self.conv2d_3(x)))

        return x
