"""Neural network models for the course.

AssignmentCNN follows the architecture required in Assignment 2:
64x64x3 input -> Conv(16) -> ReLU -> MaxPool -> Conv(32) -> ReLU -> MaxPool
-> Flatten -> FC(100) -> ReLU -> FC(10)

GAN (Generator + Discriminator) follows the architecture required in Assignment 3
for 28x28 MNIST digits.
"""

import torch.nn as nn
import torch.nn.functional as F


class AssignmentCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # 3 input channels (RGB) -> 16 feature maps; padding=1 keeps the 64x64 size
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        # 16 -> 32 feature maps; padding=1 keeps the size
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # Halves height and width
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # After two poolings the image is 16x16 (64 -> 32 -> 16) with 32 channels
        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 16, 32, 32]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 32, 16, 16]
        x = x.view(x.size(0), -1)             # Flatten -> [B, 8192]
        x = F.relu(self.fc1(x))               # [B, 100]
        x = self.fc2(x)                       # [B, 10], one score per class
        return x


class Generator(nn.Module):
    """Turns a noise vector of size 100 into a 1 x 28 x 28 image."""

    def __init__(self, z_dim: int = 100):
        super().__init__()
        self.z_dim = z_dim
        # Noise vector -> 7 * 7 * 128 numbers, reshaped to 128 maps of 7x7
        self.fc = nn.Linear(z_dim, 7 * 7 * 128)
        # 7x7 -> 14x14
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(True)
        # 14x14 -> 28x28, one channel (grayscale)
        self.deconv2 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()  # pixel values between -1 and 1

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 128, 7, 7)       # [B, 128, 7, 7]
        x = self.act1(self.bn1(self.deconv1(x)))  # [B, 64, 14, 14]
        x = self.tanh(self.deconv2(x))         # [B, 1, 28, 28]
        return x


class Discriminator(nn.Module):
    """Takes a 1 x 28 x 28 image and returns the probability that it is real."""

    def __init__(self):
        super().__init__()
        # 28x28 -> 14x14
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        # 14x14 -> 7x7
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 7 * 7, 1)
        self.sigmoid = nn.Sigmoid()  # turns the score into a probability

    def forward(self, x):
        x = self.act1(self.conv1(x))             # [B, 64, 14, 14]
        x = self.act2(self.bn2(self.conv2(x)))   # [B, 128, 7, 7]
        x = self.flatten(x)                      # [B, 6272]
        x = self.sigmoid(self.fc(x))             # [B, 1], probability of "real"
        return x


class GAN(nn.Module):
    """Holds the generator and the discriminator together."""

    def __init__(self, z_dim: int = 100):
        super().__init__()
        self.z_dim = z_dim
        self.generator = Generator(z_dim)
        self.discriminator = Discriminator()


def get_model(model_name: str) -> nn.Module:
    """Return a model by name, as required by the helper_lib activity."""
    if model_name == "CNN":
        return AssignmentCNN()
    if model_name == "GAN":
        return GAN()
    raise ValueError(f"Unknown model name: {model_name}. Available: CNN, GAN")