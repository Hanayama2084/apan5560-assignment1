"""Neural network models for the course.

AssignmentCNN follows the architecture required in Assignment 2:
64x64x3 input -> Conv(16) -> ReLU -> MaxPool -> Conv(32) -> ReLU -> MaxPool
-> Flatten -> FC(100) -> ReLU -> FC(10)
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


def get_model(model_name: str) -> nn.Module:
    """Return a model by name, as required by the helper_lib activity."""
    if model_name == "CNN":
        return AssignmentCNN()
    raise ValueError(f"Unknown model name: {model_name}. Available: CNN")