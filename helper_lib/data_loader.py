"""Data loading for the CIFAR10 dataset."""

import torch
from torchvision import datasets, transforms

# The Assignment 2 CNN expects 64x64 inputs, while CIFAR10 images are 32x32
IMAGE_SIZE = 64

CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def get_transform():
    """Resize to 64x64 and convert to a tensor.

    The API uses this same function, so images are prepared
    exactly the same way for training and for prediction.
    """
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )


def get_data_loader(data_dir, batch_size=32, train=True):
    """Download CIFAR10 (if needed) and return a DataLoader."""
    dataset = datasets.CIFAR10(
        root=data_dir, train=train, download=True, transform=get_transform()
    )
    # Shuffle only the training data; the test data keeps a fixed order
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return loader