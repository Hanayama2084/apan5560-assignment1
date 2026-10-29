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

def get_mnist_loader(data_dir, batch_size=128, train=True):
    """Download MNIST (if needed) and return a DataLoader for the GAN.

    Pixels are scaled to the range [-1, 1] so they match the Tanh output
    of the generator.
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),                 # pixels in [0, 1]
            transforms.Normalize((0.5,), (0.5,)),  # (x - 0.5) / 0.5 -> [-1, 1]
        ]
    )
    dataset = datasets.MNIST(root=data_dir, train=train, download=True, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train)
