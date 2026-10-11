"""Small helpers: choose a device, save and load trained weights."""

import os

import torch

DEFAULT_MODEL_PATH = "models/cnn_cifar10.pth"


def get_device():
    """Use the Apple GPU (MPS) or an NVIDIA GPU if available, otherwise the CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_model(model, path=DEFAULT_MODEL_PATH):
    """Save the trained weights (the model's state_dict) to a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
    return path


def load_model(model, path=DEFAULT_MODEL_PATH, device="cpu"):
    """Load trained weights into a model and switch it to evaluation mode."""
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model