"""CIFAR10 image classifier served by the API (Assignment 2)."""

import io

import torch
from PIL import Image

from helper_lib.data_loader import CLASSES, get_transform
from helper_lib.model import get_model
from helper_lib.utils import load_model


class ImageClassifier:
    """Loads the trained CNN once and predicts the CIFAR10 class of an image."""

    def __init__(self, weights_path: str = "models/cnn_cifar10.pth"):
        self.device = torch.device("cpu")
        self.model = load_model(get_model("CNN"), weights_path, device=self.device)
        # Same preprocessing as training: resize to 64x64, convert to tensor
        self.transform = get_transform()

    def predict(self, image_bytes: bytes) -> dict:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x = self.transform(image).unsqueeze(0)  # add a batch dimension: [1, 3, 64, 64]
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0]  # 10 scores -> 10 probabilities
        top_prob, top_idx = torch.max(probs, 0)
        return {
            "predicted_class": CLASSES[top_idx.item()],
            "confidence": round(top_prob.item(), 4),
            "probabilities": {c: round(p.item(), 4) for c, p in zip(CLASSES, probs)},
        }