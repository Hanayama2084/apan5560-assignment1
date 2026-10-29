"""MNIST digit generator served by the API (Assignment 3)."""

import torch

from helper_lib.generator import generate_png_bytes
from helper_lib.model import get_model
from helper_lib.utils import load_model


class DigitGenerator:
    """Loads the trained GAN generator once and draws new handwritten digits."""

    def __init__(self, weights_path: str = "models/gan_generator.pth"):
        self.device = torch.device("cpu")
        self.model = get_model("GAN")
        # Only the generator is needed to draw digits; the discriminator was used for training
        load_model(self.model.generator, weights_path, device=self.device)

    def generate_png(self, num_samples: int = 10, seed: int | None = None) -> bytes:
        """Return num_samples new digits as one PNG grid. A fixed seed gives the same digits."""
        if seed is not None:
            torch.manual_seed(seed)
        return generate_png_bytes(self.model, self.device, num_samples)
