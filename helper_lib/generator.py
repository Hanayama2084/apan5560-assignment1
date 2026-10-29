"""Generate new digit images with a trained GAN generator."""

import io

import torch
from torchvision.utils import make_grid, save_image


def _sample(model, device, num_samples):
    """Feed random noise into the generator and return images in [0, 1]."""
    generator = model.generator.to(device)
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.z_dim, device=device)
        images = generator(z).cpu()
    return (images + 1) / 2  # Tanh output [-1, 1] -> [0, 1]


def generate_samples(model, device, num_samples=10):
    """Generate num_samples digits and show them in a grid."""
    import matplotlib.pyplot as plt

    images = _sample(model, device, num_samples)
    grid = make_grid(images, nrow=min(num_samples, 10), padding=2)
    plt.figure(figsize=(num_samples, 2))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.axis("off")
    plt.show()


def generate_png_bytes(model, device="cpu", num_samples=10):
    """Generate digits and return them as one PNG image (used by the API)."""
    images = _sample(model, device, num_samples)
    buffer = io.BytesIO()
    save_image(images, buffer, format="png", nrow=min(num_samples, 10), padding=2)
    return buffer.getvalue()
