"""Train the Assignment 3 GAN on MNIST and save the generator weights.

Run from the project root:
    uv run python train_gan.py
"""

import torch
import torch.nn as nn
import torch.optim as optim

from helper_lib.data_loader import get_mnist_loader
from helper_lib.generator import generate_png_bytes
from helper_lib.model import get_model
from helper_lib.trainer import train_gan
from helper_lib.utils import get_device, save_model

EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 0.0002
GENERATOR_PATH = "models/gan_generator.pth"


def main():
    torch.manual_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    train_loader = get_mnist_loader("data", batch_size=BATCH_SIZE, train=True)

    model = get_model("GAN")
    criterion = nn.BCELoss()
    optimizer_d = optim.Adam(model.discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_g = optim.Adam(model.generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    model = train_gan(
        model, train_loader, criterion, (optimizer_d, optimizer_g), device=device, epochs=EPOCHS
    )
    save_model(model.generator, GENERATOR_PATH)

    # Save a preview so we can check the digits look right
    with open("gan_samples.png", "wb") as f:
        f.write(generate_png_bytes(model, device, num_samples=10))
    print("Preview saved to gan_samples.png")


if __name__ == "__main__":
    main()
