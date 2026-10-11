"""Train the Assignment 2 CNN on CIFAR10 and save the weights.

Run from the project root:
    uv run python train.py
"""

import torch
import torch.nn as nn
import torch.optim as optim

from helper_lib.data_loader import get_data_loader
from helper_lib.evaluator import evaluate_model
from helper_lib.model import get_model
from helper_lib.trainer import train_model
from helper_lib.utils import get_device, save_model

EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001


def main():
    torch.manual_seed(42)  # makes the results repeatable
    device = get_device()
    print(f"Using device: {device}")

    train_loader = get_data_loader("data", batch_size=BATCH_SIZE, train=True)
    test_loader = get_data_loader("data", batch_size=BATCH_SIZE, train=False)

    model = get_model("CNN")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    trained_model = train_model(
        model, train_loader, criterion, optimizer, device=device, epochs=EPOCHS
    )
    evaluate_model(trained_model, test_loader, criterion, device=device)
    save_model(trained_model)


if __name__ == "__main__":
    main()