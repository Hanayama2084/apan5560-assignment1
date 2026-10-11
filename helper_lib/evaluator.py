"""Evaluation on the test set, moved from the Module 4 CNN practical."""

import torch


def evaluate_model(model, data_loader, criterion, device="cpu"):
    """Return the average loss and the accuracy (in %) on the given data."""
    model.to(device)
    model.eval()  # evaluation mode
    total_loss = 0.0
    correct, total = 0, 0
    with torch.no_grad():  # no gradients needed when we only make predictions
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total
    print(f"Test loss: {avg_loss:.4f}  Test accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy