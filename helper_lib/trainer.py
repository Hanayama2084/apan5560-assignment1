"""Training loop, moved from the Module 4 CNN practical."""

import torch


def train_model(model, data_loader, criterion, optimizer, device="cpu", epochs=10):
    """Train the model for a number of epochs and return it."""
    model.to(device)
    for epoch in range(epochs):
        model.train()  # put the model in training mode before each epoch
        running_loss = 0.0
        running_correct, running_total = 0, 0

        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()              # clear gradients from the previous batch
            outputs = model(inputs)            # forward pass: 10 scores per image
            loss = criterion(outputs, labels)  # how wrong the scores are
            loss.backward()                    # compute gradients
            optimizer.step()                   # update the parameters

            _, predicted = torch.max(outputs, 1)  # class with the highest score
            running_correct += (predicted == labels).sum().item()
            running_total += labels.size(0)
            running_loss += loss.item()

        avg_loss = running_loss / len(data_loader)
        accuracy = running_correct / running_total
        print(f"Epoch {epoch + 1}/{epochs}  loss: {avg_loss:.4f}  accuracy: {accuracy:.3f}")
    return model