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

def train_gan(model, data_loader, criterion, optimizer, device="cpu", epochs=10):
    """Train a GAN (generator + discriminator) and return it.

    optimizer is a pair (optimizer_d, optimizer_g): one optimizer for the
    discriminator and one for the generator. criterion is BCELoss.
    """
    optimizer_d, optimizer_g = optimizer
    generator = model.generator.to(device)
    discriminator = model.discriminator.to(device)

    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        running_d, running_g = 0.0, 0.0

        for real_images, _ in data_loader:  # labels are not needed
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            real_labels = torch.ones(batch_size, 1, device=device)   # 1 = real
            fake_labels = torch.zeros(batch_size, 1, device=device)  # 0 = fake

            # 1) Train the discriminator: real images -> 1, fake images -> 0
            z = torch.randn(batch_size, model.z_dim, device=device)
            fake_images = generator(z)
            optimizer_d.zero_grad()
            loss_real = criterion(discriminator(real_images), real_labels)
            loss_fake = criterion(discriminator(fake_images.detach()), fake_labels)
            loss_d = loss_real + loss_fake
            loss_d.backward()
            optimizer_d.step()

            # 2) Train the generator: make the discriminator say 1 for fakes
            optimizer_g.zero_grad()
            loss_g = criterion(discriminator(fake_images), real_labels)
            loss_g.backward()
            optimizer_g.step()

            running_d += loss_d.item()
            running_g += loss_g.item()

        print(
            f"Epoch {epoch + 1}/{epochs}  "
            f"D loss: {running_d / len(data_loader):.4f}  "
            f"G loss: {running_g / len(data_loader):.4f}"
        )
    return model
