import os
import torch
import torch.nn as nn
import torch.optim as optim

from src.model import CNN
from src.data_loader import get_data_loaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def main():
    train_loader, test_loader = get_data_loaders()

    model = CNN().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_acc = 0

    for epoch in range(10):
        loss = train_one_epoch(model, train_loader, criterion, optimizer)
        acc = evaluate(model, test_loader)

        print(f"Epoch {epoch+1}")
        print(f"Loss: {loss:.4f}, Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/cnn_model.pth")

    print(f"Best Accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()