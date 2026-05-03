# src/evaluate.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from src.model import CNN
from src.data_loader import get_data_loaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(path="models/cnn_model.pth"):
    model = CNN().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


def evaluate_model(model, loader):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(cm):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(10)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.tight_layout()
    plt.show()


def main():
    # Load data
    _, test_loader = get_data_loaders(batch_size=64)

    # Load model
    model = load_model()

    # Evaluate
    preds, labels = evaluate_model(model, test_loader)

    # Accuracy
    accuracy = (preds == labels).mean()
    print(f"\n🎯 Test Accuracy: {accuracy:.4f}\n")

    # Classification Report
    print("📊 Classification Report:")
    print(classification_report(labels, preds))

    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    print("🧩 Confusion Matrix:")
    print(cm)

    plot_confusion_matrix(cm)


if __name__ == "__main__":
    main()