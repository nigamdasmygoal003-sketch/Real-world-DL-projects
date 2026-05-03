"""Train the mask / no-mask CNN and export models/mask_model.pth."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src.mask_model import MaskClassifierCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a mask / no-mask CNN classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/mask_dataset"), help="Root dataset directory.")
    parser.add_argument("--output", type=Path, default=Path("models/mask_model.pth"), help="Output checkpoint path.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="cpu", help="Training device, defaults to cpu.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def build_transforms():
    train_transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def prepare_dataloaders(args: argparse.Namespace):
    train_transform, eval_transform = build_transforms()
    dataset = datasets.ImageFolder(args.data_dir)

    if len(dataset.classes) != 2:
        raise ValueError(
            f"Expected exactly 2 classes in {args.data_dir}, found {dataset.classes}. "
            "Create subfolders named Mask and No Mask."
        )

    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Dataset is too small for the requested validation split.")

    generator = torch.Generator().manual_seed(args.seed)
    train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=generator)

    train_subset.dataset = datasets.ImageFolder(args.data_dir, transform=train_transform)
    val_subset.dataset = datasets.ImageFolder(args.data_dir, transform=eval_transform)

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    return dataset.classes, train_loader, val_loader


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

        preds = logits.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / max(1, total_samples)
    avg_acc = total_correct / max(1, total_samples)
    return avg_loss, avg_acc


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not args.data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {args.data_dir}. "
            "Expected ImageFolder layout like data/mask_dataset/Mask and data/mask_dataset/No Mask."
        )

    device = torch.device(args.device)
    class_names, train_loader, val_loader = prepare_dataloaders(args)

    model = MaskClassifierCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Training on {device} with classes: {class_names}")
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, optimizer=None)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "class_names": class_names,
                    "input_size": [128, 128],
                    "best_val_acc": best_val_acc,
                },
                args.output,
            )
            print(f"Saved best checkpoint to {args.output}")

    print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
