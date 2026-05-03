from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
from src.custom_dataset import CustomDigitDataset


def get_data_loaders(batch_size=64):

    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.rotate(-90).transpose(0)),

        transforms.RandomRotation(20),
        transforms.RandomAffine(0, translate=(0.2, 0.2), scale=(0.8, 1.2)),

        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    emnist_train = datasets.EMNIST(
        root="data/",
        split="digits",
        train=True,
        download=True,
        transform=transform
    )

    emnist_test = datasets.EMNIST(
        root="data/",
        split="digits",
        train=False,
        download=True,
        transform=transform
    )

    custom_dataset = CustomDigitDataset("data/custom")

    # 🔥 Balance datasets
    combined_train = ConcatDataset([
        emnist_train,
        custom_dataset,
        custom_dataset,
        custom_dataset
    ])

    train_loader = DataLoader(combined_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(emnist_test, batch_size=batch_size)

    return train_loader, test_loader