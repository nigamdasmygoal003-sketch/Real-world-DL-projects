import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class CustomDigitDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []

        for label in range(10):
            folder = os.path.join(root_dir, str(label))
            if not os.path.exists(folder):
                continue

            for file in os.listdir(folder):
                path = os.path.join(folder, file)
                self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def preprocess(self, img):
        # Resize safely
        img = cv2.resize(img, (28, 28))

        # Normalize
        img = img / 255.0

        return img

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        # 🔴 Handle broken images
        if img is None:
            img = np.zeros((28, 28), dtype=np.uint8)

        img = self.preprocess(img)

        img = torch.tensor(img).float().unsqueeze(0)

        return img, label