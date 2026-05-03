import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 🔥 AUTO CALCULATE FEATURE SIZE
        self._to_linear = None
        self._get_conv_output()

        self.fc = nn.Sequential(
            nn.Linear(self._to_linear, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )

    def _get_conv_output(self):
        with torch.no_grad():
            x = torch.randn(1, 1, 28, 28)
            x = self.conv(x)
            self._to_linear = x.view(1, -1).shape[1]

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)