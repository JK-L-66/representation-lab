"""Shared CNN encoder used to predict image translations from latent differences."""

import torch
from torch import nn


class Encoder(nn.Module):
    """Map a (batch, 1, 64, 64) image tensor to a 32-dimensional latent vector."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 8 * 8, 32)

    def forward(self, x):
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        h = self.pool(h)
        return self.fc(self.flatten(h))


if __name__ == "__main__":
    encoder = Encoder()
    x = torch.zeros((1, 1, 64, 64))
    with torch.no_grad():
        z = encoder(x)
    print("Input:", x.shape)
    print("Latent:", z.shape)
    print("Parameters:", sum(parameter.numel() for parameter in encoder.parameters()))
