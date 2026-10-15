
import torch
import torch.nn as nn
from data import make_translation_example


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        # 在初始化阶段创建网络层
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)

        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1024, 32)

    def forward(self, x):
        # 在前向传播中调用已创建的网络层
        h1 = self.relu(self.conv1(x))
        h2 = self.relu(self.conv2(h1))

        h3 = self.pool(h2)
        a = self.flatten(h3)

        z = self.fc(a)
        return z

if __name__ == "__main__":
    encoder = Encoder()

    x = torch.zeros((1, 1, 64, 64))
    z = encoder(x)

    print("Input:", x.shape)
    print("Latent:", z.shape)
    print("Parameters:", sum(p.numel() for p in encoder.parameters()))
    z1 = encoder(x)
    z2 = encoder(x)

    print("Maximum difference:", (z1 - z2).abs().max().item())