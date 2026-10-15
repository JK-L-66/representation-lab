
import torch

# 假设有 80 张 64×64 灰度图
images = torch.zeros((80, 64, 64))

print("Before:", images.shape)

# 在索引 1 处增加一个维度
images = images.unsqueeze(1)

print("After:", images.shape)
