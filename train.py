
import torch
from data import make_translation_example
from model import Encoder

# 1. 创建一个已知位移的样本
original, moved, dx, dy = make_translation_example(10, -6)

# 2. 准备神经网络输入
x = torch.from_numpy(moved).unsqueeze(0).unsqueeze(0)

# 3. 正确答案
target = torch.tensor([[dx, dy]], dtype=torch.float32)

# 4. 创建 Encoder 和预测头
encoder = Encoder()
head = torch.nn.Linear(32, 2)

# 5. 计算预测值
z = encoder(x)
prediction = head(z)

# 6. 计算预测误差
loss = torch.nn.functional.mse_loss(prediction, target)

print("Latent shape:", z.shape)
print("Prediction:", prediction.detach())
print("Target:", target)
print("Loss:", loss.item())
loss.backward()

print("Head bias gradient:", head.bias.grad)

# 暂时只更新预测头的 bias
optimizer = torch.optim.SGD(
    list(encoder.parameters()) + list(head.parameters()),
    lr=0.001
)

weight_before = encoder.conv1.weight.detach().clone()

# 根据刚才计算的梯度，更新参数
optimizer.step()

weight_after = encoder.conv1.weight.detach()

difference = (weight_after - weight_before).abs().max().item()

print("Conv1 maximum weight change:", difference)
print("Bias after:", head.bias.detach())

# 使用更新后的参数重新预测
with torch.no_grad():
    new_prediction = head(encoder(x))
    new_loss = torch.nn.functional.mse_loss(
        new_prediction, target
    )

print("Loss before:", loss.item())
print("Loss after:", new_loss.item())
print("New prediction:", new_prediction)
