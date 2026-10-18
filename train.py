"""Legacy fixed-start, single-image experiment; not the main paired model."""

import torch
from data import build_dataset
from model import Encoder


def main():
    torch.manual_seed(42)

    # 生成100组数据
    X_original, X_moved, shifts = build_dataset(
        n_samples=100,
        seed=42
    )

    # 划分训练集和测试集
    X_train = X_moved[:80]
    X_test = X_moved[80:]

    y_train = shifts[:80]
    y_test = shifts[80:]

    print("Train images:", X_train.shape)
    print("Test images:", X_test.shape)
    print("Train labels:", y_train.shape)
    print("Test labels:", y_test.shape)

    X_train = torch.from_numpy(X_train).unsqueeze(1)
    X_test = torch.from_numpy(X_test).unsqueeze(1)

    y_train = torch.from_numpy(y_train).float()
    y_test = torch.from_numpy(y_test).float()

    print("PyTorch train:", X_train.shape)
    print("PyTorch test:", X_test.shape)
    print("Target dtype:", y_train.dtype)
    batch_size = 16

    x_batch = X_train[:batch_size]
    y_batch = y_train[:batch_size]

    print("Batch images:", x_batch.shape)
    print("Batch labels:", y_batch.shape)

    # 模型和优化器在循环外创建
    encoder = Encoder()
    head = torch.nn.Linear(32, 2)

    optimizer = torch.optim.SGD(
        list(encoder.parameters()) + list(head.parameters()),
        lr=0.002
    )


    encoder.train()
    head.train()
    for epoch in range(1, 31):
        total_loss = 0.0
        indices = torch.randperm(len(X_train))
        for start in range(0, len(X_train), batch_size):
            end = start + batch_size

            batch_indices = indices[start:end]

            x_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]


            # 清空旧梯度
            optimizer.zero_grad()

            z = encoder(x_batch)
            prediction = head(z)

            loss = torch.nn.functional.mse_loss(prediction, y_batch)
            loss.backward()

            # 更新参数
            optimizer.step()

            total_loss += loss.item() * len(x_batch)


        average_loss = total_loss / len(X_train)


        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {average_loss:.4f}")

    # Evaluation: 使用训练结束后的固定模型参数

    encoder.eval()
    head.eval()

    with torch.no_grad():
        # 训练集预测
        train_prediction = head(encoder(X_train))

        train_loss = torch.nn.functional.mse_loss(
            train_prediction, y_train
        )

        # 测试集预测
        test_prediction = head(encoder(X_test))

        test_loss = torch.nn.functional.mse_loss(
            test_prediction, y_test
        )

    print("Final Train MSE:", train_loss.item())
    print("Final Test MSE:", test_loss.item())

    print("Train prediction shape:", train_prediction.shape)
    print("Test prediction shape:", test_prediction.shape)



if __name__ == "__main__":
    main()
