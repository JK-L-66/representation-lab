
import numpy as np
from data import build_dataset, make_circle_image, translate


def predict_translation(original, moved):
    # 提取前景像素位置
    positions = np.argwhere(moved > 0.5)
    original_positions = np.argwhere(original > 0.5)

    if len(positions) == 0 or len(original_positions) == 0:
        raise ValueError("No foreground pixels found")

    # 计算原图和平移图的质心
    center_row, center_col = positions.mean(axis=0)
    original_center_row, original_center_col = original_positions.mean(axis=0)

    # 预测水平与垂直位移
    pred_dx = center_col - original_center_col
    pred_dy = center_row - original_center_row

    return pred_dx, pred_dy


if __name__ == "__main__":

    # Test 1: square dataset
    X_original, X_moved, shifts = build_dataset(100, seed=42)

    total_squared_error = 0.0

    for i in range(80, 100):
        original = X_original[i]
        moved = X_moved[i]

        pred_dx, pred_dy = predict_translation(original, moved)
        true_dx, true_dy = shifts[i]

        total_squared_error += (
            (true_dx - pred_dx) ** 2
            + (true_dy - pred_dy) ** 2
        )

    baseline_mse = total_squared_error / (20 * 2)
    print("Square Baseline MSE:", baseline_mse)

    # Test 2: circle
    circle = make_circle_image()
    moved_circle = translate(circle, dx=7, dy=-4)

    pred_dx, pred_dy = predict_translation(circle, moved_circle)

    print("Circle target:", (7, -4))
    print("Circle prediction:", (pred_dx, pred_dy))
