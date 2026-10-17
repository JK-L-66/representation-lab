from logging import exception

import numpy as np

from data import make_translation_example
from data import build_dataset

X_original, X_moved, shifts = build_dataset(100, seed=42)
def predict_translation(original, moved):
    positions = np.argwhere(moved > 0.5)
    original_positions = np.argwhere(original > 0.5)

    if len(positions) == 0 or len(original_positions) == 0:
        raise ValueError("No foreground pixels found")
    # 3. 分别计算两个图像的质心
    center = positions.mean(axis=0)
    center_row, center_col = center
    original_center = original_positions.mean(axis=0)
    original_center_row, original_center_col = original_center
    # 4. 计算水平位移和垂直位移
    pred_dx = center_col - original_center_col
    pred_dy = center_row - original_center_row
    return pred_dx, pred_dy
total_squared_error = 0.0
for i in range(80, 100):
    original = X_original[i]
    moved = X_moved[i]

    pred_dx, pred_dy = predict_translation(original, moved)
    true_dx, true_dy = shifts[i]
    total_squared_error += (true_dx-pred_dx) ** 2+(true_dy-pred_dy) ** 2

    print("Target:", shifts[i])
    print("Prediction:", pred_dx, pred_dy)
baseline_mse = total_squared_error / (20 * 2)
print("Baseline Test MSE:", baseline_mse)

