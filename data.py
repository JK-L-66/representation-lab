import dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


# image = np.zeros((5, 5), dtype=np.float32)
#
# image[2, 2] = 1.0
# shifted = np.zeros_like(image)
# shifted[:-2,:] = image[2:,:]
# print(shifted)
# shifted[1:,2:]=image[:-1,:-2]
# print(shifted)

def translate(image, dx, dy):
    height, width = image.shape
    shifted = np.zeros_like(image)

    if abs(dx) >= width or abs(dy) >= height:
        return shifted

    # 源区域起始位置
    src_row = max(0, -dy)
    src_col = max(-dx,0)

    # 目标区域起始位置
    dst_row = max(0, dy)
    dst_col =max(dx,0)

    # 能够复制的行数和列数
    copy_height = height - abs(dy)
    copy_width = width - abs(dx)

    shifted[
        dst_row:dst_row + copy_height,
        dst_col:dst_col + copy_width
    ] = image[
        src_row:src_row + copy_height,
        src_col:src_col + copy_width
    ]

    return shifted
#


# 创建 64×64 图像
image = np.zeros((64, 64), dtype=np.float32)

# 创建 16×16 白色方块
image[24:40, 24:40] = 1.0

# 平移图像
moved = translate(image, dx=10, dy=-6)

#
# # 测试 1：左移 2，下移 1
# result = translate(image, -2, 1)
#
# assert result[2, 1] == 1.0
# assert result.sum() == 1.0
#
# #测试 2：不移动
# result = translate(image, 0, 0)
#
# assert np.array_equal(result, image)
# assert np.array_equal(dataset[2][0], dataset[2][1])
#
# # 测试 3：完全移出边界
# result = translate(image, 5, 0)
#
# assert np.all(result == 0)
#
# print("All tests passed!")

#使用已经创建的 image
def make_square_image(size=64, square_size=16, top=24, left=24):
    image = np.zeros((size, size), dtype=np.float32)
    image[top:top + square_size, left:left + square_size] = 1.0
    return image
def make_translation_example(dx, dy, size=64, square_size=16, top=24, left=24):
    image = make_square_image(size=size, square_size=square_size, top=top, left=left)
    moved = translate(image, dx, dy)
    return image, moved, dx, dy

# image, moved, dx, dy = make_translation_example(10, -6)
rng = np.random.default_rng(42)

examples = []

while len(examples) < 100:
    dx = rng.integers(-12, 13)
    dy = rng.integers(-12, 13)

    # 只保存没有出现过的位移组合
    if  (dx,dy) not in examples:
        examples.append((dx, dy))

print("样本数量:", len(examples))
print("不同位移组合数量:", len(set(examples)))

# print(examples)
dataset = []

for dx, dy in examples:
    sample=make_translation_example(dx, dy)
    dataset.append(sample)
X_original = []
X_moved = []
shifts = []

for original, moved, dx, dy in dataset:
    X_original.append(original)
    X_moved.append(moved)
    shifts.append((dx, dy))

X_original = np.array(X_original, dtype=np.float32)
X_moved = np.array(X_moved, dtype=np.float32)
shifts = np.array(shifts, dtype=np.int64)

print("X_original shape:", X_original.shape)
print("X_moved shape:", X_moved.shape)
print("shifts shape:", shifts.shape)
X_original_train = X_original[:80]
X_original_test = X_original[80:]

X_moved_train = X_moved[:80]
X_moved_test = X_moved[80:]

shifts_train = shifts[:80]
shifts_test = shifts[80:]
print("Train:", X_original_train.shape,
      X_moved_train.shape, shifts_train.shape)

print("Test:", X_original_test.shape,
      X_moved_test.shape, shifts_test.shape)
train_pairs = set(map(tuple, shifts_train))
test_pairs = set(map(tuple, shifts_test))

assert train_pairs.isdisjoint(test_pairs)
# print("第一组 shift:", shifts[0])
# print("样本总数:", len(dataset))
# print("第一组 dx:", dataset[0][2])
# print("第一组 dy:", dataset[0][3])
# print("第一组原图形状:", dataset[0][0].shape)
# assert np.array_equal(dataset[2][0], dataset[2][1])
# fig, axes = plt.subplots(4, 2, figsize=(8, 12))
# axes[2, 1].imshow(dataset[2][1], cmap="gray", vmin=0, vmax=1)
# plt.show()
# fig, axes = plt.subplots(4, 2, figsize=(8, 12))
# for i in range(len(dataset)):
#     original, moved, dx, dy = dataset[i]
#
#     axes[i, 0].imshow(original, cmap="gray", vmin=0, vmax=1)
#     axes[i, 1].imshow(moved, cmap="gray", vmin=0, vmax=1)
#     axes[i,0].set_title("Original")
#     axes[i, 1].set_title(f"dx={dx}, dy={dy}")
# plt.tight_layout()
# plt.show()