import numpy as np


def translate(image, dx, dy):
    """
    Translate a 2D image with zero padding.

    dx > 0: move right
    dy > 0: move down
    """
    height, width = image.shape
    shifted = np.zeros_like(image)

    if abs(dx) >= width or abs(dy) >= height:
        return shifted

    # 源区域起始位置
    src_row = max(0, -dy)
    src_col = max(-dx, 0)

    # 目标区域起始位置
    dst_row = max(0, dy)
    dst_col = max(dx, 0)

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


def make_square_image(size=64, square_size=16, top=24, left=24):
    image = np.zeros((size, size), dtype=np.float32)
    image[top:top + square_size, left:left + square_size] = 1.0
    return image


def make_translation_example(dx, dy, size=64, square_size=16, top=24, left=24):
    image = make_square_image(size=size, square_size=square_size, top=top, left=left)
    moved = translate(image, dx, dy)
    return image, moved, dx, dy


if __name__ == "__main__":
    test_image = np.zeros((5, 5), dtype=np.float32)
    test_image[2, 2] = 1.0

    result = translate(test_image, dx=-1, dy=2)

    assert result[4, 1] == 1.0
    assert result.sum() == 1.0
    assert np.array_equal(translate(test_image, 0, 0), test_image)
    assert np.all(translate(test_image, 5, 0) == 0)

    print("Translation tests passed!")

    # 后面继续放随机数据集生成代码
    rng = np.random.default_rng(42)

    examples = []

    while len(examples) < 100:
        dx = rng.integers(-12, 13)
        dy = rng.integers(-12, 13)

        # 只保存没有出现过的位移组合
        if (dx, dy) not in examples:
            examples.append((dx, dy))

    print("样本数量:", len(examples))
    print("不同位移组合数量:", len(set(examples)))

    dataset = []

    for dx, dy in examples:
        sample = make_translation_example(dx, dy)
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
