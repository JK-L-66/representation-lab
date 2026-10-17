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


def build_dataset(n_samples=100, seed=42):
    if not 1 <= n_samples <= 625:
        raise ValueError("n_samples must be between 1 and 625")

    rng = np.random.default_rng(seed)


    examples = []

    while len(examples) < n_samples:
        dx = rng.integers(-12, 13)
        dy = rng.integers(-12, 13)

        if (dx, dy) not in examples:
            examples.append((dx, dy))


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



    return X_original, X_moved, shifts


if __name__ == "__main__":
    X_original, X_moved, shifts = build_dataset()

    print("Original:", X_original.shape)
    print("Moved:", X_moved.shape)
    print("Shifts:", shifts.shape)

    assert X_original.shape == (100, 64, 64)
    assert X_moved.shape == (100, 64, 64)
    assert shifts.shape == (100, 2)

    assert len(set(map(tuple, shifts))) == 100

    print("Dataset tests passed!")