"""Generate binary-shape image pairs for 2D translation experiments.

Images use (row, column) indexing, while shift labels use (dx, dy):
positive dx moves right and positive dy moves down.
"""

import numpy as np


def translate(image, dx, dy):
    """Translate a 2D image with zero padding (without wraparound)."""
    height, width = image.shape
    shifted = np.zeros_like(image)

    if abs(dx) >= width or abs(dy) >= height:
        return shifted

    src_row = max(0, -dy)
    src_col = max(0, -dx)
    dst_row = max(0, dy)
    dst_col = max(0, dx)
    copy_height = height - abs(dy)
    copy_width = width - abs(dx)

    shifted[
        dst_row:dst_row + copy_height,
        dst_col:dst_col + copy_width,
    ] = image[
        src_row:src_row + copy_height,
        src_col:src_col + copy_width,
    ]
    return shifted


def make_square_image(size=64, square_size=16, top=24, left=24):
    """Create a binary image containing one filled square."""
    image = np.zeros((size, size), dtype=np.float32)
    image[top:top + square_size, left:left + square_size] = 1.0
    return image


def make_translation_example(dx, dy, size=64, square_size=16, top=24, left=24):
    """Return an original square, translated square, and displacement labels."""
    image = make_square_image(size=size, square_size=square_size, top=top, left=left)
    moved = translate(image, dx, dy)
    return image, moved, dx, dy


def build_dataset(n_samples=100, seed=42):
    """Legacy fixed-position squares with unique sampled displacements."""
    if not 1 <= n_samples <= 625:
        raise ValueError("n_samples must be between 1 and 625")

    rng = np.random.default_rng(seed)
    examples = []
    while len(examples) < n_samples:
        dx = rng.integers(-12, 13)
        dy = rng.integers(-12, 13)
        if (dx, dy) not in examples:
            examples.append((dx, dy))

    dataset = [make_translation_example(dx, dy) for dx, dy in examples]
    originals = np.array([row[0] for row in dataset], dtype=np.float32)
    moved = np.array([row[1] for row in dataset], dtype=np.float32)
    labels = np.array([row[2:] for row in dataset], dtype=np.int64)
    return originals, moved, labels


def make_circle_image(size=64, radius=8, center_row=32, center_col=32):
    """Create a filled circle using a Euclidean-distance pixel mask."""
    rows, cols = np.indices((size, size))
    mask = ((center_row - rows) ** 2 + (center_col - cols) ** 2 <= radius ** 2)
    return mask.astype(np.float32)


def make_random_square_pair(rng):
    """Sample a square's position and a translation using the supplied RNG."""
    top = rng.integers(12, 37)
    left = rng.integers(12, 37)
    dx = rng.integers(-12, 13)
    dy = rng.integers(-12, 13)
    return make_translation_example(dx, dy, top=top, left=left)


def build_paired_dataset(n_samples=1000, seed=42):
    """Generate square image pairs for the main shared-encoder experiment."""
    rng = np.random.default_rng(seed)
    originals = []
    moved = []
    labels = []

    for _ in range(n_samples):
        original, transformed, dx, dy = make_random_square_pair(rng)
        originals.append(original)
        moved.append(transformed)
        labels.append((dx, dy))

    return (
        np.array(originals, dtype=np.float32),
        np.array(moved, dtype=np.float32),
        np.array(labels, dtype=np.int64),
    )


def make_random_circle_pair(rng):
    """Sample an unclipped circle and a translation from the supplied RNG."""
    radius = 8
    center_row = rng.integers(20, 44)
    center_col = rng.integers(20, 44)
    dx = rng.integers(-12, 13)
    dy = rng.integers(-12, 13)

    original = make_circle_image(
        radius=radius,
        center_row=center_row,
        center_col=center_col,
    )
    moved = translate(original, dx, dy)
    return original, moved, dx, dy


if __name__ == "__main__":
    square_originals, square_moved, square_labels = build_paired_dataset()
    assert square_originals.shape == square_moved.shape == (1000, 64, 64)
    assert square_labels.shape == (1000, 2)
    assert np.all(square_originals.sum(axis=(1, 2)) == 256)
    assert np.array_equal(square_originals.sum(axis=(1, 2)),
                          square_moved.sum(axis=(1, 2)))

    rng_a = np.random.default_rng(2026)
    rng_b = np.random.default_rng(2026)
    for _ in range(100):
        sample_a = make_random_circle_pair(rng_a)
        sample_b = make_random_circle_pair(rng_b)
        assert all(np.array_equal(a, b) for a, b in zip(sample_a, sample_b))
        assert sample_a[0].sum() == sample_a[1].sum() == 197

    print("All data checks passed (shapes, uncropped objects, reproducibility).")
