"""Evaluate square/circle translation and latent stability without retraining."""
from pathlib import Path

import numpy as np
import torch
from torch import nn

from baseline import predict_translation
from data import (
    build_paired_dataset,
    make_random_circle_pair,
    make_square_image,
    translate,
)
from model import Encoder

CHECKPOINT_PATH = Path(__file__).resolve().parent / "checkpoints" / "paired_square.pt"


def to_images(images):
    return torch.from_numpy(np.asarray(images, dtype=np.float32)).unsqueeze(1)


def predict_pairs(encoder, head, originals, moved):
    with torch.no_grad():
        delta = encoder(moved) - encoder(originals)
        return head(delta)


def centroid_mse(originals, moved, labels):
    errors = []
    for original, transformed, target in zip(originals, moved, labels):
        predicted = np.array(predict_translation(original, transformed))
        errors.append((predicted - target) ** 2)
    return float(np.mean(errors))


def evaluate_translations(encoder, head, checkpoint):
    original_np, moved_np, shifts_np = build_paired_dataset(
        n_samples=checkpoint["n_samples"], seed=checkpoint["seed"]
    )
    test_idx = checkpoint["test_idx"].cpu().numpy()
    square_originals = original_np[test_idx]
    square_moved = moved_np[test_idx]
    square_labels = shifts_np[test_idx]

    square_prediction = predict_pairs(
        encoder, head, to_images(square_originals), to_images(square_moved)
    )
    square_y = torch.from_numpy(square_labels).float()
    square_mse = nn.functional.mse_loss(square_prediction, square_y).item()
    square_baseline = centroid_mse(square_originals, square_moved, square_labels)

    rng = np.random.default_rng(2026)
    circle_samples = [make_random_circle_pair(rng) for _ in range(100)]
    circle_originals = np.stack([sample[0] for sample in circle_samples])
    circle_moved = np.stack([sample[1] for sample in circle_samples])
    circle_labels = np.array([sample[2:] for sample in circle_samples], dtype=np.float32)
    circle_y = torch.from_numpy(circle_labels)
    circle_prediction = predict_pairs(
        encoder, head, to_images(circle_originals), to_images(circle_moved)
    )
    circle_mse = nn.functional.mse_loss(circle_prediction, circle_y).item()
    circle_baseline = centroid_mse(circle_originals, circle_moved, circle_labels)
    circle_errors = circle_prediction - circle_y
    scale = ((circle_prediction * circle_y).sum() / (circle_y**2).sum()).item()

    print(f"Square Test MSE: {square_mse:.6f}")
    print(f"Square Centroid MSE: {square_baseline:.6f}")
    print(f"Circle Zero-shot MSE: {circle_mse:.6f}")
    print(f"Circle Centroid MSE: {circle_baseline:.6f}")
    print("Circle signed error (dx, dy):", circle_errors.mean(dim=0).tolist())
    print("Circle MAE (dx, dy):", circle_errors.abs().mean(dim=0).tolist())
    print(f"Circle displacement scale: {scale:.6f}")
    print("Circle examples:")
    for i in range(5):
        print(f"  target={circle_y[i].tolist()}, predicted={circle_prediction[i].tolist()}")


def evaluate_latent_stability(encoder):
    rng = np.random.default_rng(2027)
    originals = [
        make_square_image(top=int(rng.integers(12, 37)),
                          left=int(rng.integers(12, 37)))
        for _ in range(20)
    ]
    original_tensor = to_images(originals)

    with torch.no_grad():
        z_original = encoder(original_tensor)
        deltas = []
        for dx, dy in ((7, -4), (-7, 4)):
            moved = [translate(original, dx, dy) for original in originals]
            deltas.append(encoder(to_images(moved)) - z_original)

    mean_first = deltas[0].mean(dim=0)
    mean_second = deltas[1].mean(dim=0)
    within_first = torch.linalg.vector_norm(deltas[0] - mean_first, dim=1).mean()
    within_second = torch.linalg.vector_norm(deltas[1] - mean_second, dim=1).mean()
    between = torch.linalg.vector_norm(mean_first - mean_second)

    print(f"Within distance (7, -4): {within_first.item():.6f}")
    print(f"Within distance (-7, 4): {within_second.item():.6f}")
    print(f"Between-shift distance: {between.item():.6f}")


def main():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    encoder = Encoder()
    head = nn.Linear(32, 2)
    encoder.load_state_dict(checkpoint["encoder"])
    head.load_state_dict(checkpoint["head"])
    encoder.eval()
    head.eval()

    evaluate_translations(encoder, head, checkpoint)
    evaluate_latent_stability(encoder)


if __name__ == "__main__":
    main()


