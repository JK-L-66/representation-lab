"""Create presentation-ready figures from the trained paired-CNN checkpoint.

Requires the existing data.py, evaluate.py, model.py and checkpoints/paired_square.pt.
Run: python plot_results.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save images without opening a GUI window.
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from data import make_random_circle_pair
from evaluate import predict_pairs, to_images
from model import Encoder

ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT / "checkpoints" / "paired_square.pt"
FIGURES_PATH = ROOT / "figures"
CIRCLE_TEST_SEED = 2026
CIRCLE_TEST_SIZE = 100

# Consistent, readable palette; no external fonts or style packages required.
INK = "#17243B"
MUTED = "#596579"
GRID = "#E5EAF1"
BLUE = "#2563EB"
TEAL = "#099C89"
ORANGE = "#EA7A39"
PAGE = "#FFFFFF"


def _clean_axes(ax):
    ax.set_facecolor(PAGE)
    for name in ("top", "right"):
        ax.spines[name].set_visible(False)
    for name in ("bottom", "left"):
        ax.spines[name].set_color("#CAD3DF")
    ax.tick_params(colors=MUTED, labelsize=10, length=0, pad=7)
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def plot_training_curve(history):
    """Training epoch-average loss and validation end-of-epoch loss."""
    if not history:
        raise ValueError("Checkpoint lacks training history. Run train_paired.py once.")

    epochs = np.array([row["epoch"] for row in history])
    train = np.array([row["train_mse"] for row in history])
    val = np.array([row["validation_mse"] for row in history])

    fig, ax = plt.subplots(figsize=(9.6, 5.5), facecolor=PAGE)
    fig.subplots_adjust(left=0.105, right=0.90, bottom=0.20, top=0.72)
    _clean_axes(ax)

    ax.plot(epochs, train, color=BLUE, marker="o", markersize=8,
            linewidth=2.7, label="Training · epoch average", zorder=4)
    ax.plot(epochs, val, color=TEAL, marker="s", markersize=7.5,
            linewidth=2.7, label="Validation · epoch end", zorder=4)
    ax.set_yscale("log")
    ax.set_xticks(epochs)
    ax.set_xlabel("Epoch", color=INK, labelpad=10, fontsize=11)
    ax.set_ylabel("Mean squared error  ·  log scale", color=INK, labelpad=12, fontsize=11)
    ax.legend(loc="upper right", frameon=False, fontsize=10, labelcolor=INK)

    # Use header/subtitle instead of crowding the plotting area.
    fig.text(0.105, 0.925, "TRAINING DYNAMICS", fontsize=9, weight="bold", color=TEAL)
    fig.text(0.105, 0.84, "Learning to predict 2D translations", fontsize=19,
             weight="bold", color=INK)
    fig.text(0.105, 0.775, "Shared encoder  ·  800 training / 100 validation pairs  ·  5 epochs",
             fontsize=10, color=MUTED)
    fig.text(0.105, 0.07,
             "Note: training values average batches during each epoch; validation is measured after the epoch.",
             fontsize=9, color=MUTED)
    path = FIGURES_PATH / "training_curve.png"
    fig.savefig(path, dpi=200, facecolor=PAGE)
    plt.close(fig)
    print("Saved:", path)


def plot_circle_zero_shot(encoder, head):
    """Prediction vs. target, split by displacement component to avoid clutter."""
    rng = np.random.default_rng(CIRCLE_TEST_SEED)
    samples = [make_random_circle_pair(rng) for _ in range(CIRCLE_TEST_SIZE)]
    originals = np.stack([item[0] for item in samples])
    moved = np.stack([item[1] for item in samples])
    targets = np.asarray([item[2:] for item in samples], dtype=np.float32)
    predictions = predict_pairs(
        encoder, head, to_images(originals), to_images(moved)
    ).numpy()
    mse = float(np.mean((predictions - targets) ** 2))
    scale = float(np.sum(predictions * targets) / np.sum(targets ** 2))

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 6.1), facecolor=PAGE)
    fig.subplots_adjust(left=0.085, right=0.955, bottom=0.20,
                        top=0.73, wspace=0.21)
    for ax, column, color, label in zip(
        axes, [0, 1], [BLUE, ORANGE], ["Horizontal shift  ·  dx", "Vertical shift  ·  dy"]
    ):
        _clean_axes(ax)
        ax.scatter(targets[:, column], predictions[:, column],
                   s=40, alpha=0.72, color=color, edgecolors="white",
                   linewidths=0.5, zorder=3)
        ax.plot([-13, 13], [-13, 13], color="#9098A6", linewidth=1.5,
                linestyle="--", label="Ideal", zorder=2)
        ax.plot([-13, 13], [-13 * scale, 13 * scale], color=TEAL,
                linewidth=2, alpha=0.85, linestyle=":",
                label="Pooled scale", zorder=2)
        ax.set_xlim(-13, 13)
        ax.set_ylim(-13, 13)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(label, fontsize=11, weight="bold", color=INK, pad=11, loc="left")
        ax.set_xlabel("True displacement (px)", fontsize=10, color=INK, labelpad=8)
        ax.set_ylabel("Predicted displacement (px)", fontsize=10, color=INK, labelpad=8)

    fig.text(0.085, 0.945, "OUT-OF-DISTRIBUTION GENERALIZATION", fontsize=9,
             weight="bold", color=TEAL)
    fig.text(0.085, 0.866, "Unseen shape, familiar translation task", fontsize=19,
             weight="bold", color=INK)
    fig.text(0.085, 0.806,
             f"100 circle pairs  ·  MSE {mse:.3f} px²  ·  pooled displacement scale {scale:.3f}",
             fontsize=10.5, color=MUTED)
    fig.text(0.085, 0.081, "– –  Ideal prediction (y = x)       · · ·  Pooled scale through origin",
             color=MUTED, fontsize=9)
    fig.text(0.085, 0.045,
             "Model was trained only on squares. Pooled scale is descriptive, not a calibrated model.",
             color=MUTED, fontsize=9)

    path = FIGURES_PATH / "circle_zero_shot_scatter.png"
    fig.savefig(path, dpi=200, facecolor=PAGE)
    plt.close(fig)
    print("Saved:", path)
    print(f"Circle zero-shot MSE: {mse:.6f}")
    print(f"Circle displacement scale: {scale:.6f}")


def main():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint missing: {CHECKPOINT_PATH}. Run train_paired.py first."
        )
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)

    plot_training_curve(checkpoint.get("history"))

    encoder = Encoder()
    head = nn.Linear(32, 2)
    encoder.load_state_dict(checkpoint["encoder"])
    head.load_state_dict(checkpoint["head"])
    encoder.eval()
    head.eval()
    plot_circle_zero_shot(encoder, head)


if __name__ == "__main__":
    main()
