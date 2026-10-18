"""Train a shared-encoder CNN to predict translations from image pairs."""
from pathlib import Path

import torch
from torch import nn

from data import build_paired_dataset
from model import Encoder

SEED = 42
N_SAMPLES = 1000
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
CHECKPOINT_PATH = Path(__file__).resolve().parent / "checkpoints" / "paired_square.pt"


def main():
    torch.manual_seed(SEED)

    original_np, moved_np, shifts_np = build_paired_dataset(
        n_samples=N_SAMPLES, seed=SEED
    )
    originals = torch.from_numpy(original_np).unsqueeze(1)
    moved = torch.from_numpy(moved_np).unsqueeze(1)
    labels = torch.from_numpy(shifts_np).float()

    encoder = Encoder()
    head = nn.Linear(32, 2)

    # Use the same random split order as the original experiment.
    indices = torch.randperm(N_SAMPLES)
    train_idx = indices[:800]
    val_idx = indices[800:900]
    test_idx = indices[900:]
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()),
        lr=LEARNING_RATE,
    )

    history = []

    for epoch in range(EPOCHS):
        encoder.train()
        head.train()
        permutation = torch.randperm(len(train_idx))
        total_loss = 0.0

        for start in range(0, len(train_idx), BATCH_SIZE):
            batch_idx = train_idx[permutation[start:start + BATCH_SIZE]]
            original_batch = originals[batch_idx]
            moved_batch = moved[batch_idx]
            label_batch = labels[batch_idx]

            optimizer.zero_grad()
            delta_z = encoder(moved_batch) - encoder(original_batch)
            prediction = head(delta_z)
            loss = nn.functional.mse_loss(prediction, label_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_idx)

        encoder.eval()
        head.eval()
        with torch.no_grad():
            val_delta = encoder(moved[val_idx]) - encoder(originals[val_idx])
            val_prediction = head(val_delta)
            val_mse = nn.functional.mse_loss(val_prediction, labels[val_idx])

        train_mse = total_loss / len(train_idx)
        validation_mse = val_mse.item()
        history.append({
            "epoch": epoch + 1,
            "train_mse": train_mse,
            "validation_mse": validation_mse,
        })

        print(
            f"Epoch {epoch + 1}: "
            f"Train MSE = {train_mse:.6f}, "
            f"Validation MSE = {validation_mse:.6f}"
        )

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "head": head.state_dict(),
            "seed": SEED,
            "epochs": EPOCHS,
            "n_samples": N_SAMPLES,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
            "history": history,
        },
        CHECKPOINT_PATH,
    )
    print("Model saved:", CHECKPOINT_PATH)


if __name__ == "__main__":
    main()
