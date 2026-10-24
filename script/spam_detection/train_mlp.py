
import copy
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from preprocess import prepare_data


# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Training parameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MAX_EPOCHS = 30
PATIENCE = 5


class SpamMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * len(y)
            total_samples += len(y)

    return total_loss / total_samples


def main():
    X_train, X_test, y_train, y_test = prepare_data()

    # Validation set: 10% of training data
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.1,
        random_state=42,
        stratify=y_train,
    )

    # Convert sparse TF-IDF matrices to float32
    X_fit = X_fit.toarray().astype(np.float32)
    X_val = X_val.toarray().astype(np.float32)
    X_test = X_test.toarray().astype(np.float32)

    y_fit = np.asarray(y_fit, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int64)

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_fit),
            torch.from_numpy(y_fit),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_val),
            torch.from_numpy(y_val),
        ),
        batch_size=BATCH_SIZE,
    )

    # Start with CPU for compatibility
    device = torch.device("cpu")
    print("\nTraining device:", device)

    model = SpamMLP(input_dim=X_fit.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    history = []
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        total_loss = 0
        total_samples = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            total_samples += len(y)

        train_loss = total_loss / total_samples
        val_loss = evaluate_loss(
            model, val_loader, criterion, device
        )

        history.append((epoch, train_loss, val_loss))

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    model.load_state_dict(best_state)

    # Save trained model
    torch.save(
        model.state_dict(),
        RESULTS_DIR / "spam_mlp.pt",
    )

    # Plot convergence
    epochs, train_losses, val_losses = zip(*history)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("MLP Training Convergence")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "mlp_loss.png",
        dpi=200,
    )
    plt.close()

    with open(
        RESULTS_DIR / "mlp_history.csv", "w", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        writer.writerows(history)

    # Evaluate on held-out test set
    model.eval()

    with torch.no_grad():
        X_test_tensor = torch.from_numpy(X_test).to(device)
        logits = model(X_test_tensor)
        probabilities = torch.sigmoid(logits).cpu().numpy()

    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, zero_division=0
        ),
    }

    print("\n========== MLP TEST RESULTS ==========")
    for name, value in metrics.items():
        print(f"{name.capitalize()}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    with open(
        RESULTS_DIR / "mlp_metrics.csv", "w", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerows(metrics.items())

    print("\nResults saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
