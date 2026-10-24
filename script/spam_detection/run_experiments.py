import joblib
import copy
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

SEED = 42
EPOCHS = 30
BATCH_SIZE = 64
PATIENCE = 5
DEVICE = torch.device("cpu")

CONFIGS = [
    ("A", 64, 32, 0.3),
    ("B", 128, 64, 0.3),
    ("C", 256, 128, 0.3),
    ("D", 128, 64, 0.0),
    ("E", 128, 64, 0.5),
]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data():
    path = ROOT / "data" / "enron_spam_data.csv"
    df = pd.read_csv(path)

    df["Subject"] = df["Subject"].fillna("")
    df["Message"] = df["Message"].fillna("")

    df["text"] = (
        df["Subject"] + " " + df["Message"]
    ).str.strip()

    labels = (
        df["Spam/Ham"]
        .astype(str)
        .str.lower()
        .str.strip()
    )
    df["label"] = labels.map({"spam": 1, "ham": 0})

    df = df.dropna(subset=["label"])
    df = df[df["text"] != ""]
    df = df.drop_duplicates(subset=["text"])

    # 80% train+validation, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"].astype(int),
        test_size=0.2, random_state=SEED,
        stratify=df["label"]
    )

    # 90% of training portion for fitting,
    # 10% for validation
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train,
        test_size=0.1, random_state=SEED,
        stratify=y_train
    )

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    # Fit vocabulary ONLY on the fitting set
    A = vectorizer.fit_transform(X_fit)
    B = vectorizer.transform(X_val)
    C = vectorizer.transform(X_test)

    print("Total:", len(df))
    print("Fit:", A.shape)
    print("Validation:", B.shape)
    print("Test:", C.shape)

    return (
        A, B, C,
        y_fit.to_numpy(),
        y_val.to_numpy(),
        y_test.to_numpy()
    )


class SpamMLP(nn.Module):
    def __init__(self, input_dim, h1, h2, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def batches(X, y, shuffle=False):
    indices = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, len(indices), BATCH_SIZE):
        ids = indices[start:start + BATCH_SIZE]
        xb = torch.from_numpy(
            X[ids].toarray().astype(np.float32)
        ).to(DEVICE)
        yb = torch.tensor(
            y[ids], dtype=torch.float32,
            device=DEVICE
        )
        yield xb, yb


def predict(model, X):
    model.eval()
    probabilities = []

    with torch.no_grad():
        for xb, _ in batches(
            X, np.zeros(X.shape[0])
        ):
            probs = torch.sigmoid(model(xb))
            probabilities.extend(probs.cpu().numpy())

    return (np.array(probabilities) >= 0.5).astype(int)


def metrics(y, predictions):
    return {
        "Accuracy": accuracy_score(y, predictions),
        "Precision": precision_score(
            y, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y, predictions, zero_division=0
        ),
        "F1": f1_score(
            y, predictions, zero_division=0
        ),
    }


def evaluate_loss(model, X, y, criterion):
    model.eval()
    total = 0.0

    with torch.no_grad():
        for xb, yb in batches(X, y):
            loss = criterion(model(xb), yb)
            total += loss.item() * len(yb)

    return total / len(y)


def train_model(name, h1, h2, dropout,
                X_fit, X_val, y_fit, y_val):

    set_seed()
    model = SpamMLP(
        X_fit.shape[1], h1, h2, dropout
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001
    )
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    best_state = None
    patience = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0

        for xb, yb in batches(
            X_fit, y_fit, shuffle=True
        ):
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(yb)

        train_loss = total / len(y_fit)
        val_loss = evaluate_loss(
            model, X_val, y_val, criterion
        )

        history.append({
            "Epoch": epoch,
            "Train Loss": train_loss,
            "Val Loss": val_loss
        })

        print(
            f"{name} Epoch {epoch:02d} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(
                model.state_dict()
            )
            patience = 0
        else:
            patience += 1

        if patience >= PATIENCE:
            break

    model.load_state_dict(best_state)
    scores = metrics(y_val, predict(model, X_val))

    return model, scores, history


def save_csv(filename, rows):
    if not rows:
        return

    with open(RESULTS / filename, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    set_seed()
    X_fit, X_val, X_test, y_fit, y_val, y_test = (
        load_data()
    )

    ablation_results = []
    trained_models = {}
    histories = {}

    for name, h1, h2, dropout in CONFIGS:
        print(f"\n===== Experiment {name} =====")

        model, scores, history = train_model(
            name, h1, h2, dropout,
            X_fit, X_val, y_fit, y_val
        )

        trained_models[name] = model
        histories[name] = history

        ablation_results.append({
            "Experiment": name,
            "Hidden1": h1,
            "Hidden2": h2,
            "Dropout": dropout,
            **scores
        })

    save_csv("ablation_results.csv", ablation_results)

    # Select using validation F1 only
    best = max(
        ablation_results,
        key=lambda row: row["F1"]
    )
    best_name = best["Experiment"]
    best_model = trained_models[best_name]

    print("\nBest validation configuration:", best_name)

    # Evaluate the chosen model on the test set once
    final_results = []

    mlp_scores = metrics(
        y_test, predict(best_model, X_test)
    )
    final_results.append({
        "Model": f"MLP ({best_name})",
        **mlp_scores
    })

    # Baselines train on same fitting subset
    baselines = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=SEED
        ),
        "Linear SVM": LinearSVC(
            max_iter=5000, random_state=SEED
        ),
    }

    for name, baseline in baselines.items():
        baseline.fit(X_fit, y_fit)
        scores = metrics(
            y_test, baseline.predict(X_test)
        )
        final_results.append({
            "Model": name, **scores
        })

    save_csv("final_model_comparison.csv", final_results)

    # Save chosen MLP weights
    torch.save(
        best_model.state_dict(),
        RESULTS / "best_spam_mlp.pt"
    )

    # Save training history
    history = histories[best_name]
    save_csv("best_mlp_history.csv", history)

    # Training convergence plot
    plt.figure(figsize=(8, 5))
    plt.plot(
        [r["Epoch"] for r in history],
        [r["Train Loss"] for r in history],
        label="Training Loss"
    )
    plt.plot(
        [r["Epoch"] for r in history],
        [r["Val Loss"] for r in history],
        label="Validation Loss"
    )
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title(f"MLP Convergence - Experiment {best_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        RESULTS / "final_loss_curve.png", dpi=200
    )
    plt.close()

    print("\n===== ABLATION RESULTS =====")
    print(pd.DataFrame(ablation_results).to_string(index=False))

    print("\n===== FINAL TEST RESULTS =====")
    print(pd.DataFrame(final_results).to_string(index=False))

    print("\nSaved to:", RESULTS)


if __name__ == "__main__":
    main()
