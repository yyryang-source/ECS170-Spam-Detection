"""Shared training/evaluation helpers for ECS 170 CNN experiments."""
import csv
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

SEED = 42
ROOT = Path(__file__).resolve().parents[2]


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def classification_metrics(targets, predictions, average="binary"):
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, average=average, zero_division=0
    )
    return {
        "Accuracy": float(accuracy_score(targets, predictions)),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
    }


def save_csv(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def plot_loss(histories, path, title):
    """Plot one line per model for validation loss and train/val for chosen model separately."""
    plt.figure(figsize=(7, 4.5))
    for name, history in histories.items():
        plt.plot([r["Epoch"] for r in history],
                 [r["Val Loss"] for r in history], label=f"{name} validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_train_val(history, path, title):
    plt.figure(figsize=(7, 4.5))
    for key in ("Train Loss", "Val Loss"):
        plt.plot([r["Epoch"] for r in history], [r[key] for r in history], label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_confusion(targets, predictions, labels, path, title):
    cm = confusion_matrix(targets, predictions, labels=list(range(len(labels))))
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(len(labels)), labels, rotation=50, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_architecture(steps, path, title):
    """Save a small publication-ready architecture schematic for the report."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(min(12, 1.85 * len(steps)), 2.6))
    ax.set_xlim(0, len(steps) * 1.85)
    ax.set_ylim(0, 2.5)
    ax.axis("off")
    for i, label in enumerate(steps):
        x = i * 1.85 + 0.05
        ax.add_patch(FancyBboxPatch((x, 0.78), 1.6, 0.88,
                                   boxstyle="round,pad=0.05", linewidth=1,
                                   edgecolor="#476987", facecolor="#EAF1F7"))
        ax.text(x + 0.8, 1.22, label, ha="center", va="center", fontsize=9)
        if i < len(steps)-1:
            ax.add_patch(FancyArrowPatch((x+1.65, 1.22), (x+1.83, 1.22),
                                         arrowstyle="-|>", mutation_scale=9,
                                         color="#476987"))
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
