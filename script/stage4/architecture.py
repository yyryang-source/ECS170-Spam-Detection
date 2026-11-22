"""Task-specific, reproducible diagrams of the Stage 4 recurrent models."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def plot_recurrent_architecture(task, kind, embedding_dim, hidden_dim, layers,
                                dropout, vocab_size, destination):
    if task not in ("classification", "generation"):
        raise ValueError("task must be classification or generation")
    if kind not in ("rnn", "lstm", "gru"):
        raise ValueError("kind must be rnn, lstm, or gru")
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    head = ("Last non-pad state → binary logits" if task == "classification"
            else "Each time step → vocabulary logits")
    boxes = [
        ("Token IDs", "pad / unk / bos / eos"),
        ("Embedding", f"trainable, {embedding_dim}d"),
        (kind.upper(), f"{layers} layer(s), {hidden_dim} hidden"),
        ("Dropout", f"p = {dropout:.2f}"),
        ("Task head", head),
    ]
    fig, ax = plt.subplots(figsize=(11.5, 2.8))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 2.8)
    ax.axis("off")
    for i, (name, info) in enumerate(boxes):
        x = 0.18 + 2.3 * i
        ax.add_patch(FancyBboxPatch(
            (x, 0.7), 2.05, 1.3,
            boxstyle="round,pad=0.15,rounding_size=0.14",
            linewidth=1.2, edgecolor="#35465a", facecolor="#eaf1f8"))
        ax.text(x + 1.025, 1.65, name, ha="center", va="center",
                fontsize=12, fontweight="bold", color="#1d354a")
        ax.text(x + 1.025, 1.16, info, ha="center", va="center",
                fontsize=7.9, color="#35465a", wrap=True)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + 2.31, 1.35), xytext=(x + 2.07, 1.35),
                        arrowprops={"arrowstyle": "->", "lw": 1.4})
    title = "IMDb sentiment classification" if task == "classification" else "Joke next-token generation"
    ax.set_title(f"{title} — {kind.upper()} — vocabulary {vocab_size:,}",
                 fontsize=12, pad=9)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
