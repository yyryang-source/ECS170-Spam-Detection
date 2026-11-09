"""Detailed CNN architecture plot for ECS170 Stage 3 reports."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def plot_architecture(config, dataset, classes, channels, path, params):
    from script.stage3.models import Stage3CNN
    widths = (32, 64) if config == "baseline" else (32, 64, 128)
    kernel = 5 if config == "kernel5" else 3
    spatial = {"mnist": "28x28", "cifar10": "32x32", "orl": "64x64"}[dataset]
    blocks = [f"Block {i+1}\nConv {kernel}x{kernel} x2\nBN, ReLU, MaxPool\n{width} channels"
              for i, width in enumerate(widths)]
    labels = [f"Input\n{channels} x {spatial}"] + blocks + [
        "Adaptive Global\nAverage Pool",
        "FC 128 + ReLU\nDropout",
        f"Linear logits\n{len(classes)} classes"
    ]
    n = len(labels)
    fig, ax = plt.subplots(figsize=(min(16, n*2.05), 3.2))
    ax.set_xlim(-0.12, n*2.02)
    ax.set_ylim(0, 2)
    ax.axis("off")
    for i, label in enumerate(labels):
        x = i*2.02 + 0.05
        is_block = 1 <= i <= len(widths)
        fill = "#e6f5ef" if is_block else "#eaf1fb"
        edge = "#338465" if is_block else "#4168a3"
        ax.add_patch(FancyBboxPatch((x, .48), 1.85, 1.02,
                                   boxstyle="round,pad=0.035",
                                   linewidth=1.35, facecolor=fill, edgecolor=edge))
        ax.text(x+.925, .99, label, ha="center", va="center",
                fontsize=8.3, color="#18283d")
        if i < n-1:
            ax.add_patch(FancyArrowPatch((x+1.9, .99), (x+2.02, .99),
                                         arrowstyle="-|>", mutation_scale=12,
                                         linewidth=1.1, color="#35485f"))
    residual = "with residual shortcuts" if config != "baseline" else "no residual shortcuts"
    fig.suptitle(
        f"ECS170 {dataset.upper()} — {config} ({residual}) | "
        f"kernel={kernel}, trainable parameters={params:,}",
        fontsize=10.5, fontweight="bold"
    )
    fig.text(.5, .075, "Two convolutions per block; logits use CrossEntropyLoss "
             "(softmax applied implicitly).", ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0,.13,1,.92))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
