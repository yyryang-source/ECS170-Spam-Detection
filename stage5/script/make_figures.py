#!/usr/bin/env python3
"""Rebuild report figures from saved experiment artifacts; never retrain.

Run from any directory: ``python script/make_figures.py``. PNG and PDF copies
are written to report/figures/. Figure metadata records data sources, PCA
explained variance, and the exact ablation statistics shown.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import MaxNLocator
import numpy as np
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("cora", "pubmed", "citeseer")
NAMES = {"cora": "Cora", "pubmed": "Pubmed", "citeseer": "Citeseer"}
TRAIN, VALID, NEUTRAL = "#246A9A", "#C67822", "#60656D"
CLASS_COLORS = ("#2474A8", "#D18A20", "#8A529E", "#43836B", "#BA5268", "#5C65B5", "#81754B")


def save(fig, directory: Path, name: str) -> None:
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{extension}", dpi=200, facecolor="white")
    plt.close(fig)


def source_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def style(ax, *, grid="y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#949AA2")
    ax.tick_params(color="#949AA2", labelcolor="#30343A", length=3)
    ax.grid(axis=grid, color="#E5E7EA", linewidth=0.55)
    ax.set_axisbelow(True)


def architecture(directory: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 2.35))
    ax.set(xlim=(0, 7.9), ylim=(-0.25, 2.4))
    ax.axis("off")
    y, height = 0.72, 0.66
    nodes = [
        (0.08, 0.87, "$X$\n$N\\times F$", "#F3F4F6"),
        (1.56, 1.16, "GCN 1\n$F\\rightarrow16$", "#E6F0F6"),
        (3.32, 1.12, "$H$ = ReLU\n$N\\times16$", "#E8F1EB"),
        (5.04, 1.16, "GCN 2\n$16\\rightarrow C$", "#E6F0F6"),
        (6.81, 0.95, "Logits\n$N\\times C$", "#F3F4F6"),
    ]
    for x, width, label, color in nodes:
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02,rounding_size=0.06",
                                   edgecolor="#6E7A85", linewidth=.8, facecolor=color))
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=10)
    for left, right in zip(nodes[:-1], nodes[1:]):
        ax.add_patch(FancyArrowPatch((left[0] + left[1] + .03, y + height / 2),
                                    (right[0] - .03, y + height / 2), arrowstyle="-|>",
                                    mutation_scale=10, linewidth=.9, color=NEUTRAL))
    ax.text(1.25, .47, "Dropout\n$p=0.5$", ha="center", va="center", fontsize=9.5)
    ax.text(4.74, .47, "Dropout\n$p=0.5$", ha="center", va="center", fontsize=9.5)
    ax.text(3.88, .47, "Node\nembeddings", ha="center", va="center", fontsize=9.5, color="#43634E")
    ax.text(7.29, .47, "argmax\nclass label", ha="center", va="center", fontsize=9.5)
    ax.text(3.95, 2.14, r"$\hat A=\tilde D^{-1/2}(A+I)\tilde D^{-1/2}$", ha="center", va="center", fontsize=11)
    for target in (2.14, 5.62):
        ax.add_patch(FancyArrowPatch((3.95 + (-.8 if target < 3.95 else .8), 1.94),
                                    (target, 1.42), arrowstyle="-|>",
                                    mutation_scale=10, linewidth=.85, color=NEUTRAL))
    ax.text(3.95, -.12, "Each GCN layer: $\\hat A(XW)+b$  |  Loss: cross-entropy on training nodes only",
            ha="center", va="center", fontsize=9.5)
    fig.subplots_adjust(left=.01, right=.99, bottom=.06, top=.98)
    save(fig, directory, "architecture")


def learning_curves(results: Path, directory: Path, metadata: dict) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(7, 6.4), sharex=True)
    entries = []
    max_epoch, max_loss = 0, 0.
    for name in DATASETS:
        run = results / name / "baseline" / "seed_42"
        with (run / "history.csv").open(newline="") as stream:
            history = list(csv.DictReader(stream))
        data = {key: np.array([float(row[key]) for row in history]) for key in history[0]}
        metrics = json.loads((run / "metrics.json").read_text())
        entries.append((name, data, metrics))
        max_epoch = max(max_epoch, data["epoch"].max())
        max_loss = max(max_loss, data["train_loss"].max(), data["val_loss"].max())
    for row, (name, data, metrics) in enumerate(entries):
        for col, (metric, ylabel) in enumerate((("loss", "Cross-entropy"), ("accuracy", "Accuracy (%)"))):
            ax = axes[row, col]
            scale = 100 if metric == "accuracy" else 1
            ax.plot(data["epoch"], data[f"train_{metric}"] * scale, color=TRAIN, linewidth=1.45)
            ax.plot(data["epoch"], data[f"val_{metric}"] * scale, color=VALID, linewidth=1.4, linestyle=(0, (4, 1.5)))
            ax.axvline(metrics["best_epoch"], color=NEUTRAL, linewidth=.9, linestyle=":")
            ax.set(title=f"{NAMES[name]}: {metric}", ylabel=ylabel, xlim=(0, max_epoch + 3))
            ax.set_ylim((0, 102) if metric == "accuracy" else (0, max_loss * 1.05))
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.set_xticks(np.linspace(0, max_epoch, 4))
            style(ax)
            if row == 2:
                ax.set_xlabel("Training epoch")
        axes[row, 0].text(.97, .90, f"Selected epoch: {metrics['best_epoch']}", transform=axes[row, 0].transAxes,
                          ha="right", va="top", fontsize=9, color=NEUTRAL,
                          bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": 1.})
        metadata["learning_curves"][name] = {
            "source": source_path(results / name / "baseline" / "seed_42" / "history.csv"),
            "best_epoch": metrics["best_epoch"], "epochs_ran": metrics["epochs_ran"],
        }
    handles = [Line2D([], [], color=TRAIN, lw=1.5, label="Train (dropout off)"),
               Line2D([], [], color=VALID, lw=1.5, ls=(0, (4, 1.5)), label="Validation"),
               Line2D([], [], color=NEUTRAL, lw=1., ls=":", label="Lowest validation loss")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .99), ncol=3, frameon=False, fontsize=9)
    fig.subplots_adjust(left=.10, right=.985, bottom=.085, top=.91, wspace=.30, hspace=.53)
    save(fig, directory, "learning_curves")


def embeddings(results: Path, directory: Path, metadata: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.95))
    for ax, name in zip(axes, DATASETS):
        run = results / name / "baseline" / "seed_42"
        path = run / "embeddings.npz"
        data = np.load(path, allow_pickle=False)
        dataset = json.loads((run / "dataset.json").read_text())
        # Full SVD has no random sampling and fits PCA to all nodes.
        pca = PCA(n_components=2, svd_solver="full")
        projected = pca.fit_transform(data["embeddings"])
        var = 100 * pca.explained_variance_ratio_
        labels = data["labels"]
        # Draw in a fixed random order to avoid systematically covering a class.
        order = np.random.default_rng(42).permutation(len(labels))
        colors = [CLASS_COLORS[int(label)] for label in labels[order]]
        ax.scatter(projected[order, 0], projected[order, 1], c=colors,
                   s=3.2 if name != "pubmed" else 1.5, alpha=.45 if name != "pubmed" else .28,
                   linewidths=0, rasterized=True)
        ax.set(title=f"{NAMES[name]} ({len(labels):,} nodes)",
               xlabel=f"PC1 ({var[0]:.1f}%)", ylabel=f"PC2 ({var[1]:.1f}%)")
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(3))
        style(ax, grid="both")
        handles = [Line2D([], [], color=CLASS_COLORS[i], marker="o", linestyle="", markersize=4,
                          label=str(i)) for i in range(len(dataset["classes"]))]
        ax.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(.5, -.30),
                  frameon=False, fontsize=9, handletextpad=.2, columnspacing=.55, borderaxespad=0.)
        metadata["embeddings_pca"][name] = {
            "source": source_path(path),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "node_count": len(labels), "embedding_dim": int(data["embeddings"].shape[1]),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "class_ids": {str(i): label for i, label in enumerate(dataset["classes"])},
            "scope": "All graph nodes, colored by true labels; qualitative visualization only",
        }
    fig.subplots_adjust(left=.075, right=.99, bottom=.31, top=.88, wspace=.44)
    save(fig, directory, "embeddings_pca")


def ablation(results: Path, directory: Path, metadata: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.0))
    positions = np.arange(len(DATASETS))
    width = .30
    for index, (variant, label, color) in enumerate((("baseline", "Hidden width 16", TRAIN),
                                                    ("hidden32", "Hidden width 32", VALID))):
        means, deviations = [], []
        for name in DATASETS:
            values = [100 * json.loads((results / name / variant / f"seed_{seed}" / "metrics.json").read_text())["accuracy"]
                      for seed in (42, 43, 44)]
            means.append(float(np.mean(values)))
            deviations.append(float(np.std(values, ddof=1)))
            metadata["ablation"].append({"dataset": name, "variant": variant, "seeds": [42, 43, 44],
                                         "accuracy_percent_mean": means[-1],
                                         "accuracy_percent_sample_sd": deviations[-1]})
        centers = positions + (index - .5) * width
        ax.bar(centers, means, width=width, color=color, label=label, yerr=deviations,
               capsize=3, error_kw={"elinewidth": 1., "ecolor": "#343941"}, alpha=.88)
        for center, mean, sd in zip(centers, means, deviations):
            ax.text(center, mean + sd + 1.8, f"{mean:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set(xticks=positions, xticklabels=[NAMES[name] for name in DATASETS],
           ylabel="Test accuracy (%)", ylim=(0, 100))
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.18), ncol=2, frameon=False)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    style(ax)
    ax.text(.5, -.23, "Error bars: sample SD across seeds 42, 43, 44 on a fixed split", transform=ax.transAxes,
            ha="center", va="top", fontsize=9.5, color=NEUTRAL)
    fig.subplots_adjust(left=.09, right=.985, bottom=.23, top=.80)
    save(fig, directory, "ablation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "report" / "figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 10,
                         "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "axes.titlepad": 7,
                         "text.color": "#30343A", "axes.labelcolor": "#30343A"})
    metadata = {"learning_curves": {}, "embeddings_pca": {}, "ablation": []}
    architecture(args.output)
    learning_curves(args.results.resolve(), args.output, metadata)
    embeddings(args.results.resolve(), args.output, metadata)
    ablation(args.results.resolve(), args.output, metadata)
    (args.output / "figure_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote four PNG/PDF figure pairs and figure_metadata.json to {args.output}")


if __name__ == "__main__":
    main()
