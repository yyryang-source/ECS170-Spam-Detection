"""Stage 5 GCN training and reporting utilities.

Validation-only checkpoint/architecture selection; held-out test scored once.
"""
import copy
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn

from script.cnn.common import (
    classification_metrics, plot_architecture, plot_confusion,
    plot_loss, plot_train_val, save_csv, save_json
)
from script.stage5.model import CONFIGS, GCN, symmetric_normalized_adjacency


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(value):
    # PyTorch MPS sparse.mm is not uniformly supported; never construct dense A.
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "mps":
        raise ValueError("Sparse GCN does not use MPS; use --device cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested, but not available")
    return torch.device(value)


def masked_metrics(logits, labels, mask):
    targets = labels[mask].detach().cpu().numpy()
    predictions = logits[mask].argmax(1).detach().cpu().numpy()
    return classification_metrics(targets, predictions, average="macro"), targets, predictions


def train_one(graph, adjacency, config, opts, device):
    if config not in CONFIGS:
        raise ValueError(f"Unknown configuration: {config}")
    seed_everything(opts.seed)
    model = GCN(
        graph.x.shape[1], graph.num_classes, **CONFIGS[config]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=opts.learning_rate,
        weight_decay=opts.weight_decay)
    best_loss = float("inf")
    best_state = None
    stale = 0
    history = []
    x, y = graph.x.to(device), graph.y.to(device)
    train, validation = graph.train_mask.to(device), graph.val_mask.to(device)
    for epoch in range(1, opts.epochs+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_logits = model(x, adjacency)
        train_loss = nn.functional.cross_entropy(train_logits[train], y[train])
        train_loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(x, adjacency)
            val_loss = nn.functional.cross_entropy(logits[validation], y[validation])
            scores, _, _ = masked_metrics(logits, y, validation)
        history.append({
            "Epoch": epoch,
            "Train Loss": float(train_loss.item()),
            "Val Loss": float(val_loss.item()),
            "Val F1": scores["F1"]
        })
        if opts.verbose or epoch == 1 or epoch % 10 == 0:
            print(f"{config} epoch={epoch:03d} train={train_loss.item():.4f} "
                  f"val={val_loss.item():.4f} val_macro_F1={scores['F1']:.4f}", flush=True)
        if val_loss.item() < best_loss-1e-6:
            best_loss = val_loss.item()
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= opts.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scores, _, _ = masked_metrics(model(x, adjacency), y, validation)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, scores, history, params


def _plot_embedding(embedding, labels, path, seed):
    coords = embedding.detach().cpu().numpy()
    classes = labels.detach().cpu().numpy()
    if coords.shape[1] < 2:
        return
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(coords), size=min(3000, len(coords)), replace=False)
    xy = PCA(n_components=2, random_state=seed).fit_transform(coords[indices])
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(xy[:, 0], xy[:, 1], c=classes[indices], s=7,
                         cmap="tab20", alpha=0.65, linewidth=0)
    ax.set_title("Trained GCN node embeddings (PCA; labels for visualization only)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(scatter, ax=ax, label="Node class")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run_experiments(graph, output_dir, configs, opts):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not configs or len(set(configs)) != len(configs):
        raise ValueError("Select one or more unique GCN configurations")
    device = select_device(opts.device)
    adjacency = symmetric_normalized_adjacency(
        graph.edge_index, len(graph.y), device=device)
    evaluations, trained, histories = [], {}, {}
    for config in configs:
        model, validation, history, params = train_one(
            graph, adjacency, config, opts, device)
        trained[config] = model
        histories[config] = history
        evaluations.append({
            "Model": config, "Parameters": params, **validation
        })
        save_csv(output / f"{config}_history.csv", history)
    save_csv(output / "ablation_validation.csv", evaluations)
    winner = max(evaluations, key=lambda r: r["F1"])["Model"]
    selected = trained[winner]
    selected.eval()
    with torch.no_grad():
        logits, embedding = selected(
            graph.x.to(device), adjacency, return_embedding=True)
        # Exactly one test evaluation after all validation-based decisions.
        test_metrics, truth, pred = masked_metrics(
            logits, graph.y.to(device), graph.test_mask.to(device))
    test_row = {"Model": winner, **test_metrics}
    save_csv(output / "final_test_metrics.csv", [test_row])

    plot_train_val(histories[winner], output / "best_loss_curve.png",
                   f"GCN {winner} training convergence")
    plot_loss(histories, output / "ablation_val_loss.png",
              "GCN architecture ablation: validation loss")
    plot_confusion(truth, pred, [str(i) for i in range(graph.num_classes)],
                   output / "confusion_matrix.png", "Held-out node classification")
    params = CONFIGS[winner]
    plot_architecture(
        [f"Features {graph.x.shape[1]}", "Sparse normalized A",
         f"{params['depth']} GCN layers", f"Hidden {params['hidden']}",
         f"Dropout {params['dropout']}", f"{graph.num_classes} labels"],
        output / "architecture.png", f"Stage 5 {winner} GCN")
    _plot_embedding(embedding, graph.y, output / "node_embedding_pca.png", opts.seed)
    torch.save({
        "state_dict": {k: v.cpu() for k, v in selected.state_dict().items()},
        "config": winner, "configuration": params,
        "input_dim": graph.x.shape[1], "classes": graph.num_classes
    }, output / "best_gcn.pt")
    summary = {
        "task": "transductive node classification",
        "dataset": graph.provenance,
        "quick_smoke_test_not_for_report": bool(opts.quick),
        "device": str(device),
        "seed": opts.seed,
        "epochs_max": opts.epochs,
        "early_stopping_patience": opts.patience,
        "learning_rate": opts.learning_rate,
        "weight_decay": opts.weight_decay,
        "normalized_adjacency_nonzeros": adjacency._nnz(),
        "checkpoint_selection": "minimum validation cross entropy",
        "architecture_selection": "highest validation macro F1",
        "test_used_for_selection": False,
        "configs": {k: CONFIGS[k] for k in configs},
        "validation": evaluations,
        "selected_model": winner,
        "final_test": test_row
    }
    save_json(output / "experiment_summary.json", summary)
    print("=== VALIDATION ABLATIONS ===")
    for row in evaluations:
        print(row)
    print("BEST:", winner, "TEST:", test_metrics, "OUT:", str(output))
    return summary
