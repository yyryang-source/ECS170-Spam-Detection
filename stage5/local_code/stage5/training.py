"""Full-batch transductive training with validation-only checkpoint selection."""
import copy
import csv
import json
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from .model import GCN, GraphOperator


@dataclass
class Config:
    hidden_dim: int = 16
    depth: int = 2
    dropout: float = 0.5
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    max_epochs: int = 300
    patience: int = 50
    seed: int = 42
    device: str = "cpu"
    threads: int = 4


def evaluate_predictions(truth, prediction, classes):
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, prediction, labels=list(range(len(classes))), average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "precision_macro": float(precision), "recall_macro": float(recall), "f1_macro": float(f1),
        "confusion_matrix": confusion_matrix(truth, prediction, labels=list(range(len(classes)))).tolist(),
        "per_class": classification_report(truth, prediction, labels=list(range(len(classes))),
                                            target_names=classes, output_dict=True, zero_division=0),
    }


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def train_graph(data, config, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if config.max_epochs < 1 or config.patience < 1 or config.threads < 1:
        raise ValueError("epochs, patience and threads must be positive")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.threads)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    # CPU is the reproducibility reference. MPS edge reductions may vary slightly.
    torch.use_deterministic_algorithms(device.type == "cpu")
    x = data["graph"]["X"].to(device)
    y = data["graph"]["y"].to(device)
    adj = GraphOperator.from_scipy(data["graph"]["utility"]["A_scipy"], device=device)
    splits = {k: v.to(device) for k, v in data["train_test_val"].items()}
    train_idx, val_idx, test_idx = (splits[k] for k in ("idx_train", "idx_val", "idx_test"))
    model = GCN(x.shape[1], config.hidden_dim, len(data["metadata"]["classes"]), config.dropout, config.depth).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_loss, best_epoch, best_state = float("inf"), 0, None
    history = []
    synchronize(device)
    start = time.perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, adj)
        objective = F.cross_entropy(logits[train_idx], y[train_idx])
        if not torch.isfinite(objective):
            raise RuntimeError(f"Nonfinite training loss at epoch {epoch}")
        objective.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(x, adj)
            train_loss = F.cross_entropy(logits[train_idx], y[train_idx]).item()
            val_loss = F.cross_entropy(logits[val_idx], y[val_idx]).item()
            train_acc = (logits[train_idx].argmax(1) == y[train_idx]).float().mean().item()
            val_acc = (logits[val_idx].argmax(1) == y[val_idx]).float().mean().item()
        history.append({"epoch": epoch, "train_objective_dropout": objective.item(),
                        "train_loss": train_loss, "val_loss": val_loss,
                        "train_accuracy": train_acc, "val_accuracy": val_acc})
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        if epoch == 1 or epoch % 50 == 0:
            print(f"{data['metadata']['dataset']} seed={config.seed} h={config.hidden_dim} epoch={epoch} train={train_loss:.4f} val={val_loss:.4f}", flush=True)
        if epoch - best_epoch >= config.patience:
            break
    synchronize(device)
    elapsed = time.perf_counter() - start
    model.load_state_dict(best_state)
    model.eval()
    # The held-out test labels are first used here, after training/model selection.
    with torch.no_grad():
        logits = model(x, adj)
        test_loss = F.cross_entropy(logits[test_idx], y[test_idx]).item()
        predictions = logits.argmax(1).cpu().numpy()
        embeddings = model.embed(x, adj).cpu().numpy()
    truth = data["graph"]["y"].numpy()
    test_numpy = data["train_test_val"]["idx_test"].numpy()
    metrics = evaluate_predictions(truth[test_numpy], predictions[test_numpy], data["metadata"]["classes"])
    metrics.update({"test_loss": test_loss, "best_epoch": best_epoch, "best_val_loss": best_loss,
                    "epochs_ran": len(history), "training_seconds": elapsed,
                    "train_accuracy_at_best": history[best_epoch - 1]["train_accuracy"],
                    "validation_accuracy_at_best": history[best_epoch - 1]["val_accuracy"],
                    "config": asdict(config), "dataset": data["metadata"]["dataset"],
                    "parameter_count": sum(v.numel() for v in model.parameters()),
                    "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                    "numpy": np.__version__, "platform": platform.platform(),
                                    "machine": platform.machine(), "device": str(device)}})
    write_json(out / "metrics.json", metrics)
    write_json(out / "dataset.json", data["metadata"])
    write_json(out / "splits.json", {k: v.tolist() for k, v in data["train_test_val"].items()})
    with (out / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    with (out / "test_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_index", "node_id", "true_label", "predicted_label"])
        classes = data["metadata"]["classes"]
        for i in test_numpy:
            writer.writerow([i, data["graph"]["node_ids"][i], classes[truth[i]], classes[predictions[i]]])
    np.savez_compressed(out / "embeddings.npz", embeddings=embeddings, labels=truth,
                        predictions=predictions, node_ids=np.asarray(data["graph"]["node_ids"]))
    torch.save({"model_state": best_state, "config": asdict(config),
                "in_features": x.shape[1], "classes": data["metadata"]["classes"],
                "dataset": data["metadata"], "best_epoch": best_epoch}, out / "model.pt")
    print(f"DONE {data['metadata']['dataset']} seed={config.seed} test_acc={metrics['accuracy']:.4f} macro_f1={metrics['f1_macro']:.4f} best_epoch={best_epoch}", flush=True)
    return metrics
