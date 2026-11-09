"""ECS 170 Stage 3: CNN recognition on MNIST, ORL and CIFAR-10.

Runs a baseline and controlled CNN ablations, exports report-ready evidence.
No test set is used for model/configuration selection.
"""
import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from script.cnn.common import (classification_metrics, get_device,
                               plot_architecture, plot_confusion, plot_loss,
                               plot_train_val, save_csv, save_json,
                               seed_everything)
from script.stage3.datasets import load_stage3_data
from script.stage3.models import Stage3CNN

ALL_CONFIGS = ("baseline", "residual", "kernel5")


def evaluate(model, loader, device, criterion):
    model.eval()
    running_loss, count = 0., 0
    targets, predictions = [], []
    with torch.no_grad():
        for images, y in loader:
            images, y = images.to(device), y.to(device)
            logits = model(images)
            running_loss += criterion(logits, y).item()*len(y)
            count += len(y)
            targets.extend(y.cpu().tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    if count == 0:
        raise ValueError("Empty evaluation split")
    scores = classification_metrics(targets, predictions, average="macro")
    return running_loss/count, scores, targets, predictions


def training_loader(dataset, batch_size, seed):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=0, generator=torch.Generator().manual_seed(seed))


def train_one(config, train_data, val_loader, in_ch, nclasses, device, opts):
    seed_everything(opts.seed)
    model = Stage3CNN(in_ch, nclasses, config, opts.dropout).to(device)
    train_loader = training_loader(train_data, opts.batch_size, opts.seed)
    criterion = nn.CrossEntropyLoss(label_smoothing=opts.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opts.learning_rate,
                                  weight_decay=opts.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2)
    best_loss, best_state, wait, history = float("inf"), None, 0, []
    for epoch in range(1, opts.epochs+1):
        model.train()
        total, train_sum = 0, 0.
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += len(targets)
            train_sum += loss.item()*len(targets)
        val_loss, val_score, _, _ = evaluate(model, val_loader, device, criterion)
        scheduler.step(val_loss)
        row = {"Epoch": epoch, "Train Loss": train_sum/total,
               "Val Loss": val_loss, "Val F1": val_score["F1"],
               "LR": optimizer.param_groups[0]["lr"]}
        history.append(row)
        print(f"{config}: epoch={epoch:02d} train={row['Train Loss']:.4f} "
              f"val={val_loss:.4f} val_macro_F1={row['Val F1']:.4f}", flush=True)
        if val_loss < best_loss-1e-5:
            best_loss, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= opts.patience:
                print(f"{config}: early stopping", flush=True)
                break
    model.load_state_dict(best_state)
    _, val_scores, _, _ = evaluate(model, val_loader, device, criterion)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, val_scores, history, param_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mnist", "orl", "cifar10"), required=True)
    parser.add_argument("--source", choices=("standard", "local"), default="standard",
                        help="Use local instructor files for report-faithful experiments")
    parser.add_argument("--data-root", default=None,
                        help="Local NPZ or class directories, otherwise torchvision download dir")
    parser.add_argument("--configs", default="baseline,residual,kernel5",
                        help="Comma-separated CNN configurations")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test only: scores must NOT be used in a report")
    opts = parser.parse_args()
    config_names = [x.strip() for x in opts.configs.split(",") if x.strip()]
    if not config_names or len(set(config_names)) != len(config_names) or any(
            x not in ALL_CONFIGS for x in config_names):
        parser.error("Specify distinct configurations among baseline,residual,kernel5")
    if min(opts.epochs, opts.batch_size, opts.patience) < 1:
        parser.error("epochs/batch-size/patience must be positive")
    if opts.learning_rate <= 0 or opts.weight_decay < 0 or not 0 <= opts.dropout < 1:
        parser.error("Invalid optimizer/dropout settings")
    if not 0 <= opts.label_smoothing < 1:
        parser.error("label-smoothing must be in [0,1)")
    if opts.dataset == "orl" and opts.source == "standard":
        parser.error("ORL has no standard torchvision loader; specify --source local --data-root PATH")
    if opts.source == "local" and not opts.data_root:
        parser.error("--source local requires --data-root pointing to instructor dataset files")
    root = Path(opts.data_root) if opts.data_root else ROOT/"data"/"images"
    device = get_device(opts.device)
    seed_everything(opts.seed)
    (train_data, val_data, test_data), class_names, provenance = load_stage3_data(
        opts.dataset, opts.source, root, opts.seed, opts.quick)
    if opts.quick:
        opts.epochs = min(opts.epochs, 2)
    out = ROOT/"results"/"stage3"/opts.dataset/("quick" if opts.quick else "full")
    out.mkdir(parents=True, exist_ok=True)
    val_loader = DataLoader(val_data, batch_size=opts.batch_size, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=opts.batch_size, num_workers=0)
    evaluations, trained, histories = [], {}, {}
    for config in config_names:
        print(f"\nDATASET={opts.dataset} CONFIG={config} DEVICE={device}", flush=True)
        model, scores, hist, params = train_one(
            config, train_data, val_loader, provenance["channels"],
            len(class_names), device, opts)
        trained[config] = model
        histories[config] = hist
        evaluations.append({"Model": config, "Parameters": params, **scores})
        save_csv(out/f"{config}_loss_history.csv", hist)
    save_csv(out/"ablation_validation.csv", evaluations)
    winner = max(evaluations, key=lambda r: r["F1"])["Model"]
    criterion = nn.CrossEntropyLoss(label_smoothing=opts.label_smoothing)
    test_loss, test_scores, y_true, y_pred = evaluate(
        trained[winner], test_loader, device, criterion)
    save_csv(out/"final_test_metrics.csv",
             [{"Dataset": opts.dataset, "Model": winner, "Test Loss": test_loss, **test_scores}])
    plot_train_val(histories[winner], out/"best_loss_curve.png",
                   f"{opts.dataset.upper()} {winner} - Train/Validation Loss")
    plot_loss(histories, out/"ablation_val_loss.png", f"{opts.dataset.upper()} CNN Ablations")
    plot_confusion(y_true, y_pred, class_names, out/"confusion_matrix.png",
                   f"{opts.dataset.upper()} Held-out Test Predictions")
    plot_architecture([
        "Input image", "Conv+BN+ReLU", "Residual blocks" if winner != "baseline" else "Conv blocks",
        "MaxPool", "Global Pool", "FC 128", f"{len(class_names)} classes"
    ], out/"architecture.png", f"Stage 3 - {opts.dataset.upper()} ({winner})")
    torch.save({
        "model": trained[winner].cpu().state_dict(),
        "config": winner, "classes": class_names,
        "channels": provenance["channels"], "dropout": opts.dropout
    }, out/"best_model.pt")
    save_json(out/"experiment_settings.json", {
        "author": "Yourui Yang", "course": "ECS170 Fall 2026", "dataset": opts.dataset,
        "quick_smoke_test_not_for_report": opts.quick,
        "device": str(device), "provenance": provenance, "model_configs": config_names,
        "optimizer": "AdamW", "loss": "CrossEntropyLoss",
        "seed": opts.seed, "epochs": opts.epochs, "patience": opts.patience,
        "batch_size": opts.batch_size, "learning_rate": opts.learning_rate,
        "weight_decay": opts.weight_decay, "label_smoothing": opts.label_smoothing,
        "dropout": opts.dropout,
        "winner_selection": "validation macro F1",
        "selected_model": winner, "test_evaluated_after_selection": True,
        "final_test_scores": test_scores,
    })
    print("\n=== VALIDATION ABLATIONS ===")
    for row in evaluations:
        print(row)
    print("\nBEST:", winner, "TEST:", test_scores, "OUT:", out, flush=True)


if __name__ == "__main__":
    main()
