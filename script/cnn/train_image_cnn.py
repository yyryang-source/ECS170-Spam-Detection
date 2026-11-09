"""2D CNN image classification: Fashion-MNIST (default) or CIFAR-10."""
import argparse
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from common import (ROOT, SEED, classification_metrics, get_device, plot_architecture,
                    plot_confusion, plot_loss, plot_train_val, save_csv, save_json, seed_everything)

CONFIGS = [("A_two_blocks", (32, 64), 0.25),
           ("B_three_blocks", (32, 64, 128), 0.25),
           ("C_more_dropout", (32, 64, 128), 0.50)]


class ImageCNN(nn.Module):
    def __init__(self, in_channels=1, classes=10, widths=(32, 64), dropout=0.25):
        super().__init__()
        layers = []
        incoming = in_channels
        for width in widths:
            layers += [nn.Conv2d(incoming, width, kernel_size=3, padding=1),
                       nn.ReLU(), nn.MaxPool2d(kernel_size=2)]
            incoming = width
        self.net = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                 nn.Dropout(dropout), nn.Linear(incoming, classes))

    def forward(self, x):
        return self.net(x)


def make_data(dataset, quick=False):
    from torchvision import datasets, transforms
    if dataset == "fashion-mnist":
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.2860,), (0.3530,))])
        source, channels = datasets.FashionMNIST, 1
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                             (0.2470, 0.2435, 0.2616))])
        source, channels = datasets.CIFAR10, 3
    root = str(ROOT/"data"/"images")
    training = source(root=root, train=True, transform=transform, download=True)
    test = source(root=root, train=False, transform=transform, download=True)
    val_size = round(len(training) * 0.1)
    fit, val = random_split(training, [len(training)-val_size, val_size],
                            generator=torch.Generator().manual_seed(SEED))
    if quick:
        fit = Subset(fit, range(min(len(fit), 512)))
        val = Subset(val, range(min(len(val), 128)))
        test = Subset(test, range(min(len(test), 128)))
    return (fit, val, test), training.classes, channels


def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, total, targets, predictions = 0., 0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item() * len(y)
            total += len(y)
            predictions.extend(logits.argmax(1).cpu().tolist())
            targets.extend(y.cpu().tolist())
    return total_loss/total, classification_metrics(targets, predictions, "macro"), targets, predictions


def fit(name, widths, dropout, channels, classes, train, val, device, epochs, patience):
    seed_everything()
    model = ImageCNN(channels, classes, widths, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history, stale, best_loss, best_state = [], 0, float("inf"), None
    for epoch in range(1, epochs+1):
        model.train()
        total_loss, total = 0., 0
        for x, y in train:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()*len(y)
            total += len(y)
        val_loss, _, _, _ = evaluate(model, val, device)
        history.append({"Epoch": epoch, "Train Loss": total_loss/total, "Val Loss": val_loss})
        print(f"{name}: epoch={epoch} train={total_loss/total:.4f} validation={val_loss:.4f}", flush=True)
        if val_loss < best_loss-1e-6:
            best_loss, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model, evaluate(model, val, device)[1], history


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("fashion-mnist", "cifar10"), default="fashion-mnist")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--device", default="auto")
    p.add_argument("--quick", action="store_true", help="Smoke test, not report scores")
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.patience) < 1:
        p.error("epochs, batch-size, patience >= 1")
    seed_everything()
    device = get_device(args.device)
    output = ROOT/"results"/"image_cnn"/args.dataset/("quick" if args.quick else "full")
    output.mkdir(parents=True, exist_ok=True)
    (fit_data, val_data, test_data), classes, channels = make_data(args.dataset, args.quick)
    if args.quick:
        args.epochs = min(args.epochs, 2)
    train = DataLoader(fit_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val = DataLoader(val_data, batch_size=args.batch_size, num_workers=0)
    test = DataLoader(test_data, batch_size=args.batch_size, num_workers=0)
    rows, models, histories = [], {}, {}
    for name, widths, dropout in CONFIGS:
        model, metrics, history = fit(name, widths, dropout, channels, len(classes),
                                      train, val, device, args.epochs, args.patience)
        rows.append({"Model": name, "Channels": str(widths), "Dropout": dropout, **metrics})
        models[name], histories[name] = model, history
        save_csv(output/f"{name}_history.csv", history)
    save_csv(output/"ablation_validation.csv", rows)
    winner = max(rows, key=lambda r: r["F1"])["Model"]
    _, test_metrics, truth, pred = evaluate(models[winner], test, device)
    save_csv(output/"final_test_metrics.csv", [{"Model": winner, **test_metrics}])
    widths, dropout = next((w, d) for n, w, d in CONFIGS if n == winner)
    plot_architecture([f"Image ({channels} ch.)", f"Conv2D blocks {widths}",
                       "Global AvgPool", "Dropout + Linear", "10 classes"],
                      output/"model_architecture.png", "2D CNN - Image Classification")
    plot_train_val(histories[winner], output/"best_loss_curve.png", f"Image CNN - {winner}")
    plot_loss(histories, output/"ablation_val_loss.png", "Image CNN Validation Loss")
    plot_confusion(truth, pred, classes, output/"confusion_matrix.png",
                   "Image CNN Test Confusion Matrix")
    torch.save({"state_dict": models[winner].cpu().state_dict(), "classes": classes,
                "channels": channels, "widths": widths, "dropout": dropout},
               output/"best_image_cnn.pt")
    save_json(output/"experiment_metadata.json", {
        "dataset": args.dataset, "quick_smoke_test": args.quick, "seed": SEED,
        "split": {"train": len(fit_data), "val": len(val_data), "test": len(test_data)},
        "selection": "validation macro F1", "test_used_for_selection": False,
        "best_config": winner, "test_metrics": test_metrics})
    print("Best:", winner, "Final test:", test_metrics, "Output:", output)


if __name__ == "__main__":
    main()
