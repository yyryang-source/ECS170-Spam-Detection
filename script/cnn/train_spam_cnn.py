"""1D CNN spam detector; Enron CSV schema: Subject, Message, Spam/Ham."""
import argparse
import copy
import re
from collections import Counter
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from common import (ROOT, SEED, classification_metrics, get_device, plot_architecture,
                    plot_confusion, plot_loss, plot_train_val, save_csv, save_json, seed_everything)

CONFIGS = [("A_small", (32,), 0.25), ("B_deep", (32, 64), 0.25),
           ("C_dropout", (32, 64), 0.50)]
TOKENS = re.compile(r"(?u)\b\w+\b")


def tokenize(text):
    return TOKENS.findall(text.lower())


def split_data(path):
    df = pd.read_csv(path)
    required = {"Subject", "Message", "Spam/Ham"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(df.columns))}")
    df["text"] = (df["Subject"].fillna("").astype(str) + " " +
                  df["Message"].fillna("").astype(str)).str.strip()
    df["label"] = df["Spam/Ham"].astype(str).str.lower().str.strip().map({"ham": 0, "spam": 1})
    df = df.dropna(subset=["label"])
    df = df[df["text"].ne("")].drop_duplicates(subset=["text"])
    train_x, test_x, train_y, test_y = train_test_split(
        df["text"].tolist(), df["label"].astype(int).tolist(),
        test_size=0.2, random_state=SEED, stratify=df["label"])
    fit_x, val_x, fit_y, val_y = train_test_split(
        train_x, train_y, test_size=0.1, random_state=SEED, stratify=train_y)
    return (fit_x, fit_y), (val_x, val_y), (test_x, test_y)


def vocabulary(texts, max_words):
    count = Counter(word for text in texts for word in tokenize(text))
    return {word: i+2 for i, (word, _) in enumerate(
        sorted(count.items(), key=lambda item: (-item[1], item[0]))[:max_words-2])}


def encode(texts, vocab, length):
    values = np.zeros((len(texts), length), dtype=np.int64)
    for i, text in enumerate(texts):
        ids = [vocab.get(word, 1) for word in tokenize(text)[:length]]
        values[i, :len(ids)] = ids
    return torch.from_numpy(values)


def loader(part, vocab, length, batch, shuffle=False):
    x, y = part
    return DataLoader(TensorDataset(encode(x, vocab, length), torch.tensor(y, dtype=torch.float32)),
                      batch_size=batch, shuffle=shuffle)


class SpamCNN(nn.Module):
    def __init__(self, vocab_size, channels=(32, 64), dropout=0.25):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64, padding_idx=0)
        layers = []
        incoming = 64
        for outgoing in channels:
            layers.extend([nn.Conv1d(incoming, outgoing, 3, padding=1), nn.ReLU()])
            incoming = outgoing
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.AdaptiveMaxPool1d(1), nn.Flatten(),
                                        nn.Dropout(dropout), nn.Linear(incoming, 1))

    def forward(self, tokens):
        x = self.embedding(tokens).transpose(1, 2)
        return self.classifier(self.features(x)).squeeze(1)


def evaluate(model, batches, device):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    loss_sum, count, y_true, y_pred = 0., 0, [], []
    with torch.no_grad():
        for x, y in batches:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += criterion(logits, y).item() * len(y)
            count += len(y)
            y_true.extend(y.cpu().int().tolist())
            y_pred.extend((torch.sigmoid(logits) >= 0.5).cpu().int().tolist())
    return loss_sum/count, classification_metrics(y_true, y_pred), y_true, y_pred


def fit(name, channels, dropout, vocab_size, train, val, device, epochs, patience):
    seed_everything()
    model = SpamCNN(vocab_size, channels, dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_state, best_loss, stale, history = None, float("inf"), 0, []
    for epoch in range(1, epochs+1):
        model.train()
        loss_sum, count = 0., 0
        for x, y in train:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            count += len(y)
        val_loss, _, _, _ = evaluate(model, val, device)
        history.append({"Epoch": epoch, "Train Loss": loss_sum/count, "Val Loss": val_loss})
        print(f"{name}: epoch={epoch} train={loss_sum/count:.4f} validation={val_loss:.4f}", flush=True)
        if val_loss < best_loss - 1e-6:
            best_loss, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model, evaluate(model, val, device)[1], history


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default=str(ROOT/"data"/"enron_spam_data.csv"))
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--vocab-size", type=int, default=20000)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--device", default="auto")
    p.add_argument("--quick", action="store_true", help="Smoke test, not report scores")
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.patience) < 1 or args.max_len < 3 or args.vocab_size < 3:
        p.error("epochs, batch-size, patience >= 1; max-len and vocab-size >= 3")
    seed_everything()
    device = get_device(args.device)
    output = ROOT/"results"/"spam_cnn"/("quick" if args.quick else "full")
    output.mkdir(parents=True, exist_ok=True)
    fit_part, val_part, test_part = split_data(args.csv)
    if args.quick:
        fit_part, val_part, test_part = (
            (fit_part[0][:512], fit_part[1][:512]),
            (val_part[0][:128], val_part[1][:128]),
            (test_part[0][:128], test_part[1][:128]))
        args.epochs = min(2, args.epochs)
    vocab = vocabulary(fit_part[0], args.vocab_size)
    train = loader(fit_part, vocab, args.max_len, args.batch_size, shuffle=True)
    val = loader(val_part, vocab, args.max_len, args.batch_size)
    test = loader(test_part, vocab, args.max_len, args.batch_size)
    rows, models, histories = [], {}, {}
    for name, channels, dropout in CONFIGS:
        model, metrics, history = fit(name, channels, dropout, len(vocab)+2,
                                      train, val, device, args.epochs, args.patience)
        rows.append({"Model": name, "Channels": str(channels), "Dropout": dropout, **metrics})
        models[name], histories[name] = model, history
        save_csv(output/f"{name}_history.csv", history)
    save_csv(output/"ablation_validation.csv", rows)
    winner = max(rows, key=lambda r: r["F1"])["Model"]
    _, test_metrics, truth, pred = evaluate(models[winner], test, device)
    save_csv(output/"final_test_metrics.csv", [{"Model": winner, **test_metrics}])
    channels, dropout = next((c, d) for n, c, d in CONFIGS if n == winner)
    plot_architecture(["Email text", "Token IDs", "Embedding", f"Conv1D {channels}",
                       "Global MaxPool", "Spam / Ham"], output/"model_architecture.png",
                      "1D CNN - Spam Detection")
    plot_train_val(histories[winner], output/"best_loss_curve.png", f"Spam CNN - {winner}")
    plot_loss(histories, output/"ablation_val_loss.png", "Spam CNN Validation Loss")
    plot_confusion(truth, pred, ["Ham", "Spam"], output/"confusion_matrix.png",
                   "Spam CNN Test Confusion Matrix")
    torch.save({"state_dict": models[winner].cpu().state_dict(), "vocab": vocab,
                "max_len": args.max_len, "channels": channels, "dropout": dropout},
               output/"best_spam_cnn.pt")
    save_json(output/"experiment_metadata.json", {
        "dataset": "Enron-Spam", "quick_smoke_test": args.quick, "seed": SEED,
        "split": {"train": len(fit_part[0]), "val": len(val_part[0]), "test": len(test_part[0])},
        "max_len": args.max_len, "vocab_size": len(vocab)+2,
        "selection": "validation binary F1", "test_used_for_selection": False,
        "best_config": winner, "test_metrics": test_metrics})
    print("Best:", winner, "Final test:", test_metrics, "Output:", output)


if __name__ == "__main__":
    main()
