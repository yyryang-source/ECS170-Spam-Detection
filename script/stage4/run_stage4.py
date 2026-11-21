#!/usr/bin/env python3
"""ECS 170 Stage 4: run instructor-data RNN/LSTM/GRU experiments."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import sys

# Must be set before torch initialization on macOS.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

# Supports invocation as python script/stage4/run_stage4.py from repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from script.stage4.data import (BOS, EOS, PAD, UNK, CourseData, JokesDataset,
                                ReviewsDataset, collate_jokes, collate_reviews,
                                make_vocab, split_items, split_stratified, tokenize)
from script.stage4.models import JokeRNN, SentimentRNN, load_glove_matrix


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for(name):
    if name != "auto":
        if name == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable: select --device cpu")
        if name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable: select --device cpu")
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def metrics_binary(y_true, y_pred):
    n = len(y_true)
    accuracy = sum(a == b for a, b in zip(y_true, y_pred)) / max(n, 1)
    p_values, r_values, f_values = [], [], []
    for cls in (0, 1):
        tp = sum(a == cls and b == cls for a, b in zip(y_true, y_pred))
        fp = sum(a != cls and b == cls for a, b in zip(y_true, y_pred))
        fn = sum(a == cls and b != cls for a, b in zip(y_true, y_pred))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        p_values.append(p)
        r_values.append(r)
        f_values.append(f)
    return {"accuracy": accuracy, "macro_precision": sum(p_values) / 2,
            "macro_recall": sum(r_values) / 2, "macro_f1": sum(f_values) / 2}


def evaluate_classification(model, loader, device):
    model.eval()
    truths, preds, total_loss = [], [], 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for tokens, lengths, labels in loader:
            tokens, labels = tokens.to(device), labels.to(device)
            logits = model(tokens, lengths)
            total_loss += criterion(logits, labels).item()
            truths.extend(labels.cpu().tolist())
            preds.extend(logits.argmax(-1).cpu().tolist())
    return {"loss": total_loss / max(len(truths), 1), **metrics_binary(truths, preds)}


def evaluate_generation(model, loader, device):
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss(ignore_index=PAD, reduction="sum")
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss_sum += criterion(logits.reshape(-1, logits.size(-1)),
                                  targets.reshape(-1)).item()
            mask = targets.ne(PAD)
            correct += ((logits.argmax(-1) == targets) & mask).sum().item()
            count += mask.sum().item()
    loss = loss_sum / max(count, 1)
    return {"loss": loss, "perplexity": math.exp(min(loss, 30)),
            "token_accuracy": correct / max(count, 1)}


def save_curve(history, destination, title):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot([r["epoch"] for r in history], [r["train_loss"] for r in history],
            label="Train")
    ax.plot([r["epoch"] for r in history], [r["val_loss"] for r in history],
            label="Validation")
    ax.set(xlabel="Epoch", ylabel="Cross-entropy loss", title=title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def save_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def set_embedding(model, matrix):
    if matrix is not None:
        with torch.no_grad():
            model.embedding.weight.copy_(matrix)


def train_classification(args, source, device, output):
    examples = source.load_reviews("train", 240 if args.quick else None)
    test_examples = source.load_reviews("test", 80 if args.quick else None)
    train, val = split_stratified(examples, seed=args.seed)
    vocab = make_vocab((x for x, _ in train), max_size=args.max_vocab,
                       min_freq=1 if args.quick else 2)
    (output / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    make_loader = lambda items, shuffle: DataLoader(
        ReviewsDataset(items, vocab, args.max_review_tokens),
        batch_size=args.batch_size, shuffle=shuffle, num_workers=0,
        collate_fn=collate_reviews)
    train_loader = make_loader(train, True)
    val_loader = make_loader(val, False)
    test_loader = make_loader(test_examples, False)

    weights = None
    if args.glove:
        weights, found = load_glove_matrix(vocab, args.embedding_dim, args.glove)
        print(f"IMDb GloVe coverage: {found}/{len(vocab)} tokens")
    results = []
    for kind in args.models:
        set_seed(args.seed)
        model = SentimentRNN(kind, len(vocab), args.embedding_dim,
                             args.hidden_dim, args.layers, args.dropout).to(device)
        set_embedding(model, weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        history, best_score, best_loss = [], -1.0, float("inf")
        checkpoint = output / f"{kind}_best.pt"
        for epoch in range(1, (1 if args.quick else args.epochs_classification) + 1):
            model.train()
            loss_sum, seen = 0.0, 0
            for tokens, lengths, labels in train_loader:
                tokens, labels = tokens.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(tokens, lengths), labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_sum += loss.item() * len(labels)
                seen += len(labels)
            val_metrics = evaluate_classification(model, val_loader, device)
            history.append({"epoch": epoch, "train_loss": loss_sum / max(seen, 1),
                            "val_loss": val_metrics["loss"],
                            "val_macro_f1": val_metrics["macro_f1"]})
            print(f"IMDb {kind} epoch {epoch}: val F1={val_metrics['macro_f1']:.4f}")
            current = val_metrics["macro_f1"]
            if current > best_score or (current == best_score and val_metrics["loss"] < best_loss):
                best_score, best_loss = current, val_metrics["loss"]
                torch.save(model.state_dict(), checkpoint)
        model.load_state_dict(torch.load(checkpoint, map_location=device,
                                         weights_only=True))
        test_metrics = evaluate_classification(model, test_loader, device)
        result = {"model": kind, "parameters": sum(p.numel() for p in model.parameters()),
                  "best_val_macro_f1": best_score, **test_metrics}
        results.append(result)
        save_csv(output / f"{kind}_history.csv", history)
        save_curve(history, output / f"{kind}_loss.png", f"IMDb sentiment — {kind.upper()}")

    save_csv(output / "test_metrics.csv", results)
    settings = {"dataset": "Instructor IMDb movie reviews", "quick": args.quick,
                "train": len(train), "validation": len(val),
                "instructor_test": len(test_examples), "vocab_size": len(vocab),
                "max_review_tokens": args.max_review_tokens,
                "selected_by": "validation macro F1; checkpoint test evaluated once"}
    (output / "settings.json").write_text(json.dumps(settings, indent=2))
    return results


def detokenize(words):
    text = " ".join(words)
    for mark in (".", ",", "!", "?", ":", ";"):
        text = text.replace(" " + mark, mark)
    return text.replace(" '", "'")


def sample_generation(model, vocab, prompt_words, device, max_new_tokens,
                      temperature=0.85, top_k=10):
    inv = {v: k for k, v in vocab.items()}
    initial = [vocab.get(t, UNK) for t in prompt_words]
    tokens = [BOS] + initial
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            ids = torch.tensor([tokens], dtype=torch.long, device=device)
            logits = model(ids)[0, -1] / max(temperature, 1e-5)
            logits[PAD] = -float("inf")
            logits[BOS] = -float("inf")
            logits[UNK] = -float("inf")
            k = min(max(1, top_k), logits.numel())
            values, indices = torch.topk(logits, k)
            next_id = int(indices[torch.multinomial(torch.softmax(values, 0), 1)].item())
            if next_id == EOS:
                break
            tokens.append(next_id)
    return detokenize([inv.get(t, "<unk>") for t in tokens[1:]])


def train_generation(args, source, device, output):
    jokes = source.load_jokes()
    if args.quick:
        jokes = jokes[:180]
    train, val = split_items(jokes, seed=args.seed)
    vocab = make_vocab(train, max_size=args.max_joke_vocab, min_freq=1)
    (output / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    make_loader = lambda texts, shuffle: DataLoader(
        JokesDataset(texts, vocab, args.max_joke_tokens),
        batch_size=args.batch_size, shuffle=shuffle,
        num_workers=0, collate_fn=collate_jokes)
    train_loader, val_loader = make_loader(train, True), make_loader(val, False)

    # Every reference comes from the training partition; no held-out text is shown
    # to the model for prompt/reference comparison.
    prompts = []
    for joke in train:
        words = tokenize(joke)
        if len(words) >= 3 and all(any(c.isalpha() for c in w) for w in words[:3]):
            if words[:3] not in [p for p, _ in prompts]:
                prompts.append((words[:3], joke))
        if len(prompts) == 3:
            break

    weights = None
    if args.glove:
        weights, found = load_glove_matrix(vocab, args.embedding_dim, args.glove)
        print(f"Jokes GloVe coverage: {found}/{len(vocab)} tokens")
    results, examples = [], {}
    for kind in args.models:
        set_seed(args.seed)
        model = JokeRNN(kind, len(vocab), args.embedding_dim,
                        args.hidden_dim, args.layers, args.dropout).to(device)
        set_embedding(model, weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss(ignore_index=PAD, reduction="sum")
        history, best_val = [], float("inf")
        checkpoint = output / f"{kind}_best.pt"
        for epoch in range(1, (1 if args.quick else args.epochs_generation) + 1):
            model.train()
            sum_loss, total_tokens = 0.0, 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs)
                count = targets.ne(PAD).sum().item()
                loss_sum = criterion(logits.reshape(-1, logits.size(-1)),
                                     targets.reshape(-1))
                (loss_sum / max(count, 1)).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                sum_loss += loss_sum.item()
                total_tokens += count
            val_metrics = evaluate_generation(model, val_loader, device)
            history.append({"epoch": epoch, "train_loss": sum_loss / max(total_tokens, 1),
                            "val_loss": val_metrics["loss"],
                            "val_perplexity": val_metrics["perplexity"]})
            print(f"Jokes {kind} epoch {epoch}: val PPL={val_metrics['perplexity']:.2f}")
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                torch.save(model.state_dict(), checkpoint)
        model.load_state_dict(torch.load(checkpoint, map_location=device,
                                         weights_only=True))
        results.append({"model": kind,
                        "parameters": sum(p.numel() for p in model.parameters()),
                        **evaluate_generation(model, val_loader, device)})
        save_csv(output / f"{kind}_history.csv", history)
        save_curve(history, output / f"{kind}_loss.png", f"Joke generation — {kind.upper()}")
        set_seed(args.seed)
        examples[kind] = [
            {"prompt": " ".join(prompt), "training_reference": reference,
             "generated": sample_generation(model, vocab, prompt, device,
                                            args.max_new_tokens, args.temperature,
                                            args.top_k)}
            for prompt, reference in prompts
        ]

    save_csv(output / "validation_metrics.csv", results)
    (output / "generation_examples.json").write_text(
        json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")
    settings = {"dataset": "Instructor joke CSV", "quick": args.quick,
                "train": len(train), "validation": len(val), "vocab_size": len(vocab),
                "max_joke_tokens": args.max_joke_tokens,
                "no_official_test_split": True,
                "reference_provenance": "training split only",
                "model_selection": "lowest validation token cross-entropy"}
    (output / "settings.json").write_text(json.dumps(settings, indent=2))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/stage_4_data.zip",
                        help="Instructor ZIP or extracted stage_4_data directory")
    parser.add_argument("--task", choices=("both", "classification", "generation"),
                        default="both")
    parser.add_argument("--models", default="rnn,lstm,gru",
                        help="Comma-separated: rnn,lstm,gru")
    parser.add_argument("--output-dir", default="results/stage4")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"),
                        default="auto")
    parser.add_argument("--glove", default=None,
                        help="Optional pretrained GloVe .txt or .zip; dimension must match")
    parser.add_argument("--embedding-dim", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--max-review-tokens", type=int, default=256)
    parser.add_argument("--max-joke-tokens", type=int, default=64)
    parser.add_argument("--max-vocab", type=int, default=30000)
    parser.add_argument("--max-joke-vocab", type=int, default=10000)
    parser.add_argument("--epochs-classification", type=int, default=5)
    parser.add_argument("--epochs-generation", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true",
                        help="Tiny 1-epoch smoke test; never use for the report")
    args = parser.parse_args()
    args.models = [name.strip().lower() for name in args.models.split(",")]
    if not args.models or any(x not in ("rnn", "lstm", "gru") for x in args.models):
        parser.error("--models must contain rnn,lstm,gru")
    if args.max_joke_tokens < 5 or args.embedding_dim < 1:
        parser.error("Need --max-joke-tokens >= 5 and --embedding-dim >= 1")
    set_seed(args.seed)
    device = device_for(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | Instructor source: {args.data}")
    results = {}
    with CourseData(args.data) as source:
        if args.task in ("both", "classification"):
            folder = out / "classification"
            folder.mkdir(exist_ok=True)
            results["classification"] = train_classification(args, source, device, folder)
        if args.task in ("both", "generation"):
            folder = out / "generation"
            folder.mkdir(exist_ok=True)
            results["generation"] = train_generation(args, source, device, folder)
    (out / "summary.json").write_text(
        json.dumps({"device": str(device), "quick": args.quick,
                    "models": args.models, "results": results}, indent=2))
    print(f"Results saved to: {out.resolve()}")
    if args.quick:
        print("QUICK MODE: diagnostic only; do not report these scores.")


if __name__ == "__main__":
    main()
