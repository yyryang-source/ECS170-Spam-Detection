#!/usr/bin/env python3
"""Resample trained Stage 4 joke models WITHOUT retraining or changing historical scores.

Examples are qualitative only. Source training-reference jokes come from the
original generation_examples.json to permit a transparent comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from script.stage4.data import PAD, UNK, BOS, EOS, tokenize
from script.stage4.models import JokeRNN
from script.stage4.run_stage4 import detokenize


def load_model(kind: str, checkpoint: Path, vocab_size: int):
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    embedding = weights["embedding.weight"]
    hidden = weights["recurrent.weight_hh_l0"].size(1)
    layers = len([k for k in weights if k.startswith("recurrent.weight_hh_l")])
    if len(embedding) != vocab_size:
        raise ValueError("Saved vocab length differs from checkpoint embedding")
    model = JokeRNN(
        kind, vocab_size, embedding_dim=embedding.size(1),
        hidden_dim=hidden, layers=layers, dropout=0.0)
    model.load_state_dict(weights, strict=True)
    return model.eval()


def blocked_ngrams(history: list[int], n: int):
    """Return tokens that would repeat an existing n-gram if appended."""
    if n < 2 or len(history) < n - 1:
        return set()
    prefix = history[-(n - 1):]
    return {history[i + n - 1]
            for i in range(len(history) - n + 1)
            if history[i:i + n - 1] == prefix}


def sample_controlled(model, vocab, prompt, *, rng, temperature=0.8,
                      top_k=15, top_p=0.9, min_new_tokens=10,
                      max_new_tokens=40, repetition_penalty=1.15,
                      no_repeat_ngram=3):
    if not 0 < top_p <= 1 or temperature <= 0 or top_k < 1:
        raise ValueError("Invalid temperature, top_k or top_p")
    if min_new_tokens < 0 or max_new_tokens < min_new_tokens:
        raise ValueError("max_new_tokens must be >= min_new_tokens >= 0")
    if repetition_penalty < 1:
        raise ValueError("repetition_penalty must be >= 1")
    words = tokenize(prompt)
    if len(words) != 3:
        raise ValueError("Prompt must contain exactly three words/tokens")
    inv = {v: k for k, v in vocab.items()}
    tokens = [BOS] + [vocab.get(w, UNK) for w in words]
    banned = {"www", "http", "https", "com", "/", "*", "\\"}
    banned_ids = [vocab[w] for w in banned if w in vocab]
    with torch.inference_mode():
        for step in range(max_new_tokens):
            logits = model(torch.tensor([tokens], dtype=torch.long))[0, -1].clone()
            logits = logits / temperature
            logits[[PAD, BOS, UNK]] = -float("inf")
            if step < min_new_tokens:
                logits[EOS] = -float("inf")
            if banned_ids:
                logits[banned_ids] = -float("inf")
            if repetition_penalty > 1:
                for index in set(tokens[1:]):
                    if index in (PAD, UNK, BOS, EOS):
                        continue
                    if logits[index] < 0:
                        logits[index] *= repetition_penalty
                    else:
                        logits[index] /= repetition_penalty
            for index in blocked_ngrams(tokens[1:], no_repeat_ngram):
                logits[index] = -float("inf")
            finite_count = int(torch.isfinite(logits).sum().item())
            if finite_count == 0:
                break
            n = min(top_k, finite_count)
            values, indices = torch.topk(logits, n)
            probs = torch.softmax(values, dim=0)
            # Nucleus filter over descending logits, retaining the first
            # element that reaches/exceeds the probability mass cutoff.
            cdf = torch.cumsum(probs, 0)
            cutoff = int((cdf < top_p).sum().item()) + 1
            probs = probs[:cutoff]
            indices = indices[:cutoff]
            probs = probs / probs.sum()
            selected = int(indices[torch.multinomial(probs, 1, generator=rng)].item())
            if selected == EOS:
                break
            tokens.append(selected)
    return detokenize([inv.get(t, "<unk>") for t in tokens[1:]])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-dir", required=True,
                   help="Completed results/stage4/<experiment> directory with local .pt files")
    p.add_argument("--models", default="rnn,lstm,gru")
    p.add_argument("--num-samples", type=int, default=3)
    p.add_argument("--min-new-tokens", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.15)
    p.add_argument("--no-repeat-ngram", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None)
    args = p.parse_args(argv)
    if args.num_samples < 1 or args.no_repeat_ngram < 2:
        p.error("Require num-samples >= 1 and no-repeat-ngram >= 2")
    root = Path(args.experiment_dir).expanduser() / "generation"
    vocab = json.loads((root / "vocab.json").read_text(encoding="utf-8"))
    originals = json.loads(
        (root / "generation_examples.json").read_text(encoding="utf-8"))
    models = [x.strip().lower() for x in args.models.split(",") if x.strip()]
    if not models or any(x not in ("rnn", "lstm", "gru") for x in models):
        p.error("--models must contain rnn,lstm,gru")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    results = {}
    for kind in models:
        model = load_model(kind, root / f"{kind}_best.pt", len(vocab))
        samples = []
        for example in originals[kind]:
            candidates = [
                sample_controlled(
                    model, vocab, example["prompt"], rng=rng,
                    temperature=args.temperature, top_k=args.top_k,
                    top_p=args.top_p, min_new_tokens=args.min_new_tokens,
                    max_new_tokens=args.max_new_tokens,
                    repetition_penalty=args.repetition_penalty,
                    no_repeat_ngram=args.no_repeat_ngram)
                for _ in range(args.num_samples)
            ]
            samples.append({
                "prompt": example["prompt"],
                "training_reference": example["training_reference"],
                "original_generated": example["generated"],
                "additional_samples": candidates,
            })
        results[kind] = samples
    payload = {
        "note": "Qualitative resampling from original saved models; no retraining or score changes.",
        "generation_settings": {
            key: getattr(args, key) for key in (
                "seed", "num_samples", "min_new_tokens", "max_new_tokens",
                "temperature", "top_k", "top_p",
                "repetition_penalty", "no_repeat_ngram")
        },
        "models": results,
    }
    destination = Path(args.output) if args.output else root / "resampled_examples.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved qualitative examples to {destination}")
    return payload


if __name__ == "__main__":
    main()
