"""Read the actual instructor Stage 4 ZIP or its extracted directory."""
from __future__ import annotations

from collections import Counter
import csv
import html
import io
from pathlib import Path
import random
import re
import zipfile

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

SPECIAL = ("<pad>", "<unk>", "<bos>", "<eos>")
PAD, UNK, BOS, EOS = range(4)
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+|[^\w\s]", re.I)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(html.unescape(text).lower().replace("<br />", " "))


class CourseData:
    """Read both original Mac-created ZIP and a normal extracted directory."""

    def __init__(self, source: str | Path):
        self.source = Path(source).expanduser()
        if not self.source.exists():
            raise FileNotFoundError(
                f"Missing Stage 4 data: {self.source}. Download stage_4_data.zip "
                "from the instructor and pass --data PATH."
            )
        self.archive = self.source.is_file()
        if self.archive:
            self._zip = zipfile.ZipFile(self.source)
            names = self._zip.namelist()
            self.prefix = "stage_4_data/" if "stage_4_data/text_generation/data" in names else ""
            if not any(x.endswith("text_generation/data") for x in names):
                raise ValueError("This is not the instructor's Stage 4 ZIP.")
        else:
            self._zip = None
            self.root = self.source / "stage_4_data" if (
                self.source / "stage_4_data").is_dir() else self.source
            if not (self.root / "text_generation" / "data").exists():
                raise ValueError("Expected stage_4_data/text_generation/data under --data.")

    def close(self):
        if self._zip is not None:
            self._zip.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def read_text(self, relative: str) -> str:
        if self.archive:
            return self._zip.read(self.prefix + relative).decode("utf-8", errors="replace")
        return (self.root / relative).read_text(encoding="utf-8", errors="replace")

    def review_paths(self, split: str, label: str) -> list[str]:
        prefix = f"text_classification/{split}/{label}/"
        if self.archive:
            return sorted(
                n[len(self.prefix):] for n in self._zip.namelist()
                if n.startswith(self.prefix + prefix)
                and n.endswith(".txt") and "/._" not in n
            )
        return sorted(p.relative_to(self.root).as_posix()
                      for p in (self.root / prefix).glob("*.txt"))

    def load_reviews(self, split: str, per_class_limit: int | None = None):
        examples = []
        for label, value in (("neg", 0), ("pos", 1)):
            paths = self.review_paths(split, label)
            if per_class_limit is not None:
                paths = paths[:per_class_limit]
            for p in paths:
                examples.append((self.read_text(p), value))
        if not examples:
            raise ValueError(f"No IMDb reviews found for {split}")
        return examples

    def load_jokes(self) -> list[str]:
        raw = self.read_text("text_generation/data")
        rows = csv.DictReader(io.StringIO(raw))
        if not rows.fieldnames or "Joke" not in rows.fieldnames:
            raise ValueError("Expected the instructor's CSV columns ID,Joke")
        jokes = [row["Joke"].strip() for row in rows if row.get("Joke", "").strip()]
        if not jokes:
            raise ValueError("No joke examples found")
        return jokes


def split_stratified(examples, val_fraction=0.1, seed=42):
    rng = random.Random(seed)
    groups = {}
    for text, label in examples:
        groups.setdefault(label, []).append((text, label))
    train, val = [], []
    for group in groups.values():
        rng.shuffle(group)
        n = max(1, round(len(group) * val_fraction))
        if n >= len(group):
            raise ValueError("Need at least two training examples for every class")
        val.extend(group[:n])
        train.extend(group[n:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def split_items(items, val_fraction=0.1, seed=42):
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    n = max(1, round(len(shuffled) * val_fraction))
    if n >= len(shuffled):
        raise ValueError("Need more examples for the generation split")
    return shuffled[n:], shuffled[:n]


def make_vocab(texts, max_size=30000, min_freq=2):
    counts = Counter(word for text in texts for word in tokenize(text))
    vocab = {word: i for i, word in enumerate(SPECIAL)}
    for word, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        if len(vocab) >= max_size:
            break
        if count >= min_freq and word not in vocab:
            vocab[word] = len(vocab)
    return vocab


def encode(text, vocab, max_tokens):
    return [vocab.get(token, UNK) for token in tokenize(text)[:max_tokens]] or [UNK]


class ReviewsDataset(Dataset):
    def __init__(self, examples, vocab, max_tokens=256):
        self.examples, self.vocab, self.max_tokens = examples, vocab, max_tokens

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        text, label = self.examples[index]
        return torch.tensor(encode(text, self.vocab, self.max_tokens)), label


def collate_reviews(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    return pad_sequence(sequences, batch_first=True, padding_value=PAD), lengths, torch.tensor(labels)


class JokesDataset(Dataset):
    def __init__(self, texts, vocab, max_tokens=64):
        self.texts, self.vocab, self.max_tokens = texts, vocab, max_tokens

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        ids = [BOS] + encode(self.texts[index], self.vocab, self.max_tokens - 2) + [EOS]
        return torch.tensor(ids[:-1]), torch.tensor(ids[1:])


def collate_jokes(batch):
    xs, ys = zip(*batch)
    return (pad_sequence(xs, batch_first=True, padding_value=PAD),
            pad_sequence(ys, batch_first=True, padding_value=PAD))
