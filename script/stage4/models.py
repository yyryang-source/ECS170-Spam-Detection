"""Three comparable PyTorch RNN/LSTM/GRU baselines, MPS-compatible."""
import torch
from torch import nn

from .data import PAD


def recurrent_layer(kind, input_dim, hidden_dim, layers, dropout):
    cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[kind]
    return cls(input_dim, hidden_dim, num_layers=layers, batch_first=True,
               dropout=dropout if layers > 1 else 0.0)


class SentimentRNN(nn.Module):
    def __init__(self, kind, vocab_size, embedding_dim=100, hidden_dim=128,
                 layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD)
        self.recurrent = recurrent_layer(kind, embedding_dim, hidden_dim, layers, dropout)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 2))

    def forward(self, tokens, lengths):
        # Unidirectional: padding after the last real token cannot affect its state.
        # No packed sequences are needed, which also avoids MPS packing issues.
        outputs, _ = self.recurrent(self.embedding(tokens))
        idx = (lengths.to(tokens.device) - 1).clamp(min=0)
        final = outputs[torch.arange(tokens.size(0), device=tokens.device), idx]
        return self.classifier(final)


class JokeRNN(nn.Module):
    def __init__(self, kind, vocab_size, embedding_dim=100, hidden_dim=128,
                 layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD)
        self.recurrent = recurrent_layer(kind, embedding_dim, hidden_dim, layers, dropout)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tokens):
        output, _ = self.recurrent(self.embedding(tokens))
        return self.classifier(self.dropout(output))


def load_glove_matrix(vocab, dim, filename):
    """Stream only requested tokens from the large Stanford text or ZIP file."""
    import zipfile
    from pathlib import Path

    path = Path(filename).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"GloVe file not found: {path}")
    matrix = torch.empty((len(vocab), dim), dtype=torch.float32)
    nn.init.normal_(matrix, std=0.05)
    matrix[PAD].zero_()

    if path.suffix.lower() == ".zip":
        archive = zipfile.ZipFile(path)
        members = [n for n in archive.namelist() if n.endswith(".txt") and not n.startswith("__MACOSX")]
        if not members:
            archive.close()
            raise ValueError("Could not find a .txt vector file inside the GloVe ZIP")
        stream = archive.open(members[0])
    else:
        archive = None
        stream = open(path, "rb")

    count = 0
    try:
        for raw in stream:
            fields = raw.decode("utf-8", errors="replace").strip().split()
            if len(fields) != dim + 1:
                continue
            word = fields[0]
            if word in vocab:
                try:
                    matrix[vocab[word]] = torch.tensor([float(v) for v in fields[1:]])
                    count += 1
                except ValueError:
                    pass
    finally:
        stream.close()
        if archive is not None:
            archive.close()
    return matrix, count
