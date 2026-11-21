"""Fast unit tests. They do not claim scores on the instructor test sets."""
import io
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from script.stage4.data import (CourseData, JokesDataset, ReviewsDataset,
                                 collate_jokes, collate_reviews, make_vocab,
                                 split_stratified, tokenize)
from script.stage4.models import JokeRNN, SentimentRNN


class Stage4Tests(unittest.TestCase):
    def test_instructor_zip_layout(self):
        with tempfile.TemporaryDirectory() as d:
            filename = Path(d) / "stage_4_data.zip"
            with zipfile.ZipFile(filename, "w") as z:
                z.writestr("stage_4_data/text_generation/data",
                           'ID,Joke\n1,"What did the cat do?"\n2,"Why did the dog bark?"\n')
                for split in ("train", "test"):
                    for label in ("pos", "neg"):
                        for i in range(2):
                            z.writestr(f"stage_4_data/text_classification/{split}/{label}/{i}.txt",
                                       f"{label} movie example {i}")
            with CourseData(filename) as source:
                self.assertEqual(len(source.load_reviews("train")), 4)
                self.assertEqual(len(source.load_reviews("test")), 4)
                self.assertEqual(len(source.load_jokes()), 2)
                train, val = split_stratified(source.load_reviews("train"), .5, 42)
                self.assertEqual((len(train), len(val)), (2, 2))

    def test_tokenization_and_batches(self):
        examples = [("Great movie!", 1), ("Bad movie.", 0)]
        vocab = make_vocab([x[0] for x in examples], min_freq=1)
        x, lengths, y = collate_reviews([ReviewsDataset(examples, vocab)[i] for i in range(2)])
        self.assertEqual(x.size(0), 2)
        self.assertEqual(lengths.tolist(), [3, 3])
        self.assertEqual(y.tolist(), [1, 0])
        self.assertEqual(tokenize("Hello, world!"), ["hello", ",", "world", "!"])
        gen_vocab = make_vocab(["what did the cat say?", "why did the dog bark?"], min_freq=1)
        dataset = JokesDataset(["what did the cat say?"], gen_vocab)
        a, b = collate_jokes([dataset[0]])
        self.assertEqual(a.shape, b.shape)

    def test_three_models_backward(self):
        for kind in ("rnn", "lstm", "gru"):
            classifier = SentimentRNN(kind, 30, embedding_dim=8, hidden_dim=12)
            tokens = torch.tensor([[4, 5, 0], [6, 7, 8]])
            logits = classifier(tokens, torch.tensor([2, 3]))
            self.assertEqual(tuple(logits.shape), (2, 2))
            logits.sum().backward()
            generator = JokeRNN(kind, 30, embedding_dim=8, hidden_dim=12)
            next_logits = generator(tokens)
            self.assertEqual(tuple(next_logits.shape), (2, 3, 30))
            next_logits.sum().backward()


if __name__ == "__main__":
    unittest.main()
