"""Offline unit checks for optional decoding-only Stage 4 inference."""
import tempfile
import unittest
from pathlib import Path

import torch

from script.stage4.models import JokeRNN
from script.stage4.resample_generation import (
    blocked_ngrams, load_model, sample_controlled,
)


class Stage4ResamplingTests(unittest.TestCase):
    def test_ngram_repetition_filter(self):
        self.assertEqual(blocked_ngrams([1, 2, 3, 1, 2], 3), {3})
        self.assertEqual(blocked_ngrams([1, 2], 3), set())

    def test_checkpoint_loading_and_three_word_sampling(self):
        vocab = {
            "<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3,
            "i'm": 4, "very": 5, "keen": 6, "joke": 7, "funny": 8,
        }
        torch.manual_seed(42)
        network = JokeRNN("gru", len(vocab), embedding_dim=5,
                          hidden_dim=7, layers=1)
        with tempfile.TemporaryDirectory() as d:
            checkpoint = Path(d) / "gru_best.pt"
            torch.save(network.state_dict(), checkpoint)
            reloaded = load_model("gru", checkpoint, len(vocab))
            answer = sample_controlled(
                reloaded, vocab, "i'm very keen",
                rng=torch.Generator().manual_seed(42),
                top_k=5, top_p=.9, min_new_tokens=2, max_new_tokens=5)
            self.assertTrue(answer.startswith("i'm very keen"))
            self.assertGreater(len(answer.split()), 3)
            with self.assertRaises(ValueError):
                sample_controlled(reloaded, vocab, "two words",
                                  rng=torch.Generator())


if __name__ == "__main__":
    unittest.main()
