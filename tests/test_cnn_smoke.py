"""Basic forward-pass tests for both CNNs."""
import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"script"/"cnn"))
from train_spam_cnn import SpamCNN, vocabulary, encode
from train_image_cnn import ImageCNN

class TestCNN(unittest.TestCase):
    def test_spam(self):
        vocab = vocabulary(["Free money", "Meeting on Friday"], 20)
        x = encode(["Free FREE money", "new unknown word"], vocab, 12)
        self.assertEqual(tuple(x.shape), (2, 12))
        self.assertEqual(x[1, 0].item(), 1)
        self.assertEqual(tuple(SpamCNN(len(vocab)+2, (8, 16))(x).shape), (2,))

    def test_images(self):
        for channels, size in [(1, 28), (3, 32)]:
            x = torch.randn((2, channels, size, size))
            model = ImageCNN(channels, 10, widths=(8, 16, 32))
            self.assertEqual(tuple(model(x).shape), (2, 10))

if __name__ == "__main__":
    unittest.main()
