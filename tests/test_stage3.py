"""Offline shape + instructor-style local dataset loading smoke tests."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

STAGE3 = Path(__file__).resolve().parents[1]/"script"/"stage3"
sys.path.insert(0, str(STAGE3))
from models import Stage3CNN
from datasets import load_stage3_data


class Stage3Tests(unittest.TestCase):
    def test_shapes_all_three_tasks(self):
        for channels, classes, size in [(1, 10, 28), (1, 40, 64), (3, 10, 32)]:
            for config in ("baseline", "residual", "kernel5"):
                model = Stage3CNN(channels, classes, config)
                with torch.no_grad():
                    output = model(torch.randn(2, channels, size, size))
                self.assertEqual(tuple(output.shape), (2, classes))

    def test_orl_local_face_folders_three_rgb_equal_channels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for class_idx in range(4):
                d = root/f"s{class_idx+1}"
                d.mkdir()
                for i in range(10):
                    gray = np.full((112, 92), 25+class_idx*40+i, dtype=np.uint8)
                    rgb = np.stack([gray]*3, axis=-1)
                    Image.fromarray(rgb).save(d/f"{i}.png")
            (fit, val, test), labels, meta = load_stage3_data(
                "orl", "local", root, seed=42, quick=False)
            self.assertEqual(len(labels), 4)
            self.assertEqual(len(fit)+len(val)+len(test), 40)
            x, y = fit[0]
            self.assertEqual(tuple(x.shape), (1, 64, 64))
            self.assertTrue(0 <= y < 4)

    def test_instructor_npz_preserves_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"mnist.npz"
            rng = np.random.default_rng(9)
            x_train = rng.integers(0, 256, (100, 28, 28), dtype=np.uint8)
            y_train = np.tile(np.arange(10), 10)
            x_test = rng.integers(0, 256, (20, 28, 28), dtype=np.uint8)
            y_test = np.tile(np.arange(10), 2)
            np.savez(path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
            (fit, val, test), classes, meta = load_stage3_data(
                "mnist", "local", path, seed=42)
            self.assertEqual(len(fit)+len(val), 100)
            self.assertEqual(len(test), 20)
            self.assertTrue(meta["prepartitioned_test"])
            self.assertEqual(len(classes), 10)
            self.assertEqual(tuple(fit[0][0].shape), (1, 28, 28))


if __name__ == "__main__":
    unittest.main()
