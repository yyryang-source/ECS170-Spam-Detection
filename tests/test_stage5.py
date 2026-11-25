"""Offline synthetic tests: Stage 5 GCN, split isolation and report artifacts."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import torch

from script.stage5.data import load_graph
from script.stage5.model import GCN, symmetric_normalized_adjacency
from script.stage5.train import run_experiments


def fixture(root, name="cora", overlap=False, out_of_range=False, with_splits=True):
    n = 90
    rng = np.random.default_rng(17)
    x = rng.uniform(0, 1, (n, 12)).astype(np.float32)
    labels = np.arange(n) % 3
    src = np.arange(n)
    dst = (src + 1) % n
    edges = np.stack((src, dst))
    if out_of_range:
        edges[1, 0] = n + 2
    arrays = dict(x=x, y=labels, edge_index=edges)
    if with_splits:
        train = np.arange(54)
        val = np.arange(54, 72)
        test = np.arange(72, 90)
        if overlap:
            test[0] = 0
        arrays.update(train_idx=train, val_idx=val, test_idx=test)
    path = Path(root)/f"{name}.npz"
    np.savez(path, **arrays)
    return path


def opts():
    return SimpleNamespace(seed=42, device="cpu", epochs=3, patience=2,
                           learning_rate=0.01, weight_decay=0.0005,
                           verbose=False, quick=True,
                           config_names=("baseline", "wide"))


class TestStage5(unittest.TestCase):
    def test_normalization_and_sparse_forward(self):
        edges = torch.tensor([[0, 1, 0], [1, 0, 1]])
        adj = symmetric_normalized_adjacency(edges, 2)
        self.assertEqual(adj.layout, torch.sparse_coo)
        self.assertEqual(adj._nnz(), 4)
        torch.testing.assert_close(adj.to_dense(), torch.full((2, 2), 0.5))
        model = GCN(5, 3, hidden=8, depth=3)
        logits, embeddings = model(torch.rand(2, 5), adj, return_embedding=True)
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertEqual(tuple(embeddings.shape), (2, 8))
        logits.sum().backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_local_masks_and_no_leakage(self):
        with tempfile.TemporaryDirectory() as folder:
            path = fixture(folder)
            graph = load_graph("cora", "local", path)
            self.assertEqual(graph.provenance["split_origin"], "provided")
            self.assertEqual(graph.provenance["train"], 54)
            self.assertEqual(graph.provenance["val"], 18)
            self.assertEqual(graph.provenance["test"], 18)
            self.assertFalse(torch.any(graph.train_mask & graph.test_mask))
            self.assertEqual(tuple(graph.x.shape), (90, 12))
            archive_path = Path(folder)/"course.zip"
            with ZipFile(archive_path, "w", ZIP_DEFLATED) as archive:
                archive.write(path, arcname="stage_5_data/cora.npz")
            zipped = load_graph("cora", "local", archive_path)
            torch.testing.assert_close(graph.y, zipped.y)

    def test_bad_splits_and_edges_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            for variant in ("overlap", "out_of_range", "missing"):
                p = fixture(folder, overlap=variant == "overlap",
                            out_of_range=variant == "out_of_range",
                            with_splits=variant != "missing")
                with self.assertRaises(ValueError):
                    load_graph("cora", "local", p)
            missing = load_graph("cora", "local", p, allow_generated_split=True)
            self.assertEqual(missing.provenance["split_origin"], "generated_stratified_60_20_20")

    def test_complete_training_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            graph = load_graph("cora", "local", fixture(folder))
            out = Path(folder)/"out"
            summary = run_experiments(graph, out, ("baseline", "wide"), opts())
            self.assertTrue(summary["test_used_for_selection"] is False)
            self.assertTrue(summary["quick_smoke_test_not_for_report"])
            self.assertIn("F1", summary["final_test"])
            for path in ("experiment_summary.json", "ablation_validation.csv",
                         "best_loss_curve.png", "architecture.png",
                         "confusion_matrix.png", "node_embedding_pca.png",
                         "final_test_metrics.csv", "best_gcn.pt"):
                self.assertTrue((out/path).exists(), path)


if __name__ == "__main__":
    unittest.main()
