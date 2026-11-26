"""Independent checks of the assignment's split quotas and graph preprocessing."""

import unittest
import tempfile
from pathlib import Path

import numpy as np

from local_code.stage5.data import load_graph, normalized_adjacency, split_indices


class SplitTests(unittest.TestCase):
    def labels_for(self, dataset):
        # Counts include the exact limiting Cora class:20 train+150 test+10 val.
        counts = {
            'cora': [298, 418, 818, 426, 217, 180, 351],
            'citeseer': [249, 596, 701, 508, 668, 590],
            'pubmed': [4103, 7739, 7875],
        }[dataset]
        labels = np.repeat(np.arange(len(counts)), counts)
        np.random.default_rng(991).shuffle(labels)
        return labels

    def test_assignment_quotas_disjoint_and_exhaustive_membership(self):
        for dataset, nclass, test_per_class in (
            ('cora', 7, 150), ('citeseer', 6, 200), ('pubmed', 3, 200)
        ):
            with self.subTest(dataset=dataset):
                labels = self.labels_for(dataset)
                splits = split_indices(labels, dataset, seed=42)
                expected = {'idx_train': 20, 'idx_val': 10, 'idx_test': test_per_class}
                for key, per_class in expected.items():
                    selected = np.asarray(splits[key])
                    self.assertEqual(selected.ndim, 1)
                    self.assertTrue(np.issubdtype(selected.dtype, np.integer))
                    self.assertEqual(len(selected), nclass * per_class)
                    self.assertEqual(len(np.unique(selected)), len(selected))
                    self.assertTrue(((selected >= 0) & (selected < len(labels))).all())
                    np.testing.assert_array_equal(
                        np.bincount(labels[selected], minlength=nclass),
                        np.full(nclass, per_class),
                    )
                combined = np.concatenate([splits[key] for key in expected])
                self.assertEqual(len(np.unique(combined)), len(combined))
                if dataset == 'cora':
                    # Every node of the smallest class belongs to exactly one split.
                    self.assertEqual(int((labels[combined] == 5).sum()), 180)

    def test_seed_reproducibility_and_sensitivity(self):
        for dataset in ('cora', 'citeseer', 'pubmed'):
            with self.subTest(dataset=dataset):
                labels = self.labels_for(dataset)
                first = split_indices(labels, dataset, seed=42)
                repeat = split_indices(labels, dataset, seed=42)
                other = split_indices(labels, dataset, seed=43)
                for key in ('idx_train', 'idx_val', 'idx_test'):
                    np.testing.assert_array_equal(first[key], repeat[key])
                    self.assertFalse(np.array_equal(np.sort(first[key]), np.sort(other[key])))

    def test_insufficient_class_population_fails(self):
        # Cora requires180 examples in every class, with no replacement.
        labels = np.repeat(np.arange(7), [200, 200, 200, 200, 200, 179, 200])
        with self.assertRaises(ValueError):
            split_indices(labels, 'cora', seed=42)


class GraphDataTests(unittest.TestCase):
    def test_normalization_matches_hand_computation_with_existing_loop(self):
        # Reciprocal links remain one edge; node0's raw loop survives adding I.
        edges = np.array([[0, 0], [0, 1], [1, 0], [1, 2]])
        result = normalized_adjacency(edges, 4).toarray()
        expected = np.array([
            [2 / 3, 1 / 3, 0, 0],
            [1 / 3, 1 / 3, 1 / np.sqrt(6), 0],
            [0, 1 / np.sqrt(6), 1 / 2, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(result, result.T)
        self.assertTrue(np.isfinite(result).all())

    def write_fixture(self, root):
        folder = Path(root) / 'stage_5_data' / 'cora'
        folder.mkdir(parents=True)
        rows = [
            f'paper-{i} {"0 0" if i == 0 else "2 6"} {"zeta" if i < 180 else "alpha"}'
            for i in range(360)
        ]
        (folder / 'node').write_text('\n'.join(rows) + '\n')
        (folder / 'link').write_text('paper-0 paper-0\npaper-0 paper-1\n')
        return folder

    def test_loader_normalizes_features_and_preserves_node_identity(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_fixture(root)
            normalized = load_graph(root, 'cora')
            raw = load_graph(root, 'cora', normalize_features=False)
        graph, meta = normalized['graph'], normalized['metadata']
        self.assertEqual(meta['classes'], ['alpha', 'zeta'])
        self.assertEqual(graph['node']['paper-179'], 179)
        self.assertEqual(graph['utility']['reverse_idx'][179], 'paper-179')
        self.assertEqual(int(graph['y'][0]), 1)
        self.assertEqual(int(graph['y'][180]), 0)
        np.testing.assert_array_equal(graph['X'][0], [0, 0])
        np.testing.assert_allclose(graph['X'][1:].numpy(), np.tile([.25, .75], (359, 1)))
        np.testing.assert_array_equal(raw['graph']['X'][1], [2, 6])
        self.assertTrue(np.isfinite(graph['X'].numpy()).all())
        self.assertEqual(meta['raw_self_loops'], 1)
        self.assertEqual(meta['isolated_nodes_excluding_self'], 358)
        self.assertEqual(meta['normalized_adjacency_nnz'], 362)
        self.assertEqual(meta['split_counts'], {'idx_train': 40, 'idx_val': 20, 'idx_test': 300})

    def test_loader_rejects_unknown_endpoints_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as root:
            folder = self.write_fixture(root)
            (folder / 'link').write_text('paper-0 missing-paper\n')
            with self.assertRaisesRegex(ValueError, 'unknown node'):
                load_graph(root, 'cora')
            (folder / 'link').write_text('paper-0 paper-1\n')
            with (folder / 'node').open('a') as handle:
                handle.write('paper-0 2 6 zeta\n')
            with self.assertRaisesRegex(ValueError, 'Duplicate node IDs'):
                load_graph(root, 'cora')


if __name__ == '__main__':
    unittest.main()
