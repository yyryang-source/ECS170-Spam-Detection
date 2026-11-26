"""Numerical and gradient checks for the independently implemented GCN."""

import unittest

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from local_code.stage5.model import GCN, GraphConvolution, GraphOperator


class GCNTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        # A normalized three-node path with self loops.
        graph = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=np.float32)
        inv = np.diag(graph.sum(1) ** -0.5)
        self.matrix = sp.coo_matrix(inv @ graph @ inv)
        self.dense = torch.from_numpy(self.matrix.toarray())
        self.sparse = self.dense.to_sparse().coalesce()
        self.x = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])

    def test_layer_matches_hand_computation(self):
        layer = GraphConvolution(2, 2)
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([[1., -1.], [0.5, 2.]]))
            layer.bias.copy_(torch.tensor([0.25, -0.5]))
        expected = self.dense @ (self.x @ layer.weight) + layer.bias
        torch.testing.assert_close(layer(self.x, self.sparse), expected)

    def test_edge_aggregation_matches_sparse_and_gradients(self):
        sparse_op = GraphOperator.from_scipy(self.matrix)
        edge_op = GraphOperator.from_scipy(self.matrix, backend="edge")
        x1 = self.x.clone().requires_grad_()
        x2 = self.x.clone().requires_grad_()
        y1 = sparse_op.matmul(x1)
        y2 = edge_op.matmul(x2)
        torch.testing.assert_close(y1, y2)
        y1.square().sum().backward()
        y2.square().sum().backward()
        torch.testing.assert_close(x1.grad, x2.grad)

    def test_logits_dimensions_and_finite_gradients(self):
        for depth in (2, 3):
            model = GCN(2, 8, 3, dropout=0.25, depth=depth)
            logits = model(self.x, self.sparse)
            self.assertEqual(tuple(logits.shape), (3, 3))
            F.cross_entropy(logits, torch.tensor([0, 1, 2])).backward()
            for parameter in model.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_eval_and_embedding_are_deterministic(self):
        model = GCN(2, 5, 3, dropout=0.8)
        model.eval()
        torch.testing.assert_close(model(self.x, self.sparse), model(self.x, self.sparse))
        model.train()
        first = model.embed(self.x, self.sparse)
        second = model.embed(self.x, self.sparse)
        torch.testing.assert_close(first, second)
        self.assertEqual(tuple(first.shape), (3, 5))
        self.assertTrue(model.training)
        self.assertFalse(first.requires_grad)

    def test_two_layer_forward_matches_explicit_equation(self):
        model = GCN(2, 4, 3, dropout=0.)
        model.eval()
        a, b = model.layers
        hidden = F.relu(self.dense @ (self.x @ a.weight) + a.bias)
        expected = self.dense @ (hidden @ b.weight) + b.bias
        torch.testing.assert_close(model(self.x, self.sparse), expected)
        torch.testing.assert_close(model.embed(self.x, self.sparse), hidden)

    def test_invalid_configuration_rejected(self):
        for depth in (1, 2.5, True):
            with self.assertRaises(ValueError):
                GCN(2, 3, 2, depth=depth)
        for dropout in (-0.1, 1.):
            with self.assertRaises(ValueError):
                GCN(2, 3, 2, dropout=dropout)
        with self.assertRaises(ValueError):
            GraphOperator.from_scipy(sp.coo_matrix((2, 3)))


if __name__ == "__main__":
    unittest.main()
