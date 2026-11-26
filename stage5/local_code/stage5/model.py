"""A GCN without PyTorch Geometric or a prebuilt graph-convolution layer.

The caller supplies A_hat = D^(-1/2) (A + I) D^(-1/2). Each layer applies
``A_hat @ (X @ W) + b``. CPU/CUDA use sparse matrix multiplication; the MPS
backend uses an equivalent edge-wise sum and never creates a dense N x N
adjacency matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class GraphOperator:
    """Fixed graph propagation operator, stored on the selected device.

    ``backend='edge'`` is also available on CPU for a numerical cross-check.
    This class stores an already normalized graph; it does not normalize it.
    """

    num_nodes: int
    backend: str
    sparse: Tensor | None = None
    rows: Tensor | None = None
    cols: Tensor | None = None
    values: Tensor | None = None

    @classmethod
    def from_scipy(
        cls,
        adjacency: Any,
        device: str | torch.device = "cpu",
        *,
        backend: str | None = None,
    ) -> "GraphOperator":
        """Copy a square SciPy sparse matrix into a PyTorch operator."""
        import numpy as np

        device = torch.device(device)
        backend = backend or ("edge" if device.type == "mps" else "sparse")
        if backend not in {"sparse", "edge"}:
            raise ValueError("backend must be 'sparse' or 'edge'")
        if adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("adjacency must be square")
        coo = adjacency.tocoo(copy=True)
        coo.sum_duplicates()
        rows = torch.tensor(np.asarray(coo.row), dtype=torch.long, device=device)
        cols = torch.tensor(np.asarray(coo.col), dtype=torch.long, device=device)
        values = torch.tensor(np.asarray(coo.data), dtype=torch.float32, device=device)
        if not bool(torch.isfinite(values).all().item()):
            raise ValueError("adjacency contains non-finite values")
        if backend == "edge":
            return cls(coo.shape[0], backend, rows=rows, cols=cols, values=values)
        sparse = torch.sparse_coo_tensor(
            torch.stack((rows, cols)), values, coo.shape, device=device,
            check_invariants=True,
        ).coalesce()
        return cls(coo.shape[0], backend, sparse=sparse)

    def matmul(self, support: Tensor) -> Tensor:
        """Aggregate neighboring rows of a dense N x channels matrix."""
        if support.ndim != 2 or support.shape[0] != self.num_nodes:
            raise ValueError("support must have shape [num_nodes, channels]")
        if self.backend == "sparse":
            if self.sparse is None:
                raise ValueError("sparse backend is missing its tensor")
            return torch.sparse.mm(self.sparse, support)
        if self.backend != "edge" or any(
            value is None for value in (self.rows, self.cols, self.values)
        ):
            raise ValueError("edge backend is missing its indices or values")
        messages = support[self.cols] * self.values[:, None]
        result = support.new_zeros((self.num_nodes, support.shape[1]))
        return result.index_add_(0, self.rows, messages)


def _propagate(adjacency: Tensor | GraphOperator, support: Tensor) -> Tensor:
    if isinstance(adjacency, GraphOperator):
        return adjacency.matmul(support)
    if adjacency.layout == torch.strided:
        return adjacency @ support
    return torch.sparse.mm(adjacency, support)


class GraphConvolution(nn.Module):
    """Learnable GCN layer: A_hat @ (X @ W) + b."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        if in_features < 1 or out_features < 1:
            raise ValueError("feature dimensions must be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, adjacency: Tensor | GraphOperator) -> Tensor:
        output = _propagate(adjacency, x @ self.weight)
        return output if self.bias is None else output + self.bias

    def extra_repr(self) -> str:
        return f"{self.in_features}, {self.out_features}, bias={self.bias is not None}"


class GCN(nn.Module):
    """Full-batch node classifier with configurable graph-convolution depth.

    Depth counts graph-convolution layers, including the classifier. For the
    default depth=2, the architecture is input dropout -> GCN -> ReLU ->
    hidden dropout -> GCN -> logits. CrossEntropyLoss consumes these logits.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.5,
        depth: int = 2,
    ):
        super().__init__()
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 2:
            raise ValueError("depth must be an integer of at least 2")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        dimensions = [in_features] + [hidden_dim] * (depth - 1) + [num_classes]
        self.layers = nn.ModuleList(
            GraphConvolution(a, b) for a, b in zip(dimensions[:-1], dimensions[1:])
        )
        self.dropout = float(dropout)
        self.depth = depth

    def _hidden(
        self, x: Tensor, adjacency: Tensor | GraphOperator, *, training: bool
    ) -> Tensor:
        x = F.dropout(x, p=self.dropout, training=training)
        for index, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x, adjacency))
            # Leave the final hidden activation intact for embedding export.
            if index < len(self.layers) - 2:
                x = F.dropout(x, p=self.dropout, training=training)
        return x

    def forward(self, x: Tensor, adjacency: Tensor | GraphOperator) -> Tensor:
        hidden = self._hidden(x, adjacency, training=self.training)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return self.layers[-1](hidden, adjacency)

    @torch.no_grad()
    def embed(self, x: Tensor, adjacency: Tensor | GraphOperator) -> Tensor:
        """Export deterministic last-hidden-layer embeddings without dropout.

        This method does not change the caller's train/eval mode and does not
        retain an autograd graph. It works even if called during train mode.
        """
        return self._hidden(x, adjacency, training=False)
