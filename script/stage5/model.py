"""Memory-bounded sparse Graph Convolutional Network, no PyG ops required."""
import torch
from torch import nn


def symmetric_normalized_adjacency(edge_index, num_nodes, device="cpu"):
    """Binary undirected A+I normalized as D^-1/2 (A+I) D^-1/2.

    Duplicate edges do NOT increase edge weights. Stays sparse even for Pubmed.
    """
    if num_nodes < 1 or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, edges]")
    edge_index = edge_index.to(device="cpu", dtype=torch.long)
    if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= num_nodes):
        raise ValueError("Edge index outside node range")
    loops = torch.arange(num_nodes, dtype=torch.long)
    idx = torch.cat([edge_index, edge_index.flip(0), torch.stack((loops, loops))], 1)
    raw = torch.sparse_coo_tensor(
        idx, torch.ones(idx.shape[1]), (num_nodes, num_nodes)).coalesce()
    index = raw.indices()
    binary_values = torch.ones(index.shape[1], dtype=torch.float32)
    degree = torch.zeros(num_nodes, dtype=torch.float32)
    degree.scatter_add_(0, index[0], binary_values)
    degree_inverse_sqrt = degree.clamp(min=1).pow(-0.5)
    values = degree_inverse_sqrt[index[0]] * degree_inverse_sqrt[index[1]]
    adjacency = torch.sparse_coo_tensor(
        index, values, (num_nodes, num_nodes)).coalesce()
    return adjacency.to(device)


class GCN(nn.Module):
    """One sparse propagation per layer; embedding is last hidden activation."""
    def __init__(self, num_features, num_classes, hidden=64, depth=2, dropout=0.5):
        super().__init__()
        if min(num_features, num_classes, hidden) < 1 or depth < 2:
            raise ValueError("Need positive dimensions and depth >= 2")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        sizes = [num_features] + [hidden] * (depth - 1) + [num_classes]
        self.layers = nn.ModuleList(
            [nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adjacency, return_embedding=False):
        if adjacency.layout != torch.sparse_coo:
            raise ValueError("Expected coalesced sparse COO adjacency")
        for layer in self.layers[:-1]:
            x = self.dropout(x)
            x = torch.sparse.mm(adjacency, layer(x)).relu()
        embedding = x
        x = torch.sparse.mm(adjacency, self.layers[-1](self.dropout(x)))
        if return_embedding:
            return x, embedding
        return x


CONFIGS = {
    "baseline": {"hidden": 64, "depth": 2, "dropout": 0.5},
    "deep": {"hidden": 64, "depth": 3, "dropout": 0.5},
    "wide": {"hidden": 128, "depth": 2, "dropout": 0.5},
    "no_dropout": {"hidden": 64, "depth": 2, "dropout": 0.0},
}
