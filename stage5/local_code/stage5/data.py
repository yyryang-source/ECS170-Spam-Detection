"""Read the instructor's text graphs and construct the required balanced splits."""
from pathlib import Path
import hashlib
import numpy as np
import scipy.sparse as sp
import torch


def split_indices(labels, dataset, seed=42):
    """Fixed class-balanced split: 20 train, 150/200 test, 10 validation per class.

    ReadMe.pdf specifies train/test sizes. Validation is a documented extension.
    Sampling is without replacement. Class labels are used only for stratification.
    """
    if dataset not in {"cora", "citeseer", "pubmed"}:
        raise ValueError(f"Unsupported dataset: {dataset}")
    rng = np.random.default_rng(seed)
    test_count = 150 if dataset == "cora" else 200
    parts = {"idx_train": [], "idx_val": [], "idx_test": []}
    for c in np.unique(labels):
        nodes = rng.permutation(np.flatnonzero(labels == c))
        if len(nodes) < 20 + test_count + 10:
            raise ValueError(f"Class {c} has too few nodes for the requested split")
        parts["idx_train"].extend(nodes[:20])
        parts["idx_test"].extend(nodes[20:20 + test_count])
        parts["idx_val"].extend(nodes[20 + test_count:30 + test_count])
    return {k: np.sort(np.asarray(v, dtype=np.int64)) for k, v in parts.items()}


def normalized_adjacency(edges, n):
    """Match the supplied loader: symmetrize with max, then add I (keep raw loops)."""
    adjacency = sp.coo_matrix((np.ones(len(edges), dtype=np.float32),
                               (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
    adjacency = adjacency.maximum(adjacency.T)
    adjacency = adjacency + sp.eye(n, dtype=np.float32, format="csr")
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    inv = sp.diags(np.power(degrees, -0.5))
    return (inv @ adjacency @ inv).astype(np.float32).tocoo()


def load_graph(data_root, name, split_seed=42, normalize_features=True):
    root = Path(data_root)
    if (root / "stage_5_data").is_dir():
        root = root / "stage_5_data"
    folder = root / name
    if not (folder / "node").is_file():
        raise FileNotFoundError(f"Expected {folder / 'node'}; extract stage_5_data.zip first")
    ids, raw_features, raw_labels = [], [], []
    with (folder / "node").open() as handle:
        for line in handle:
            tokens = line.split()
            if not tokens:
                continue
            ids.append(tokens[0])
            raw_features.append(np.asarray(tokens[1:-1], dtype=np.float32))
            raw_labels.append(tokens[-1])
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate node IDs")
    features = np.stack(raw_features)
    if not np.isfinite(features).all() or (features < 0).any():
        raise ValueError("Features must be finite and nonnegative")
    classes = sorted(set(raw_labels))
    label_map = {name: i for i, name in enumerate(classes)}
    labels = np.asarray([label_map[v] for v in raw_labels], dtype=np.int64)
    idx_map = {node_id: i for i, node_id in enumerate(ids)}
    raw_edges = np.loadtxt(folder / "link", dtype=str, ndmin=2)
    if raw_edges.shape[1] != 2:
        raise ValueError("Each link must contain two node IDs")
    try:
        edges = np.asarray([[idx_map[a], idx_map[b]] for a, b in raw_edges], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Link references unknown node {exc}") from exc
    adjacency = normalized_adjacency(edges, len(ids))
    if normalize_features:
        totals = features.sum(axis=1, keepdims=True)
        features = np.divide(features, totals, out=np.zeros_like(features), where=totals != 0)
    splits = split_indices(labels, name, split_seed)
    undirected = {(min(a, b), max(a, b)) for a, b in edges if a != b}
    used = np.unique(edges[edges[:, 0] != edges[:, 1]])
    stats = {
        "dataset": name, "nodes": len(ids), "features": features.shape[1],
        "classes": classes, "class_counts": np.bincount(labels).tolist(),
        "raw_link_rows": len(edges), "undirected_nonself_edges": len(undirected),
        "raw_self_loops": int((edges[:, 0] == edges[:, 1]).sum()),
        "isolated_nodes_excluding_self": len(ids) - len(used),
        "normalized_adjacency_nnz": adjacency.nnz,
        "split_seed": split_seed,
        "split_counts": {k: len(v) for k, v in splits.items()},
        "split_class_counts": {k: np.bincount(labels[v], minlength=len(classes)).tolist() for k, v in splits.items()},
        "feature_normalization": "row_l1" if normalize_features else "none",
        "self_loop_policy": "preserve raw loops, then add identity (matches supplied loader)",
        "sha256": {f: hashlib.sha256((folder / f).read_bytes()).hexdigest() for f in ("node", "link")},
    }
    return {
        "graph": {"node": idx_map, "node_ids": ids, "edge": edges,
                  "X": torch.from_numpy(features), "y": torch.from_numpy(labels),
                  "utility": {"A_scipy": adjacency, "reverse_idx": dict(enumerate(ids))}},
        "train_test_val": {k: torch.from_numpy(v) for k, v in splits.items()},
        "metadata": stats,
    }
