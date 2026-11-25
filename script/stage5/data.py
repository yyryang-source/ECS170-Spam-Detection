"""ECS170 Stage 5 citation-graph datasets and safe, explicit split validation."""
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile
import hashlib
import io

import numpy as np
import torch
from sklearn.model_selection import train_test_split

NAMES = ("cora", "citeseer", "pubmed")
PLANETOID = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}


@dataclass
class GraphData:
    x: torch.Tensor
    y: torch.Tensor
    edge_index: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    provenance: dict

    @property
    def num_classes(self):
        return int(torch.unique(self.y).numel())


def _get(data, *keys):
    for key in keys:
        if key in data:
            return np.asarray(data[key])
    return None


def _mask(data, part, n):
    raw = _get(data, part + "_mask")
    if raw is not None:
        if raw.shape != (n,) or raw.dtype.kind not in "biu" or not np.isin(raw, [0, 1]).all():
            raise ValueError(f"Invalid {part}_mask; expected {n} Boolean/0-1 entries")
        return raw.astype(bool)
    idx = _get(data, part + "_idx", part + "_indices")
    if idx is None:
        return None
    if idx.ndim != 1 or idx.dtype.kind not in "iu" or np.any(idx < 0) or np.any(idx >= n):
        raise ValueError(f"Invalid {part} indices")
    if len(np.unique(idx)) != len(idx):
        raise ValueError(f"Duplicate {part} indices")
    out = np.zeros(n, dtype=bool)
    out[idx.astype(np.int64)] = True
    return out


def _splits(data, labels, seed, allow_generated):
    n = len(labels)
    masks = {p: _mask(data, p, n) for p in ("train", "val", "test")}
    if all(x is not None for x in masks.values()):
        origin = "provided"
    elif any(x is not None for x in masks.values()):
        raise ValueError("Partial train/val/test split; need all three masks or indices")
    elif allow_generated:
        ids = np.arange(n)
        train, rest = train_test_split(ids, train_size=0.60, stratify=labels, random_state=seed)
        val, test = train_test_split(rest, train_size=0.50, stratify=labels[rest], random_state=seed)
        masks = {}
        for part, indices in (("train", train), ("val", val), ("test", test)):
            mask = np.zeros(n, dtype=bool)
            mask[indices] = True
            masks[part] = mask
        origin = "generated_stratified_60_20_20"
    else:
        raise ValueError("Missing splits. Instructor masks required or use --allow-generated-split.")
    if any(not x.any() for x in masks.values()):
        raise ValueError("Every split must contain at least one node")
    overlap = (masks["train"] & masks["val"]) | (masks["train"] & masks["test"]) | (masks["val"] & masks["test"])
    if overlap.any():
        raise ValueError("Training, validation, test splits overlap")
    return masks, origin


def _assemble(data, provenance, seed, allow_generated_split):
    x, y, edges = _get(data, "x", "features"), _get(data, "y", "labels"), _get(data, "edge_index", "edges")
    if x is None or y is None or edges is None:
        raise ValueError("Need numeric x, y and edge_index arrays")
    if x.ndim != 2 or not len(x) or x.dtype.kind not in "biuf" or not np.isfinite(x).all():
        raise ValueError("Invalid non-finite/non-numeric node feature matrix")
    if y.ndim != 1 or y.shape[0] != x.shape[0] or y.dtype.kind not in "iu":
        raise ValueError("Labels must contain one integer per node")
    if edges.ndim != 2 or edges.dtype.kind not in "iu":
        raise ValueError("edge_index must be an integer matrix")
    if edges.shape[0] != 2 and edges.shape[1] == 2:
        edges = edges.T
    if edges.shape[0] != 2 or edges.shape[1] < 1 or edges.min() < 0 or edges.max() >= len(y):
        raise ValueError("Invalid or out-of-range graph edges")
    classes, remapped = np.unique(y, return_inverse=True)
    if len(classes) < 2:
        raise ValueError("At least two node classes are required")
    masks, origin = _splits(data, remapped, seed, allow_generated_split)
    features = x.astype(np.float32)
    if (features >= 0).all():
        denom = features.sum(axis=1, keepdims=True)
        features = features / np.maximum(denom, 1e-12)
        provenance["feature_normalization"] = "row_sum"
    else:
        provenance["feature_normalization"] = "none"
    provenance.update({
        "nodes": len(y), "features": x.shape[1], "edges_raw": edges.shape[1],
        "classes": len(classes), "class_labels": [int(v) for v in classes],
        "split_origin": origin,
        **{p: int(masks[p].sum()) for p in ("train", "val", "test")}
    })
    return GraphData(
        torch.from_numpy(np.ascontiguousarray(features)),
        torch.from_numpy(remapped.astype(np.int64)),
        torch.from_numpy(np.ascontiguousarray(edges.astype(np.int64))),
        *(torch.from_numpy(masks[p]) for p in ("train", "val", "test")),
        provenance
    )


def _local(path, name):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing instructor graph file: {path}")
    if path.suffix.lower() == ".npz":
        payload = path.read_bytes()
    elif path.suffix.lower() == ".zip":
        with ZipFile(path) as archive:
            matched = [n for n in archive.namelist()
                       if n.lower().endswith(name + ".npz") and not n.startswith("__MACOSX/")]
            if len(matched) != 1:
                raise ValueError(f"ZIP must contain exactly one {name}.npz; found {matched}")
            payload = archive.read(matched[0])
    else:
        raise ValueError("Local data must be .npz or .zip containing named .npz files")
    with np.load(io.BytesIO(payload), allow_pickle=False) as raw:
        values = {k: raw[k] for k in raw.files}
    return values, {"source": "instructor_local_unverified",
                    "path": str(path), "sha256": hashlib.sha256(payload).hexdigest()}


def load_graph(name, source="planetoid", data_root="data/stage_5_data",
               seed=42, allow_generated_split=False):
    name = name.lower()
    if name not in NAMES:
        raise ValueError(f"Unknown graph {name!r}; choose {NAMES}")
    if source == "local":
        arrays, provenance = _local(data_root, name)
    elif source == "planetoid":
        try:
            from torch_geometric.datasets import Planetoid
        except ImportError as exc:
            raise RuntimeError("Install torch-geometric for public Planetoid data, or use --source local") from exc
        ds = Planetoid(root=str(Path(data_root) / name), name=PLANETOID[name], split="public")
        graph = ds[0]
        arrays = {
            "x": graph.x.cpu().numpy(), "y": graph.y.cpu().numpy(),
            "edge_index": graph.edge_index.cpu().numpy(),
            "train_mask": graph.train_mask.cpu().numpy(),
            "val_mask": graph.val_mask.cpu().numpy(),
            "test_mask": graph.test_mask.cpu().numpy(),
        }
        provenance = {"source": "public_planetoid_reference", "dataset": name,
                      "root": str(data_root), "instructor_dataset_verified": False}
    else:
        raise ValueError("source must be local or planetoid")
    return _assemble(arrays, provenance, seed, allow_generated_split)
