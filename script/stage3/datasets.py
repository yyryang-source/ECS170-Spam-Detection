"""Dataset loaders for the three datasets required by ECS 170 Stage 3.

'standard': official torchvision MNIST/CIFAR10 (a runnable reference).
'local': instructor-supplied class folders or NPZ, preserving any train/test split.
ORL requires local files; no alternate face dataset is silently substituted.
"""
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import datasets, transforms

NAMES = ("mnist", "orl", "cifar10")


def sample_to_image(array, dataset):
    a = np.asarray(array)
    if a.ndim == 1:
        if dataset == "mnist" and a.size == 784:
            a = a.reshape(28, 28)
        elif dataset == "orl" and a.size == 10304:
            a = a.reshape(112, 92)
        elif dataset == "orl" and a.size == 4096:
            a = a.reshape(64, 64)
        elif dataset == "cifar10" and a.size == 3072:
            a = a.reshape(3, 32, 32).transpose(1, 2, 0)
        else:
            raise ValueError(f"Unsupported flattened image size {a.size} for {dataset}")
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[:, :, 0]
    if a.dtype != np.uint8:
        a = a.astype(np.float32)
        if a.min() >= 0 and a.max() <= 1.01:
            a = a * 255
        a = np.uint8(np.clip(a, 0, 255))
    return Image.fromarray(a)


class NpzDataset(Dataset):
    def __init__(self, x, y, kind, label_map):
        if len(x) != len(y):
            raise ValueError("Image count != label count")
        self.x = x
        self.targets = [label_map[int(label)] for label in y]
        self.kind = kind

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return sample_to_image(self.x[index], self.kind), self.targets[index]


class TransformSubset(Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, label = self.dataset[self.indices[index]]
        return self.transform(image), int(label)


def transforms_for(name, augment=True):
    if name == "mnist":
        steps = [transforms.Resize((28, 28)), transforms.Grayscale(num_output_channels=1)]
        if augment:
            steps += [transforms.RandomAffine(10, translate=(0.05, 0.05))]
        return transforms.Compose(steps + [
            transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    if name == "orl":
        steps = [transforms.Resize((64, 64)), transforms.Grayscale(num_output_channels=1)]
        if augment:
            steps += [transforms.RandomAffine(5, translate=(0.03, 0.03))]
        return transforms.Compose(steps + [
            transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    if name == "cifar10":
        steps = [transforms.Resize((32, 32)), transforms.Lambda(lambda im: im.convert("RGB"))]
        if augment:
            steps += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        return transforms.Compose(steps + [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    raise ValueError(name)


def _targets(ds):
    return np.asarray(ds.targets, dtype=int)


def _partition(indices, labels, test_size, seed):
    labels = np.asarray(labels)[np.asarray(indices)]
    stratify = labels if min(np.unique(labels, return_counts=True)[1]) >= 2 else None
    return train_test_split(indices, test_size=test_size, stratify=stratify, random_state=seed)


def _npz_load(path, name):
    with np.load(path, allow_pickle=False) as f:
        keys = {k.lower(): k for k in f.files}
        def value(*names):
            for n in names:
                if n in keys:
                    return f[keys[n]]
            return None
        x_train, y_train = value("x_train", "train_images"), value("y_train", "train_labels")
        x_test, y_test = value("x_test", "test_images"), value("y_test", "test_labels")
        if x_train is None:
            x_train, y_train = value("x", "images"), value("y", "labels")
        if x_train is None or y_train is None:
            raise ValueError("NPZ needs x_train/y_train (+ optional x_test/y_test) or images/labels")
        if (x_test is None) != (y_test is None):
            raise ValueError("Test images and labels must be provided together")
        labels = np.unique(np.concatenate([y_train, y_test]) if y_test is not None else y_train)
        label_map = {int(label): i for i, label in enumerate(labels)}
        tr = NpzDataset(x_train, y_train, name, label_map)
        te = NpzDataset(x_test, y_test, name, label_map) if x_test is not None else None
    return tr, te, [str(x) for x in labels]


def _local_datasets(root, name):
    root = Path(root)
    if root.is_file() and root.suffix.lower() == ".npz":
        return _npz_load(root, name)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset not found at {root}. Supply an NPZ or image folders via --data-root.")
    if (root/"train").is_dir() and (root/"test").is_dir():
        train, test = datasets.ImageFolder(root/"train"), datasets.ImageFolder(root/"test")
        if train.classes != test.classes:
            raise ValueError("Class folder names differ between local train/ and test/")
        return train, test, train.classes
    train = datasets.ImageFolder(root)
    return train, None, train.classes


def load_stage3_data(name, source, data_root, seed=42, quick=False):
    """Return transformed train/validation/test splits, class names, provenance."""
    if name not in NAMES:
        raise ValueError(f"Dataset must be one of {NAMES}")
    if name == "orl" and source != "local":
        raise ValueError("ORL requires instructor-provided local files (--source local)")
    if source == "standard":
        if name == "mnist":
            train_raw = datasets.MNIST(root=str(data_root), train=True, download=True)
            test_raw = datasets.MNIST(root=str(data_root), train=False, download=True)
        elif name == "cifar10":
            train_raw = datasets.CIFAR10(root=str(data_root), train=True, download=True)
            test_raw = datasets.CIFAR10(root=str(data_root), train=False, download=True)
        else:
            raise ValueError(name)
        classes = [str(x) for x in train_raw.classes]
    elif source == "local":
        train_raw, test_raw, classes = _local_datasets(data_root, name)
    else:
        raise ValueError(source)

    if len(classes) < 2:
        raise ValueError("Classification requires at least two classes")
    train_ids = np.arange(len(train_raw))
    if test_raw is None:
        train_ids, test_ids = _partition(train_ids, _targets(train_raw), 0.20, seed)
        test_raw = train_raw
    else:
        test_ids = np.arange(len(test_raw))

    # ORL has only ~10 images per identity; preserve more images for validation.
    val_fraction = 0.25 if name == "orl" else 0.10
    train_ids, val_ids = _partition(train_ids, _targets(train_raw), val_fraction, seed)

    if quick:
        rng = np.random.default_rng(seed)
        train_ids = rng.permutation(train_ids)[:min(len(train_ids), 256)]
        val_ids = rng.permutation(val_ids)[:min(len(val_ids), 96)]
        test_ids = rng.permutation(test_ids)[:min(len(test_ids), 96)]

    fit = TransformSubset(train_raw, train_ids, transforms_for(name, True))
    val = TransformSubset(train_raw, val_ids, transforms_for(name, False))
    test = TransformSubset(test_raw, test_ids, transforms_for(name, False))
    return (fit, val, test), classes, {
        "source": source, "data_root": str(data_root),
        "train": len(fit), "validation": len(val), "test": len(test),
        "channels": 3 if name == "cifar10" else 1,
        "prepartitioned_test": (source == "standard" or
                                (source == "local" and train_raw is not test_raw)),
    }
