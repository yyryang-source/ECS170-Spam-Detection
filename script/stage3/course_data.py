"""Read the instructor's prepartitioned Stage 3 archive."""

import pickle
from zipfile import ZipFile

import numpy as np
from numpy.core.multiarray import _reconstruct, scalar


COURSE_FILES = {"mnist": "MNIST", "orl": "ORL", "cifar10": "CIFAR"}


class CourseUnpickler(pickle.Unpickler):
    # The course files store NumPy images and integer labels. Never import
    # other globals or resolve external objects from the pickle.
    allowed = {
        ("numpy.core.multiarray", "_reconstruct"): _reconstruct,
        ("numpy.core.multiarray", "scalar"): scalar,
        ("numpy._core.multiarray", "_reconstruct"): _reconstruct,
        ("numpy._core.multiarray", "scalar"): scalar,
        ("numpy", "ndarray"): np.ndarray,
        ("numpy", "dtype"): np.dtype,
    }

    def find_class(self, module, name):
        if (module, name) not in self.allowed:
            raise pickle.UnpicklingError(f"Unsupported dataset global: {module}.{name}")
        return self.allowed[module, name]


def load_course_archive(path, name):
    member = f"stage_3_data/{COURSE_FILES[name]}"
    with ZipFile(path) as archive, archive.open(member) as stream:
        data = CourseUnpickler(stream).load()

    if not isinstance(data, dict) or data.keys() != {"train", "test"}:
        raise ValueError(f"{member} must contain the instructor's train and test splits")

    def arrays(split):
        records = data[split]
        images = np.stack([record["image"] for record in records])
        labels = np.array([record["label"] for record in records])
        if labels.ndim != 1 or labels.dtype.kind not in "iu":
            raise ValueError(f"{member}/{split} must have one integer label per image")
        pixels = images.astype(np.uint8, copy=False)
        if not np.array_equal(pixels, images):
            raise ValueError(f"{member}/{split} must contain integer pixels from 0 to 255")
        return pixels, labels

    return arrays("train"), arrays("test")
