"""Stage 3 Dataset_Loader: inherits the official ECS170 dataset base class.

Supports the instructor ZIP, class folders, NPZ files, and public MNIST/CIFAR-10
reference downloads. ORL is deliberately local-only: do not replace faces with
an unrelated benchmark.
"""
from pathlib import Path
from local_code.base_class.dataset import dataset
from script.stage3.datasets import load_stage3_data


class Dataset_Loader(dataset):
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None,
                 dataset_key="mnist", source="standard", seed=42,
                 quick=False):
        super().__init__(dName, dDescription)
        self.dataset_key = dataset_key
        self.source = source
        self.seed = seed
        self.quick = quick

    def load(self):
        if not self.dataset_source_folder_path:
            raise ValueError("Set dataset_source_folder_path before load().")
        root = Path(self.dataset_source_folder_path)
        if self.dataset_source_file_name:
            root = root / self.dataset_source_file_name
        partitions, classes, provenance = load_stage3_data(
            name=self.dataset_key, source=self.source, data_root=root,
            seed=self.seed, quick=self.quick)
        fit, validation, test = partitions
        self.data = {
            "train": fit, "validation": validation, "test": test,
            "classes": classes, "provenance": provenance,
        }
        print("loading data...", self.dataset_key,
              "train=", len(fit), "validation=", len(validation),
              "test=", len(test))
        return self.data
