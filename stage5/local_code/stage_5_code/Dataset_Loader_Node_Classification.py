"""Adapted instructor loader with deterministic labels and assignment-required splits."""
from local_code.base_class.dataset import dataset
from local_code.stage5.data import load_graph
from local_code.stage5.model import GraphOperator


class Dataset_Loader(dataset):
    def __init__(self, seed=42, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
        self.seed = 42 if seed is None else seed
        self.normalize_features = True

    def load(self):
        from pathlib import Path
        folder = Path(self.dataset_source_folder_path)
        root = folder.parent if folder.name == self.dataset_name else folder
        result = load_graph(root, self.dataset_name, self.seed, self.normalize_features)
        # Preserve the graph['utility']['A'] tensor contract from the provided loader.
        result["graph"]["utility"]["A"] = GraphOperator.from_scipy(result["graph"]["utility"]["A_scipy"], "cpu").sparse
        return result
