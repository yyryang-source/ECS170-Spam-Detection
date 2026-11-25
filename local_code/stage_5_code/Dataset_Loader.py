"""Stage 5 Dataset_Loader inherits the instructor's dataset base class."""
from local_code.base_class.dataset import dataset
from script.stage5.data import load_graph


class Dataset_Loader(dataset):
    def __init__(self, dName, source, data_root, seed=42,
                 allow_generated_split=False):
        super().__init__(dName, "Citation network with labeled nodes")
        self.source = source
        self.data_root = data_root
        self.seed = seed
        self.allow_generated_split = allow_generated_split
        self.data = None

    def load(self):
        self.data = load_graph(
            self.dataset_name, self.source, self.data_root,
            seed=self.seed, allow_generated_split=self.allow_generated_split)
        return self.data
