"""Stage 5 GCN method using the instructor's MethodModule API."""
from local_code.base_class.method import method
from script.stage5.train import run_experiments


class Method_GCN(method):
    data = None

    def __init__(self, dataset, opts, out_dir):
        super().__init__(f"GCN_{dataset}", "Sparse graph convolutional neural network")
        self.dataset_name = dataset
        self.options = opts
        self.out_dir = out_dir

    def run(self, trainData=None, trainLabel=None, testData=None):
        if self.data is None:
            raise ValueError("Setting must supply loaded graph to method.data")
        return run_experiments(
            self.data, self.out_dir, self.options.config_names, self.options)
