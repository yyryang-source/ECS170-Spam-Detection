from local_code.base_class.method import method
from local_code.stage5.training import train_graph


class Method_GCN(method):
    def __init__(self, config, output_dir):
        super().__init__("GCN", "Custom PyTorch graph convolutional network")
        self.config, self.output_dir = config, output_dir

    def run(self):
        return train_graph(self.data, self.config, self.output_dir)
