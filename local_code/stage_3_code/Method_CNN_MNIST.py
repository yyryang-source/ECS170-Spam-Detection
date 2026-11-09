"""MNIST MethodModule, following Method_MLP.py from the official code template."""
from local_code.stage_3_code.Method_CNN import Method_CNN


class Method_CNN_MNIST(Method_CNN):
    def __init__(self, mName="CNN MNIST", mDescription="MNIST handwritten digit recognition"):
        super().__init__(mName, mDescription, dataset_key="mnist")
