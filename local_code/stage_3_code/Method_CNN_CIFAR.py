"""CIFAR-10 object-recognition MethodModule, following the official code template."""
from local_code.stage_3_code.Method_CNN import Method_CNN


class Method_CNN_CIFAR(Method_CNN):
    def __init__(self, mName="CNN CIFAR", mDescription="CIFAR-10 object recognition"):
        super().__init__(mName, mDescription, dataset_key="cifar10")
