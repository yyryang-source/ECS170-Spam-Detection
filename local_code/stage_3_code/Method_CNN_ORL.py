"""ORL face-recognition MethodModule, following the official code template."""
from local_code.stage_3_code.Method_CNN import Method_CNN


class Method_CNN_ORL(Method_CNN):
    # Keep coarse spatial information about the aligned facial features.
    pool_size = 4

    def __init__(self, mName="CNN ORL", mDescription="ORL face recognition"):
        super().__init__(mName, mDescription, dataset_key="orl")
