from local_code.stage_5_code.Method_GCN import Method_GCN


class Method_GCN_Citeseer(Method_GCN):
    def __init__(self, opts, out_dir):
        super().__init__("citeseer", opts, out_dir)
