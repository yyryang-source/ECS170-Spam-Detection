from local_code.stage_5_code.Method_GCN import Method_GCN


class Method_GCN_Cora(Method_GCN):
    def __init__(self, opts, out_dir):
        super().__init__("cora", opts, out_dir)
