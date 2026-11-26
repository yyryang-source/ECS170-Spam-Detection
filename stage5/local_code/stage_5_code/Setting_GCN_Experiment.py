from local_code.base_class.setting import setting


class Setting_GCN_Experiment(setting):
    def __init__(self):
        super().__init__("Stage 5", "Transductive node classification")

    def load_run_save_evaluate(self):
        self.method.data = self.dataset.load()
        metrics = self.method.run()
        self.evaluate.data = metrics
        self.result.data = {"metrics": metrics, "evaluation": self.evaluate.evaluate()}
        self.result.save()
        return metrics
