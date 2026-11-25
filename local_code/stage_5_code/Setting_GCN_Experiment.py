"""Instructor's Setting.prepare -> load/run/save/evaluate pattern."""
from local_code.base_class.setting import setting


class Setting_GCN_Experiment(setting):
    def __init__(self):
        super().__init__("Stage 5 citation node classification", "Transductive GCN")

    def load_run_save_evaluate(self):
        graph = self.dataset.load()
        self.method.data = graph
        learned_result = self.method.run()
        self.result.data = learned_result
        self.result.save()
        self.evaluate.data = learned_result
        return self.evaluate.evaluate(), None
