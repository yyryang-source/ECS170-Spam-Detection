"""Base SettingModule class (ECS170 instructor template)."""
import abc

class setting:
    setting_name = None
    setting_description = None
    dataset = None
    method = None
    result = None
    evaluate = None

    def __init__(self, sName=None, sDescription=None):
        self.setting_name = sName
        self.setting_description = sDescription

    def prepare(self, sDataset, sMethod, sResult, sEvaluate):
        self.dataset = sDataset
        self.method = sMethod
        self.result = sResult
        self.evaluate = sEvaluate

    def print_setup_summary(self):
        print("dataset:", self.dataset.dataset_name, ", method:", self.method.method_name,
              ", setting:", self.setting_name, ", result:", self.result.result_name,
              ", evaluation:", self.evaluate.evaluate_name)

    @abc.abstractmethod
    def load_run_save_evaluate(self):
        return
