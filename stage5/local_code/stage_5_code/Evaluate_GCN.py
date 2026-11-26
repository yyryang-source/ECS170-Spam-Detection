from local_code.base_class.evaluate import evaluate
from local_code.stage5.training import evaluate_predictions


class Evaluate_GCN(evaluate):
    def __init__(self):
        super().__init__("classification metrics", "Accuracy and macro precision/recall/F1")

    def evaluate(self):
        if "truth" in self.data:
            return evaluate_predictions(self.data["truth"], self.data["prediction"], self.data["classes"])
        return {key: self.data[key] for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro")}
