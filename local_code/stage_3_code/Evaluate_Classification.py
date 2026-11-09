"""Macro-averaged multiclass metrics, following ECS170 Evaluate_Accuracy."""
from local_code.base_class.evaluate import evaluate
from script.cnn.common import classification_metrics


class Evaluate_Classification(evaluate):
    data = None

    def evaluate(self):
        if self.data is None:
            raise ValueError("Set evaluator.data first")
        return classification_metrics(
            self.data["true_y"], self.data["pred_y"], average="macro"
        )
