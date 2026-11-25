"""Instructor-style Evaluate; the method's held-out test is scored once."""
from local_code.base_class.evaluate import evaluate


class Evaluate_Node_Classification(evaluate):
    def __init__(self):
        super().__init__("macro node classification", "Accuracy, macro P/R/F1")
        self.data = None

    def evaluate(self):
        if self.data is None or "final_test" not in self.data:
            raise ValueError("Missing saved test evaluation")
        return {key: self.data["final_test"][key]
                for key in ("Accuracy", "Precision", "Recall", "F1")}
