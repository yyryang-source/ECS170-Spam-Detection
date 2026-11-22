"""Instructor-template EvaluateModule; selection never uses IMDb test metrics."""
from local_code.base_class.evaluate import evaluate


class Evaluate_RNN(evaluate):
    data = None

    def evaluate(self):
        if self.data is None:
            raise ValueError("Set evaluator.data to Method_RNN output")
        runs = self.data["training"]
        answer = {}
        if "classification" in runs:
            rows = runs["classification"]
            if not rows:
                raise ValueError("Empty classification metrics")
            # IMPORTANT: the winner is selected on validation macro F1 only.
            selected = max(rows, key=lambda row: row["best_val_macro_f1"])
            answer["classification"] = {
                "selected_model": selected["model"],
                "selection_metric": "validation macro F1",
                "validation_macro_f1": selected["best_val_macro_f1"],
                "official_test": {
                    key: selected[key] for key in
                    ("accuracy", "macro_precision", "macro_recall", "macro_f1")
                },
                "all_models": rows,
            }
        if "generation" in runs:
            rows = runs["generation"]
            if not rows:
                raise ValueError("Empty generation metrics")
            selected = min(rows, key=lambda row: row["loss"])
            answer["generation"] = {
                "selected_model": selected["model"],
                "selection_metric": "validation token cross-entropy",
                "validation": {
                    key: selected[key] for key in
                    ("loss", "perplexity", "token_accuracy")
                },
                "official_test_set": False,
                "all_models": rows,
            }
        self.data = answer
        return answer
