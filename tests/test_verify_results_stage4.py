"""Offline checks for course result-auditing safeguards."""
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from script.stage4.verify_results import (
    EXPERIMENTS, EvidenceError, create_submission_zip, inspect_archive,
)


def table(rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def create_synthetic_archive(path, *, wrong_f1=False):
    with ZipFile(path, "w") as archive:
        archive.writestr("results/stage4/summary.json", json.dumps({
            "quick": True, "models": ["rnn", "lstm", "gru"]
        }))
        for run in EXPERIMENTS:
            base = f"results/stage4/{run}/"
            archive.writestr(base + "summary.json", json.dumps({
                "quick": False, "models": ["rnn", "lstm", "gru"]
            }))
            for task in ("classification", "generation"):
                root = base + task + "/"
                settings = (
                    {"quick": False, "train": 22500,
                     "validation": 2500, "instructor_test": 25000,
                     "vocab_size": 30000}
                    if task == "classification" else
                    {"quick": False, "train": 1460,
                     "validation": 162, "no_official_test_split": True,
                     "vocab_size": 4491}
                )
                archive.writestr(root + "settings.json", json.dumps(settings))
                vocab = {f"word{i}": i for i in range(settings["vocab_size"])}
                archive.writestr(root + "vocab.json", json.dumps(vocab))
                metrics = []
                for model in ("rnn", "lstm", "gru"):
                    history = []
                    for epoch in range(1, 6 if task == "classification" else 31):
                        row = {"epoch": epoch, "train_loss": 1.0,
                               "val_loss": 2.0 if epoch == 1 else 2.2}
                        if task == "classification":
                            row["val_macro_f1"] = .8 if epoch == 1 else .7
                        history.append(row)
                    archive.writestr(root + model + "_history.csv", table(history))
                    archive.writestr(root + model + "_loss.png", b"PNG")
                    if task == "classification":
                        metrics.append({
                            "model": model, "parameters": 1234,
                            "best_val_macro_f1": .1 if wrong_f1 and run == "full" and model == "rnn" else .8,
                            "accuracy": .75, "macro_precision": .75,
                            "macro_recall": .75, "macro_f1": .75
                        })
                    else:
                        metrics.append({
                            "model": model, "parameters": 4321,
                            "loss": 2.0, "perplexity": 7.3890560989,
                            "token_accuracy": .2
                        })
                outfile = "test_metrics.csv" if task == "classification" else "validation_metrics.csv"
                archive.writestr(root + outfile, table(metrics))
                if task == "generation":
                    examples = {
                        model: [{"prompt": "one two three",
                                 "training_reference": "one two three joke",
                                 "generated": "one two three story"} for _ in range(3)]
                        for model in ("rnn", "lstm", "gru")
                    }
                    archive.writestr(root + "generation_examples.json", json.dumps(examples))


class VerifyResultsTests(unittest.TestCase):
    def test_six_runs_36_rows_and_exclude_quick(self):
        with tempfile.TemporaryDirectory() as d:
            raw = Path(d) / "raw.zip"
            cleaned = Path(d) / "clean.zip"
            create_synthetic_archive(raw)
            with ZipFile(raw) as archive:
                result = inspect_archive(archive)
                self.assertEqual(result["verified_model_task_runs"], 36)
                self.assertEqual(result["official_runs"], 6)
                self.assertTrue(result["warnings"])
                self.assertEqual(create_submission_zip(archive, cleaned), 120)
            with ZipFile(cleaned) as archive:
                self.assertNotIn("results/stage4/summary.json", archive.namelist())
                self.assertEqual(inspect_archive(archive)["status"], "pass")

    def test_mismatched_validation_metric_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            raw = Path(d) / "incorrect.zip"
            create_synthetic_archive(raw, wrong_f1=True)
            with ZipFile(raw) as archive:
                with self.assertRaisesRegex(EvidenceError, "best validation F1"):
                    inspect_archive(archive)


if __name__ == "__main__":
    unittest.main()
