"""Stage 5 report tests with synthetic summaries; never publish these scores."""
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from script.stage5.build_report import build_report, load_evidence


def build_fixture(root, source="public_planetoid_reference"):
    for name in ("cora", "citeseer", "pubmed"):
        path = Path(root)/name/"full"
        path.mkdir(parents=True)
        fake_scores = {"Model": "baseline", "Accuracy": 0.5,
                       "Precision": 0.5, "Recall": 0.5, "F1": 0.5}
        summary = {
            "dataset": {"source": source, "sha256": "synthetic",
                        "nodes": 90, "features": 12, "edges_raw": 90,
                        "classes": 3, "train": 54, "val": 18, "test": 18},
            "seed": 42, "learning_rate": 0.01, "weight_decay": 0.0005,
            "epochs_max": 2, "early_stopping_patience": 2,
            "quick_smoke_test_not_for_report": False,
            "test_used_for_selection": False,
            "selected_model": "baseline", "final_test": fake_scores,
            "validation": [{**fake_scores, "Parameters": 100}]
        }
        (path/"experiment_summary.json").write_text(json.dumps(summary))
        for figure in ("architecture.png", "best_loss_curve.png",
                       "ablation_val_loss.png", "confusion_matrix.png"):
            Image.new("RGB", (700, 250), "white").save(path/figure)


class TestReportSafety(unittest.TestCase):
    def test_pdf_generated_from_all_three_saved_summaries(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            build_fixture(root)
            pdf = root/"report.pdf"
            evidence = load_evidence(root, "reference")
            build_report(evidence, pdf, "reference")
            self.assertTrue(pdf.read_bytes().startswith(b"%PDF-"))

    def test_reject_wrong_source_and_quick_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            build_fixture(root)
            with self.assertRaises(ValueError):
                load_evidence(root, "instructor")
            summary_file = root/"cora/full/experiment_summary.json"
            data = json.loads(summary_file.read_text())
            data["quick_smoke_test_not_for_report"] = True
            summary_file.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                load_evidence(root, "reference")


if __name__ == "__main__":
    unittest.main()
