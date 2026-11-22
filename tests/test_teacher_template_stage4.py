"""Exercise instructor-template Stage 4 inheritance and real training pipeline.

Small fabricated texts test program wiring only; never report these scores.
"""
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from local_code.base_class.dataset import dataset
from local_code.base_class.method import method
from local_code.base_class.result import result
from local_code.base_class.evaluate import evaluate
from local_code.base_class.setting import setting

from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Evaluate_RNN import Evaluate_RNN
from local_code.stage_4_code.Method_RNN import Method_RNN
from local_code.stage_4_code.Result_Saver import Result_Saver
from local_code.stage_4_code.Setting_RNN_Experiment import Setting_RNN_Experiment
from script.stage_4_script.script_rnn import main


class Stage4TemplateTests(unittest.TestCase):
    def test_instructor_base_classes(self):
        for actual, base in (
            (Dataset_Loader, dataset), (Method_RNN, method),
            (Result_Saver, result), (Evaluate_RNN, evaluate),
            (Setting_RNN_Experiment, setting),
        ):
            self.assertTrue(issubclass(actual, base))

    def test_evaluation_selects_only_using_validation(self):
        e = Evaluate_RNN("scorer", "")
        e.data = {"training": {
            "classification": [
                {"model": "rnn", "best_val_macro_f1": .83, "accuracy": .71,
                 "macro_precision": .72, "macro_recall": .71, "macro_f1": .71},
                {"model": "gru", "best_val_macro_f1": .81, "accuracy": .99,
                 "macro_precision": .99, "macro_recall": .99, "macro_f1": .99},
            ],
            "generation": [
                {"model": "rnn", "loss": 4.9, "perplexity": 134.29, "token_accuracy": .2},
                {"model": "gru", "loss": 4.8, "perplexity": 121.51, "token_accuracy": .1},
            ]
        }}
        chosen = e.evaluate()
        self.assertEqual(chosen["classification"]["selected_model"], "rnn")
        self.assertEqual(chosen["generation"]["selected_model"], "gru")
        self.assertFalse(chosen["generation"]["official_test_set"])

    def test_end_to_end_instructor_style_quick(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "stage_4_data.zip"
            with ZipFile(archive, "w") as z:
                for split in ("train", "test"):
                    for label in ("pos", "neg"):
                        for i in range(6):
                            z.writestr(
                                f"stage_4_data/text_classification/{split}/{label}/{i}.txt",
                                f"This is a {label} review number {i}."
                            )
                rows = "ID,Joke\n" + "".join(
                    f'{i},"The {i} cat sits on a mat and tells a joke."\n'
                    for i in range(20)
                )
                z.writestr("stage_4_data/text_generation/data", rows)
            output = root / "out"
            scores = main([
                "--data", str(archive), "--output-dir", str(output),
                "--quick", "--device", "cpu", "--models", "rnn",
                "--embedding-dim", "8", "--hidden-dim", "12",
                "--batch-size", "4", "--max-vocab", "100",
                "--max-joke-vocab", "100",
            ])
            self.assertIn("classification", scores)
            self.assertIn("generation", scores)
            for task in ("classification", "generation"):
                self.assertTrue((output / task / "model_architecture.png").is_file())
                self.assertTrue((output / task / "vocab.json").is_file())
            self.assertTrue((output / "template_summary.json").is_file())
            self.assertTrue((output / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
