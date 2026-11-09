"""Smoke tests for the ECS170 official base-class style Stage 3 implementation.

No external dataset download is required. Synthetic images verify wiring only;
never include their numerical scores in a class report.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from local_code.base_class.dataset import dataset
from local_code.base_class.evaluate import evaluate
from local_code.base_class.method import method
from local_code.base_class.result import result
from local_code.base_class.setting import setting

from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN_MNIST import Method_CNN_MNIST
from local_code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL
from local_code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Evaluate_Classification import Evaluate_Classification
from local_code.stage_3_code.Setting_CNN_Experiment import Setting_CNN_Experiment


class TestOfficialTemplate(unittest.TestCase):
    def test_inheritance(self):
        self.assertTrue(issubclass(Dataset_Loader, dataset))
        self.assertTrue(issubclass(Method_CNN_MNIST, method))
        self.assertTrue(issubclass(Method_CNN_ORL, method))
        self.assertTrue(issubclass(Method_CNN_CIFAR, method))
        self.assertTrue(issubclass(Result_Saver, result))
        self.assertTrue(issubclass(Evaluate_Classification, evaluate))
        self.assertTrue(issubclass(Setting_CNN_Experiment, setting))

    def test_orl_class_folder_end_to_end(self):
        """Run the actual setting.prepare -> load_run_save_evaluate path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = root / "orl"
            for label_idx in range(4):
                class_dir = images / f"s{label_idx + 1}"
                class_dir.mkdir(parents=True)
                for i in range(10):
                    pixels = np.full((64, 64), label_idx * 48 + i,
                                     dtype=np.uint8)
                    Image.fromarray(pixels).save(class_dir / f"{i:02d}.png")

            dataset_obj = Dataset_Loader(
                "ORL smoke", "synthetic test only", dataset_key="orl",
                source="local", quick=True)
            dataset_obj.dataset_source_folder_path = str(images)

            method_obj = Method_CNN_ORL("CNN ORL", "")
            method_obj.max_epoch = 1
            method_obj.batch_size = 8
            method_obj.patience = 1
            method_obj.device_name = "cpu"

            saver_obj = Result_Saver("saver", "")
            saver_obj.result_destination_folder_path = str(root / "output")
            saver_obj.result_destination_file_name = "experiment_summary.json"

            evaluator_obj = Evaluate_Classification("macro metrics", "")
            setting_obj = Setting_CNN_Experiment("official template integration", "")
            setting_obj.configurations = ("baseline",)
            setting_obj.prepare(
                dataset_obj, method_obj, saver_obj, evaluator_obj)
            scores, _ = setting_obj.load_run_save_evaluate()
            self.assertIn("Accuracy", scores)
            self.assertTrue((root / "output" /
                             "experiment_summary.json").is_file())
            self.assertTrue((root / "output" /
                             "model_architecture.png").is_file())
            self.assertTrue((root / "output" /
                             "final_test_metrics.csv").is_file())


if __name__ == "__main__":
    unittest.main()
