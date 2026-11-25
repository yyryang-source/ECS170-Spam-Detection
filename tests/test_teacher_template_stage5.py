"""Offline instructor-template Stage 5 integration test."""
import tempfile
import unittest
from pathlib import Path

from tests.test_stage5 import fixture, opts
from local_code.base_class.dataset import dataset
from local_code.base_class.method import method
from local_code.base_class.setting import setting
from local_code.base_class.evaluate import evaluate
from local_code.base_class.result import result
from local_code.stage_5_code.Dataset_Loader import Dataset_Loader
from local_code.stage_5_code.Method_GCN_Cora import Method_GCN_Cora
from local_code.stage_5_code.Method_GCN_Citeseer import Method_GCN_Citeseer
from local_code.stage_5_code.Method_GCN_Pubmed import Method_GCN_Pubmed
from local_code.stage_5_code.Setting_GCN_Experiment import Setting_GCN_Experiment
from local_code.stage_5_code.Evaluate_Node_Classification import Evaluate_Node_Classification
from local_code.stage_5_code.Result_Saver import Result_Saver


class TestTeacherStage5(unittest.TestCase):
    def test_inherits_official_interfaces(self):
        self.assertTrue(issubclass(Dataset_Loader, dataset))
        for cls in (Method_GCN_Cora, Method_GCN_Citeseer, Method_GCN_Pubmed):
            self.assertTrue(issubclass(cls, method))
        self.assertTrue(issubclass(Setting_GCN_Experiment, setting))
        self.assertTrue(issubclass(Evaluate_Node_Classification, evaluate))
        self.assertTrue(issubclass(Result_Saver, result))

    def test_template_flow(self):
        with tempfile.TemporaryDirectory() as folder:
            path = fixture(folder)
            output = Path(folder)/"submission"
            options = opts()
            options.epochs = 2
            options.config_names = ("baseline",)
            ds = Dataset_Loader("cora", "local", str(path))
            model = Method_GCN_Cora(options, output)
            setting_obj = Setting_GCN_Experiment()
            saver = Result_Saver(output)
            scorer = Evaluate_Node_Classification()
            setting_obj.prepare(ds, model, saver, scorer)
            scores, _ = setting_obj.load_run_save_evaluate()
            self.assertIn("Accuracy", scores)
            self.assertTrue((output/"template_summary.json").is_file())
            self.assertEqual(saver.load()["selected_model"], "baseline")


if __name__ == "__main__":
    unittest.main()
