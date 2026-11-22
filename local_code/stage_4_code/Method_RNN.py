"""Instructor-template MethodModule reusing the tested Stage 4 PyTorch trainer.

The historical runner and its 36 existing result sets are intentionally unchanged.
"""
import json
from pathlib import Path

from local_code.base_class.method import method
from script.stage4.data import CourseData
from script.stage4.run_stage4 import (
    device_for, set_seed, train_classification, train_generation,
)


class Method_RNN(method):
    """Train RNN/LSTM/GRU on IMDb, jokes, or both, using instructor data."""

    data = None
    args = None
    result_destination_folder_path = None

    def run(self, trainData=None, trainLabel=None, testData=None):
        if self.args is None:
            raise ValueError("Set method.args to the parsed Stage 4 settings")
        source_data = trainData if trainData is not None else self.data
        if not source_data or "source_path" not in source_data:
            raise ValueError("Load instructor data with Dataset_Loader first")
        args = self.args
        output = Path(self.result_destination_folder_path)
        output.mkdir(parents=True, exist_ok=True)
        set_seed(args.seed)
        device = device_for(args.device)
        print(f"Template Stage 4 | device={device} | source={source_data['source_path']}")
        results = {}
        with CourseData(source_data["source_path"]) as source:
            if args.task in ("both", "classification"):
                folder = output / "classification"
                folder.mkdir(exist_ok=True)
                results["classification"] = train_classification(
                    args, source, device, folder)
            if args.task in ("both", "generation"):
                folder = output / "generation"
                folder.mkdir(exist_ok=True)
                results["generation"] = train_generation(
                    args, source, device, folder)
        payload = {
            "source": source_data, "device": str(device),
            "quick": bool(args.quick), "models": list(args.models),
            "training": results,
        }
        (output / "summary.json").write_text(
            json.dumps({"device": str(device), "quick": bool(args.quick),
                        "models": list(args.models), "results": results},
                       indent=2), encoding="utf-8")
        self.data = payload
        return payload
