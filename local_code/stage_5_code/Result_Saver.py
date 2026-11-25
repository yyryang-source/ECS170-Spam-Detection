"""Instructor-style ResultModule, exporting a reproducible summary."""
import json
from pathlib import Path
from local_code.base_class.result import result


class Result_Saver(result):
    def __init__(self, folder):
        super().__init__("stage5 result", "GNN experiment summary")
        self.result_destination_folder_path = str(folder)
        self.result_destination_file_name = "template_summary.json"
        self.data = None

    def save(self):
        if self.data is None:
            raise ValueError("No data to save")
        folder = Path(self.result_destination_folder_path)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / self.result_destination_file_name).write_text(
            json.dumps(self.data, indent=2), encoding="utf-8")

    def load(self):
        return json.loads(
            (Path(self.result_destination_folder_path) /
             self.result_destination_file_name).read_text(encoding="utf-8"))
