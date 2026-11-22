"""Instructor-template ResultModule for one Stage 4 experiment."""
import json
from pathlib import Path

from local_code.base_class.result import result


class Result_Saver(result):
    data = None
    result_destination_folder_path = None
    result_destination_file_name = "template_summary.json"

    def _file(self):
        if not self.result_destination_folder_path:
            raise ValueError("Set result_destination_folder_path first")
        root = Path(self.result_destination_folder_path)
        root.mkdir(parents=True, exist_ok=True)
        return root / self.result_destination_file_name

    def save(self):
        if self.data is None:
            raise ValueError("Set result.data first")
        destination = self._file()
        destination.write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False),
            encoding="utf-8")
        return destination

    def load(self):
        return json.loads(self._file().read_text(encoding="utf-8"))
