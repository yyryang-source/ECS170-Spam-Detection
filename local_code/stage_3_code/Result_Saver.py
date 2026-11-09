"""ECS170 ResultModule implementation for reproducible Stage 3 results."""
import json
from pathlib import Path
from local_code.base_class.result import result


class Result_Saver(result):
    data = None
    result_destination_folder_path = None
    result_destination_file_name = "stage_3_experiment.json"

    def _output_file(self):
        root = Path(self.result_destination_folder_path)
        root.mkdir(parents=True, exist_ok=True)
        return root / self.result_destination_file_name

    def save(self):
        if self.data is None:
            raise ValueError("Set result.data before saving")
        path = self._output_file()
        path.write_text(json.dumps(self.data, indent=2, default=str),
                        encoding="utf-8")
        print("saving results...", path)

    def load(self):
        return json.loads(self._output_file().read_text(encoding="utf-8"))
