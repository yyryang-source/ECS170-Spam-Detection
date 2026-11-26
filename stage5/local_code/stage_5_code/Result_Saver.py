import json
from pathlib import Path
from local_code.base_class.result import result


class Result_Saver(result):
    def __init__(self, path):
        super().__init__("GCN results", "JSON")
        self.path = Path(path)

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2) + "\n")

    def load(self):
        self.data = json.loads(self.path.read_text())
        return self.data
