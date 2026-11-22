"""Instructor-template DatasetModule for both supplied Stage 4 text datasets."""
from pathlib import Path

from local_code.base_class.dataset import dataset
from script.stage4.data import CourseData


class Dataset_Loader(dataset):
    """Validate and describe instructor data without modifying its train/test split."""

    dataset_source_folder_path = "data/stage_4_data.zip"

    def load(self):
        path = Path(self.dataset_source_folder_path).expanduser()
        with CourseData(path) as source:
            counts = {
                f"imdb_{split}_{sentiment}": len(source.review_paths(split, sentiment))
                for split in ("train", "test")
                for sentiment in ("neg", "pos")
            }
            counts["jokes"] = len(source.load_jokes())
        if any(value == 0 for value in counts.values()):
            raise ValueError(f"Missing IMDb class/split or jokes in instructor ZIP: {counts}")
        self.data = {"source_path": str(path), "counts": counts,
                     "classification_test_is_official": True,
                     "generation_test_is_official": False}
        return self.data
