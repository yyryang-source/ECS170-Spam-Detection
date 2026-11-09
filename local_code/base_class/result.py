"""Base ResultModule class (ECS170 instructor template)."""
import abc

class result:
    data = None
    result_name = None
    result_description = None
    result_destination_folder_path = None
    result_destination_file_name = None

    def __init__(self, rName=None, rType=None):
        self.result_name = rName
        self.result_description = rType

    @abc.abstractmethod
    def save(self):
        return

    @abc.abstractmethod
    def load(self):
        return
