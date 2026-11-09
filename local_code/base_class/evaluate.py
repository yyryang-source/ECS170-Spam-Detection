"""Base Evaluation class (ECS170 instructor template)."""
import abc

class evaluate:
    evaluate_name = None
    evaluate_description = None
    data = None

    def __init__(self, eName=None, eDescription=None):
        self.evaluate_name = eName
        self.evaluate_description = eDescription

    @abc.abstractmethod
    def evaluate(self):
        return
