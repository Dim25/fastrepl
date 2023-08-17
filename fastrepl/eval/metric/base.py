from abc import ABC, abstractmethod


class BaseMetric(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def compute(self, predictions, references, **kwargs):
        pass
