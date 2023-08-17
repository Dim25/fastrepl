from abc import ABC, abstractmethod


class BaseMetric(ABC):
    @abstractmethod
    def compute(self, predictions, references, **kwargs):
        ...
