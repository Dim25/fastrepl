from abc import ABC, abstractmethod


class BaseMetricEval(ABC):
    @abstractmethod
    def compute(self, predictions, references, **kwargs):
        ...
