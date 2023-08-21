from abc import ABC, abstractmethod


# TODO: LLM based eval should return cost
class BaseModelEval(ABC):
    @abstractmethod
    def compute(self, sample: str, context: str) -> str:
        ...
