from abc import ABC, abstractmethod


class BaseEval(ABC):
    @abstractmethod
    def compute(self, sample: str, context: str) -> str:
        ...

    @abstractmethod
    def is_interactive(self) -> bool:
        ...
